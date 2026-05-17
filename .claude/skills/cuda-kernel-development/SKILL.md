---
name: cuda-kernel-development
description: Guide for development and performance optimization of CUDA kernels.
---

# Performance Optimization

Here is a list of common performance optimization techniques for CUDA kernels to follow when developing CUDA kernels. This also applies to writing the method or configuration in which the way the kernels are launched.

---

## Memory Hierarchy and Access Patterns

### Global Memory Coalescing
Threads in a warp should access consecutive memory addresses so the hardware can merge (coalesce) them into a single wide transaction. A 32-thread warp accessing 4-byte elements should touch a single 128-byte cache line. Strided or scattered access patterns issue multiple transactions, multiplying effective memory bandwidth cost. Transpose kernels and gather/scatter patterns are common coalescing pitfalls; use shared memory staging to fix them.

### Shared Memory Usage
Shared memory (SMEM) is on-chip, ~100x lower latency than global memory, and shared within a thread block. Use it to:
- Stage tiles of global data for reuse across multiple computations (tiling/blocking strategy).
- Reduce redundant global memory loads when multiple threads in a block need the same data.
- Stage results for inter-thread communication within a block.

Declare with `__shared__`. Size is configurable at launch via dynamic shared memory (`extern __shared__`). Balance occupancy against per-block SMEM usage.

### Shared Memory Bank Conflicts
Shared memory is divided into 32 banks (for 32-bit accesses). When two or more threads in a warp access addresses mapping to the same bank (but different words), the accesses serialize. Avoid by:
- Padding arrays: add one element per row (e.g., `float tile[32][33]`) to shift bank assignments across columns.
- XOR swizzling: permute the column index with the row index so threads in a warp land on distinct banks without padding overhead.
- Ensuring stride-1 or stride-coprime access patterns in SMEM.

### Vectorized Memory Access
Use 128-bit vector loads/stores (`float4`, `int4`, `uint4`, `double2`) to maximize memory transaction width per instruction. This improves throughput when data is naturally aligned and consecutive. Prefer `float4` loads to four separate `float` loads when processing four elements per thread. Cast to `float4*` after verifying alignment (16-byte aligned base pointer). Also reduces instruction count and can help hide latency through better ILP.

### L1/L2 Cache Utilization
- Use `__ldg()` (load through texture cache) for read-only data that benefits from the texture cache path (Kepler+). On Volta+ `__ldg` is less critical but still valid.
- Prefer access patterns with high spatial and temporal locality to improve L2 hit rate.
- Use `cudaFuncSetCacheConfig` or `__launch_bounds__` hints to tune the L1/shared split on architectures that allow it.

### Registers and Register Pressure
Each SM has a fixed register file. More registers per thread = fewer concurrent warps = lower occupancy. Mitigate by:
- Limiting live variables (manual register reuse, `#pragma unroll` with small bounds).
- Using `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` to cap register usage.
- Spilling to local memory (which maps to global memory) is expensive; profile with `--ptxas-options=-v` to see spill counts.

---

## Thread and Warp-Level Programming

### Warp Tiling
Assign a contiguous tile of output to each warp rather than individual elements. This enables:
- Warp-level register reuse across a tile without synchronization.
- Better exploitation of tensor cores (which operate on warp-wide data).
- Improved cache line reuse for input tiles loaded by the warp.

Common pattern: a thread block tile is subdivided into warp tiles, which are further subdivided into thread tiles (each thread owns multiple output elements in registers).

### Warp Shuffle Intrinsics
Threads in a warp can exchange data directly through registers without going through shared memory using shuffle intrinsics:
- `__shfl_sync(mask, val, srcLane)` — broadcast a value from a specific lane.
- `__shfl_xor_sync(mask, val, laneMask)` — butterfly reduction pattern.
- `__shfl_up/down_sync` — prefix scan patterns.

Use these for warp-level reductions, prefix sums, and data exchange that would otherwise require a round-trip through shared memory.

### Warp-Level Primitives (Vote and Match)
- `__ballot_sync(mask, predicate)` — returns a 32-bit mask of which lanes satisfy a predicate; useful for conditional divergence analysis.
- `__any_sync` / `__all_sync` — reduction of a boolean across a warp.
- `__match_any_sync` / `__match_all_sync` (Volta+) — find lanes with the same value; useful for work aggregation and memory coalescing in irregular kernels.

### Warp Synchronization (`__syncwarp`)
Within a warp, execution is implicitly synchronous up to Volta. From Volta onward, independent thread scheduling means warp threads can diverge. Use `__syncwarp(mask)` to synchronize lanes within a warp and ensure memory visibility before inter-lane communication. Always specify the active mask explicitly rather than relying on `0xFFFFFFFF`.

### Thread Block Synchronization
`__syncthreads()` synchronizes all threads in a block and issues a memory fence for shared memory. Use it after writing to shared memory and before reading from shared memory written by other threads. Prefer `__syncthreads()` at well-defined pipeline stages rather than scattered throughout a kernel; misplaced syncs can cause deadlock in divergent code.

### Divergence Minimization
Threads in a warp execute in lockstep (SIMT). Conditional branches where different threads in a warp take different paths (warp divergence) serialize the execution of both paths. Minimize by:
- Structuring workloads so warp threads follow the same control path.
- Moving condition checks outside inner loops.
- Using predicated execution (the compiler does this; avoid forcing branches with `__builtin_expect` unless profiled).
- Grouping similar work (e.g., via sorting or binning) before launch.

---

## Occupancy and Launch Configuration

### Occupancy Optimization
Occupancy is the ratio of active warps to maximum warps per SM. Higher occupancy helps hide latency but is not always optimal. Use `cudaOccupancyMaxPotentialBlockSize` to find the block size that maximizes occupancy. Profile with `ncu --set roofline` and the occupancy section to find whether a kernel is latency-bound or throughput-bound before tuning.

Key resources limiting occupancy per SM:
- Registers per thread (hard cap: 255).
- Shared memory per block.
- Thread blocks per SM (hardware limit per architecture).

### Grid and Block Sizing
- Block size should be a multiple of 32 (warp size). Common choices: 128, 256, 512.
- Ensure the grid covers all output elements; use ceiling division: `(N + blockDim.x - 1) / blockDim.x`.
- Persistent kernels (grid size = SM count × waves) can improve throughput for memory-bound kernels by reducing launch overhead and improving cache reuse across waves.

### Thread Coarsening
Assign multiple output elements per thread (thread coarsening) to:
- Increase arithmetic intensity and amortize memory load costs.
- Reduce the number of threads, lowering register file pressure from oversubscription.
- Enable better register-level reuse of loaded data across multiple outputs.

Typical pattern: each thread computes a small `THREAD_TILE_M × THREAD_TILE_N` output tile in registers.

---

## Instruction-Level Optimizations

### Loop Unrolling
`#pragma unroll N` unrolls inner loops by factor N, reducing loop overhead and enabling the compiler to pipeline independent instructions. Full unroll (`#pragma unroll`) is useful for small fixed-trip-count loops. Excessive unrolling increases register pressure; tune based on profiler feedback.

### Fused Multiply-Add (FMA)
The GPU executes FMA (`a * b + c`) as a single instruction. Write expressions as FMA-friendly patterns; avoid breaking them with intermediate stores. Use `__fmaf_rn` for explicit single-precision FMA when needed.

### Fast Math
`--use_fast_math` (or `__fdividef`, `__sinf`, `__expf` intrinsics) trades accuracy for throughput on transcendental operations. Appropriate for ML and graphics workloads where 24-bit mantissa precision is acceptable.

### Instruction Throughput Awareness
Integer division and modulo are expensive on GPU. Replace with bitwise ops when divisor is a power of two (`x & (N-1)` instead of `x % N`, `x >> k` instead of `x / (1<<k)`). Use this to compute bank indices, tile offsets, and warp lane IDs cheaply.

---

## Asynchronous Execution and Pipelining

### CUDA Streams
Independent kernels and memory copies in different streams can overlap execution. Use multiple streams to pipeline host-device transfers with kernel execution (copy-compute overlap). `cudaMemcpyAsync` with a non-default stream enables this.

### Double / Multi-Buffering (Software Pipelining)
While the compute stage processes tile N, the memory stage prefetches tile N+1 into a second buffer. Requires two (or more) shared memory buffers and careful synchronization. Reduces stalls caused by waiting for global memory loads to complete before computation begins.

### `cp.async` / `cuda::memcpy_async` (Ampere+)
Ampere introduced hardware-accelerated asynchronous global-to-shared memory copies via the `cp.async` PTX instruction (exposed as `cuda::memcpy_async` in the CUDA cooperative groups API, or via CUTLASS/CuTe abstractions). This decouples the global load from the SMEM write, enabling the warp to continue executing other instructions while the copy completes. Use `__pipeline_commit()` and `__pipeline_wait_prior()` (or `cuda::pipeline`) to manage the pipeline stages.

---

## Tensor Core and Matrix Multiplication

### Tensor Cores (Volta+)
Tensor cores perform 4×4 (Volta) or 16×16 (Ampere, via WMMA) matrix multiply-accumulate in a single warp-level operation. Access through:
- `nvcuda::wmma` API (all tensor-core generations, high portability).
- `mma.sync` PTX instructions (explicit, maximum control).
- CUTLASS/CuTe (recommended for production kernels).

Fragment layouts (row-major vs. column-major) must match the tensor core requirement; mismatches require layout transforms in shared memory before loading.

### WMMA Fragment Loading and Layout
Tensor core input fragments have specific memory layout requirements. Load from shared memory using `wmma::load_matrix_sync` with the correct stride. Ensure SMEM tiles are laid out to enable coalesced fragment loads; this often requires the swizzling patterns described above.

---

## Architecture-Specific Optimizations

### Ampere (SM80, A100)
- **`cp.async`**: asynchronous global-to-shared copies; enables software-pipelined GEMMs without stalling warps on loads.
- **TF32 tensor cores**: 10-bit mantissa with FP32 range; enabled via `--tf32` in cuBLAS or explicit WMMA `nvcuda::wmma::precision::tf32` fragments.
- **Sparsity (2:4 structured sparsity)**: tensor cores can operate on 50%-sparse weight matrices at 2× throughput; requires offline pruning to 2:4 pattern.
- **L2 cache partitioning**: A100 allows reserving a portion of the 40MB L2 for specific data via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` and `cudaStreamAttrValue` access policy windows. Use to pin frequently reused data (e.g., weight matrices) in L2.
- **Async copy + barrier (`cp.async.commit_group` / `cp.async.wait_group`)**: fine-grained pipeline stage management.

### Hopper (SM90, H100)
- **Tensor Memory Accelerator (TMA)**: hardware unit that performs bulk async transfers between global and shared memory with address generation offloaded from the SM. Replaces `cp.async` for large tiles. Use `cuda::experimental::tma` or CuTe `TmaCopyAtom`.
- **Warpgroup MMA (`wgmma.mma_async`)**: warp-group-level (128-thread) matrix instructions that operate directly on shared memory descriptors, enabling the SM to issue MMA and TMA concurrently. Exposed via `nvcuda::experimental::wgmma` or CuTe.
- **Distributed shared memory**: thread blocks in the same GPC can directly access each other's shared memory via a hardware switch, eliminating global memory round-trips for inter-block communication.
- **Thread block clusters**: a new hierarchy between blocks and the grid. Blocks in a cluster are co-scheduled on the same GPC. Launch with `cudaLaunchKernelEx` and `__cluster_dims__`. Use `cluster.sync()` for cluster-wide barriers.
- **FP8 tensor cores**: E4M3 and E5M2 formats for 2× throughput over FP16 at the cost of reduced precision; requires careful scaling to avoid overflow.

### Blackwell (SM100, B100/B200)
- **5th-gen tensor cores with FP4/FP6**: MX-format (microscaling) data types enable extremely high throughput for inference; block-level scaling factors are stored alongside quantized weights.
- **NVLINK 5.0 and NVSwitch**: higher bisection bandwidth for multi-GPU workloads.
- **Second-generation TMA**: enhanced TMA with multicast support, allowing a single TMA descriptor to deliver data to multiple CTAs simultaneously (reduces redundant global memory traffic in attention-like kernels).
- **Steam (Streaming Execution and Memory)**: improved overlap of compute and memory with dedicated hardware units.
- **FP8 fast accumulate**: reduced precision accumulation path for transformer inference with minimal accuracy loss when combined with dynamic scaling.

---

## Profiling and Debugging Tools

### Nsight Compute (`ncu`)
Primary per-kernel profiler. Key sections:
- `--set roofline`: plots achieved FLOPS vs. bandwidth to identify whether a kernel is compute-bound or memory-bound.
- `Memory Workload Analysis`: shows transactions, L1/L2 hit rates, SMEM bank conflicts.
- `Warp State Statistics`: shows stall reasons (long scoreboard = memory stalls, sync = `__syncthreads` stalls).
- `Source Counters`: annotates source lines with stall counts.

### Nsight Systems (`nsys`)
System-wide timeline profiler. Use to identify stream overlap, identify CPU-GPU sync points, and view kernel launch overhead.

### `nvcc` PTX and SASS Inspection
- `nvcc -ptx` / `-cubin`: generate PTX or SASS for manual inspection.
- `cuobjdump --dump-sass`: disassemble a compiled binary to SASS.
- `--ptxas-options=-v`: print register count, SMEM usage, and spill statistics at compile time.

---

## General Best Practices

- **Profile before optimizing**: use `ncu` to identify the actual bottleneck (memory bandwidth, compute, or latency) before applying any technique.
- **Roofline model**: compute arithmetic intensity (FLOPs / bytes) of your kernel and compare to the hardware roofline to understand headroom.
- **Avoid global memory atomics in hot paths**: batch atomic operations using shared memory local reduction first, then a single global atomic per block.
- **Minimize host-device synchronization**: `cudaDeviceSynchronize()` flushes the pipeline; prefer stream-based synchronization (`cudaStreamSynchronize`) and events (`cudaEventRecord` / `cudaEventSynchronize`).
- **Kernel fusion**: fuse elementwise or memory-bound operations into a single kernel to avoid round-trips through global memory. Commonly applied to activation functions, normalization, and residual additions after GEMM.
- **Use CUTLASS / CuTe for matmul kernels**: CUTLASS provides battle-tested, architecture-specialized GEMM implementations. CuTe (part of CUTLASS 3.x) provides composable tensor and copy abstractions that map directly to hardware primitives (TMA, WGMMA) and are the recommended building block for custom Hopper/Blackwell kernels.
