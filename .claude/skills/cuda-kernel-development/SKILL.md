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

### Constant Memory
`__constant__` memory is a 64KB read-only region cached in a dedicated constant cache. Reads from constant memory are broadcast to all threads in a warp for free when all threads access the same address — making it ideal for kernel parameters, lookup tables, and weight matrices that all threads share uniformly. Declare with `__constant__ float table[N];` and initialize with `cudaMemcpyToSymbol`. Divergent access (different threads reading different addresses) serializes into multiple cache transactions, so constant memory performs poorly for non-uniform access patterns.

### Texture Memory
The texture cache provides hardware-accelerated 2D spatial locality and optional hardware interpolation. On Kepler+, `__ldg()` routes loads through the texture/L1 cache for read-only global pointers, which is the simplest way to use the texture path. For structured 2D data with 2D locality, texture objects (`cudaTextureObject_t`) can outperform direct global access by exploiting 2D cache line geometry. Texture memory also provides free boundary clamping and normalized coordinate addressing for image workloads.

### Pointer Aliasing: `__restrict__`
By default, the compiler assumes output pointers may alias input pointers, preventing reordering of loads and stores. Declaring pointer parameters as `__restrict__` tells the compiler that no two pointers alias each other, enabling it to generate more aggressive load/store schedules and avoid redundant reloads. Apply to all non-aliasing pointer parameters in kernels: `__global__ void kernel(float* __restrict__ out, const float* __restrict__ in)`.

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

### Cooperative Groups
The Cooperative Groups API (`#include <cooperative_groups.h>`) provides flexible, explicit synchronization scopes:
- `cg::thread_block tb = cg::this_thread_block()` — equivalent to `__syncthreads()` scope.
- `cg::tiled_partition<N>(tb)` — creates a sub-group of N threads; supports warp-level operations (shuffle, vote) on arbitrary tile sizes.
- `cg::grid_group g = cg::this_grid()` — grid-wide sync, only valid in cooperative kernels launched with `cudaLaunchCooperativeKernel`.
- `cg::cluster_group` (Hopper+) — cluster-wide barrier for thread block clusters.

Prefer Cooperative Groups over raw `__syncthreads` / `__syncwarp` for new kernels; they make the synchronization scope explicit and compose correctly with templated tile sizes.

### Memory Fences
Memory fences control the visibility ordering of memory operations across threads, without synchronization:
- `__threadfence_block()` — ensures all prior writes to shared or global memory are visible to threads in the same block before subsequent reads.
- `__threadfence()` — device-wide fence; ensures prior writes are visible to all threads on the GPU. Required for producer-consumer patterns between blocks (e.g., a block writing results to global memory for another block to read).
- `__threadfence_system()` — system-wide fence, including host memory; needed for unified memory or mapped host memory patterns.

Fences are cheaper than full barriers (`__syncthreads`) when you only need ordering, not synchronization. Use them in lock-free data structure implementations, inter-block pipelines, and flag-based producer-consumer designs.

### Grid-Stride Loops
Instead of mapping one thread to one element, a grid-stride loop iterates over all elements with stride equal to the total grid size. This decouples problem size from grid size, enabling a fixed-size persistent grid and handling arbitrarily large inputs:
```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    // process element i
}
```
Benefits: better cache reuse across iterations within a thread, simpler launch configuration, natural fit for persistent kernel patterns. Use when N is much larger than the maximum grid size or when tuning the grid for occupancy independently of problem size.

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

### Inline PTX
When CUDA C++ lacks an intrinsic for a hardware feature, inline PTX provides direct access:
```cpp
asm volatile("op.modifier dest, src;" : "=r"(out) : "r"(in));
```
Common uses: explicit prfm prefetch instructions, special register reads (`%laneid`, `%smid`, `%clock64`), control of rounding modes, and accessing new hardware features before the CUDA toolkit exposes them in C++. Use `asm volatile` (not `asm`) to prevent the compiler from reordering or eliminating the instruction. Keep inline PTX minimal and well-commented — it is not portable across architectures.

### Prefetch Intrinsics
Explicitly issue prefetches to pull data into L1 or L2 cache ahead of when it is needed:
- `asm("prefetch.global.L1 [%0];" :: "l"(ptr))` — L1 prefetch (PTX).
- `__builtin_prefetch(ptr, 0, 3)` — compiler prefetch hint (less reliable on GPU).

Prefetching is most effective when there is a known, predictable access pattern and sufficient arithmetic to hide the prefetch latency. Ampere `cp.async` is generally preferred over software prefetch for structured tile loads.

### Half-Precision (FP16/BF16) Arithmetic
`__half` (FP16) and `__nv_bfloat16` (BF16) types offer 2× throughput and 2× storage density compared to FP32 on Volta+ hardware. Key points:
- Use `half2` / `__nv_bfloat162` packed types for 2-wide SIMD operations on 16-bit data — nearly 2× additional throughput.
- FP16 has 10-bit mantissa (limited dynamic range); BF16 has 7-bit mantissa but FP32 exponent range (preferred for training).
- Accumulate in FP32 for numerical stability: compute in FP16, accumulate with `float` accumulators, convert output with `__float2half`.
- Tensor cores natively consume FP16/BF16 inputs and accumulate in FP32; explicit mixed-precision is the default for transformer workloads.

---

## Asynchronous Execution and Pipelining

### CUDA Streams
Independent kernels and memory copies in different streams can overlap execution. Use multiple streams to pipeline host-device transfers with kernel execution (copy-compute overlap). `cudaMemcpyAsync` with a non-default stream enables this.

### Double / Multi-Buffering (Software Pipelining)
While the compute stage processes tile N, the memory stage prefetches tile N+1 into a second buffer. Requires two (or more) shared memory buffers and careful synchronization. Reduces stalls caused by waiting for global memory loads to complete before computation begins.

### Atomic Operations
Atomic operations serialize concurrent updates to shared locations. Use correctly to avoid both data races and unnecessary serialization:
- **Scope matters**: prefer `atomicAdd_block` (block-scope) or `atomicAdd_system` (system-scope, Unified Memory) over the default device-scope when the contention is local. Block-scope atomics are dramatically faster.
- **Reduction pattern**: for global reductions, perform a partial reduction in shared memory first (using shuffle or `__syncthreads`), then issue one atomic per block to global memory. Never issue one atomic per thread for reductions.
- **Ordering**: atomics are not a memory fence by default; use `__threadfence()` before/after an atomic that signals another block.
- **`atomicCAS` for lock-free patterns**: compare-and-swap is the building block for lock-free queues and work stealing, but spin-waiting in a kernel stalls warps — prefer GPU-native work distribution (e.g., cooperative launch, device-side queues) when possible.
- **FP32 atomics**: `atomicAdd` on `float*` is natively supported from Kepler+; use it without emulation. FP16 atomics require Volta+.

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

## Host-Device Transfer Optimization

### Pinned (Page-Locked) Memory
`cudaMallocHost` / `cudaHostAlloc` allocate page-locked host memory that the DMA engine can access directly, bypassing the OS page-fault mechanism. This increases peak host-to-device bandwidth by 2–4× compared to pageable memory. Pinned memory is a limited resource; allocate it for long-lived transfer buffers only, not temporary allocations. Free with `cudaFreeHost`.

### Asynchronous Transfers and Copy-Compute Overlap
`cudaMemcpyAsync` with a non-default stream issues the transfer without blocking the host or other streams. Overlap execution pattern:
1. Stream A: `cudaMemcpyAsync` H2D for batch N.
2. Stream B: kernel processing batch N-1.
3. Alternate between two streams for continuous overlap.

Requires pinned host memory for the source buffer. Profile with Nsight Systems to verify actual overlap.

### Unified Memory and Prefetching
`cudaMallocManaged` allocates memory accessible from both CPU and GPU, with the runtime migrating pages on demand. On-demand migration causes page faults and is slow for production workloads. Mitigate with explicit prefetch:
- `cudaMemPrefetchAsync(ptr, size, device, stream)` — migrate pages to the target device before the kernel runs.
- `cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device)` — hint that data will be read-only; the runtime may create read-only replicas on multiple devices.

Unified Memory is useful for code simplicity and for workloads that access only a subset of a large dataset, but explicit `cudaMalloc` + `cudaMemcpy` gives higher and more predictable bandwidth.

### Zero-Copy Memory
`cudaHostAlloc` with `cudaHostAllocMapped` creates host memory directly accessible from the GPU over PCIe without a copy. Useful when data is accessed only once or sparsely by the GPU (avoids the copy cost), but peak bandwidth is limited to PCIe bandwidth (~32 GB/s on PCIe 5.0 vs. 3+ TB/s HBM). Best for streaming workloads that are not bandwidth-bound on the GPU side.

---

## CUDA Graphs

### Motivation
Every CUDA kernel launch has ~5–10 µs of CPU-side overhead for driver work. For workloads with many short kernels (inference pipelines, frame-rate-sensitive simulations), this overhead dominates. CUDA Graphs capture a sequence of kernels, memcopies, and dependencies as a graph, and re-execute the entire graph with a single `cudaGraphLaunch` call after the first run, reducing launch overhead to ~1–2 µs total.

### Graph Capture
```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// issue kernels and memcpies as usual
kernelA<<<grid, block, 0, stream>>>();
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
kernelB<<<grid, block, 0, stream>>>();
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
```
After capture, call `cudaGraphLaunch(graphExec, stream)` on every iteration. Destroy with `cudaGraphExecDestroy` / `cudaGraphDestroy`.

### Graph Updates
When only kernel parameters (not graph topology) change between iterations, use `cudaGraphExecKernelNodeSetParams` to update a node without re-instantiating the graph. For structural changes (different number of nodes or edges), re-capture. Graph update is cheaper than re-instantiation.

### Limitations
- Streams used during capture must be non-default (created with `cudaStreamNonBlocking` or `cudaStreamCreate`).
- CPU callbacks inside a captured region execute asynchronously as host nodes — do not use them for timing or debug output.
- Dynamic grid/block sizes require re-capture or explicit node parameter updates.

### When to Use
Apply CUDA Graphs when kernel launch overhead is measurable (many short kernels, high iteration rate). Avoid for kernels with highly dynamic shapes that require frequent re-capture.

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

### Compute Sanitizer
`compute-sanitizer` (replaces `cuda-memcheck`) detects memory errors at runtime:
- `compute-sanitizer --tool memcheck`: out-of-bounds accesses, uninitialized reads, misaligned accesses.
- `compute-sanitizer --tool racecheck`: shared memory data races (missing `__syncthreads`).
- `compute-sanitizer --tool synccheck`: divergent `__syncthreads` calls.
- `compute-sanitizer --tool initcheck`: use of uninitialized device memory.

Run these early during kernel development — they catch bugs that are invisible to correctness tests on small inputs. Sanitizer mode adds 10–100× overhead; use only for debugging, not benchmarking.

### NVTX Annotations
The NVIDIA Tools Extension (`nvtx3/nvToolsExt.h`) lets you annotate regions of host and device code with named markers that appear in Nsight Systems timelines:
```cpp
nvtxRangePushA("forward_pass");
kernel<<<grid, block>>>();
nvtxRangePop();
```
Use NVTX to correlate high-level algorithm stages with kernel execution in the timeline, making it easy to identify which stage is causing latency. Zero overhead when profiling is disabled; link with `-lnvToolsExt`.

---

## Common Algorithm Patterns

### Parallel Reduction
The canonical GPU reduction proceeds in two phases: (1) warp-level reduction using shuffle intrinsics, then (2) block-level reduction through shared memory, then (3) one atomic per block to global memory.
```cpp
// Warp reduction (no SMEM needed)
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
// Lane 0 holds the warp partial sum; write to SMEM, then reduce across warps
```
Avoid the naive tree-reduction-in-SMEM pattern (requires many `__syncthreads`); the shuffle-first approach halves the number of sync points and eliminates most SMEM bank conflicts.

### Prefix Scan (Inclusive/Exclusive)
Block-wide scans use the Kogge-Stone or Blelloch (up-sweep / down-sweep) pattern in shared memory. For large arrays, use a two-pass approach: (1) block-local scan, (2) scan of per-block totals, (3) add block offsets back. CUB (`cub::BlockScan`, `cub::DeviceScan`) provides optimized implementations; prefer them over hand-rolled scans unless the scan is embedded inside a larger custom kernel.

### Histogram
Naive per-thread global atomics serialize badly under contention. Better approaches:
- **Private SMEM histograms**: each block maintains a local histogram in shared memory (using `atomicAdd` on SMEM), then merges to global at the end. Reduces global contention by the block count.
- **Subwarp privatization**: multiple private copies per block (one per warp or per N threads) to further reduce SMEM conflicts.
- **Sorting-based**: sort elements by bin key, then use a segmented reduce. Avoids atomics entirely but adds sort overhead.

### Persistent Kernels
A persistent kernel launches exactly as many thread blocks as the GPU can run simultaneously (SM count × max blocks per SM), then processes work items from a device-side queue in a loop. Benefits:
- Eliminates repeated kernel launch overhead for iterative algorithms.
- Improves L2 cache reuse across iterations.
- Enables dynamic load balancing (fast blocks pull more work).

Pattern: use an atomic counter in global memory as a work queue index. Each block atomically claims the next chunk of work with `atomicAdd`. Requires careful memory ordering (`__threadfence`) between work production and consumption.

---

## General Best Practices

- **Profile before optimizing**: use `ncu` to identify the actual bottleneck (memory bandwidth, compute, or latency) before applying any technique.
- **Roofline model**: compute arithmetic intensity (FLOPs / bytes) of your kernel and compare to the hardware roofline to understand headroom.
- **Avoid global memory atomics in hot paths**: batch atomic operations using shared memory local reduction first, then a single global atomic per block.
- **Minimize host-device synchronization**: `cudaDeviceSynchronize()` flushes the pipeline; prefer stream-based synchronization (`cudaStreamSynchronize`) and events (`cudaEventRecord` / `cudaEventSynchronize`).
- **Kernel fusion**: fuse elementwise or memory-bound operations into a single kernel to avoid round-trips through global memory. Commonly applied to activation functions, normalization, and residual additions after GEMM.
- **Use CUTLASS / CuTe for matmul kernels**: CUTLASS provides battle-tested, architecture-specialized GEMM implementations. CuTe (part of CUTLASS 3.x) provides composable tensor and copy abstractions that map directly to hardware primitives (TMA, WGMMA) and are the recommended building block for custom Hopper/Blackwell kernels.
- **Use CUB for device-wide primitives**: `cub::DeviceReduce`, `cub::DeviceScan`, `cub::DeviceSort`, and `cub::DeviceHistogram` provide architecture-tuned implementations of common collective operations. Prefer them over hand-rolled block-level collectives for anything device-wide.
- **Memory pool allocation**: `cudaMallocAsync` / `cudaFreeAsync` (CUDA 11.2+) use a per-stream memory pool that avoids synchronizing the device on every allocation/free. Essential for inference loops and dynamic workloads that allocate many small tensors per iteration.
