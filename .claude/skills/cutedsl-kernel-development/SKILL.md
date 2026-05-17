---
name: cutedsl-kernel-development
description: Guide for development and performance optimization of CuTeDSL kernels.
---

# Performance Optimization

CuTeDSL is the Python-embedded DSL for NVIDIA's CuTe (part of CUTLASS 3.x). It lets you write GPU kernels in Python that compile to PTX/SASS via MLIR. CuTe's core abstraction is the **layout** — a function from a logical coordinate space to a flat memory offset — and composition of layouts expresses everything from tiling to warp-level partitioning. Performance requires understanding how layouts map to hardware: memory transactions, tensor cores, TMA, and register allocation.

---

## Core Programming Model

### `@cute.jit` Kernels
Decorate a Python function with `@cute.jit` to compile it as a GPU kernel. Inputs are typed with `cute.Tensor`, `cute.Int32`, `cute.Float32`, etc. The function body executes per-thread; use `cute.arch.thread_idx()` and `cute.arch.block_idx()` to get the CUDA thread/block coordinates.

```python
import cutlass.cute as cute

@cute.jit
def my_kernel(A: cute.Tensor, B: cute.Tensor, N: cute.Int32):
    tid = cute.arch.thread_idx().x
    bid = cute.arch.block_idx().x
    ...
```

Call site: wrap the kernel in a host function that constructs `cute.Tensor` objects from raw pointers and launches with a grid/block config.

### Kernel Launch Pattern
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def kernel(A: cute.Tensor, ...):
    ...

def solve(a: torch.Tensor, ...):
    # Build cute.Tensor wrappers
    A = cute.make_tensor(a.data_ptr(), cute.make_layout(...))
    grid  = (cdiv(M, BM), cdiv(N, BN), 1)
    block = (NTHREADS, 1, 1)
    kernel[grid, block](A, ...)
```

---

## Layout System

CuTe layouts are the foundation of every optimization. A layout is `(shape, stride)` and maps a logical index to a memory offset.

### `make_layout`, `make_shape`, `make_stride`
```python
layout = cute.make_layout(
    cute.make_shape(M, N),
    cute.make_stride(N, 1)   # row-major: stride along M is N, along N is 1
)
```
Column-major (Fortran order): `make_stride(1, M)`.

### Hierarchical (Nested) Shapes
Shapes and strides can be nested tuples, expressing multi-level tiling directly in the layout. This is CuTe's killer feature: a `(BM, BN) : (N, 1)` layout and a `((BM//WM, WM), (BN//WN, WN)) : ((WM*N, N), (WN, 1))` layout are related by layout algebra, not by index arithmetic in your code.

### `make_tensor`
Attach a layout to a pointer to produce a tensor:
```python
gA = cute.make_tensor(ptr, cute.make_layout(cute.make_shape(M, K), cute.make_stride(K, 1)))
```
For shared memory tensors, use a shared memory allocator (see Shared Memory section).

### `local_tile`
Slice a global tensor into a tile owned by the current thread block:
```python
# gA has shape (M, K); extract (BM, BK) tile at block (bm, bk)
tA = cute.local_tile(gA, cute.make_shape(BM, BK), cute.make_coord(bm, bk))
```
`local_tile` returns a view — no data is copied; only the layout offset changes.

### `local_partition`
Partition a tile across threads so each thread owns a disjoint subtile:
```python
# Partition (BM, BK) tile across (NTHREADS,) threads
thr_A = cute.local_partition(tA, cute.make_layout(NTHREADS), thread_idx)
```
For 2D thread layouts (warps × lanes), pass a 2D thread layout.

---

## Memory Access and Copy

### TiledCopy
A `TiledCopy` describes how a group of threads cooperatively move a tile of data. You pick a **copy atom** (a hardware instruction like `cp.async`, `ldmatrix`, TMA) and then tile it across threads and the problem shape.

```python
copy_atom = cute.make_tiled_copy(
    cute.Copy_Atom(cute.AutoVectorizingCopyWithAssumedAlignment(128), dtype),
    thr_layout,   # layout of participating threads
    val_layout,   # elements per thread per instruction
)
```

Call `cute.copy(tiled_copy, src, dst)` to issue the cooperative copy.

### Global-to-Shared Copy (Coalesced Loads)
Structure the thread-to-element mapping so consecutive thread IDs map to consecutive memory addresses. The `AutoVectorizingCopy` atom selects the widest possible vectorized load (up to 128-bit / `float4`) automatically when alignment allows.

Always ensure the global tensor's innermost stride is 1 (contiguous in the fast dimension) for coalesced access. If inputs arrive transposed, apply the transposition in SMEM using a swizzled layout rather than in global memory.

### `ldmatrix` for Tensor Core Inputs
On Ampere/Hopper, use `ldmatrix` to load SMEM fragments directly into the tensor core register layout without explicit index arithmetic:
```python
copy_atom = cute.Copy_Atom(cute.SM80_CP_ASYNC_CACHEGLOBAL, dtype)   # global→SMEM
ldmatrix_atom = cute.Copy_Atom(cute.SM75_U32x4_LDSM_N, dtype)       # SMEM→registers
```
`ldmatrix` loads 8×8 or 16×8 tiles in the exact register layout expected by `mma` instructions, eliminating manual swizzle/permute steps.

---

## Shared Memory

### Allocating SMEM Tensors
Use the SMEM allocator inside `@cute.jit` functions:
```python
smem_A = cute.make_tensor(
    cute.make_smem_ptr(smem_buf, dtype),
    cute.make_layout(cute.make_shape(BM, BK), cute.make_stride(BK, 1))
)
```
Size the SMEM buffer at launch: `smem_bytes = (BM * BK + BK * BN) * sizeof(dtype)`.

### Swizzled SMEM Layouts (Bank Conflict Elimination)
Raw row-major SMEM layouts cause bank conflicts when loading columns (e.g., fragment loads for tensor cores). CuTe's **swizzle** applies an XOR permutation to column indices so that all threads in a warp land on distinct banks.

```python
swizzle = cute.make_swizzle(M=3, S=3, B=3)  # parameters depend on dtype and tile size
layout  = cute.composition(swizzle, cute.make_layout(...))
```
For FP16 tiles of width 64: `Swizzle<3,3,3>` is the canonical choice. For FP32 width 32: `Swizzle<2,3,2>`. Prefer swizzled layouts over padding (`+1` column tricks) for tensor core workloads because swizzling preserves the tile's power-of-two size, which is required by `ldmatrix` and `mma`.

### `__syncthreads` Equivalent
Call `cute.arch.syncthreads()` (maps to `__syncthreads`) after cooperative SMEM writes before any thread reads data written by another thread.

---

## MMA (Tensor Core) Operations

### TiledMMA
Describes how to tile an MMA atom across threads and the output matrix:
```python
tiled_mma = cute.make_tiled_mma(
    cute.MMA_Atom(cute.SM80_16x8x16_F32F16F16F32_TN()),  # Ampere FP16 tensor core
    cute.make_layout(cute.make_shape(WARPS_M, WARPS_N)),  # warp layout
    cute.make_layout(cute.make_shape(MMA_M, MMA_N)),      # MMA tile repetitions per warp
)
```

Common atoms:
- Ampere FP16 → FP32: `SM80_16x8x16_F32F16F16F32_TN`
- Ampere BF16 → FP32: `SM80_16x8x16_F32BF16BF16F32_TN`
- Hopper FP16 → FP32 (WGMMA): `SM90_64x64x16_F32F16F16_SS`

### `cute.gemm`
Issue the tensor core MMA using partitioned register fragments:
```python
cute.gemm(tiled_mma, acc_frag, A_frag, B_frag, acc_frag)
```
`acc_frag`, `A_frag`, `B_frag` must already be partitioned per thread via `tiled_mma.partition_C / partition_A / partition_B`. All layouts must match what the atom expects — CuTe's layout algebra tracks this automatically if you use `tiled_mma` consistently.

### Accumulator Initialization
```python
acc = cute.make_tensor(cute.make_fragment_like(tiled_mma.partition_C(gC)))
cute.fill(acc, 0.0)
```

---

## Software Pipelining

### Double Buffering (Ampere)
Overlap global→SMEM copies with tensor core compute using two alternating SMEM buffers:
```python
# smem has shape (2, BM, BK) — ping-pong buffers
smem = cute.make_tensor(..., cute.make_shape(2, BM, BK), ...)

for k in range(num_tiles):
    buf      = k % 2
    next_buf = (k + 1) % 2
    cute.copy(copy_atom, gA[k+1], smem_A[next_buf])  # prefetch next tile
    cute.arch.syncthreads()
    cute.gemm(tiled_mma, acc, smem_A[buf], smem_B[buf], acc)  # compute on current
    cute.arch.syncthreads()
```

### `cp.async` Pipelining (Ampere)
Use async copy atoms (`SM80_CP_ASYNC_CACHEGLOBAL`) so the load instruction returns immediately while the data transfer continues in the background. Commit groups and wait:
```python
cute.copy(async_copy_atom, src, smem_dst)   # issues cp.async, non-blocking
cute.arch.cp_async_commit_group()            # commit current group
cute.arch.cp_async_wait_group(0)             # wait until 0 groups are pending
cute.arch.syncthreads()
```
With `wait_group(1)` (instead of 0) you wait only until there is at most 1 pending group, allowing compute to run while the next group transfers.

### TMA Pipelining (Hopper)
On SM90, use TMA copy atoms and pipeline barrier objects:
```python
tma_atom = cute.Copy_Atom(cute.SM90_TMA_LOAD(), dtype)
barrier  = cute.arch.ClusterTransactionBarrier()
# Producer warp issues TMA; consumer warps wait on barrier
cute.copy(tma_atom, desc, smem_dst)
barrier.arrive_and_expect_tx(expected_bytes)
barrier.wait(phase)
```
TMA off-loads address generation from SMs, enabling higher MMA utilization on Hopper.

---

## Reduction Operations

### Warp-Level Reduction
CuTeDSL exposes warp shuffle intrinsics for within-warp reductions without SMEM:
```python
val = cute.arch.shfl_xor_sync(0xFFFFFFFF, val, 16)
val = cute.arch.shfl_xor_sync(0xFFFFFFFF, val, 8)
val = cute.arch.shfl_xor_sync(0xFFFFFFFF, val, 4)
val = cute.arch.shfl_xor_sync(0xFFFFFFFF, val, 2)
val = cute.arch.shfl_xor_sync(0xFFFFFFFF, val, 1)
# lane 0 now holds the warp-wide reduction
```

### Block-Level Reduction via SMEM
Write warp partial results to SMEM, synchronize, then have one warp reduce them:
```python
smem_partial[warp_id] = warp_sum          # one value per warp
cute.arch.syncthreads()
if warp_id == 0:
    total = smem_partial[lane_id]          # each lane of warp 0 reads one partial
    # apply warp reduction again
```

### Online Softmax (Flash Attention Pattern)
Maintain running `(max, sum)` statistics as tiles are loaded; rescale the accumulator when the running max increases:
```python
m_old = m_new
m_new = max(m_old, row_max(scores_tile))
scale = exp(m_old - m_new)
acc   = acc * scale + exp(scores_tile - m_new) @ V_tile
lse   = lse * scale + row_sum(exp(scores_tile - m_new))
```
Final output: `acc / lse`. This single-pass algorithm is essential for flash attention; it avoids materializing the full attention matrix.

---

## Program Ordering and L2 Reuse

### Swizzled / Grouped Grid Ordering
Default row-major program ordering causes every CTA row to be scheduled before the next, destroying L2 reuse of the B (or K/V) matrix. Use a swizzled ordering where consecutive CTA IDs share the same row of B tiles:
```python
# Python host code
pid_m = (pid % num_pid_in_group) % group_m
pid_n = (pid % num_pid_in_group) // group_m
# group_m ≈ 8 works well for most GEMM shapes
```
This is exactly the Triton grouped-GEMM trick; apply the same idea when launching CuTeDSL kernels.

### Persistent Kernels
Launch exactly `SM_count × waves` blocks. Each block loops over multiple output tiles using an atomic counter as a work queue. This improves L2 hit rate across tiles and amortizes launch overhead for many-tile problems.

---

## Dtype and Precision

### FP16 / BF16 Input, FP32 Accumulation
Always accumulate in FP32; convert inputs to FP16/BF16 only at load time:
```python
a_frag_f16 = cute.cast(a_frag, cute.Float16)
cute.gemm(tiled_mma, acc_f32, a_frag_f16, b_frag_f16, acc_f32)
```
Tensor core atoms enforce this automatically: `F32F16F16F32` accumulates FP16 inputs into FP32 accumulators.

### FP8 (Hopper / Blackwell)
Use `SM90_64x64x32_F32E4M3E4M3_SS` or similar atoms for FP8 WGMMA. Apply per-tensor or per-block scaling factors to compensate for the reduced dynamic range. CuTe handles the register layout; you supply the scale tensors.

### TF32 (Ampere)
FP32 inputs can use TF32 tensor cores via `SM80_16x8x8_F32TF32TF32F32_TN`. This gives ~8× higher throughput than CUDA cores at the cost of ~3 bits of mantissa. Enable when numerical accuracy requirements allow.

---

## Tile Size Selection

| Kernel type | BM | BN / BK | Warps | Notes |
|---|---|---|---|---|
| GEMM (large) | 128 | 128 / 64 | 4–8 | Classic Ampere config |
| GEMM (thin) | 64 | 128 / 32 | 4 | For small M (inference) |
| Attention Q×K | 64–128 | 64 / 64 | 4 | seqlen tile |
| Softmax / norm | —  | 256–1024 | 4–8 | 1D reduction per row |

Tile sizes must be compile-time constants (they fold into the layout algebra). Tune via `constexpr` template parameters or Python-level autotuning loops.

---

## Architecture-Specific Features

### Ampere (SM80, A100)
- `cp.async`: async global→SMEM copy; non-blocking warp.
- `ldmatrix`: SMEM→registers in MMA layout, 4 or 16 `uint32` in one instruction.
- TF32 tensor cores via `SM80_16x8x8_F32TF32TF32F32_TN` atom.
- L2 cache partitioning: pin weight matrices with `cudaStreamAttrValue` access policy.

### Hopper (SM90, H100)
- **TMA**: hardware unit off-loads address generation for bulk SMEM fills. Use `SM90_TMA_LOAD` / `SM90_TMA_STORE` copy atoms. Requires a TMA descriptor built on the host.
- **WGMMA** (`wgmma.mma_async`): warp-group-level (128 threads) async MMA that reads operands from SMEM descriptors, enabling full compute/memory overlap. Use `SM90_*_SS` MMA atoms.
- **Thread block clusters**: co-schedule up to 16 blocks on the same GPC for distributed shared memory access and TMA multicast. Launch with `cluster_dims` in the kernel launch config.
- **Pipeline barrier** (`cutlass::arch::ClusterTransactionBarrier`): replaces `cp.async.wait_group` for TMA-based pipelines.
- Prefer 3–5 pipeline stages (double or triple buffering) on Hopper to fully hide TMA latency.

### Blackwell (SM100, B200)
- 5th-gen tensor cores with FP4/FP6 (MX formats) for inference.
- Second-generation TMA with multicast: one descriptor can fill SMEM in multiple CTAs simultaneously.
- FP8 fast accumulate path for transformer inference.

---

## Debugging and Profiling

### Correctness First
Use `torch.testing.assert_close` or `np.testing.assert_allclose` with a PyTorch reference before any performance tuning. CuTe's layout errors produce wrong outputs, not crashes.

### `cute.printf` / `print` Inside Kernels
```python
cute.arch.printf("tid=%d val=%f\n", tid, val)
```
Disable before benchmarking — device printf serializes across the GPU.

### Nsight Compute (`ncu`)
Profile CuTeDSL-generated kernels the same as any CUDA kernel:
```bash
ncu --set full python my_kernel.py
```
Key sections:
- **Memory Workload Analysis**: L1/L2 hit rates, SMEM bank conflicts, transaction counts.
- **Compute Workload Analysis**: MMA pipe utilization, warp stall reasons.
- **Roofline**: determine if compute-bound or bandwidth-bound.
- **Source / PTX**: inspect generated PTX to verify vector loads, `ldmatrix`, and `cp.async` are emitted.

### Inspecting Generated Code
CuTeDSL compiles through MLIR; enable verbose output to inspect the IR or PTX when debugging unexpected performance:
```python
import os
os.environ["CUTE_JIT_VERBOSE"] = "1"
```

### Benchmarking
```python
import torch
starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)
for _ in range(10): my_kernel(...)    # warmup
starter.record()
for _ in range(100): my_kernel(...)
ender.record()
torch.cuda.synchronize()
ms = starter.elapsed_time(ender) / 100
```
Compare achieved throughput (TFLOPS or GB/s) against the hardware roofline, not just wall time.

---

## Common Algorithm Patterns

### Tiled GEMM (SMEM-Staged)
```
for k_tile in range(K // BK):
    cooperative_copy(gA[bm, k_tile] → smA)     # BM×BK global→SMEM
    cooperative_copy(gB[k_tile, bn] → smB)     # BK×BN global→SMEM
    syncthreads()
    for k in range(BK // MMA_K):
        load_fragment(smA, regA)                # SMEM→registers via ldmatrix
        load_fragment(smB, regB)
        mma(acc, regA, regB)                    # tensor core MMA
    syncthreads()
store(acc → gC[bm, bn])
```

### Flash Attention
```
for kv_tile in range(seq_len // BN):
    load K_tile, V_tile into SMEM
    scores = Q_tile @ K_tile.T               # BM×BN score tile
    apply causal mask (if needed)
    online_softmax_update(scores, m, lse)    # rescale acc in-place
    acc += softmax(scores) @ V_tile
output = acc / lse
```
Key optimizations: fuse the softmax into the Q×K loop to avoid materializing `seq_len × seq_len` attention weights. Use BN = 64 or 128 for the KV tile; larger BN improves MMA utilization but increases SMEM pressure.

### Parallel Reduction (Row-wise)
```
# Each block reduces one or more rows
partial = warp_reduce(row_elements[lane_id::WARP_SIZE])
smem_partials[warp_id] = partial
syncthreads()
if warp_id == 0:
    result = warp_reduce(smem_partials[lane_id])
    if lane_id == 0: out[row] = result
```

---

## General Best Practices

- **Use layout algebra, not index arithmetic**: express tiling, partitioning, and transposition as layout compositions. Index arithmetic in loops is a sign you should encode the pattern in the layout instead.
- **Start with the MMA atom and work outward**: pick the MMA atom first, derive the thread-level register layout from it, then design the SMEM and global-copy layouts to feed it correctly.
- **Match copy and MMA tile sizes**: `BM × BK` for SMEM-A and the MMA atom's `M × K` must be consistent so that fragment loads map directly from SMEM without redundant stores.
- **Use swizzled SMEM layouts for all tensor core operand tiles**: bank conflicts in SMEM loads are the single most common performance killer in GEMM-like kernels.
- **Fuse elementwise ops into the epilogue**: apply bias, activation, and output scaling in the register-level epilogue before writing to global memory. This avoids a second kernel launch and a global memory round-trip.
- **Prefer `cp.async` / TMA over synchronous loads**: synchronous global loads stall the warp; async copies allow computation to proceed in parallel.
- **Profile before guessing**: SMEM bank conflicts, uncoalesced global loads, and low MMA utilization are distinct bottlenecks with distinct fixes. Nsight Compute tells you which is dominant.
- **Validate layout correctness on small problems**: test with M=N=K=16 before scaling up. Layout bugs produce subtle wrong-output errors that are easier to debug at small scale.
- **Reference CUTLASS examples**: CUTLASS 3.x ships reference kernels for GEMM, convolution, and attention using CuTe. The `examples/` directory is the canonical source for layout configurations tuned per architecture.
