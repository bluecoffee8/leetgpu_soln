---
name: gluon-kernel-development
description: Guide for development and performance optimization of gluon DSL kernels.
---

# Performance Optimization

Gluon is Triton's lower-level GPU programming model. It shares Triton's compiler stack and Python syntax but targets the `ttg` (Triton GPU) IR directly, bypassing the standard `tt` IR layer. This grants explicit control over tensor layouts, shared memory allocation, warp specialization, and architecture-specific instructions — at the cost of portability. Use Gluon when Triton's automatic optimizations leave a meaningful performance gap and you need to close it with hardware-specific tuning.

---

## Core Programming Model

### `@gluon.jit` Kernels
Decorate a Python function with `@gluon.jit` to compile it as a GPU kernel. Each kernel invocation corresponds to one thread block (CTA) on the GPU. Inputs arrive as raw global memory pointers; tensor arguments must be wrapped in `TensorDescriptor` objects for TMA-based access.

```python
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

@gluon.jit
def my_kernel(in_ptr, out_ptr, N: gl.constexpr, BLOCK: gl.constexpr):
    pid    = gl.program_id(0)
    layout = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    offs   = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask   = offs < N
    x = gl.load(in_ptr + offs, mask=mask)
    gl.store(out_ptr + offs, x, mask=mask)
```

### Kernel Launch Pattern
```python
def solve(x: torch.Tensor) -> torch.Tensor:
    out  = torch.empty_like(x)
    N    = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    my_kernel[grid](x, out, N, BLOCK, num_warps=4)
    return out
```

`num_warps` and `maxnreg` are passed as launch-time keywords; they are not kernel arguments.

---

## Layout System

Layouts are Gluon's central abstraction. A layout defines how the elements of a tile are distributed across threads, warps, and CTAs. Every tensor in Gluon has a layout; mismatches require an explicit `gl.convert_layout()` call.

### `BlockedLayout` — The Standard Layout
```python
gl.BlockedLayout(
    size_per_thread=[1, 4],    # elements per thread per dimension
    threads_per_warp=[16, 2],  # thread distribution within one warp
    warps_per_cta=[2, 2],      # warp distribution within the CTA
    order=[1, 0],              # [1,0] = row-major, [0,1] = col-major
)
```
- All values must be powers of two.
- `order` controls which dimension varies fastest and thus coalescing.
- A helper for coalesced access to row-major 2D tensors:
  ```python
  def coalesced_layout_2d(num_warps):
      return gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
  ```

### `SliceLayout` — Reduce a Dimension
```python
row_layout = gl.SliceLayout(dim=1, parent=layout_2d)
col_layout = gl.SliceLayout(dim=0, parent=layout_2d)
```
Use when computing per-row or per-column offsets from a 2D layout.

### `NVMMADistributedLayout` — Tensor Core Accumulator
Required for the output of WGMMA (Hopper) or TMEM (Blackwell) MMA operations.
```python
c_layout = gl.NVMMADistributedLayout(
    version=[3, 0],             # [2,0]=Ampere, [3,0]=Hopper WGMMA
    warps_per_cta=[4, 1],
    instr_shape=[16, N, K],     # [m, n, k] of the MMA instruction
)
```

### `DotOperandLayout` — Tensor Core Inputs in Registers
For loading the A operand into registers for WGMMA:
```python
a_reg_layout = gl.DotOperandLayout(
    operand_index=0,            # 0=A, 1=B
    parent=c_layout,
    k_width=32 // dtype.primitive_bitwidth,
)
```

### `DistributedLinearLayout` — Maximum Expressiveness
Encodes the lane/warp/block basis vectors explicitly. Zero-cost reshape, permute, and transpose operations are possible when the layout allows it.
```python
gl.DistributedLinearLayout(
    reg_bases=[],
    lane_bases=[[1], [2], [4], [8], [16]],
    warp_bases=[[32], [64]],
    block_bases=[],
    shape=[128],
)
```

### Layout Conversion
```python
value = gl.convert_layout(value, target_layout)
# assert_trivial=True to catch unexpected SMEM round-trips at compile time:
value = gl.convert_layout(value, target_layout, assert_trivial=True)
```
Prefer layouts that make all inter-thread communication trivial. Conversions that require shared memory are expensive.

### Tensor Creation with Layouts
```python
indices = gl.arange(0, BLOCK, layout=layout)
zeros   = gl.zeros((M, N), dtype=gl.float32, layout=c_layout)
```

---

## Memory Operations

### Global Memory Load/Store
```python
x    = gl.load(ptr + offsets, mask=mask, other=0.0)
gl.store(ptr + offsets, value, mask=mask)
```
Ensure the innermost stride of the access pattern is 1 and the layout's `order` matches for full coalescing.

### Shared Memory Allocation
```python
smem = gl.allocate_shared_memory(dtype, shape, layout)
```
`layout` here is a shared memory layout (e.g., `NVMMASharedLayout`). The allocation is scoped to the kernel invocation.

Common shared memory layouts:
- `gl.NVMMASharedLayout.get_default_for(shape, dtype)` — for WGMMA B operand.
- `gl.SwizzledSharedLayout(...)` — manual XOR swizzle to avoid bank conflicts.
- `gl.PaddedSharedLayout(...)` — column padding alternative to swizzling.

### `shared_memory_descriptor` Load/Store
Shared memory buffers are accessed as descriptors, not raw pointers.
```python
# Store from registers to SMEM
smem.store(reg_tensor)

# Load from SMEM to registers (target_layout specifies register layout)
reg_tensor = smem.load(target_layout)

# Index into a multi-buffered SMEM (e.g., shape=[2, M, K])
buf = smem.index(i % 2)
```

### `fence_async_shared()`
Required to order CPU-proxy (register/SMEM) stores against async-proxy (TMA) operations on the same shared memory region.
```python
smem.store(value)
fence_async_shared()                          # fence before TMA async store
tma.async_copy_shared_to_global(desc, coord, smem)
```
Also required when a TMA load into SMEM is followed by a register load from that SMEM that a non-TMA instruction will consume.

---

## TMA (Tensor Memory Accelerator) — Hopper+

TMA off-loads N-dimensional address generation from SMs to a dedicated hardware unit, dramatically increasing the effective bandwidth for structured tile loads and stores.

### TensorDescriptor
Built on the host from a PyTorch tensor and a tile shape:
```python
from triton.experimental.gluon.runtime import TensorDescriptor

desc = TensorDescriptor.from_tensor(
    tensor,          # PyTorch tensor (must be contiguous along innermost dim)
    block_shape,     # tile shape (list of ints matching tensor rank)
    smem_layout,     # shared memory layout for the destination buffer
)
```
The descriptor is passed to the kernel as a regular argument.

### Async Load into SMEM
```python
from triton.experimental.gluon.language.nvidia import hopper as nvh

bar  = gl.allocate_shared_memory(gl.int64, [1], nvh.mbarrier.MBarrierLayout())
nvh.mbarrier.init(bar, count=1)

smem = gl.allocate_shared_memory(desc.dtype, desc.block_type.shape, desc.layout)

nvh.mbarrier.expect(bar, desc.block_type.nbytes)
nvh.tma.async_load(desc, [tile_row * BM, tile_col * BN], bar, smem)
nvh.mbarrier.wait(bar, phase=0)
```

### Async Store from SMEM to Global
```python
smem.store(result)
fence_async_shared()
nvh.tma.async_copy_shared_to_global(desc, [out_row * BM, out_col * BN], smem)
nvh.tma.store_wait(pendings=0)
```

### mbarrier Lifecycle
```
init(count)         — initialize; phase starts at 0 (not-yet-complete)
expect(bar, nbytes) — declare how many bytes this phase expects
async_load(...)     — issue the TMA; arrives automatically on completion
wait(bar, phase)    — block until phase is complete
arrive(count, pred) — manually arrive (used in warp specialization)
invalidate(bar)     — release after final use (optional but good practice)
```
Phase flips on every completion: 0 → 1 → 0 → … Track with `phase ^= 1`.

---

## MMA Operations

### Warp-Group MMA (WGMMA) — Hopper
WGMMA is an asynchronous warp-group-level (≥4 warps) MMA instruction that reads the B operand from shared memory descriptors, avoiding explicit register loads for B.

```python
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma_init, warpgroup_mma, warpgroup_mma_wait,
)

# Instruction shape constraints
# k = 256 // a_dtype.primitive_bitwidth (fixed)
# m = 16 (fixed)
# n = positive multiple of 8, up to 256
instr_shape = [16, INSTR_N, 256 // dtype.primitive_bitwidth]

c_layout = gl.NVMMADistributedLayout(
    version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=instr_shape
)

# Accumulator must live in registers with c_layout
acc = warpgroup_mma_init(
    gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=c_layout)
)

# A can be in registers (DotOperandLayout) or SMEM
a_reg_layout = gl.DotOperandLayout(0, parent=c_layout, k_width=32 // dtype.primitive_bitwidth)
a = a_smem.load(a_reg_layout)

# B must be in SMEM; use NVMMASharedLayout.get_default_for(shape, dtype)
acc = warpgroup_mma(a, b_smem, acc, is_async=True, use_acc=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

`use_acc=False` on the first iteration zeros the accumulator implicitly instead of reading the initial `c` value — avoids a SMEM round-trip.

After `warpgroup_mma_wait(0)` completes, it is safe to write new data into the shared memory operand buffers used by that MMA.

### Warp Distribution for WGMMA
`warps_per_cta=[M_warps, N_warps]` tiles the warp group across the output tile. The minimum indivisible unit is `[4, 1]`, so M_warps ≥ 4. Scale by doubling along either dimension.

Constraint: `BLOCK_M / 16 × BLOCK_N / n ≤ M_warps × N_warps` must hold; tune `n` (INSTR_N) accordingly.

### 5th-Gen Tensor Cores (TMEM) — Blackwell
On Blackwell (SM100+), accumulators live in TMEM — a 2D per-SM memory space (128 rows × 512 columns of 32-bit cells). Use the `tcgen05` API.

```python
from triton.experimental.gluon.language.nvidia import blackwell as nvb

tmem_ptr = nvb.tcgen05.allocate_tensor_memory(shape, dtype)
nvb.tcgen05.mma(a_smem, b_smem, tmem_ptr, use_acc=False)  # first iteration
nvb.tcgen05.mma(a_smem, b_smem, tmem_ptr, use_acc=True)   # subsequent iterations
nvb.tcgen05.mma_wait()

# Convert TMEM output to registers for epilogue
reg_layout = nvb.tcgen05.get_reg_layout(BLOCK_M, BLOCK_N, dtype)
result = gl.convert_layout(tmem_ptr.load(), reg_layout)
```

TMEM allocation constraints:
- Number of columns must be a power of 2 in [32, 512].
- Each warp accesses only 32 of the 128 rows; a full warp group (4 warps) covers all 128.
- Only 2D tensors are supported.

---

## Shared Memory Pipelining

### Double Buffering (Generic Pattern)
```python
# Allocate 2 SMEM buffers: shape = [2, BM, BK]
a_smem = gl.allocate_shared_memory(dtype, [2, BM, BK], smem_layout)
b_smem = gl.allocate_shared_memory(dtype, [2, BK, BN], smem_layout)

acc   = warpgroup_mma_init(...)
phase = 0

for k in range(num_k_tiles):
    buf  = k % 2
    buf  = a_smem.index(buf)
    # Async load into current buffer
    nvh.mbarrier.expect(bar, nbytes)
    nvh.tma.async_load(a_desc, [off_m, k * BK], bar, a_smem.index(k % 2))
    nvh.tma.async_load(b_desc, [k * BK, off_n], bar, b_smem.index(k % 2))
    nvh.mbarrier.wait(bar, phase=phase)
    phase ^= 1

    # MMA on current buffer, prefetch next
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    acc = warpgroup_mma(a_smem.index(k % 2), b_smem.index(k % 2), acc, is_async=True)

acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

For deeper pipelines, maintain a ring of `num_stages` buffers and track which barriers correspond to which stage.

---

## Warp Specialization — Hopper+

Warp specialization assigns distinct roles to different warp groups in the same CTA. It lowers the programming model one level below Triton's uniform SPMD model.

### `gl.warp_specialize()`
```python
gl.warp_specialize(
    [
        (partition_fn_A, (arg1, arg2, ...)),
        (partition_fn_B, (arg1, arg2, ...)),
    ],
    [warp_count_A, warp_count_B],      # warps per partition
    [min_regs_A,   min_regs_B],        # minimum registers per partition
)
```
- The **default partition** (the code that calls `warp_specialize`) always has `num_warps` warps and is the only partition that receives tensor arguments.
- Worker partitions receive explicitly passed arguments only; they share the same SMEM as the default partition.
- Warp/register counts are allocated in warpgroup granularity (multiples of 4).
- Minimum register count is 24; total register budget = `maxnreg × (num_warps + sum(worker_warps)) × 32`.

### Producer-Consumer Pattern
```python
@gluon.jit
def elementwise_kernel(in_desc, out_desc, N, BLOCK: gl.constexpr, num_warps: gl.constexpr):
    # Allocate shared barriers and ring buffers
    load_bars  = gl.allocate_shared_memory(gl.int64, [2], nvh.mbarrier.MBarrierLayout())
    store_bars = gl.allocate_shared_memory(gl.int64, [2], nvh.mbarrier.MBarrierLayout())
    smem       = gl.allocate_shared_memory(dtype, [2, BLOCK], smem_layout)

    nvh.mbarrier.init(load_bars.index(0), count=1)
    nvh.mbarrier.init(load_bars.index(1), count=1)
    nvh.mbarrier.init(store_bars.index(0), count=1)
    nvh.mbarrier.init(store_bars.index(1), count=1)

    gl.warp_specialize(
        [
            (load_partition,    (in_desc,  load_bars, smem, N, BLOCK)),
            (compute_partition, (load_bars, store_bars, smem, BLOCK)),
            (store_partition,   (out_desc, store_bars, smem, N, BLOCK)),
        ],
        [1, num_warps - 2, 1],
        [24, 232, 24],
    )

def load_partition(in_desc, load_bars, smem, N, BLOCK: gl.constexpr):
    for i in range(gl.cdiv(N, BLOCK)):
        buf   = smem.index(i % 2)
        bar   = load_bars.index(i % 2)
        phase = i // 2 & 1
        nvh.mbarrier.expect(bar, buf.nbytes)
        nvh.tma.async_load(in_desc, [i * BLOCK], bar, buf)
        nvh.mbarrier.wait(bar, phase=phase)         # load → compute handoff

def compute_partition(load_bars, store_bars, smem, BLOCK: gl.constexpr):
    for i in range(gl.cdiv(N, BLOCK)):
        # Wait for data from load partition
        nvh.mbarrier.wait(load_bars.index(i % 2), phase=i // 2 & 1)
        data = smem.index(i % 2).load(compute_layout)
        result = elementwise_fn(data)
        smem.index(i % 2).store(result)
        fence_async_shared()
        nvh.mbarrier.arrive(store_bars.index(i % 2), count=1)
```

### Ring Buffer Phase Tracking
When a ring of `num_buffers` barriers cycles through, the phase flips every `num_buffers` iterations:
```python
index = i % num_buffers
phase = (i // num_buffers) & 1
bar   = bars.index(index)
nvh.mbarrier.wait(bar, phase=phase)
```

---

## Persistent Kernels

A persistent kernel launches exactly `SM_count × waves` blocks (or fewer) and loops over output tiles internally using an atomic work counter. This improves L2 reuse and amortizes launch overhead.

```python
@gluon.jit
def persistent_kernel(work_counter_ptr, ...):
    pid = gl.program_id(0)
    # Claim the next tile
    while True:
        tile_id = gl.atomic_add(work_counter_ptr, 1)
        if tile_id >= total_tiles:
            break
        off_m = (tile_id // num_n_tiles) * BM
        off_n = (tile_id %  num_n_tiles) * BN
        # ... compute tile ...
```

Persistent kernels pair naturally with warp specialization: the load partition fetches continuously while the compute partition processes claimed tiles. Launch with `grid = (num_sms,)`.

---

## L2 Cache–Friendly Tile Ordering

Default linear tile ordering destroys L2 reuse for the B matrix in GEMM-like kernels. Use grouped / swizzled ordering (same as Triton best practice):

```python
GROUP_M = 8
num_m   = triton.cdiv(M, BM)
num_n   = triton.cdiv(N, BN)

pid          = gl.program_id(0)
group_id     = pid // (GROUP_M * num_n)
first_m      = group_id * GROUP_M
group_size_m = min(num_m - first_m, GROUP_M)
pid_m        = first_m + (pid % group_size_m)
pid_n        = (pid % (group_size_m * num_n)) // group_size_m
```

Consecutive PIDs now share the same column of B tiles in L2.

---

## Synchronization Primitives

### CTA Barrier
```python
gl.barrier()          # equivalent to __syncthreads() — full CTA barrier
```

### Warp Shuffle (SMEM-Free Warp Reductions)
```python
val = gl.inline_asm_elementwise(
    "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
    [val, offset], dtype=val.dtype,
)
```
Or use `gl.reduce(tensor, axis, combine_fn)` for block-level reductions which the compiler lowers efficiently.

### Block-Level Reduction
```python
result = gl.sum(tensor, axis=0)    # reduction along axis 0
result = gl.max(tensor, axis=1)    # reduction along axis 1
```
Gluon emits warp shuffle + SMEM partial reduction automatically.

---

## Architecture-Specific Notes

### Ampere (SM80, A100)
- Use `@gluon.jit` with `BlockedLayout` and manually issued `cp.async` for SMEM staging.
- Warp specialization is not available; use double-buffered SMEM with `__pipeline_commit` / `__pipeline_wait`.
- TMA is not available; stage tiles through cooperative global→SMEM copies.

### Hopper (SM90, H100)
- Primary target for Gluon. Full API: TMA, WGMMA, mbarrier, warp specialization.
- Use `NVMMADistributedLayout(version=[3, 0], ...)` for WGMMA accumulators.
- Use `NVMMASharedLayout.get_default_for(shape, dtype)` for WGMMA operand SMEM.
- Set `num_warps ≥ 4` (warpgroup minimum for WGMMA).
- Prefer `is_async=True` for all WGMMA calls to enable overlap with TMA.
- 3–5 pipeline stages typically saturate the MMA pipeline.

### Blackwell (SM100, B200)
- Use `nvb.tcgen05` APIs for TMEM-based accumulation and 5th-gen tensor cores.
- FP8 and FP4/MX block scaling via `nvb.tcgen05.blocked_scaled_mma`.
- Second-gen TMA with multicast: one descriptor fills SMEM across multiple CTAs simultaneously.
- Pipelining schedule becomes a 3:2 load:MMA ratio with dual TMEM accumulators for maximum overlap.

### AMD (CDNA3 / MI300X)
- `gl.amd.cdna3.buffer_load(base_ptr, offsets, ...)` — hardware buffer load with implicit bounds handling.
- Wave size is 64 threads; set `threads_per_warp=[64]` in `BlockedLayout`.
- Matrix operations via `gl.amd.mfma_*` intrinsics.
- Layouts and tile sizes must be re-tuned; NVIDIA configs do not transfer.

---

## Dtype and Precision

### FP16 / BF16 Input, FP32 Accumulation
Cast inputs to lower precision only at load time; accumulate in FP32:
```python
a = gl.cast(a, gl.float16)
b = gl.cast(b, gl.float16)
# WGMMA F32F16F16F32 accumulates FP16 inputs into FP32 accumulator
acc = warpgroup_mma(a, b_smem, acc, is_async=True)
```

### FP8 (Hopper / Blackwell)
Use E4M3 or E5M2 types with per-tensor or per-block scales:
```python
a = gl.cast(a, gl.float8e4nv)    # E4M3 (preferred for most models)
b = gl.cast(b, gl.float8e5)      # E5M2
```
Apply scale factors before accumulation or use `blocked_scaled_mma` on Blackwell.

### TF32 (Ampere)
FP32 inputs with TF32 tensor cores (~10-bit mantissa) yield ~8× throughput over CUDA cores:
```python
# Use NVMMADistributedLayout(version=[2, 0], ...) with TF32 dtype inputs
# allow_tf32 analogous to Triton's tl.dot(allow_tf32=True)
```

---

## Tile Size Selection

| Kernel type            | BM  | BN  | BK  | Warps | Notes                          |
|------------------------|-----|-----|-----|-------|--------------------------------|
| GEMM (Hopper, large)   | 128 | 128 | 64  | 8     | WGMMA; 2 warpgroups            |
| GEMM (Hopper, thin M)  | 64  | 128 | 64  | 4     | 1 warpgroup; inference shapes  |
| GEMM (Blackwell)       | 128 | 256 | 128 | 8+    | TMEM acc; FP8 w/ scale         |
| Flash Attention Q×K    | 64  | 64  | 64  | 4–8   | seqlen tile; online softmax    |
| Elementwise (warp-spec)| —   | —   | —   | 4–8   | 1 load + N compute + 1 store   |

Tile sizes must be compile-time `gl.constexpr` constants. Autotune via a Python loop over configs before kernel launch.

---

## Debugging and Profiling

### Correctness First
Test against `torch.testing.assert_close`. Run on small shapes (M=N=K=16) first — layout bugs produce wrong outputs, not crashes.

### `gl.device_print` Inside Kernels
```python
gl.device_print("tid val:", gl.program_id(0), val)
```
Disable before benchmarking; device print serializes execution.

### `TRITON_INTERPRET=1`
```bash
TRITON_INTERPRET=1 python my_kernel.py
```
Runs Gluon kernels on CPU in interpreted mode. Allows Python `print` and `pdb` for debugging. Much slower than GPU; use only for correctness.

### Nsight Compute (`ncu`)
```bash
ncu --set full python my_kernel.py
```
Key sections:
- **Memory Workload Analysis**: L1/L2 hit rates, SMEM bank conflicts, TMA transaction counts.
- **Warp State Statistics**: stall reasons — `Long Scoreboard` = memory stall, `Barrier` = `mbarrier.wait` stall.
- **Compute Workload Analysis**: WGMMA/TMEM utilization, async MMA pipeline depth.
- **Roofline**: determine compute-bound vs. memory-bound.

### PTX / LLVM IR Inspection
```python
my_kernel[grid](..., num_warps=4)        # compile once
print(my_kernel.asm['ptx'])              # inspect generated PTX
print(my_kernel.asm['ttgir'])            # Triton GPU IR (layouts visible here)
```
The `ttgir` is especially useful for verifying that layout decisions produced the intended IR-level representation.

### Benchmarking
```python
import torch
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
for _ in range(10): my_kernel[grid](...)   # warmup
start.record()
for _ in range(100): my_kernel[grid](...)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
```
Report achieved TFLOPS or GB/s against the hardware roofline, not raw milliseconds.

---

## Common Algorithm Patterns

### Tiled GEMM (Hopper, WGMMA + TMA, Pipelined)
```
Allocate 2×(BM×BK + BK×BN) SMEM buffers and 2 mbarriers

Prologue: prefetch tile 0 into buffer 0

for k_tile in range(K // BK):
    wait mbarrier for buffer[k_tile % 2]         # wait for TMA load
    issue TMA loads for tile k_tile+1 into buffer[(k_tile+1) % 2]
    warpgroup_mma_wait(outstanding=0)             # wait for prev MMA
    warpgroup_mma(a[buf], b_smem[buf], acc, async=True)  # issue MMA

warpgroup_mma_wait(outstanding=0)
epilogue: convert acc → output dtype, store via TMA
```

### Online Softmax / Flash Attention
```
for kv_tile in range(seq_len // BN):
    load K_tile, V_tile via TMA into SMEM
    warpgroup_mma Q_tile × K_tile.T  →  scores (BM × BN)
    apply causal mask
    m_new = max(m_old, row_max(scores))
    scale = exp(m_old - m_new)
    acc   = acc * scale + exp(scores - m_new) @ V_tile
    lse   = lse * scale + row_sum(exp(scores - m_new))

output = acc / lse
```
Use BN = 64 or 128 for the KV tile; online softmax avoids materializing the full `seq_len × seq_len` matrix.

### Warp-Specialized Persistent GEMM (Blackwell)
```
Load partition (1 warp):  TMA load A and B tiles into ring buffer
MMA partition (M warps):  TMEM accumulate; double-buffered accumulator
Epilogue partition (1 warp): TMA store results; signal load partition

All partitions loop over tiles claimed from a global atomic counter.
```

---

## General Best Practices

- **Start with `BlockedLayout`; switch to `LinearLayout` only when needed**: `BlockedLayout` covers most cases and is easier to reason about. Linear layouts are needed for exotic permutations or when the compiler cannot infer zero-cost conversions.
- **Design layouts top-down from the MMA atom**: choose the MMA instruction shape and `NVMMADistributedLayout` first, then derive SMEM operand layouts and global access layouts to feed it without conversion overhead.
- **Verify layouts in `ttgir` before profiling**: a layout bug that forces an unexpected SMEM round-trip can silently cut throughput in half. Use `print(my_kernel.asm['ttgir'])` to inspect.
- **Use `assert_trivial=True` on `convert_layout`**: annotate conversions you expect to be register-only; the compiler will error if it needs SMEM.
- **Keep warp specialization for genuinely asymmetric work**: load warps need only 24 registers; compute warps need 128–256. Over-allocating to load warps wastes SM register budget and reduces occupancy.
- **Use TMA for all structured tile loads on Hopper+**: TMA increases effective bandwidth by off-loading address computation from SMs and enabling independent scheduling of load and compute.
- **Match `fence_async_shared()` precisely**: too many fences serialize operations unnecessarily; too few cause correctness bugs. Place one fence when transitioning between async-proxy and CPU-proxy access to the same SMEM region.
- **Fuse epilogue into the kernel**: apply bias, activation, and type conversion in register-level epilogue before the final TMA store to global memory. Avoid launching a second kernel for elementwise ops.
- **Profile SMEM bank conflicts in `ncu` before assuming layout correctness**: bank conflicts can halve SMEM throughput silently. Target zero conflicts in the "Memory Workload Analysis" section.
- **Portability trade-off is real**: Gluon kernels are architecture-specific. Maintain a Triton fallback for older GPUs unless you are certain about deployment targets.
- **Reference Triton's `python/tutorials/gluon/`**: these are the canonical, up-to-date examples for every major Gluon feature. Always cross-check API usage against the latest tutorial source.
