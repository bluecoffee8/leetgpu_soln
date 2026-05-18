---
name: pyptx-kernel-development
description: Guide for development and performance optimization of kernels written using pyptx.
---

# Performance Optimization

pyptx is a Python DSL for handwritten PTX on NVIDIA GPUs (Ampere sm_80, Ada sm_89, Hopper sm_90a, Blackwell sm_100a/sm_120). The core contract: **one Python call = one PTX instruction**. There is no optimizer, no autotuner, no tensor IR between the Python function and the PTX it emits.

Install:
```bash
pip install 'pyptx[torch]'   # PyTorch eager + torch.compile
pip install 'pyptx[jax]'     # JAX jit
pip install 'pyptx[all]'     # Both
pip install ninja             # Drops PyTorch dispatch overhead from ~34 µs to ~14 µs
```

---

## Core Programming Model

### Trace-Time Execution
A `@kernel` function runs at **trace time** in Python, not at kernel execution time. Every line of Python executes when the kernel is first called (or compiled), generating PTX. Control structures like `for` loops and `if` statements at the Python level are **unrolled** into flat PTX — they do not become GPU loops or branches unless you emit them explicitly via `ptx.loop(...)` or `ptx.if_(...)`.

Three namespaces drive authoring:
- `reg.*` — allocates PTX registers (no instruction emitted).
- `smem.*` — allocates shared memory.
- `ptx.*` — emits PTX instructions, one per call.

### `@kernel` Decorator
```python
from pyptx import kernel, reg, smem, ptx, Tile
from pyptx.types import bf16, f32, u32

@kernel(
    in_specs=(Tile("M", "K", bf16), Tile("K", "N", bf16)),
    out_specs=(Tile("M", "N", f32),),
    grid=lambda M, N, K: (N // 64, M // 64),
    block=(128, 1, 1),
    arch="sm_90a",
)
def gemm(A, B, C):
    ...
```

Key parameters:
- `in_specs` / `out_specs`: tuple of `Tile(dim0, dim1, dtype)` or `Tile(size, dtype)` descriptors that type-check inputs and compute byte offsets. Dimensions can be integer literals (specialized) or symbolic strings (generic).
- `grid`: integer 3-tuple or a callable `lambda *shape_dims: (gx, gy, gz)`. The lambda receives the symbolic dimensions in the order they appear in `Tile`.
- `block`: integer 3-tuple `(tx, ty, tz)` — thread block size.
- `arch`: `"sm_80"`, `"sm_89"`, `"sm_90a"`, `"sm_100a"`, `"sm_120"`, or `"auto"` (picks the current GPU at trace time).
- `smem`: optional int — static shared memory bytes per block (needed when using `smem.base()` / `extern_smem=True`).
- `extern_smem=True`: declares shared memory as an extern `__shared__` byte array; required for raw `smem.base()` addressing.
- `version`: optional `(major, minor)` PTX ISA version tuple; required for Blackwell (e.g., `version=(8, 7)`).

### One Kernel, Three Runtime Paths
The same kernel object dispatches from PyTorch eager, `torch.compile`, and JAX without modification:
```python
# PyTorch eager
out = gemm(a, b)

# torch.compile
out = torch.compile(gemm)(a, b)

# JAX jit
out = jax.jit(gemm)(a, b)
```

Dispatch tiers (PyTorch):
- **CUDA graph replay**: ~4 µs per launch.
- **Turbo eager** (with `ninja`): ~14 µs (cached C++ extension).
- **`torch.compile`**: ~14–22 µs (custom_op path).

### `arch="auto"`
Selects the right target for the installed GPU at trace time. Validated on T4, A100, L4, H100, B200, RTX Pro 6000 Blackwell. Use this for kernels that should run portably; use a specific arch string when you need ISA features that vary across targets.

---

## Types (`pyptx.types`)

| Type | PTX scalar type | Bits |
|------|----------------|------|
| `f32` | `.f32` | 32 |
| `f64` | `.f64` | 64 |
| `f16` | `.f16` | 16 |
| `bf16` | `.bf16` | 16 |
| `u32` | `.u32` | 32 |
| `u64` | `.u64` | 64 |
| `s32` | `.s32` | 32 |
| `b32` | `.b32` | 32 (opaque, for packed bf16/fp16 pairs) |
| `b64` | `.b64` | 64 |

Use `b32` to hold two packed `bf16` values in a single register (required by `mma.sync` fragment layouts).

---

## Register Allocation (`reg`)

Registers are allocated in Python, not emitted as PTX. The DSL tracks liveness automatically.

```python
# Scalar register (no init value — uninitialized in PTX)
x = reg.scalar(f32)

# Scalar with compile-time initial value (emits a mov.f32 to a float literal)
x = reg.scalar(f32, init=0.0)

# Register array: N independent scalar registers
arr = reg.array(f32, 32)   # arr[0], arr[1], ..., arr[31]

# Seed from a special register value
tid = reg.from_(ptx.special.tid.x(), u32)   # emits mov.u32 tid, %tid.x
```

### Arithmetic on Registers
Python operators on `Reg` objects emit exactly one PTX instruction per operation:
```python
ptr + offset        # emits add.s64
tid * 4             # emits mul.wide.u32 (u32 × u32 → s64 when mixed with ptr)
tid & (WARP_SIZE-1) # emits and.b32
tid >> 5            # emits shr.b32
```

This means Python-level arithmetic is **not free** — each operator is an instruction. Hoist invariants outside loops just as you would in hand-written PTX.

---

## Shared Memory (`smem`)

### Named Allocation
```python
# Allocate a 2D SMEM array
partials = smem.alloc(f32, (num_warps, 1))  # shape is (rows, cols)
stats = smem.alloc(f32, (1, 1))

# Access with integer indices (emits ld.shared / st.shared automatically)
partials[warp_id, 0] = value   # emits st.shared.f32
value = partials[i, 0]         # emits ld.shared.f32
```

### WGMMA Tile (Hopper)
```python
# Shared memory tile with swizzled layout for WGMMA input
sA = smem.wgmma_tile(bf16, (64, 16), major="K")   # A matrix: K is contiguous
sB = smem.wgmma_tile(bf16, (16, 64), major="MN")  # B matrix: MN is contiguous
```
`smem.wgmma_tile` applies the CUTLASS-compatible swizzle (B32/B64/B128 with `Swizzle<B,4,3>`) to eliminate shared memory bank conflicts during WGMMA operand loads.

### Raw SMEM Base (Extern Mode)
For kernels that manage SMEM layout manually (e.g., ring buffers for `cp.async`):
```python
@kernel(..., smem=SMEM_BYTES, extern_smem=True)
def my_kernel(...):
    smem_base = smem.base()   # raw s64 pointer to __shared__ byte array
    # offset manually: smem_base + (stage * STAGE_BYTES) + thread_offset
```

---

## PTX Instructions (`ptx`)

### Special Registers
```python
ptx.special.tid.x()     # %tid.x  (thread index in x)
ptx.special.tid.y()     # %tid.y
ptx.special.ctaid.x()   # %ctaid.x (block index in x)
ptx.special.ctaid.y()
ptx.special.ntid.x()    # %ntid.x (block dimension in x)
ptx.special.nctaid.x()  # %nctaid.x (grid dimension in x)
ptx.special.laneid()    # %laneid
ptx.special.warpid()    # %warpid
ptx.special.smid()      # %smid
```

### Common Instructions
```python
# Scalar moves
ptx.inst.mov.u32(dst, src)
ptx.inst.mov.f32(dst, src)

# Arithmetic
ptx.inst.add.u32(dst, a, b)
ptx.inst.add.f32(dst, a, b)
ptx.inst.mul.f32(dst, a, b)
ptx.inst.fma.rn.f32(dst, a, b, c)   # dst = a*b + c (fused, round-to-nearest)
ptx.inst.mul.f32(dst, a, b)
ptx.inst.rsqrt.approx.f32(dst, src) # fast reciprocal square root

# Bit operations
ptx.inst.shl.b32(dst, src, shift)   # shift left
ptx.inst.shr.b32(dst, src, shift)   # shift right
ptx.inst.and_.b32(dst, a, b)
ptx.inst.or_.b32(dst, a, b)

# Global memory loads / stores (each call = one PTX ld/st)
ptx.inst.ld.global_.f32(dst, ptx.addr(ptr))
ptx.inst.ld.global_.b32(dst, ptx.addr(ptr))
ptx.inst.ld.global_.v4.f32([d0, d1, d2, d3], ptx.addr(ptr))   # 128-bit vector load
ptx.inst.st.global_.f32(ptx.addr(ptr), src)
ptx.inst.st.global_.v4.f32(ptx.addr(ptr), [s0, s1, s2, s3])   # 128-bit vector store

# Shared memory loads / stores
ptx.inst.ld.shared.b32(dst, ptx.addr(smem_ptr))
ptx.inst.st.shared.b32(ptx.addr(smem_ptr), src)
```

### `ptx.addr(ptr)`
Wraps a register (or register expression) as an address operand for load/store instructions. Required wherever PTX uses `[addr]` syntax.

### `ptx.global_ptrs(A, B, C, ...)`
Emits the parameter-pointer prologues for multiple tensors in a single call — saves boilerplate:
```python
pa, pb, pc = ptx.global_ptrs(A, B, C)
# Equivalent to three ptx.inst.ld.param.u64 / cvta.to.global.u64 sequences
```

### Synchronization
```python
ptx.bar.sync(id)                     # __syncthreads() — all threads in block, barrier id 0..15
ptx.bar.sync(id, thread_count)       # Partial barrier (first N threads)
ptx.inst.membar.gl()                 # Device-wide memory fence (__threadfence)
ptx.inst.membar.cta()                # Block-scope memory fence
```

### Warp Reductions (Sugar)
```python
ptx.warp.reduce_sum(reg_val)   # Butterfly shfl reduction; result lands in lane 0 (and all lanes after)
```
This emits the standard `shfl.sync.bfly` sequence and saves the final result back into `reg_val` in lane 0.

### Control Flow
Python `for` / `if` at the Python level are always unrolled. To emit GPU-side control flow:
```python
# Conditional block (emits predicated setp + BRA)
with ptx.if_(condition_expr):
    ...   # instructions inside are predicated

# Loop (emits a PTX loop with BRA; n_iters must be a Python int)
with ptx.loop(n_iters) as i:
    ...
```

### Return
```python
ptx.ret()   # Emit .ret PTX instruction — required at the end of every kernel
```

---

## Ampere (sm_80, A100) — Specific Features

### `cp.async` — Asynchronous Global → SMEM Copies
```python
# Issue 16-byte async copy from global to SMEM (bypass L1 cache)
ptx.cp.async_.cg(ptx.addr(smem_dst), ptx.addr(global_src), 16)

# Close pending cp.async into a commit group
ptx.cp.async_.commit_group()

# Wait until at most N groups remain in-flight (0 = wait for all)
ptx.cp.async_.wait_group(0)
```
Use `cp.async` to build double- or multi-stage SMEM ring buffers: issue loads for the next tile while computing on the current tile.

### `mma.sync` — Ampere Tensor Core MMA
```python
ptx.mma.sync(
    shape=(16, 8, 16),             # m16n8k16 — standard Ampere bf16 shape
    dtype_d=f32, dtype_a=bf16, dtype_b=bf16, dtype_c=f32,
    d=[acc[0], acc[1], acc[2], acc[3]],   # output fragment (4 f32 per lane)
    a=[a_fr[0], a_fr[1], a_fr[2], a_fr[3]],  # A fragment (4 b32 per lane)
    b=[b_fr[0], b_fr[1]],                     # B fragment (2 b32 per lane)
    c=[acc[0], acc[1], acc[2], acc[3]],   # accumulator in
    a_layout="row", b_layout="col",        # A row-major, B column-major
)
```
Fragment layout for `m16n8k16.row.col` (standard bf16 GEMM):
- A: 4 `b32` registers per lane, each holding 2 packed bf16.
- B: 2 `b32` registers per lane, each holding 2 packed bf16 (B transposed, stored as `(N, K)` row-major).
- D/C: 4 `f32` registers per lane.

### `ldmatrix` — Warp-Collective SMEM → Register Fragment Load
```python
ptx.inst.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
    [a_fr[0], a_fr[1], a_fr[2], a_fr[3]],
    ptx.addr(smem_ptr),
)
```
More efficient than per-thread `ld.shared` for loading MMA fragments; the hardware handles bank-conflict-free broadcast across lanes.

---

## Hopper (sm_90a, H100) — Specific Features

### WGMMA — Warpgroup Matrix Multiply-Accumulate
```python
# Allocate 32 f32 accumulator registers per warpgroup (128 threads)
acc = reg.array(f32, 32)

ptx.wgmma.mma_async(
    shape=(64, 64, 16),   # m64n64k16 for bf16
    dtype_d=f32, dtype_a=bf16, dtype_b=bf16,
    d=acc,
    a=sA,   # smem.wgmma_tile object (descriptor-based)
    b=sB,   # smem.wgmma_tile object
    scale_d=1,   # 1 = accumulate; 0 = zero acc before writing
)
ptx.wgmma.commit_group()    # close a wgmma commit group
ptx.wgmma.wait_group(0)     # wait for all wgmma to complete
```
WGMMA operates on 128-thread warpgroups (4 warps). The A/B operands come from shared memory descriptors built from `smem.wgmma_tile`. **SMEM must use the correct swizzled layout** (provided by `smem.wgmma_tile`) to avoid MMA correctness issues.

### TMA — Tensor Memory Accelerator
```python
# Issue a 2D async TMA load from global to SMEM (Hopper+)
ptx.tma.load_2d(smem_dst, tma_desc, c0, c1)
# TMA with multicast to a thread block cluster
ptx.tma.load_2d_multicast(smem_dst, tma_desc, c0, c1, cluster_mask)
```
TMA descriptors are built on the host and passed as kernel arguments. They encode strides, data type, and swizzle mode, offloading address generation from the SM.

### mbarriers — Asynchronous Barriers (Hopper)
```python
mbar = smem.mbarrier()               # allocate a 64-bit mbarrier in SMEM
ptx.mbarrier.init(mbar, thread_count)   # initialize with expected arrival count
ptx.mbarrier.arrive(mbar)               # signal arrival
ptx.mbarrier.wait(mbar, phase)          # wait for all arrivals (spinning on phase bit)
ptx.mbarrier.arrive_expect_tx(mbar, byte_count)  # arrive + declare async TMA bytes in flight
```
mbarriers replace `__syncthreads()` in warp-specialized pipelines. Use them to synchronize between a "producer" warpgroup (issuing TMA loads) and a "consumer" warpgroup (running WGMMA).

---

## Blackwell (sm_100a, B200) — Specific Features

### `tcgen05.mma` — Blackwell Tensor Core MMA
```python
ptx.tcgen05.mma(
    shape=(128, 256, 64),    # problem shape
    dtype_a=bf16, dtype_b=bf16, dtype_d=f32,
    d=acc_tmem,
    a=sA_desc,   # SMEM descriptor
    b=sB_desc,   # SMEM descriptor
    scale_d=1,
)
ptx.tcgen05.commit()    # commit pending tcgen05 ops
ptx.tcgen05.fence()     # fence after tcgen05 completion
```
Blackwell tcgen05 operates on **Tensor Memory (TMEM)** rather than register files. TMEM is a per-SM scratchpad that lives closer to the MMA units.

### TMEM — Tensor Memory
```python
tmem_ptr = ptx.tcgen05.alloc(rows=128, cols=256, dtype=f32)  # allocate TMEM
ptx.tcgen05.load(dst_regs, ptx.addr(tmem_ptr))               # TMEM → registers
ptx.tcgen05.store(ptx.addr(tmem_ptr), src_regs)              # registers → TMEM
```

### `cta_group` — 2-SM Cooperative MMA (Blackwell)
```python
@kernel(..., arch="sm_100a", version=(8, 7))
def gemm_2sm(A, B, C):
    # Cooperative MMA across two CTAs
    ptx.tcgen05.mma(..., cta_group=2)
```
Enables CTA pairs to collaborate on a single large MMA tile, effectively doubling tile size for better reuse.

---

## Memory Access Patterns

### Vectorized Loads (128-bit)
Prefer `v4.f32` (four floats = 128 bits) over four individual `f32` loads. One instruction, 4× throughput on aligned, consecutive accesses:
```python
ptx.inst.ld.global_.v4.f32([x0, x1, x2, x3], ptx.addr(ptr))
ptx.inst.st.global_.v4.f32(ptx.addr(ptr), [y0, y1, y2, y3])
```
Ensure the base pointer is 16-byte aligned and `items_per_thread` is a multiple of 4.

### Items Per Thread
Assign 4+ elements per thread to maximize memory-level parallelism and hide DRAM latency. More outstanding loads per thread = more latitude for the memory subsystem to reorder and coalesce:
```python
items_per_thread = N // block   # typically 4, 8, or 16
use_v4 = items_per_thread >= 4 and items_per_thread % 4 == 0
```

### Strided Thread Loads
For 1D kernels with `block` threads and `items_per_thread` per thread:
```python
elem_base = tid << 2   # tid * 4 (when use_v4)
for j in range(v4_iters):
    idx = elem_base + j * block * 4
    ptx.inst.ld.global_.v4.f32([...], ptx.addr(ptr + idx * elem_size))
```

### Coalescing
Threads in the same warp should load consecutive global memory addresses. Map `tid` to the *innermost* (fastest-varying) dimension. Strided accesses (e.g., loading column-major in a row-major buffer) require SMEM staging.

---

## Software Pipelining

### Double-Buffered `cp.async` Ring Buffer (Ampere)
```python
STAGES = 2
# Prologue: prime stage 0
issue_cp_async(stage=0, k_base=0)
ptx.cp.async_.commit_group()

for ki in range(n_iters):
    cur_stage = ki % STAGES
    next_stage = (ki + 1) % STAGES

    # Wait for the current stage's async copy to land
    ptx.cp.async_.wait_group(STAGES - 1)  # <= 1 group pending
    ptx.bar.sync(0)

    # Compute on cur_stage SMEM
    do_mma(cur_stage)

    # Prefetch next tile into next_stage
    if ki + 1 < n_iters:
        issue_cp_async(stage=next_stage, k_base=(ki + 1) * BK)
        ptx.cp.async_.commit_group()

ptx.cp.async_.wait_group(0)  # drain all remaining
ptx.bar.sync(0)
```

### Warp Specialization (Hopper / Blackwell)
Assign separate warpgroups to producer (TMA / cp.async) and consumer (WGMMA / tcgen05) roles. Producers issue async loads and arrive at mbarriers; consumers wait and run MMA. This decouples memory latency from compute throughput completely.

---

## Common Algorithm Patterns

### RMS Norm (Memory-Bound, Warp Reduction)
```python
# Pass 1: accumulate sum-of-squares per thread into sum_sq
# Warp-level reduction
ptx.warp.reduce_sum(sum_sq)
# Lane 0 writes partial to SMEM; all threads sync; thread 0 sums partials
with ptx.if_(lane == 0):
    partials[warp_id, 0] = sum_sq
ptx.bar.sync(0)
with ptx.if_(tid == 0):
    block_sum = reg.scalar(f32, init=0.0)
    for i in range(num_warps):
        ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
    stats[0, 0] = block_sum
ptx.bar.sync(0)
# Pass 2: load rstd, apply normalization using stored x_vals + weight W
```

### GEMM (Ampere, Tensor Cores)
```python
# One CTA computes (BM × BN) output; each warp owns a (WM × WN) sub-tile
# K-loop: load BM×BK A tile + BN×BK B tile into SMEM (via cp.async or direct ld)
# For each mma.sync shape: compute 16×8 fragment per warp, accumulate in f32 regs
# After K-loop: store per-thread D fragment to global memory
```

### Hopper GEMM (WGMMA + TMA)
```python
# 128-thread warpgroup, SMEM double-buffered with mbarriers
# Producer warpgroup: TMA load → arrive mbarrier
# Consumer warpgroup: wait mbarrier → wgmma.mma_async → wgmma.commit_group/wait
# Accumulator in 32 f32 regs/warpgroup
# Store: wgmma.wait_group(0), then per-thread st.global.v2.f32 output
```

### Parallel Reduction (Block-Wide)
1. Each thread accumulates its elements into a scalar register.
2. Warp-level butterfly via `ptx.warp.reduce_sum(val)` (result in lane 0 of each warp).
3. Lane 0 of each warp writes its partial to `smem.alloc(f32, (num_warps, 1))`.
4. `ptx.bar.sync(0)`.
5. Thread 0 sums partials from SMEM to produce the block result.
6. (Optional) single `atomicAdd` per block to global accumulator.

---

## Inspecting and Debugging

### Inspect the Generated PTX
```python
kernel_fn = build_my_kernel(M, N, K)
print(kernel_fn.ptx())   # Print the full PTX source
```
Use this to verify the instruction sequence, check register allocation, and confirm there are no unintended extra instructions from arithmetic operators.

### Transpile Existing PTX into pyptx
Convert PTX from `nvcc`, Triton, or Pallas into editable Python:
```bash
python -m pyptx.codegen kernel.ptx --sugar --name my_kernel > my_kernel.py
```
`--sugar` demangles names, raises spin-loops into `ptx.loop(...)`, collapses mbarrier-wait blocks, and groups expression chains. Round-trips are byte-identical on 218+ corpus files.

### Correctness Validation
Always validate against a reference before benchmarking:
```python
# PyTorch
torch.testing.assert_close(pyptx_out, ref_out, atol=1e-4, rtol=1e-3)

# JAX
np.testing.assert_allclose(np.array(pyptx_out), np.array(ref_out), atol=1e-4, rtol=1e-3)
```
For bf16 GEMM, use `atol=1e-2, rtol=1e-2` — reduced mantissa precision limits exact matching.

### Nsight Compute
Profile pyptx kernels like any CUDA kernel:
```bash
ncu --set roofline -o profile python my_kernel.py
```
Use Roofline + Memory Workload Analysis to determine whether the kernel is compute-bound or bandwidth-bound. The kernel name in `ncu` output matches the Python function name.

### Benchmarking
```python
import triton.testing  # triton.testing.do_bench works with any CUDA kernel

ms = triton.testing.do_bench(lambda: gemm(a, b))
tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
print(f"{tflops:.1f} TFLOPS")
```
Or use `torch.cuda.Event`:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(10): gemm(a, b)   # warmup
start.record()
for _ in range(100): gemm(a, b)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
```

---

## Architecture-Specific Dispatch

Use `arch="auto"` for portable kernels. For kernels that need architecture-specific code paths, build separate specialized functions and dispatch at runtime:
```python
device = torch.cuda.get_device_properties("cuda")
if device.major >= 10:
    kernel_fn = build_blackwell_kernel(M, N, K)
elif device.major == 9:
    kernel_fn = build_hopper_kernel(M, N, K)
else:
    kernel_fn = build_ampere_kernel(M, N, K)
```

---

## General Best Practices

- **One call = one instruction**: Every `ptx.inst.*` call is a PTX instruction. Hoist invariants out of Python loops; each Python iteration unrolls into emitted PTX.
- **Specialize per shape**: pyptx has no autotuner. Build factory functions (`build_kernel(M, N, K)`) that bake shapes into the kernel at trace time. Cache the kernel object for repeated calls with the same shape.
- **Use v4 loads aggressively**: 128-bit vector loads (`v4.f32`) are the single highest-impact memory throughput optimization for bandwidth-bound kernels. Ensure alignment and use items-per-thread ≥ 4.
- **Accumulate in f32**: Compute in bf16/f16 for throughput; accumulate in `f32` registers. Downcast output at store time.
- **Validate correctness before benchmarking**: The PTX-level control gives you rope to hang yourself with. Test with small shapes and tight tolerances before sweeping to large shapes.
- **Inspect `.ptx()`**: When performance is lower than expected, check the generated PTX for unexpected extra instructions from Python-level arithmetic on `Reg` objects. Each operator emits a PTX instruction.
- **SMEM swizzle matters**: For WGMMA (Hopper), always use `smem.wgmma_tile` — raw `smem.alloc` will produce bank conflicts and incorrect results with WGMMA operand loads.
- **mbarrier phasing**: Alternate between phase 0 and 1 on each mbarrier wait call in a pipeline loop. Using the same phase twice without reinitializing the barrier will hang.
- **Use `ptx.global_ptrs` for multi-tensor prologues**: It is concise and correct; the manual alternative requires several PTX instructions per tensor pointer.
- **Profile before specializing**: Check Nsight Compute's Roofline section first to identify whether you are compute-bound or bandwidth-bound before investing in new optimizations.
