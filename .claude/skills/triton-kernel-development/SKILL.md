---
name: triton-kernel-development
description: Guide for development and performance optimization of triton DSL kernels.
---

# Performance Optimization

Here is a list of common performance optimization techniques for Triton kernels. Triton is a Python-embedded DSL that compiles to GPU code via LLVM/MLIR. It exposes block-level parallelism and handles many low-level details automatically, but performance still requires careful attention to memory access patterns, tiling, and hardware-specific features.

---

## Core Programming Model

### Programs and PIDs
Each Triton kernel launch creates a grid of *programs*. Each program gets a unique `program_id` via `tl.program_id(axis)`. Unlike CUDA threads, a Triton program processes a *block* of elements, not a single element. The block shape is a compile-time constant (must be a power of two).

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

### Block Pointers (`tl.make_block_ptr`)
Prefer `tl.make_block_ptr` over manual offset arithmetic for 2D (and higher) tiles. Block pointers encode strides, shape, and boundary conditions, enabling the compiler to emit optimal loads and TMA-backed copies on Hopper+.

```python
ptr = tl.make_block_ptr(
    base=A, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K),
    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
)
a = tl.load(ptr, boundary_check=(0, 1))
```

`order=(1, 0)` means the last axis is contiguous (row-major), which is required for coalesced loads.

### Masking and Boundary Handling
Use masks to handle tiles that extend beyond the tensor boundary. `tl.load` and `tl.store` accept an `other` argument for out-of-bounds fill.

```python
mask = offsets < n_elements
x = tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(ptr + offsets, result, mask=mask)
```

With `tl.make_block_ptr`, pass `boundary_check=(0, 1)` instead of explicit masks.

---

## Memory Access Patterns

### Coalesced Global Memory Access
Triton programs load and store *blocks* of data. For coalesced access, the innermost dimension of the block (contiguous in memory) should map to the fastest-varying axis. Ensure `order=(1, 0)` for row-major tensors when using block pointers. Manual offset patterns should use `tl.arange` on the innermost dimension.

### Shared Memory via `tl.load` Caching
Triton manages shared memory automatically. The compiler decides when to stage data in SMEM. You can hint at this by structuring loads so that the same data is reused multiple times within a program — the compiler will cache it. Unlike CUDA, you do not manually declare `__shared__` arrays (with the exception of `tl.tensor` allocations in experimental APIs).

### Cache Hints (`cache_modifier`)
Pass `cache_modifier` to `tl.load` to control L1/L2 behavior:
- `.ca` — cache at all levels (default L1+L2).
- `.cg` — cache at L2 only (bypass L1); useful for data with low reuse.
- `.cs` — streaming, evict-first; for data accessed once.
- `.cv` — volatile, do not cache.

```python
a = tl.load(ptr, cache_modifier=".cg")
```

### Eviction Policy
`eviction_policy` parameter on `tl.load` / `tl.store`:
- `"evict_last"` — keep in cache as long as possible (for reused data).
- `"evict_first"` — evict early (for streaming data).

---

## Tiling and Blocking

### Tile Size Selection (BLOCK_M, BLOCK_N, BLOCK_K)
Tile sizes must be powers of two. Larger tiles increase register reuse and arithmetic intensity but increase register pressure and SMEM usage. Typical GEMM tile sizes:
- Small/medium tiles: `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`
- Large tiles: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64`

Always tune tile sizes as `constexpr` parameters via the autotuner.

### `tl.constexpr` Parameters
Mark tile sizes and other compile-time constants as `tl.constexpr`. This allows the compiler to specialize the kernel and enables constant folding, loop unrolling, and static shape inference.

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    ...
```

### Loop Over Tiles (K-loop in GEMM)
The standard pattern for matrix multiplication is a loop over tiles of the reduction dimension:

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    acc += tl.dot(a, b)
    a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
    b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))
```

---

## Dot Product and Matrix Multiply

### `tl.dot`
`tl.dot(a, b, acc)` maps to tensor core matrix multiply-accumulate. The input tensors must be 2D blocks with shapes compatible with the hardware MMA instruction. Both inputs must be in `float16`, `bfloat16`, `float32`, or `int8` (depending on hardware).

- `allow_tf32=True` (default on Ampere+): uses TF32 accumulation for FP32 inputs, significantly faster with minor precision loss.
- For FP16/BF16 inputs, accumulation is always in FP32.

```python
acc = tl.dot(a, b, acc, allow_tf32=True)
```

### Input Layout for `tl.dot`
`tl.dot` requires its inputs to be in a specific layout for tensor core access. The first argument (A) should be `(BLOCK_M, BLOCK_K)` and the second (B) should be `(BLOCK_K, BLOCK_N)`. Ensure load order matches: `order=(1,0)` for A (row-major) and `order=(0,1)` for B (column-major, i.e., transposed storage) to maximize SMEM bank efficiency and tensor core throughput.

### `tl.dot` Input Precision Casting
Cast inputs explicitly before `tl.dot` for mixed-precision kernels:

```python
a = tl.load(...).to(tl.float16)
b = tl.load(...).to(tl.float16)
acc += tl.dot(a, b)
```

---

## Autotuning

### `@triton.autotune`
Use the autotuner to sweep over tile sizes, pipeline stages, and warp counts:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

`key` lists arguments that invalidate the cached best config when they change (typically problem dimensions).

### `num_warps`
Controls how many warps execute a single Triton program. More warps can increase occupancy and hide latency but compete for registers. Typical range: 4–8. For memory-bound kernels, fewer warps with more aggressive pipelining often wins.

### `num_stages`
Number of software pipeline stages. Triton will automatically insert prefetching across this many stages. Higher stages hide memory latency at the cost of more registers and SMEM. Typical values: 2–5. On Hopper (with TMA), 3–5 stages are common.

### `num_ctas`
On Hopper, controls the number of CTAs in a cluster. Used when leveraging distributed shared memory or TMA multicast. Default is 1 (no clustering).

---

## Software Pipelining

### Automatic Pipelining (`num_stages`)
Setting `num_stages > 1` in the config enables Triton's automatic software pipeline. Triton will insert `cp.async` (Ampere) or TMA (Hopper) prefetches for global loads and overlap them with compute from previous iterations. No manual `cp.async` management is needed.

### Manual Prefetching
For kernels where automatic pipelining is insufficient, manually prefetch the next tile before the compute stage:

```python
a_next = tl.load(a_ptr_next)   # prefetch
tl.dot(a_cur, b_cur, acc)       # compute
a_cur = a_next                   # swap
```

---

## Reduction Operations

### `tl.sum`, `tl.max`, `tl.min`
Block-level reductions across a specified axis. These map to efficient warp-level and inter-warp reductions internally.

```python
row_max = tl.max(x, axis=1)          # max along columns
row_sum = tl.sum(tl.exp(x - row_max), axis=1)
```

### `tl.associative_scan`
Performs a prefix scan with a user-defined associative operator. Useful for cumulative sums, prefix max, and other scan primitives.

```python
cumsum = tl.associative_scan(x, axis=0, combine_fn=lambda a, b: a + b)
```

### Online Softmax Pattern
Numerically stable softmax in a single pass using the online algorithm:

```python
# Load tile, compute running max and sum simultaneously
m_i = tl.max(scores, axis=1)
p = tl.exp(scores - m_i[:, None])
l_i = tl.sum(p, axis=1)
# Update running stats across tiles and rescale accumulator
```

This avoids two separate passes over the input and is essential for flash attention implementations.

---

## Atomic Operations

### `tl.atomic_add`, `tl.atomic_max`, etc.
Triton provides atomic operations for accumulation into global memory when multiple programs write to the same location. Use sparingly in hot paths; prefer block-level reduction followed by a single atomic per block.

```python
tl.atomic_add(ptr + offsets, vals, mask=mask)
```

### `sem` Parameter
Control memory ordering for atomics: `"relaxed"`, `"acquire"`, `"release"`, `"acq_rel"`. Default is `"acq_rel"`. Use `"relaxed"` for commutative accumulations where ordering does not matter, to reduce synchronization overhead.

---

## Swizzling and Layout

### Swizzled SMEM Layouts
Triton's compiler applies swizzling automatically for `tl.dot` inputs to avoid shared memory bank conflicts. When loading tiles into the layout expected by `tl.dot`, the compiler emits swizzled shared memory stores. You do not need to implement XOR swizzling manually; it is handled when you use `tl.load` with the correct `order` parameter and `tl.dot`.

### Output Layout
The output of `tl.dot` is in a specific distributed layout across threads. Storing back to global memory via `tl.store` handles the layout transformation automatically. Avoid applying elementwise operations that break the layout before storing; fuse them as Triton elementwise ops (they are layout-transparent).

---

## Multi-Dimensional Grids

### 2D / 3D Grids
Use multiple `program_id` axes for 2D output tiles (e.g., GEMM output):

```python
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)
```

Launch with `grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N))`.

### L2 Cache–Friendly Program Ordering (Grouped / Swizzled Grid)
By default, programs are launched in row-major order. For GEMM, this means all programs in a row finish before the next row starts, leading to poor L2 reuse of the B matrix. Use a grouped/swizzled launch order to maximize L2 hits:

```python
GROUP_SIZE_M = 8
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

This ensures consecutive PIDs reuse the same B tile from L2.

---

## Architecture-Specific Features

### Ampere (A100)
- `cp.async` is used automatically when `num_stages > 1`. No manual intervention needed.
- `allow_tf32=True` in `tl.dot` uses TF32 tensor cores for FP32 inputs (~8× throughput vs. FP64).
- Target with `--target cuda:80` or inferred automatically by Triton.

### Hopper (H100)
- **TMA-backed block pointers**: `tl.make_block_ptr` on Hopper automatically uses TMA hardware for bulk async copies when `num_stages > 1`. This decouples address generation from SMs.
- **`num_stages=3` or higher**: Triton uses TMA + warpgroup MMA pipeline automatically on `sm_90a`.
- **Warpgroup MMA (`wgmma`)**: Triton emits `wgmma` instructions automatically for `tl.dot` on Hopper. To enable: compile with `triton.runtime.driver.active.utils.get_device_capability() >= (9, 0)` and use `triton.language.dot` normally.
- **FP8 support**: `tl.float8e4nv` (E4M3) and `tl.float8e5` (E5M2) dtypes. Cast with `.to(tl.float8e4nv)` before `tl.dot`.
- **Thread block clusters**: experimental in Triton; controllable via `num_ctas` in the autotune config for TMA multicast.

### AMD (MI300X / CDNA3)
- Triton supports AMD targets via HIP backend (`--target hip:gfx942`).
- Wave size is 64 threads (not 32); `tl.arange(0, 64)` for a full wave.
- `tl.dot` maps to MFMA instructions.
- Shared memory is called LDS; same principles as CUDA SMEM apply.
- Tune `num_warps` and tile sizes separately from NVIDIA targets; configs do not transfer directly.

---

## Debugging and Profiling

### `triton.testing.do_bench`
Measures kernel throughput in milliseconds. Use for quick performance iteration:

```python
ms = triton.testing.do_bench(lambda: my_kernel[grid](...))
```

### `TRITON_INTERPRET=1`
Runs Triton kernels on CPU in an interpreted mode for debugging. Allows use of Python `print` and `pdb` inside kernels. Much slower than GPU execution; only use for correctness debugging.

### `print` Inside Kernels
`tl.device_print("val", val)` prints values from the device for debugging. Disable before benchmarking as it severely impacts performance.

### Nsight Compute with Triton
Profile Triton kernels with `ncu` as you would any CUDA kernel. The generated PTX/SASS is inspectable via `ncu --import-source on`. Use the Roofline and Memory sections to identify bottlenecks. Triton-generated kernel names follow the pattern `triton__<hash>`.

### PTX / LLVM IR Inspection
Access the compiled artifacts for a kernel:
```python
pgm = my_kernel[grid](...)  # run once to compile
print(my_kernel.asm['ptx'])   # PTX
print(my_kernel.asm['ttgir']) # Triton GPU IR
print(my_kernel.asm['llir'])  # LLVM IR
```

---

## General Best Practices

- **Tile sizes must be powers of two**: Triton requires this for efficient vectorization and loop unrolling.
- **Fuse elementwise ops**: Apply activations, scaling, and bias addition inside the kernel using Triton elementwise ops rather than launching separate kernels. Avoids round-trips through global memory.
- **Avoid Python control flow based on runtime values**: All branching must be based on `tl.constexpr` values or tensor operations. Python `if` on non-constexpr values is not supported inside `@triton.jit` kernels.
- **Use `tl.cdiv` for ceiling division**: Cleaner and less error-prone than `(N + B - 1) // B` patterns.
- **Reference Triton tutorials and Flash Attention v2**: The official Triton GEMM and Flash Attention implementations are the canonical examples of best-practice tiling, pipelining, and grid ordering.
- **Profile, then tune**: Always measure with `do_bench` before and after changes. Intuitions from CUDA do not always transfer; Triton's compiler can surprise in both directions.
