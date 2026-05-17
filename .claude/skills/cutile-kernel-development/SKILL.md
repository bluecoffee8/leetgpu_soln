---
name: cutile-kernel-development
description: Guide for development and performance optimization of cutile DSL kernels.
---

# Performance Optimization

CuTile (`cuda.tile`) is NVIDIA's tile-based GPU programming model for Python. Unlike SIMT (thread-per-element) models, CuTile operates on **tiles** — contiguous blocks of data that a group of GPU threads processes collectively. The compiler automatically partitions tile operations onto threads, maps them to tensor cores, and manages shared memory. Your job is to express the algorithm in terms of tiles, tile sizes, and grid structure.

Install: `pip install cuda-tile` (requires CUDA 13.1+, driver r580+, GPU compute capability 8.x/10.x/11.x/12.x).

---

## Core Programming Model

### `@ct.kernel` and `ct.launch`

```python
import cuda.tile as ct
import torch

@ct.kernel
def my_kernel(A, B, C, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
    bidx = ct.bid(0)   # block ID along axis 0
    bidy = ct.bid(1)   # block ID along axis 1
    a_tile = ct.load(A, index=(bidx,), shape=(tile_m,))
    b_tile = ct.load(B, index=(bidy,), shape=(tile_n,))
    ...

stream = torch.cuda.current_stream()
grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
ct.launch(stream, grid, my_kernel, (A, B, C, TM, TN))
```

- `@ct.kernel` compiles the function to GPU code via Tile IR.
- `ct.Constant[int]` annotations mark compile-time constants folded into the kernel; required for tile shapes.
- `ct.launch(stream, grid, kernel, args_tuple)` queues the kernel on the given CUDA stream.
- `ct.bid(axis)` returns the block ID along the given grid axis (0, 1, or 2).
- `ct.cdiv(a, b)` — ceiling integer division, used for computing grid dimensions.

### Architecture-Specific CTA Count

```python
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))   # 2 CTAs per SM on Blackwell
def my_kernel(...):
    ...
```

---

## Data Model: Arrays vs. Tiles

**Arrays** — device memory buffers. Accept PyTorch or CuPy tensors as arguments. Support only `ct.load` and `ct.store`.

**Tiles** — immutable compile-time-shaped values that live in kernel code. All computation happens on tiles. Tile dimensions **must be powers of two**.

---

## Memory Operations

### `ct.load` / `ct.store`

```python
# 1D load: block pid loads tile_size elements starting at offset pid*tile_size
tile = ct.load(array, index=(pid,), shape=(tile_size,))

# 2D load: block (bidx, bidy) loads a (tm, tn) sub-tile
tile = ct.load(A, index=(bidx, bidy), shape=(tm, tn))

# Load with zero-padding for out-of-bounds (important for non-power-of-two sizes)
tile = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)

# Store
ct.store(C, index=(bidx, bidy), tile=result)
```

The index `(bidx, bidy)` addresses the tile in **tile coordinates**, not element coordinates. Element offset = `bidx * tm` along axis 0.

### Atomic Operations

```python
ct.atomic_add(result, (0,), partial_sum)          # result[0] += partial_sum
ct.atomic_cas(array, index, compare, val, memory_order)  # compare-and-swap
```

Use atomics for cross-block reduction (e.g., accumulating partial sums into a scalar).

---

## Compute Operations

### Elementwise Arithmetic

Standard Python operators work on tiles directly:
```python
c = a_tile + b_tile
c = a_tile * b_tile - scalar
c = a_tile / b_tile
```

### Matrix Multiplication — `ct.mma`

```python
accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
for k in range(num_tiles_k):
    a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)
    accumulator = ct.mma(a, b, accumulator)
```

`ct.mma(a, b, c)` — fused multiply-add: `c += a @ b`. Automatically uses tensor cores when the dtype and tile shape qualify.

For FP32 inputs, cast to TF32 first for tensor core throughput:
```python
dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
a = ct.load(...).astype(dtype)
```

### Reductions

```python
total = ct.sum(tile)           # sum all elements → scalar
m     = ct.max(tile)           # max reduction → scalar
m     = ct.min(tile)           # min reduction → scalar
```

### Other Tile Operations

```python
ct.full((tm, tn), 0, dtype=ct.float32)    # create a constant tile
ct.transpose(tile)                          # transpose a 2D tile
ct.astype(tile, dtype)                      # cast dtype
ct.maximum(a, b)                            # elementwise max (also usable as abs: ct.maximum(x, -x))
```

---

## Tile Shape Utilities

```python
num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))   # number of tiles along axis 1
num_blocks  = ct.num_blocks(0)                           # total blocks in grid axis 0
```

---

## Common Algorithm Patterns

### Vector Addition

```python
@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)

grid = (ct.cdiv(N, TILE_SIZE), 1, 1)
ct.launch(stream, grid, vector_add, (a, b, c, TILE_SIZE))
```

### Matrix Transpose

```python
@ct.kernel
def transpose_kernel(x, y, tm: ct.Constant[int], tn: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))
    ct.store(y, index=(bidy, bidx), tile=ct.transpose(tile))

grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
ct.launch(stream, grid, transpose_kernel, (x, y, TM, TN))
```

### Tiled GEMM with Swizzled Grid (Cache-Friendly)

Swizzle block IDs so consecutive blocks share the same row of B tiles, improving L2 reuse:

```python
def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M=8):
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n

@ct.kernel
def matmul_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bidx, bidy = swizzle_2d(A.shape[0], B.shape[1], tm, tn)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))

M, N, K = A.shape[0], B.shape[1], A.shape[1]
grid = (ct.cdiv(M, TM) * ct.cdiv(N, TN), 1, 1)   # 1D grid with swizzled dispatch
ct.launch(stream, grid, matmul_kernel, (A, B, C, TM, TN, TK))
```

### Persistent GEMM (Better SM Utilization)

Each block loops over multiple output tiles; launch exactly `NUM_SMS` blocks:

```python
@ct.kernel
def persistent_matmul_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bid = ct.bid(0)
    M, N = A.shape[0], B.shape[1]
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    num_total = ct.cdiv(M, tm) * ct.cdiv(N, tn)
    num_tile_blocks = ct.num_blocks(0)
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for current_bid in range(bid, num_total, num_tile_blocks):
        acc = ct.full((tm, tn), 0, dtype=ct.float32)
        bidx, bidy = swizzle_2d_from_bid(M, N, tm, tn, 8, current_bid)
        for k in range(num_tiles_k):
            a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
            b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
            acc = ct.mma(a, b, acc)
        ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
grid = (min(NUM_SMS, ct.cdiv(M, TM) * ct.cdiv(N, TN)), 1, 1)
ct.launch(stream, grid, persistent_matmul_kernel, (A, B, C, TM, TN, TK))
```

### Reduction with Atomics

```python
@ct.kernel
def reduce_sum_kernel(a, result, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(a, index=(pid,), shape=(tile_size,))
    partial = ct.sum(tile)
    ct.atomic_add(result, (0,), partial)

result = torch.zeros(1, dtype=a.dtype, device='cuda')
grid = (ct.cdiv(N, TILE_SIZE), 1, 1)
ct.launch(stream, grid, reduce_sum_kernel, (a, result, TILE_SIZE))
```

### Reduction without Atomics (Two-Pass)

Write partials to a buffer; reduce on the host or in a second kernel:

```python
@ct.kernel
def partial_reduce_kernel(a, partials, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(partials, index=(pid,), tile=ct.sum(tile))

num_blocks = ct.cdiv(N, TILE_SIZE)
partials = torch.zeros(num_blocks, dtype=a.dtype, device='cuda')
ct.launch(stream, (num_blocks, 1, 1), partial_reduce_kernel, (a, partials, TILE_SIZE))
total = partials.sum()   # host-side final reduction
```

---

## Performance Optimization

### Tile Size Selection

Tile dimensions must be powers of two. Larger tiles amortize launch overhead and improve tensor core utilization; smaller tiles fit more blocks in-flight.

| Kernel type     | Typical tile sizes        | Notes                                      |
|-----------------|---------------------------|--------------------------------------------|
| GEMM            | TM=TN=128, TK=32–64       | `ct.mma` uses tensor cores automatically   |
| GEMM thin M     | TM=64, TN=128–256, TK=32  | For small-batch inference                  |
| Elementwise     | 512–4096 elements/block   | Bandwidth-bound; larger = fewer launches   |
| Reduction       | 1024–8192 elements/block  | Larger tile → fewer atomic contention      |
| Transpose       | 32×32 or 64×64            | Square tiles minimize bank conflicts       |

### Swizzled Grid Ordering

Default row-major grid scheduling destroys L2 reuse of the B (or KV) matrix. Group blocks so `GROUP_SIZE_M=8` consecutive block IDs share the same B column tile. This is the single highest-impact optimization for GEMM-like workloads.

### Persistent Kernels

Launch exactly `NUM_SMS` blocks (or a small multiple). Each block loops over its share of output tiles. Benefits:
- Better L2 temporal reuse across tile iterations.
- Eliminates launch overhead for problems with many small tiles.
- GPU occupancy stays high on small matrices.

Use `ct.num_blocks(0)` inside the kernel to get the actual launch count for the stride.

### TF32 for FP32 Inputs

```python
dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
a = ct.load(A, ...).astype(dtype)
```

`ct.tfloat32` routes FP32 through TF32 tensor cores (~8× throughput over CUDA cores). Precision loss is ~3 mantissa bits; acceptable for most ML workloads.

### Zero-Padding for Non-Power-of-Two Shapes

Always pass `padding_mode=ct.PaddingMode.ZERO` when the problem dimension is not a multiple of the tile size. The compiler masks out-of-bounds loads to zero, preventing incorrect accumulation.

### Dtype Casting

```python
acc = ct.full((tm, tn), 0, dtype=ct.float32)   # always accumulate in FP32
a_f16 = ct.load(A, ...).astype(ct.float16)      # inputs in FP16/BF16
acc = ct.mma(a_f16, b_f16, acc)
result = ct.astype(acc, ct.float16)              # downcast output
ct.store(C, ..., tile=result)
```

---

## Data Types

| `cuda.tile` type  | Meaning                    |
|-------------------|----------------------------|
| `ct.float32`      | 32-bit float               |
| `ct.float16`      | 16-bit float               |
| `ct.bfloat16`     | bfloat16                   |
| `ct.tfloat32`     | TF32 (for tensor cores)    |
| `ct.int32`        | 32-bit integer             |
| `ct.float64`      | 64-bit float               |

---

## Full API Quick Reference

| API                                             | Description                                          |
|-------------------------------------------------|------------------------------------------------------|
| `@ct.kernel`                                    | Decorator to compile a function as a GPU kernel      |
| `@ct.kernel(num_ctas=ct.ByTarget(sm_100=N))`    | Kernel with architecture-specific CTA count          |
| `ct.launch(stream, grid, fn, args)`             | Launch kernel on a CUDA stream                       |
| `ct.bid(axis)`                                  | Block ID along axis 0/1/2                            |
| `ct.cdiv(a, b)`                                 | Ceiling integer division                             |
| `ct.num_tiles(array, axis, shape)`              | Number of tiles along an axis                        |
| `ct.num_blocks(axis)`                           | Total blocks in the launched grid along axis         |
| `ct.Constant[int]`                              | Type annotation for compile-time constant parameter  |
| `ct.load(arr, index, shape, padding_mode=...)`  | Load a tile from device memory                       |
| `ct.store(arr, index, tile)`                    | Write a tile to device memory                        |
| `ct.atomic_add(arr, index, val)`                | Atomic add a scalar into array                       |
| `ct.atomic_cas(arr, idx, cmp, val, order)`      | Atomic compare-and-swap                              |
| `ct.full(shape, val, dtype)`                    | Create a constant-filled tile                        |
| `ct.mma(a, b, c)`                               | Fused matrix multiply-accumulate: `c + a @ b`        |
| `ct.sum(tile)`                                  | Sum all elements → scalar                            |
| `ct.max(tile)`                                  | Max reduction → scalar                               |
| `ct.min(tile)`                                  | Min reduction → scalar                               |
| `ct.maximum(a, b)`                              | Elementwise max of two tiles (or tile and scalar)    |
| `ct.transpose(tile)`                            | Transpose a 2D tile                                  |
| `ct.astype(tile, dtype)`                        | Cast tile dtype                                      |
| `tile.astype(dtype)`                            | Same as `ct.astype`, method form                     |
| `ct.PaddingMode.ZERO`                           | Zero-pad out-of-bounds loads                         |
| `ct.ByTarget(sm_100=N, ...)`                    | Architecture-specific value selector                 |

---

## Debugging and Profiling

### Correctness Check

```python
torch.testing.assert_close(cutile_output, reference, atol=1e-2, rtol=1e-2)
```

Always validate against a reference PyTorch implementation before tuning.

### Nsight Compute

```bash
ncu --set detailed -o profile python my_kernel.py
```

The **Tile Statistics** section shows tile block counts and compiler-selected block sizes at source-line granularity. Use Memory Workload Analysis and Roofline sections to identify bandwidth vs. compute bottlenecks.

### Benchmarking

```python
import triton
quantiles = [0.5, 0.2, 0.8]
ms, lo, hi = triton.testing.do_bench(
    lambda: ct.launch(stream, grid, my_kernel, args),
    quantiles=quantiles
)
tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
```

---

## General Best Practices

- **Tile dimensions must be powers of two.** Pad inputs if necessary; use `ct.PaddingMode.ZERO` on loads.
- **Express algorithms as tile operations, not per-thread index arithmetic.** `ct.mma`, `ct.sum`, `ct.transpose`, `ct.maximum` cover the common cases.
- **Always accumulate in FP32.** Cast FP16/BF16 inputs at load time; downcast the result at store time.
- **Use TF32 for FP32 GEMM.** The ~3-bit mantissa loss is acceptable for ML; the throughput gain is 8×.
- **Use swizzled grid ordering for GEMM.** `GROUP_SIZE_M=8` is a robust default.
- **Use persistent kernels for large-grid workloads.** Launch `NUM_SMS` blocks, loop internally.
- **Prefer two-pass reduction over atomics for large N.** Atomic contention serializes blocks; writing partials and reducing on the host avoids it.
- **Profile before tuning.** L2 hit rate, tensor core utilization, and memory bandwidth are distinct bottlenecks. Nsight Compute identifies which dominates.
