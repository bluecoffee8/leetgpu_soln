import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a_ptr = a + offset
    b_ptr = b + offset
    tile_a = tl.load(a_ptr, mask=(offset < n_elements), other=0.0)
    tile_b = tl.load(b_ptr, mask=(offset < n_elements), other=0.0)
    c_ptr = c + offset
    tl.store(c_ptr, tile_a + tile_b, mask=(offset < n_elements))

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
