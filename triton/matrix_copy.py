import torch
import triton
import triton.language as tl

@triton.jit 
def matrix_copy_kernel(
    a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0) 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offset < N * N)
    tile = tl.load(a_ptr + offset, mask=mask, other=0.0)
    tl.store(b_ptr + offset, tile, mask=mask)

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N * N, BLOCK_SIZE), )

    matrix_copy_kernel[grid](
        a, b, N, BLOCK_SIZE
    )