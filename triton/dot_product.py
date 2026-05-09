import torch
import triton
import triton.language as tl


@triton.jit 
def dot_kernel(
    a, b, result, n, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0) 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n 
    a_val = tl.load(a + offset, mask=mask, other=0.0)
    b_val = tl.load(b + offset, mask=mask, other=0.0)
    partial_dot = tl.sum(a_val * b_val)
    tl.atomic_add(result, partial_dot)

# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
    dot_kernel[grid](a, b, result, n, BLOCK_SIZE)