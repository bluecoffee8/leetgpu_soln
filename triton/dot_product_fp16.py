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
    a_val.to(tl.float32)
    b_val.to(tl.float32)
    partial_dot = tl.sum(a_val * b_val)
    tl.atomic_add(result, partial_dot)

# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    BLOCK_SIZE = 64
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
    fp_result = torch.zeros((1, ), dtype=torch.float32, device='cuda')
    dot_kernel[grid](a, b, fp_result, n, BLOCK_SIZE)
    result.copy_(fp_result.to(torch.float16))