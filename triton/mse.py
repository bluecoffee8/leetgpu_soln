import torch
import triton
import triton.language as tl

@triton.jit 
def mse_kernel(predictions, targets, mse, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N 
    p = tl.load(predictions + offset, mask=mask, other=0.0)
    t = tl.load(targets + offset, mask=mask, other=0.0)
    mse_val = tl.sum((p - t) * (p - t)) / N 
    tl.atomic_add(mse, mse_val)

# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    mse_kernel[grid](predictions, targets, mse, N, BLOCK_SIZE)
