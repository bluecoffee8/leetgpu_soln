import torch
import triton
import triton.language as tl

@triton.jit 
def subarray_sum_kernel(input, output, N, S, E, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    if pid * BLOCK_SIZE > E:
        return 
    if (pid + 1) * BLOCK_SIZE <= S: 
        return 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offset <= E) & (offset >= S)
    x = tl.load(input + offset, mask=mask, other=0.0)
    tl.atomic_add(output, tl.sum(x), sem="relaxed")

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    subarray_sum_kernel[grid](input, output, N, S, E, BLOCK_SIZE)