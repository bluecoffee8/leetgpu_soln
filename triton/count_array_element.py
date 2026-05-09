import torch
import triton
import triton.language as tl

@triton.jit 
def count_array_elem_kernel(input, output, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N 
    x = tl.load(input + offset, mask=mask, other=0) 
    cnt = tl.sum(tl.where(x == K, 1, 0))
    tl.atomic_add(output, cnt)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 1024   
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    count_array_elem_kernel[grid](input, output, N, K, BLOCK_SIZE)