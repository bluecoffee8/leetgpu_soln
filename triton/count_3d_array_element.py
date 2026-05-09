import torch
import triton
import triton.language as tl

@triton.jit 
def count3d_kernel(input, output, T: tl.constexpr, P: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < T 

    x = tl.load(input + offset, mask=mask, cache_modifier=".ca")
    s = (x == P).sum(); 

    if s > 0:
        tl.atomic_add(output, s, sem="relaxed")

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int, P: int):
    BLOCK_SIZE = 1024
    T = N * M * K
    grid = (triton.cdiv(T, BLOCK_SIZE), )
    count3d_kernel[grid](input, output, T, P, BLOCK_SIZE)
