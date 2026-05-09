import torch
import triton
import triton.language as tl


@triton.jit
def swiglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    lo = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lm = (lo < N // 2)
    l = tl.load(input + lo, mask=lm, other=0.0)
    ro = N // 2 + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rm = (ro < N) 
    r = tl.load(input + ro, mask=rm, other=0.0)
    l = l * (1.0 / (1.0 + tl.exp(-l)))
    o = l * r 
    tl.store(output + lo, o, mask=(lo < N // 2))
    
# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
