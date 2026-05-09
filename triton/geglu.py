import torch
import triton
import triton.language as tl
import math


@triton.jit
def geglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    lo = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lm = (lo < N // 2)
    ro = N // 2 + lo 
    rm = (ro < N) 
    l = tl.load(input + lo, mask=lm, other=0.0)
    r = tl.load(input + ro, mask=rm, other=0.0)
    r = 0.5 * r * (1.0 + tl.erf(r / math.sqrt(2)))
    o = l * r 
    tl.store(output + lo, o, mask=lm)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
