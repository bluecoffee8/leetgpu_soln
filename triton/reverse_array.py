import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    lo = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lm = lo < N // 2
    ro = N - 1 - lo 
    rm = ro >= N // 2

    L = tl.load(input + lo, lm)
    R = tl.load(input + ro, rm)
    tl.store(input + lo, R, lm)
    tl.store(input + ro, L, rm)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)
