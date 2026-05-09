import torch
import triton
import triton.language as tl

@triton.jit 
def count2d_kernel(input, output, N, M, K, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, GROUPSIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1) 

    num_programs_pid0 = tl.num_programs(0)
    num_programs_pid1 = tl.num_programs(1)
    pid0, pid1 = tl.swizzle2d(pid0, pid1, num_programs_pid0, num_programs_pid1, GROUPSIZE)

    oN = pid0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    oM = pid1 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    x = tl.load(input + oN[:, None] * M + oM[None, :], mask=(oN[:, None] < N) & (oM[None, :] < M), other=0)
    x = tl.where(x == K, 1, 0)
    tl.atomic_add(output, tl.sum(x))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    GROUPSIZE = 8
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))
    count2d_kernel[grid](input, output, N, M, K, BLOCK_SIZE_N, BLOCK_SIZE_M, GROUPSIZE)