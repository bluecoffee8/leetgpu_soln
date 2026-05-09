import torch
import triton
import triton.language as tl

@triton.jit 
def max_subarray_sum(input, output, N, window_size: tl.constexpr, BLOCK_SIZE: tl.constexpr, BIG_BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + (-(window_size-1)) + tl.arange(0, BIG_BLOCK_SIZE)
    mask = (offset >= 0) & (offset < N)
    x = tl.load(input + offset, mask=mask, other=0.0)
    for i in range(-(window_size-1), BLOCK_SIZE):
        if (pid * BLOCK_SIZE + i >= 0) and (pid * BLOCK_SIZE + i + window_size - 1 < N):
            m = (offset >= pid * BLOCK_SIZE + i) & (offset <= pid * BLOCK_SIZE + i + window_size - 1)
            z = tl.sum(tl.where(m, x, 0))
            tl.atomic_max(output, z)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    BLOCK_SIZE = 8192
    BIG_BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE + 2 * (window_size - 1))
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    output[0] = -10 * window_size
    max_subarray_sum[grid](input, output, N, window_size, BLOCK_SIZE, BIG_BLOCK_SIZE)