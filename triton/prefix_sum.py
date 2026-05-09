import torch
import triton
import triton.language as tl

@triton.jit 
def partial_sum_kernel(data, output, partial_sum, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n 

    x = tl.load(data + offset, mask=mask, other=0.0)
    tl.store(partial_sum + pid, tl.sum(x))
    tl.store(output + offset, tl.cumsum(x), mask=mask)

@triton.jit 
def prefix_sum_kernel(output, partial_sum, n, BLOCK_SIZE: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    pid = tl.program_id(axis=0) 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n 

    data = tl.load(output + offset, mask=mask, other=0.0)
    prefix_mask = tl.arange(0, NUM_BLOCKS) < pid 
    sums = tl.load(partial_sum + tl.arange(0, NUM_BLOCKS), mask=prefix_mask, other=0.0)
    s = tl.sum(sums)
    data += s 
    tl.store(output+offset, data, mask=mask)

# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE), )
    partial_sum = torch.zeros(grid, dtype=torch.float32, device='cuda')
    partial_sum_kernel[grid](data, output, partial_sum, n, BLOCK_SIZE)
    prefix_sum_kernel[grid](output, partial_sum, n, BLOCK_SIZE, triton.next_power_of_2(grid[0]))
