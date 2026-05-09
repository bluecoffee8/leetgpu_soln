import torch
import triton
import triton.language as tl

@triton.jit 
def histogram_kernel(input, histogram, N, num_bins, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N 
    x = tl.load(input + offset, mask=mask, other=0.0)
    bins = x.to(tl.int32) 
    tl.atomic_add(histogram + bins, 1, mask=mask)

# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 2048   
    grid = lambda meta : (triton.cdiv(N, meta["BLOCK_SIZE"]), )
    histogram_kernel[grid](
        input, histogram, N, num_bins, BLOCK_SIZE
    )