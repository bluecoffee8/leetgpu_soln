import torch
import triton
import triton.language as tl

@triton.jit 
def reduce_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 
    mask = (offset < N) 
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    o = tl.sum(x)
    tl.atomic_add(output_ptr, o)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024 
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]), )
    reduce_kernel[grid](input, output, N, BLOCK_SIZE)