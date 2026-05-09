import torch
import triton
import triton.language as tl

@triton.jit 
def max_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N 
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    tl.atomic_max(out_ptr, tl.max(x))

@triton.jit 
def sumexp_kernel(x_ptr, max_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    m = tl.load(max_ptr) 
    pid = tl.program_id(axis=0) 
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N 
    x = tl.load(x_ptr + offsets, mask=mask)
    z = tl.where(mask, tl.exp(x - m), 0)
    tl.atomic_add(out_ptr, tl.sum(z))

@triton.jit 
def softmax_kernel(input, max_ptr, div_ptr, output, N, BLOCK_SIZE: tl.constexpr):
    m = tl.load(max_ptr) 
    div = tl.load(div_ptr)
    pid = tl.program_id(axis=0)
    ofs = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * pid
    mask = ofs < N 
    x = tl.load(input + ofs, mask=mask)
    z = tl.exp(x - m)
    y = tl.fdiv(z, div)
    tl.store(output + ofs, y, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 2048
    maxi = torch.zeros(1, 1, device=input.device)
    div = torch.zeros(1, 1, device=input.device)
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    max_kernel[grid](input, maxi, N, BLOCK_SIZE)
    sumexp_kernel[grid](input, maxi, div, N, BLOCK_SIZE)
    softmax_kernel[grid](input, maxi, div, output, N, BLOCK_SIZE)
