import torch
import triton
import triton.language as tl


@triton.jit
def rgb_to_grayscale_kernel(input, output, width, height, BLOCK_SIZE: tl.constexpr):
    N = width * height * 3
    pid = tl.program_id(axis=0)
    ro = pid * BLOCK_SIZE * 3 + 3 * tl.arange(0, BLOCK_SIZE)
    go = pid * BLOCK_SIZE * 3 + 3 * tl.arange(0, BLOCK_SIZE) + 1
    bo = pid * BLOCK_SIZE * 3 + 3 * tl.arange(0, BLOCK_SIZE) + 2
    rv = tl.load(input + ro, mask=(ro < N), other=0.0)
    gv = tl.load(input + go, mask=(go < N), other=0.0)
    bv = tl.load(input + bo, mask=(bo < N), other=0.0)
    o = 0.299 * rv + 0.587 * gv + 0.114 * bv 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(output + offset, o, mask=(offset < width * height))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    total_pixels = width * height
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_pixels, BLOCK_SIZE),)
    rgb_to_grayscale_kernel[grid](input, output, width, height, BLOCK_SIZE)
