import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offset < n_elements)
    tile = tl.load(input + offset, mask=mask, other=0.0)
    lrelu = tl.where(tile > 0, tile, 0.01 * tile)
    tl.store(output + offset, lrelu, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](input, output, N, BLOCK_SIZE)
