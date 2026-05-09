import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr, KERNEL_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < input_size)
    for k in range(kernel_size):
        k_val = tl.load(kernel + k)
        w_val = tl.load(input + offsets + k, mask=mask, other=0.0)
        acc += k_val * w_val 
    tl.store(output + offsets, acc, mask=(offsets < input_size - kernel_size + 1))

# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    KERNEL_SIZE = 2048
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE, KERNEL_SIZE)
