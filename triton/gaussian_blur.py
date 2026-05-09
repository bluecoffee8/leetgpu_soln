import torch
import triton
import triton.language as tl

@triton.jit 
def gaussian_blur_kernel(
    input, kernel, output,
    input_rows, input_cols, 
    kernel_rows, kernel_cols, 
    BLOCK_SIZE: tl.constexpr, 
):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    o_r = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    o_c = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            k_val = tl.load(kernel + i * kernel_cols + j)

            o_r_ = o_r - (kernel_rows // 2) + i 
            o_c_ = o_c - (kernel_cols // 2) + j
            i_val = tl.load(input + o_r_[:, None] * input_cols + o_c_[None, :],
                            mask=(0 <= o_r_[:, None]) & (o_r_[:, None] < input_rows)
                            & (0 <= o_c_[None, :]) & (o_c_[None, :] < input_cols), other=0.0)
            
            acc += k_val * i_val 

    tl.store(output + o_r[:, None] * input_cols + o_c[None, :], acc, mask=(o_r[:, None] < input_rows) & (o_c[None, :] < input_cols))

# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(input_rows, BLOCK_SIZE), triton.cdiv(input_cols, BLOCK_SIZE))
    gaussian_blur_kernel[grid](
        input, kernel, output, 
        input_rows, input_cols, 
        kernel_rows, kernel_cols, 
        BLOCK_SIZE
    )