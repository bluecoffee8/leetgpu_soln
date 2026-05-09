import torch
import triton
import triton.language as tl

@triton.jit 
def conv2d_kernel(
    input, kernel, output, 
    input_rows, input_cols, 
    kernel_rows, kernel_cols, 
    input_row_stride, input_col_stride,
    kernel_row_stride, kernel_col_stride,
    output_row_stride, output_col_stride,
    BLOCK_SIZE_R: tl.constexpr, 
    BLOCK_SIZE_C: tl.constexpr, 
):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    offset_o_r = pid0 * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offset_o_c = pid1 * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offset_o = offset_o_r[:, None] * output_row_stride + offset_o_c[None, :] * output_col_stride
    mask_o_r = offset_o_r < input_rows - kernel_rows + 1
    mask_o_c = offset_o_c < input_cols - kernel_cols + 1
    mask_o = mask_o_r[:, None] & mask_o_c[None, :]

    acc = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_C), dtype=tl.float32)
    for i in range(kernel_rows):
        for j in range(kernel_cols): 
            k_val = tl.load(kernel + i * kernel_row_stride + j * kernel_col_stride)

            offset_i_r = offset_o_r + i
            offset_i_c = offset_o_c + j 
            offset_i = offset_i_r[:, None] * input_row_stride + offset_i_c[None, :] * input_col_stride 
            mask_i_r = offset_i_r < input_rows 
            mask_i_c = offset_i_c < input_cols 
            mask_i = mask_i_r[:, None] & mask_i_c[None, :]
            i_val = tl.load(input + offset_i, mask=mask_i, other=0.0)

            acc += k_val * i_val 
    tl.store(output + offset_o, acc, mask_o)

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
    BLOCK_SIZE_R = 32
    BLOCK_SIZE_C = 32
    grid = lambda meta: (triton.cdiv(input_rows - kernel_rows + 1, BLOCK_SIZE_R), triton.cdiv(input_cols - kernel_cols + 1, BLOCK_SIZE_C))
    conv2d_kernel[grid](
        input, kernel, output, 
        input_rows, input_cols, 
        kernel_rows, kernel_cols, 
        input_cols, 1, 
        kernel_cols, 1, 
        input_cols - kernel_cols + 1, 1, 
        BLOCK_SIZE_R, BLOCK_SIZE_C
    )