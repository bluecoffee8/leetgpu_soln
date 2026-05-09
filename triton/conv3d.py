import torch
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    input, kernel, output, 
    input_depth, input_rows, input_cols,
    kernel_depth, kernel_rows, kernel_cols,
    BLOCK_SIZE_D: tl.constexpr, 
    BLOCK_SIZE_R: tl.constexpr, 
    BLOCK_SIZE_C: tl.constexpr, 
):
    pid_d = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)

    str_i_d, str_i_r, str_i_c = input_rows * input_cols, input_cols, 1 
    str_k_d, str_k_r, str_k_c = kernel_rows * kernel_cols, kernel_cols, 1
    str_o_d, str_o_r, str_o_c = (input_rows - kernel_rows + 1) * (input_cols - kernel_cols + 1), (input_cols - kernel_cols + 1), 1

    o_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    o_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    o_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_R, BLOCK_SIZE_C), dtype=tl.float32)
    for i in range(kernel_depth):
        for j in range(kernel_rows):
            for k in range(kernel_cols): 
                k_val = tl.load(kernel + i * str_k_d + j * str_k_r + k * str_k_c)

                i_d = o_d + i 
                i_r = o_r + j 
                i_c = o_c + k 
                i_o = i_d[:, None, None] * str_i_d + i_r[None, :, None] * str_i_r + i_c[None, None, :] * str_i_c
                i_d_m = i_d < input_depth 
                i_r_m = i_r < input_rows 
                i_c_m = i_c < input_cols 
                i_m = i_d_m[:, None, None] & i_r_m[None, :, None] & i_c_m[None, None, :]
                 
                i_val = tl.load(input + i_o, mask=i_m, other=0.0) 
                acc += k_val * i_val 

    o_d_m = o_d < input_depth - kernel_depth + 1
    o_r_m = o_r < input_rows - kernel_rows + 1
    o_c_m = o_c < input_cols - kernel_cols + 1
    o_o = o_d[:, None, None] * str_o_d + o_r[None, :, None] * str_o_r + o_c[None, None, :] * str_o_c
    o_m = o_d_m[:, None, None] & o_r_m[None, :, None] & o_c_m[None, None, :]
    tl.store(output + o_o, acc, mask=o_m)

# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    BLOCK_SIZE_D = 32
    BLOCK_SIZE_R = 32
    BLOCK_SIZE_C = 32

    grid = lambda meta: (triton.cdiv(input_depth - kernel_depth + 1, meta["BLOCK_SIZE_D"]),
                         triton.cdiv(input_rows - kernel_rows + 1, meta["BLOCK_SIZE_R"]),
                         triton.cdiv(input_cols - kernel_cols + 1, meta["BLOCK_SIZE_C"]))
    conv3d_kernel[grid](
        input, kernel, output, 
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, 
        BLOCK_SIZE_D, BLOCK_SIZE_R, BLOCK_SIZE_C
    )
