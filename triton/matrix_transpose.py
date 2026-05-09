import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, 
    BLOCK_SIZE_R: tl.constexpr, 
    BLOCK_SIZE_C: tl.constexpr):

    pid = tl.program_id(axis=0)
    num_pid_r = tl.cdiv(rows, BLOCK_SIZE_R)
    num_pid_c = tl.cdiv(cols, BLOCK_SIZE_C)
    pid_r = pid // num_pid_c 
    pid_c = pid % num_pid_c 
    offset_r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % rows
    offset_c = (pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)) % cols 
    i_ptrs = input + (offset_r[:, None] * stride_ir + offset_c[None, :] * stride_ic)
    o_ptrs = output + (offset_c[:, None] * stride_or + offset_r[None, :] * stride_oc)
    tile = tl.load(i_ptrs, mask=(offset_r[:, None] < rows) & (offset_c[None, :] < cols), other=0.0)
    tile_T = tl.trans(tile) 
    tl.store(o_ptrs, tile_T, mask=(offset_c[:, None] < cols) & (offset_r[None, :] < rows))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_SIZE_R"]) * triton.cdiv(cols, meta["BLOCK_SIZE_C"]), )
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, BLOCK_SIZE_R=64, BLOCK_SIZE_C=64
    )
