import torch
import triton
import triton.language as tl

@triton.jit
def max2dpool_kernel(input, output, N, C, H, W, i0, i1, i2, o0, o1, o2, kernel_size: tl.constexpr, stride, padding,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr):
    pid_nc = tl.program_id(axis=0) 
    pid_h = tl.program_id(axis=1) 
    pid_w = tl.program_id(axis=2) 

    offset_nc = pid_nc * i0
    max_val = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), -float("inf"), dtype=tl.float32)
    for i in tl.static_range(kernel_size):
        o_H = pid_h * BLOCK_SIZE_H * stride + (-padding) + i + tl.arange(0, BLOCK_SIZE_H) * stride
        for j in tl.static_range(kernel_size):
            # o_H = pid_h * BLOCK_SIZE_H * stride + (-padding) + i + tl.arange(0, BLOCK_SIZE_H) * stride
            o_W = pid_w * BLOCK_SIZE_W * stride + (-padding) + j + tl.arange(0, BLOCK_SIZE_W) * stride

            ptrs = offset_nc + o_H[:, None] * i1 + o_W[None, :] * i2
            mask = (o_H[:, None] >= 0) & (o_H[:, None] < H) & (o_W[None, :] >= 0) & (o_W[None, :] < W)

            x = tl.load(input + ptrs, mask=mask, other=-float("inf"))
            max_val = tl.maximum(max_val, x) 

    oH = (H + 2 * padding - kernel_size) // stride + 1
    oW = (W + 2 * padding - kernel_size) // stride + 1
    oo_nc = pid_nc * o0
    oo_H = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    oo_W = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    tl.store(output + oo_nc + oo_H[:, None] * o1 + oo_W[None, :] * o2, max_val, 
             mask=(oo_H[:, None] < oH) & (oo_W[None, :] < oW))

# input, output are tensors on the GPU
def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 128 
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    B = N * C
    grid = (B, triton.cdiv(H_out, BLOCK_SIZE_H), triton.cdiv(W_out, BLOCK_SIZE_W))
    input = input.reshape(B, H, W)
    output = output.reshape(B, H_out, W_out)
    max2dpool_kernel[grid](
        input, output, N, C, H, W, 
        input.stride(0), input.stride(1), input.stride(2), 
        output.stride(0), output.stride(1), output.stride(2), 
        kernel_size, stride, padding, BLOCK_SIZE_H, BLOCK_SIZE_W
    )
