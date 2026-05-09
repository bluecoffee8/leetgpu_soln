import torch
import triton
import triton.language as tl

@triton.jit 
def jacobi_stencil(input, output, rows, cols, 
                   BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, 
                   GROUPSIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1) 

    n_pid0 = tl.num_programs(axis=0) 
    n_pid1 = tl.num_programs(axis=1)
    pid0, pid1 = tl.swizzle2d(pid0, pid1, n_pid0, n_pid1, GROUPSIZE)

    acc = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_C), dtype=tl.float32)

    for z in tl.static_range(-1, 3, 2):
        oR = pid0 * BLOCK_SIZE_R + z + tl.arange(0, BLOCK_SIZE_R)
        oC = pid1 * BLOCK_SIZE_C + 0 + tl.arange(0, BLOCK_SIZE_C)
        mR = (oR[:, None] < rows) & (oR[:, None] >= 0)
        mC = (oC[None, :] < cols) & (oC[None, :] >= 0)
        offset = oR[:, None] * cols + oC[None, :]
        mask = mR & mC 
        M = tl.load(input + offset, mask=mask, other=0.0)
        acc += M 

        oR = pid0 * BLOCK_SIZE_R + 0 + tl.arange(0, BLOCK_SIZE_R)
        oC = pid1 * BLOCK_SIZE_C + z + tl.arange(0, BLOCK_SIZE_C)
        mR = (oR[:, None] < rows) & (oR[:, None] >= 0)
        mC = (oC[None, :] < cols) & (oC[None, :] >= 0)
        offset = oR[:, None] * cols + oC[None, :]
        mask = mR & mC 
        M = tl.load(input + offset, mask=mask, other=0.0)
        acc += M 

    oR = pid0 * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    oC = pid1 * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mR = (oR[:, None] < rows) & (oR[:, None] >= 0)
    mC = (oC[None, :] < cols) & (oC[None, :] >= 0)
    offset = oR[:, None] * cols + oC[None, :]
    mask = mR & mC

    M = tl.load(input + offset, mask=mask, other=0.0)
    border = (oR[:, None] == 0) | (oR[:, None] == rows-1) | (oC[None, :] == 0) | (oC[None, :] == cols-1)
    acc = tl.where(border, M, 0.25 * acc)

    tl.store(output + offset, acc, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_SIZE_R = 64
    BLOCK_SIZE_C = 64
    GROUPSIZE = 8
    grid = (triton.cdiv(rows, BLOCK_SIZE_R), triton.cdiv(cols, BLOCK_SIZE_C))
    jacobi_stencil[grid](input, output, rows, cols, BLOCK_SIZE_R, BLOCK_SIZE_C, GROUPSIZE)
    