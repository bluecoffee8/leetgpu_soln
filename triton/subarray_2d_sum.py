import torch
import triton
import triton.language as tl

@triton.jit 
def array2d_sum(input, output, N: tl.constexpr, M: tl.constexpr, 
                S_ROW: tl.constexpr, E_ROW: tl.constexpr,
                S_COL: tl.constexpr, E_COL: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    oR = S_ROW + pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    oC = S_COL + pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 
    mR = (oR[:, None] <= E_ROW) 
    mC = (oC[None, :] <= E_COL) 

    x = tl.load(input + oR[:, None] * M + oC[None, :], mask=(mR & mC), other=0.0)
    tl.atomic_add(output, tl.sum(x))

# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE = 64 
    grid = (triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE), triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE))
    array2d_sum[grid](input, output, N, M, S_ROW, E_ROW, S_COL, E_COL, BLOCK_SIZE)
