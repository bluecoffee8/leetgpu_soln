import torch
import triton
import triton.language as tl

@triton.jit 
def subarray_3d_sum(input, output, N, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1) 
    pid2 = tl.program_id(axis=2) 

    oD = S_DEP + pid0
    oR = S_ROW + pid1
    oC = S_COL + pid2 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mD = oD[:, None, None] <= E_DEP 
    # mR = oR[None, :, None] <= E_ROW 
    mC = oC <= E_COL 

    x = tl.load(input + oD * M * K + oR * K + oC, mask=mC, other=0.0)
    tl.atomic_add(output, tl.sum(x))

# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    K: int,
    S_DEP: int,
    E_DEP: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE = 1024
    grid = (E_DEP - S_DEP + 1, E_ROW - S_ROW + 1, triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE))
    subarray_3d_sum[grid](input, output, N, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL, BLOCK_SIZE)