import torch
import triton
import triton.language as tl

@triton.jit 
def rope_kernel(Q, cos, sin, output, M, D, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_D: tl.constexpr):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    o_M = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    o_D_L = pid1 * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    o_D_R = pid1 * BLOCK_SIZE_D + D // 2 + tl.arange(0, BLOCK_SIZE_D)
    m_M = o_M[:, None] < M 
    m_D_L = o_D_L[None, :] < D // 2
    m_D_R = o_D_R[None, :] < D 
    L = tl.load(Q + o_M[:, None] * D + o_D_L, mask=m_M & m_D_L, other=0.0)
    R = tl.load(Q + o_M[:, None] * D + o_D_R, mask=m_M & m_D_R, other=0.0)
    L_cos = tl.load(cos + o_M[:, None] * D + o_D_L, mask=m_M & m_D_L, other=0.0)
    R_cos = tl.load(cos + o_M[:, None] * D + o_D_R, mask=m_M & m_D_R, other=0.0)
    L_sin = tl.load(sin + o_M[:, None] * D + o_D_L, mask=m_M & m_D_L, other=0.0)
    R_sin = tl.load(sin + o_M[:, None] * D + o_D_R, mask=m_M & m_D_R, other=0.0) 

    L_output = L * L_cos + -R * L_sin 
    R_output = R * R_cos + L * R_sin 

    tl.store(output + o_M[:, None] * D + o_D_L, L_output, mask=m_M & m_D_L)
    tl.store(output + o_M[:, None] * D + o_D_R, R_output, mask=m_M & m_D_R)

# Q, cos, sin, output are tensors on the GPU
def solve(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_D = 64
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(D // 2, BLOCK_SIZE_D))
    rope_kernel[grid](Q, cos, sin, output, M, D, BLOCK_SIZE_M, BLOCK_SIZE_D)
