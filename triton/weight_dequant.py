import torch
import triton
import triton.language as tl
import math

@triton.jit 
def weight_dequant(X, S, Y, M, N, m, n, TILE_SIZE: tl.constexpr,
                   BLOCK_SIZE_m: tl.constexpr, 
                   BLOCK_SIZE_n: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1) 

    oM = pid0 * BLOCK_SIZE_m * TILE_SIZE + tl.arange(0, BLOCK_SIZE_m * TILE_SIZE)
    oN = pid1 * BLOCK_SIZE_n * TILE_SIZE + tl.arange(0, BLOCK_SIZE_n * TILE_SIZE)
    mM = (oM[:, None] < M)
    mN = (oN[None, :] < N)

    om = oM // TILE_SIZE
    on = oN // TILE_SIZE
    mm = (om[:, None] < m) 
    mn = (on[None, :] < n) 

    x = tl.load(X + oM[:, None] * N + oN[None, :], mask=(mM & mN), other=0.0)
    s = tl.load(S + om[:, None] * n + on[None, :], mask=(mm & mn), other=0.0)
    o = x * s 

    tl.store(Y + oM[:, None] * N + oN[None, :], o, mask=(mM & mN))

# X, S, Y are tensors on the GPU
def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int):
    m = math.ceil(M / TILE_SIZE) 
    n = math.ceil(N / TILE_SIZE)
    BLOCK_SIZE_m = 1
    BLOCK_SIZE_n = 1
    grid = (triton.cdiv(m, BLOCK_SIZE_m), triton.cdiv(n, BLOCK_SIZE_n))
    weight_dequant[grid](X, S, Y, M, N, m, n, TILE_SIZE, BLOCK_SIZE_m, BLOCK_SIZE_n)

