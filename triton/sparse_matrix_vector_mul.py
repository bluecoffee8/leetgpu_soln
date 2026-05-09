import torch
import triton
import triton.language as tl

@triton.jit 
def sgemm_kernel(a, x, y, M, N,
                 stride_a0, stride_a1,
                 BLOCK_SIZE_M: tl.constexpr, 
                 BLOCK_SIZE_N: tl.constexpr, 
                 GROUPSIZE: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1) 

    n_pid0 = tl.num_programs(axis=0) 
    n_pid1 = tl.num_programs(axis=1)
    pid0, pid1 = tl.swizzle2d(pid0, pid1, n_pid0, n_pid1, GROUPSIZE)

    oM = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    oN = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    oA = oM[:, None] * stride_a0 + oN[None, :] * stride_a1
    mA = (oM[:, None] < M) & (oN[None, :] < N)
    A = tl.load(a + oA, mask=mA, other=0.0)
    X = tl.load(x + oN, mask=(oN < N), other=0.0)[:, None]
    Y = tl.ravel(tl.dot(A, X))
    tl.atomic_add(y + oM, Y, mask=(oM < M), sem="relaxed")

# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    GROUPSIZE = 8 
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    sgemm_kernel[grid](A, x, y, M, N,
                       N, 1,
                       BLOCK_SIZE_M, BLOCK_SIZE_N, GROUPSIZE)

