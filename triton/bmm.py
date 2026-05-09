import torch
import triton
import triton.language as tl

@triton.jit 
def bmm_kernel(a_ptr, b_ptr, c_ptr, BATCH, M, N, K: tl.constexpr,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUPSIZE: tl.constexpr):
    hw_pid0 = tl.program_id(0)
    hw_pid1 = tl.program_id(1)

    num_programs_pid0 = tl.num_programs(0)
    num_programs_pid1 = tl.num_programs(1)
    pid0, pid1 = tl.swizzle2d(hw_pid0, hw_pid1, num_programs_pid0, num_programs_pid1, GROUPSIZE)
    pid2 = tl.program_id(2)

    # a: [B, M, K] @ b: [B, K, N]
    o_M = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    o_N = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_K = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.static_range(0, K, BLOCK_SIZE_K):
        o_MK = pid2 * M * K + o_M[:, None] * K + (k + o_K)[None, :]
        o_KN = pid2 * K * N + (k + o_K)[:, None] * N + o_N[None, :]
        m_MK = (o_M[:, None] < M) & ((k + o_K)[None, :] < K)
        m_KN = ((k + o_K)[:, None] < K) & (o_N[None, :] < N)
        A = tl.load(a_ptr + o_MK, mask=m_MK, other=0.0)
        B = tl.load(b_ptr + o_KN, mask=m_KN, other=0.0)

        C = tl.dot(A, B) 
        acc += C 
    
    tl.store(c_ptr + pid2 * M * N + o_M[:, None] * N + o_N[None, :], acc, mask=(o_M[:, None] < M) & (o_N[None, :] < N))

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUPSIZE = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), BATCH)
    bmm_kernel[grid](a, b, c, BATCH, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUPSIZE)
