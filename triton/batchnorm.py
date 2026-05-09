import torch
import triton
import triton.language as tl

@triton.jit 
def moment(input, m1, m2, N, C, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    o_N = pid0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_C = pid1 * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    x = tl.load(input + o_N[:, None] * C + o_C[None, :], mask=(o_N[:, None] < N) & (o_C[None, :] < C), other=0.0)
    tl.atomic_add(m1 + o_C, tl.sum(x, axis=0), mask=(o_C < C), sem="relaxed")
    tl.atomic_add(m2 + o_C, tl.sum(x * x, axis=0), mask=(o_C < C), sem="relaxed")

@triton.jit 
def batch_norm(input, gamma, beta, output, m1, m2, N, C, eps, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    o_N = pid0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_C = pid1 * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    x = tl.load(input + o_N[:, None] * C + o_C[None, :], mask=(o_N[:, None] < N) & (o_C[None, :] < C), other=0.0)

    m1_val = tl.load(m1 + o_C, mask=(o_C < C))[None, :]
    m2_val = tl.load(m2 + o_C, mask=(o_C < C))[None, :]

    beta_val = tl.load(beta + o_C, mask=(o_C < C))[None, :]
    gamma_val = tl.load(gamma + o_C, mask=(o_C < C))[None, :]

    output_val = (x - (m1_val / N)) / tl.sqrt((m2_val / N - m1_val * m1_val / (N * N)) + eps)
    output_val = output_val * gamma_val + beta_val 

    tl.store(output + o_N[:, None] * C + o_C[None, :], output_val, mask=(o_N[:, None] < N) & (o_C[None, :] < C))

# input, gamma, beta, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_C = 64
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(C, BLOCK_SIZE_C))
    m1 = input.new_zeros([C])
    m2 = input.new_zeros([C])
    moment[grid](input, m1, m2, N, C, BLOCK_SIZE_N, BLOCK_SIZE_C)
    batch_norm[grid](input, gamma, beta, output, m1, m2, N, C, eps, BLOCK_SIZE_N, BLOCK_SIZE_C)
