import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_k >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_an > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_ck > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_block = a + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_block = b + (offs_n[:, None] * stride_bn + offs_bk[None, :] * stride_bk)

    zero_point_A = float(zero_point_A) if isinstance(zero_point_A, int) else zero_point_A.to(tl.float32)
    zero_point_B = float(zero_point_B) if isinstance(zero_point_B, int) else zero_point_B.to(tl.float32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a_val = tl.load(a_block, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)
        b_val = tl.load(b_block, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        a_val -= zero_point_A
        b_val -= zero_point_B
        accumulator += tl.dot(a_val, b_val, allow_tf32=False)
        a_block += BLOCK_SIZE_N * stride_an 
        b_block += BLOCK_SIZE_N * stride_bn
    c_val = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_block = c + stride_cm * offs_cm[:, None] + stride_ck * offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    c_val *= (scale_A * scale_B / scale_C)
    c_val = tl.floor(c_val + float(0.5))
    c_val += zero_point_C
    c_val = tl.maximum(c_val, -128)
    c_val = tl.minimum(c_val, 127)
    tl.store(c_block, c_val, mask=c_mask)

# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    K: int,
    N: int,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"]), )

    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4
    )
