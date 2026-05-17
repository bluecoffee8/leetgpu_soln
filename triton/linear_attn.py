import torch
import triton
import triton.language as tl


@triton.jit
def _phi(x):
    # ELU(x) + 1: always positive, the standard linear attention feature map
    return tl.where(x >= 0.0, x + 1.0, tl.exp(x))


@triton.jit
def _kv_kernel(
    K_ptr, V_ptr, KV_ptr,
    M, d,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """KV[d1, d2] = sum_m phi(K[m, d1]) * V[m, d2]"""
    pid_d1 = tl.program_id(0)
    pid_d2 = tl.program_id(1)

    offs_d1 = pid_d1 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_d2 = pid_d2 * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d1 = offs_d1 < d
    mask_d2 = offs_d2 < d

    acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    for m_block in range(tl.cdiv(M, BLOCK_M)):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        mask_k = mask_m[:, None] & mask_d1[None, :]
        k = tl.load(
            K_ptr + offs_m[:, None] * d + offs_d1[None, :],
            mask=mask_k, other=0.0,
        ).to(tl.float32)
        # Re-mask after phi so padding rows/cols contribute 0, not phi(0)=1
        k_phi = tl.where(mask_k, _phi(k), 0.0)

        mask_v = mask_m[:, None] & mask_d2[None, :]
        v = tl.load(
            V_ptr + offs_m[:, None] * d + offs_d2[None, :],
            mask=mask_v, other=0.0,
        ).to(tl.float32)

        # acc += k_phi.T @ v  =>  (BLOCK_D, BLOCK_M) @ (BLOCK_M, BLOCK_D)
        acc = tl.dot(tl.trans(k_phi), v, acc)

    tl.store(
        KV_ptr + offs_d1[:, None] * d + offs_d2[None, :],
        acc,
        mask=mask_d1[:, None] & mask_d2[None, :],
    )


@triton.jit
def _ksum_kernel(
    K_ptr, Ksum_ptr,
    M, d,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Ksum[d1] = sum_m phi(K[m, d1])"""
    pid_d = tl.program_id(0)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < d

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for m_block in range(tl.cdiv(M, BLOCK_M)):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask = mask_m[:, None] & mask_d[None, :]

        k = tl.load(
            K_ptr + offs_m[:, None] * d + offs_d[None, :],
            mask=mask, other=0.0,
        ).to(tl.float32)
        k_phi = tl.where(mask, _phi(k), 0.0)
        acc += tl.sum(k_phi, axis=0)

    tl.store(Ksum_ptr + offs_d, acc, mask=mask_d)


@triton.jit
def _output_kernel(
    Q_ptr, KV_ptr, Ksum_ptr, O_ptr,
    M, d,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """O[m, j] = (phi(Q[m]) @ KV[:, j]) / (phi(Q[m]) @ Ksum)"""
    pid_m = tl.program_id(0)
    pid_d_out = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_out = pid_d_out * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d_out = offs_d_out < d

    numer = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_block in range(tl.cdiv(d, BLOCK_D)):
        offs_k = k_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_k = offs_k < d

        mask_q = mask_m[:, None] & mask_k[None, :]
        q = tl.load(
            Q_ptr + offs_m[:, None] * d + offs_k[None, :],
            mask=mask_q, other=0.0,
        ).to(tl.float32)
        q_phi = tl.where(mask_q, _phi(q), 0.0)

        # numer += q_phi @ KV[k_block, pid_d_out]  => (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_D)
        kv = tl.load(
            KV_ptr + offs_k[:, None] * d + offs_d_out[None, :],
            mask=mask_k[:, None] & mask_d_out[None, :], other=0.0,
        ).to(tl.float32)
        numer = tl.dot(q_phi, kv, numer)

        # denom += sum(q_phi * Ksum[k_block], axis=1)  => (BLOCK_M,)
        ksum = tl.load(Ksum_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        denom += tl.sum(q_phi * ksum[None, :], axis=1)

    out = numer / denom[:, None]

    tl.store(
        O_ptr + offs_m[:, None] * d + offs_d_out[None, :],
        out,
        mask=mask_m[:, None] & mask_d_out[None, :],
    )


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    BLOCK_M = 32
    # Pad d to the next power-of-2 >= 16 so tl.dot inner dims are valid
    BLOCK_D = max(16, triton.next_power_of_2(d))

    KV = torch.zeros((d, d), device=Q.device, dtype=torch.float32)
    Ksum = torch.zeros((d,), device=Q.device, dtype=torch.float32)

    # Step 1: KV = phi(K)^T @ V
    grid_kv = (triton.cdiv(d, BLOCK_D), triton.cdiv(d, BLOCK_D))
    _kv_kernel[grid_kv](K, V, KV, M, d, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D)

    # Step 2: Ksum = sum(phi(K), dim=0)
    _ksum_kernel[(triton.cdiv(d, BLOCK_D),)](K, Ksum, M, d, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D)

    # Step 3: O = (phi(Q) @ KV) / (phi(Q) @ Ksum)[:, None]
    grid_out = (triton.cdiv(M, BLOCK_M), triton.cdiv(d, BLOCK_D))
    _output_kernel[grid_out](Q, KV, Ksum, output, M, d, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D)
