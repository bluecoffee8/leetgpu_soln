import torch
import triton
import triton.language as tl

@triton.jit 
def mha(q_ptr, k_ptr, v_ptr, o_ptr, N, d_model, h, d_k, 
        stride_q0, stride_q1,
        stride_k0, stride_k1,
        stride_v0, stride_v1,
        stride_o0, stride_o1,
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_d: tl.constexpr):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    oN = pid0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mN = oN < N
    oD = pid1 * d_k + tl.arange(0, BLOCK_SIZE_d)
    mD = oD < pid1 * d_k + d_k

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_d), dtype=tl.float32)
    softmax_sum = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    softmax_max = tl.full([BLOCK_SIZE_N], float("-inf"), dtype=tl.float32)
    logit_scale = 1.0 / tl.sqrt(d_k + 0.0)

    oQ = oN[:, None] * stride_q0 + oD[None, :] * stride_q1
    mQ = mN[:, None] & mD[None, :]
    Q = tl.load(q_ptr + oQ, mask=mQ, other=0.0)

    for i in range(0, N, BLOCK_SIZE_N):
        oN_ = i + tl.arange(0, BLOCK_SIZE_N)
        mN_ = oN_ < N 
        oK = oD[:, None] * stride_k1 + oN_[None, :] * stride_k0
        mK = mD[:, None] & mN_[None, :]
        K = tl.load(k_ptr + oK, mask=mK, other=0.0)

        logits = tl.dot(Q, K, input_precision="ieee") * logit_scale 

        attn_mask = mN[:, None] & mN_[None, :]
        logits = tl.where(attn_mask, logits, float("-inf"))

        block_max = tl.max(logits, axis=-1)
        new_max = tl.maximum(block_max, softmax_max)

        alpha = tl.exp(softmax_max - new_max)
        softmax_max = new_max

        logits_shifted = logits - new_max[:, None]
        w = tl.exp(logits_shifted)
        denom = tl.sum(w, axis=1)

        softmax_sum = tl.fma(softmax_sum, alpha, denom)

        oV = oN_[:, None] * stride_v0 + oD[None, :] * stride_v1
        mV = mN_[:, None] & mD[None, :]
        V = tl.load(v_ptr + oV, mask=mV, other=0.0)

        wV = tl.dot(w, V, input_precision="ieee")
        acc = tl.fma(acc, alpha[:, None], wV)

    acc /= softmax_sum[:, None]

    oO = oN[:, None] * stride_o0 + oD[None, :] * stride_o1 
    mO = mN[:, None] & mD[None, :]
    tl.store(o_ptr + oO, acc, mask=mO)

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    BLOCK_SIZE_N = 64
    d_k = d_model // h 
    grid = (triton.cdiv(N, BLOCK_SIZE_N), h)
    BLOCK_SIZE_d = max(triton.next_power_of_2(d_k), 32)

    stride_q0, stride_q1 = Q.stride()
    stride_k0, stride_k1 = K.stride() 
    stride_v0, stride_v1 = V.stride() 
    stride_o0, stride_o1 = output.stride() 

    mha[grid](Q, K, V, output, N, d_model, h, d_k, 
                stride_q0, stride_q1,
                stride_k0, stride_k1,
                stride_v0, stride_v1,
                stride_o0, stride_o1,
                BLOCK_SIZE_N, BLOCK_SIZE_d)
