import torch
import triton
import triton.language as tl

@triton.jit 
def gqa(q_ptr, k_ptr, v_ptr, o_ptr, 
        num_q_heads, num_kv_heads, seq_len, head_dim, group_size,
        stride_q0, stride_q1, stride_q2, 
        stride_k0, stride_k1, stride_k2, 
        stride_v0, stride_v1, stride_v2, 
        stride_o0, stride_o1, stride_o2, 
        BLOCK_SIZE_L: tl.constexpr, 
        BLOCK_SIZE_D: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    kv_head = pid0 // group_size

    oL = pid1 * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mL = oL < seq_len 
    oD = tl.arange(0, BLOCK_SIZE_D)
    mD = oD < head_dim

    oQ = oL[:, None] * stride_q1 + oD[None, :] * stride_q2
    mQ = mL[:, None] & mD[None, :]
    Q = tl.load(q_ptr + pid0 * stride_q0 + oQ, mask=mQ, other=0.0)

    acc = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    softmax_sum = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)
    softmax_max = tl.full([BLOCK_SIZE_L], float("-inf"), dtype=tl.float32)
    logit_scale = 1.0 / tl.sqrt(head_dim + 0.0)

    for l in range(0, seq_len, BLOCK_SIZE_L):
        oL_ = l + tl.arange(0, BLOCK_SIZE_L)
        mL_ = oL_ < seq_len

        oK = oD[:, None] * stride_k2 + oL_[None, :] * stride_k1 
        mK = mD[:, None] & mL_[None, :]
        K = tl.load(k_ptr + kv_head * stride_k0 + oK, mask=mK, other=0.0)

        logits = tl.dot(Q, K, input_precision="ieee") * logit_scale
        
        attn_mask = mL[:, None] & mL_[None, :]
        logits = tl.where(attn_mask, logits, float("-inf"))
        block_max = tl.max(logits, axis=-1)
        new_max = tl.maximum(block_max, softmax_max)
        alpha = tl.exp(softmax_max - new_max)
        softmax_max = new_max
        logits_shifted = logits - new_max[:, None]
        w = tl.exp(logits_shifted)
        denom = tl.sum(w, axis=1)
        softmax_sum = tl.fma(softmax_sum, alpha, denom)

        oV = oL_[:, None] * stride_v1 + oD[None, :] * stride_v2
        mV = mL_[:, None] & mD[None, :]
        V = tl.load(v_ptr + kv_head * stride_v0 + oV, mask=mV, other=0.0)

        wV = tl.dot(w, V, input_precision="ieee")
        acc = tl.fma(acc, alpha[:, None], wV)

    acc /= softmax_sum[:, None]

    oO = oL[:, None] * stride_o1 + oD[None, :] * stride_o2
    mO = mL[:, None] & mD[None, :]
    tl.store(o_ptr + pid0 * stride_o0 + oO, acc, mask=mO)

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_D = max(triton.next_power_of_2(head_dim), 16)
    grid = (num_q_heads, triton.cdiv(seq_len, BLOCK_SIZE_L))

    stride_q0, stride_q1, stride_q2 = Q.stride() 
    stride_k0, stride_k1, stride_k2 = K.stride() 
    stride_v0, stride_v1, stride_v2 = V.stride() 
    stride_o0, stride_o1, stride_o2 = output.stride()

    group_size = num_q_heads // num_kv_heads

    gqa[grid](Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim, group_size,
              stride_q0, stride_q1, stride_q2, 
              stride_k0, stride_k1, stride_k2, 
              stride_v0, stride_v1, stride_v2, 
              stride_o0, stride_o1, stride_o2, 
              BLOCK_SIZE_L, BLOCK_SIZE_D)

    # group_size = num_q_heads // num_kv_heads
    # K = K.repeat_interleave(group_size, dim=0)
    # V = V.repeat_interleave(group_size, dim=0)
    # scores = torch.bmm(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)
    # torch.bmm(torch.softmax(scores, dim=-1), V, out=output)