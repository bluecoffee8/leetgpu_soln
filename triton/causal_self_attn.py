import torch
import triton
import triton.language as tl

@triton.jit 
def softmax_attn(q_ptr, k_ptr, v_ptr, o_ptr, M, N, d, 
                 stride_q0, stride_q1,
                 stride_k0, stride_k1,
                 stride_v0, stride_v1,
                 stride_o0, stride_o1,
                 BLOCK_SIZE_M: tl.constexpr, 
                 BLOCK_SIZE_N: tl.constexpr, 
                 BLOCK_SIZE_d: tl.constexpr):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    oM = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    oN_ = tl.arange(0, BLOCK_SIZE_N)
    oD = pid1 * BLOCK_SIZE_d + tl.arange(0, BLOCK_SIZE_d)

    mM = oM < M 
    mD = oD < d 

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_d), dtype=tl.float32)
    softmax_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    softmax_max = tl.full([BLOCK_SIZE_M], -torch.inf, dtype=tl.float32)
    logit_scale = 1.0 / tl.sqrt(d + 0.0)

    for i in range(0, N, BLOCK_SIZE_N):
        # logits blocks shift in (M, BLOCK_SIZE_N) row horizontally

        oN = i + oN_
        mN = oN < N 

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # get (BLOCK_SIZE_M, BLOCK_SIZE_N) logits block 
        for j in range(0, d, BLOCK_SIZE_d):
            # vertically iterate through (d, BLOCK_SIZE_N) column in K^T

            od = j + tl.arange(0, BLOCK_SIZE_d)
            md = od < d 

            # load (BLOCK_SIZE_M, BLOCK_SIZE_d) from Q
            oQ = oM[:, None] * stride_q0 + od[None, :] * stride_q1
            mQ = mM[:, None] & md[None, :]
            Q = tl.load(q_ptr + oQ, mask=mQ)

            # load (BLOCK_SIZE_d, BLOCK_SIZE_N) from K^T
            oK = od[:, None] * stride_k1 + oN[None, :] * stride_k0 
            mK = md[:, None] & mN[None, :]
            K = tl.load(k_ptr + oK, mask=mK) 

            logits += tl.dot(Q, K) 
            # tl.dot(Q, K, logits, input_precision="ieee")
        
        logits = logits * logit_scale 

        trn_mask = oM[:, None] >= oN[None, :]

        attn_mask = mM[:, None] & mN[None, :] & trn_mask # only for border cases, 
        logits = tl.where(attn_mask, logits, -torch.inf) 

        block_max = tl.max(logits, axis=-1)  # maximum softmax in the small block
        new_max = tl.maximum(block_max, softmax_max) # agg w/ running max

        alpha = tl.exp(softmax_max - new_max) # need to multiply old values here
        softmax_max = new_max # updating running max

        logits_shifted = logits - new_max[:, None] # logit value shifted
        w = tl.exp(logits_shifted)
        denom = tl.sum(w, axis=1)  # denominator of blog

        softmax_sum = tl.fma(softmax_sum, alpha, denom) # fused multiply add

        # load (BLOCK_SIZE_N, BLOCK_SIZE_d) block from V
        oV = oN[:, None] * stride_v0 + oD[None, :] * stride_v1
        mV = mN[:, None] & mD[None, :]
        V = tl.load(v_ptr + oV, mask=mV)

        wV = tl.dot(w, V)
        acc = tl.fma(acc, alpha[:, None], wV)
    
    acc /= softmax_sum[:, None]

    oO = oM[:, None] * stride_o0 + oD[None, :] * stride_o1
    mO = (oM[:, None] < M) & (oD[None, :] < d)
    tl.store(o_ptr + oO, acc.to(tl.float32), mask=mO)


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_d = 128
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(d, BLOCK_SIZE_d))
    stride_q0, stride_q1 = Q.stride()
    stride_k0, stride_k1 = K.stride() 
    stride_v0, stride_v1 = V.stride() 
    stride_o0, stride_o1 = output.stride() 
    softmax_attn[grid](Q, K, V, output, M, M, d,
                        stride_q0, stride_q1,
                        stride_k0, stride_k1,
                        stride_v0, stride_v1,
                        stride_o0, stride_o1,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_d)
