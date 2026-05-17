import torch
import triton
import triton.language as tl


@triton.jit
def spec_decode_kernel(
    draft_tokens_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    uniform_samples_ptr,
    output_tokens_ptr,
    T,
    V,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    T64 = tl.cast(T, tl.int64)
    V64 = tl.cast(V, tl.int64)

    b = pid // T64
    t = pid % T64
    bt = b * T64 + t

    token = tl.load(draft_tokens_ptr + bt).to(tl.int64)
    u = tl.load(uniform_samples_ptr + bt)

    base = bt * V64
    p_tgt = tl.load(target_probs_ptr + base + token)
    p_dft = tl.load(draft_probs_ptr + base + token)

    ratio = tl.where(p_dft > 0.0, p_tgt / p_dft, 1.0)
    accept = u < tl.minimum(ratio, 1.0)

    if accept:
        tl.store(output_tokens_ptr + bt, token)
    else:
        # Sample from adjusted distribution: max(0, target - draft) / Z
        offs = tl.arange(0, BLOCK_V).to(tl.int64)
        mask = offs < V64

        tgt = tl.load(target_probs_ptr + base + offs, mask=mask, other=0.0)
        dft = tl.load(draft_probs_ptr + base + offs, mask=mask, other=0.0)
        adj = tl.where(mask, tl.maximum(tgt - dft, 0.0), 0.0)

        adj_sum = tl.sum(adj)
        threshold = u * adj_sum

        # Inclusive prefix sum; find first index where cumsum >= threshold
        cumsum = tl.associative_scan(adj, axis=0, combine_fn=_add)
        valid = (cumsum >= threshold) & mask

        # Map invalid positions to BLOCK_V, then take global minimum index
        idx = tl.where(valid, tl.arange(0, BLOCK_V), BLOCK_V)
        result = tl.min(idx, axis=0)
        result = tl.minimum(result, V - 1)

        tl.store(output_tokens_ptr + bt, result)


@triton.jit
def _add(a, b):
    return a + b


def solve(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    output_tokens: torch.Tensor,
    B: int,
    T: int,
    V: int,
):
    BLOCK_V = triton.next_power_of_2(V)
    spec_decode_kernel[(B * T,)](
        draft_tokens,
        draft_probs,
        target_probs,
        uniform_samples,
        output_tokens,
        T=T,
        V=V,
        BLOCK_V=BLOCK_V,
    )
