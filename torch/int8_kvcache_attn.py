import torch


# Q, K_int8, V_int8, k_scale, v_scale, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_int8: torch.Tensor,
    V_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    K_float = K_int8.view(num_heads, seq_len, head_dim) * k_scale.view(num_heads, seq_len, 1).repeat_interleave(head_dim, dim=-1)
    V_float = V_int8.view(num_heads, seq_len, head_dim) * v_scale.view(num_heads, seq_len, 1).repeat_interleave(head_dim, dim=-1)
    scores = torch.matmul(Q.view(num_heads, 1, head_dim), K_float.transpose(-1, -2)).squeeze(dim=1) / (head_dim ** 0.5) # [N, S]
    sf = torch.softmax(scores, dim=-1) # [N, S]
    torch.matmul(sf.view(num_heads, 1, seq_len), V_float, out=output) # [N, S, D]
