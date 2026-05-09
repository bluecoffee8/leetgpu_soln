import torch


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
    group_size = num_q_heads // num_kv_heads
    K = K.repeat_interleave(group_size, dim=0)
    V = V.repeat_interleave(group_size, dim=0)
    scores = torch.bmm(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)
    torch.bmm(torch.softmax(scores, dim=-1), V, out=output)