import torch
import math

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
    d_model: int,
    gamma: float,
):
    scores = torch.matmul(Q, K.T) / math.sqrt(float(d_model))
    mask = torch.tril(gamma ** (torch.arange(seq_len, device=Q.device).unsqueeze(1) - torch.arange(seq_len, device=Q.device).unsqueeze(0)))
    torch.matmul(scores * mask, V, out=output)
