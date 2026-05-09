import torch


# x, w_q, scales, y are tensors on the GPU
def solve(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
):
    high_nibble = (w_q >> 4) & 0x0F
    low_nibble = w_q & 0x0F
    w = torch.stack([high_nibble, low_nibble], dim=-1).view(N, K).to(torch.float32)-8.0
    w *= scales.repeat_interleave(group_size, dim=-1).to(torch.float32)
    y[:] = torch.matmul(x.to(torch.float32), w.T).to(x.dtype)
