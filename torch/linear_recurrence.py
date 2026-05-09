import torch


# a, x, h are tensors on the GPU
def solve(a: torch.Tensor, x: torch.Tensor, h: torch.Tensor, B: int, L: int):
    h[:, 0] = x[:, 0]
    for t in range(1, L):
        h[:, t] = a[:, t] * h[:, t-1] + x[:, t]
