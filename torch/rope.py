import torch


# Q, cos, sin, output are tensors on the GPU
def solve(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int
):
    torch.add(Q * cos, torch.stack((-Q[:, D//2:], Q[:, :D//2]), dim=1).view(M, D) * sin, out=output)
