import torch


# points and indices are tensors on the GPU
def solve(x: torch.Tensor, indices: torch.Tensor, N: int):
    x = x.view(N, 3).to(torch.float64)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    dist2 = x_norm + x_norm.T - 2 * x @ x.T 
    dist2.fill_diagonal_(float("inf"))
    indices[:] = dist2.argmin(dim=1)
