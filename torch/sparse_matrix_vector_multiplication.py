import torch


# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    torch.matmul(A.view(M, N), x.view(N, 1), out=y)
