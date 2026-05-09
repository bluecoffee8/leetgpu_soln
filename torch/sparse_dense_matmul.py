import torch


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, nnz: int):
    torch.matmul(A.view(M, N), B.view(N, K), out=C.view(M, K))
