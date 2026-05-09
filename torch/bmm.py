import torch


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    torch.bmm(A.view(BATCH, M, K), B.view(BATCH, K, N), out=C.view(BATCH, M, N))
