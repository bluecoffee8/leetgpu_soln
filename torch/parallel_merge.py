import torch


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int):
    C[:], _ = torch.sort(torch.concat([A.view(-1), B.view(-1)]))
