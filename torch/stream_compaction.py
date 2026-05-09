import torch


# A, out are tensors on the GPU
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    z = A[A > 0]
    out[:len(z)] = z
