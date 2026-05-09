import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, lo: float, hi: float, N: int):
    torch.clamp(input, lo, hi, out=output)
