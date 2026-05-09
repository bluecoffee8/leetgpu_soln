import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    output[:] = 0.299 * input[0::3] + 0.587 * input[1::3] + 0.114 * input[2::3]
