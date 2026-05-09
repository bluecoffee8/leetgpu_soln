import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    m = torch.max(input)
    torch.exp(input-m, out=output)
    s = torch.sum(output)
    torch.div(output, s, out=output)
