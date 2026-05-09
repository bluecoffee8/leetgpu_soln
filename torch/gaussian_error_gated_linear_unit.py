import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    x1, x2 = torch.chunk(input, 2, dim=-1)
    torch.mul(x1, torch.nn.functional.gelu(x2), out=output)
