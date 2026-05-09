import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    vals = input.topk(k).values 
    output[:] = torch.sort(vals, descending=True).values
