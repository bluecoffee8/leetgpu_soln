import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    sums = torch.cumsum(input, dim=-1) 
    o = sums[window_size:] - sums[:N-window_size]
    output.copy_(torch.max(o, dim=-1)[0])
