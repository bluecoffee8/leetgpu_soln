import torch


# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    torch.histc(input, bins=num_bins, min=0, max=num_bins, out=histogram)
