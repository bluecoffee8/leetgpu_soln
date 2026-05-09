import torch


# values, flags, output are tensors on the GPU
def solve(values: torch.Tensor, flags: torch.Tensor, output: torch.Tensor, N: int):
    cumsum = values.to(torch.float64).cumsum(dim=0) - values.to(torch.float64)
    flagged_indices = torch.arange(0, N, device=values.device)[flags==1]
    new_indices = flagged_indices[torch.searchsorted(flagged_indices, torch.arange(0, N, device=values.device), right=True) - 1]
    output[:] = (cumsum - cumsum[new_indices]).to(output.dtype)
