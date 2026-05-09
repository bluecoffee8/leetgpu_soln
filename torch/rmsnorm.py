import torch


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    eps: float,
):
    rms = torch.sqrt(torch.mean(torch.pow(input, 2)) + eps)
    torch.add(gamma * input / rms, beta, out=output)
    # output[:] = gamma * input / rms + beta
