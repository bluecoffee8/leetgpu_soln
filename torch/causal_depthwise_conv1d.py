import torch


# x, weight, bias, output are tensors on the GPU
def solve(
    x: torch.Tensor, # shape (B, L, D)
    weight: torch.Tensor, # (D, K)
    bias: torch.Tensor, # (D, )
    output: torch.Tensor, # (B, L, D)
    B: int,
    L: int,
    D: int,
    K: int,
):
    output[:] = torch.nn.functional.conv1d(
        x.permute(0, 2, 1), 
        weight.flip(-1).unsqueeze(1), 
        bias, 
        padding=K-1,
        groups=D, 
    )[:, :, :L].permute(0, 2, 1)
