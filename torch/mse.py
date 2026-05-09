import torch


# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    mse[:] = torch.nn.functional.mse_loss(predictions, targets)
