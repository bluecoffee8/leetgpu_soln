import torch


# X, S, Y are tensors on the GPU
def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int):
    z = S.repeat_interleave(TILE_SIZE, dim=0).repeat_interleave(TILE_SIZE, dim=1)
    torch.mul(z[:M, :N], X, out=Y)
