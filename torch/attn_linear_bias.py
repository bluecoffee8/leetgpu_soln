import torch


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    qk = torch.matmul(Q, K.T)
    qk /= (d ** 0.5)
    qk += alpha * (torch.arange(M, device='cuda').reshape(M, 1) - torch.arange(N, device='cuda').reshape(1, N))
    qk = torch.softmax(qk, dim=-1)
    torch.matmul(qk, V, out=output)
