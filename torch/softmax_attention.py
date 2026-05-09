import torch


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    qk = torch.matmul(Q, K.T)
    qk /= (d ** 0.5)
    torch.softmax(qk, -1, out=qk) 
    torch.matmul(qk, V, out=output)
