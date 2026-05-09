import torch


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    masked = torch.tril(torch.matmul(Q.view(M, d), K.view(M, d).T) / (d ** 0.5)) + torch.triu(torch.full((M, M), float("-inf"), device=Q.device), diagonal=1)
    torch.matmul(torch.softmax(masked, dim=-1), V.view(M, d), out=output)    
