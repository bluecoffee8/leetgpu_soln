import torch


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    numer = torch.matmul(torch.nn.functional.elu(Q.view(M, d)) + 1, torch.matmul(torch.nn.functional.elu(K.view(M, d)).T + 1, V.view(M, d)))
    denom = torch.matmul(torch.nn.functional.elu(Q.view(M, d)) + 1, torch.sum(torch.nn.functional.elu(K.view(M, d)) + 1, axis=0))
    output[:] = numer.view(M, d) / denom.view(M, 1)