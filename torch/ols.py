import torch


# X, y, beta are tensors on the GPU
def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    X = X.view(n_samples, n_features)
    y = y.view(n_samples, 1)
    beta[:] = (torch.linalg.inv(X.T @ X) @ X.T @ y).flatten()
