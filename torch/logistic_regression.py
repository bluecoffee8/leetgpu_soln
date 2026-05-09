import torch


# X, y, beta are tensors on the GPU
def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    X = X.view(n_samples, n_features) 
    lr = 1e-6

    for _ in range(20):
        p = torch.nn.functional.sigmoid(X @ beta)
        w = p * (1 - p)
        g = X.T @ (y - p) - lr * beta
        A = X.T @ (w.unsqueeze(1) * X) + lr * torch.eye(n_features).to(X)

        delta = torch.linalg.solve(A, g) 
        beta.add_(delta)
