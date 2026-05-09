import torch


# u, delta, A, B, C, skip, y are tensors on the GPU
def solve(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    skip: torch.Tensor,
    y: torch.Tensor,
    batch: int,
    seq_len: int,
    d_model: int,
    d_state: int,
):
    A_ = torch.exp(delta.unsqueeze(-1) * A.reshape(1, 1, d_model, d_state))
    B_ = delta.unsqueeze(-1) * B.unsqueeze(2) 
    h = torch.zeros((batch, seq_len, d_model, d_state), device=A.device)
    u = u.unsqueeze(-1) 
    h[:, 0, :, :] = B_[:, 0, :, :] * u[:, 0, :, :]
    for t in range(1, seq_len):
        h[:, t, :, :] = A_[:, t, :, :] * h[:, t-1, :, :] + B_[:, t, :, :] * u[:, t, :, :]
    y[:, :, :] = torch.sum(C.unsqueeze(2) * h, dim=-1) + skip.view(1, 1, d_model) * u.squeeze(-1)