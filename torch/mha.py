import torch


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    # Q = Q.view(N, d_model)
    # K = K.view(N, d_model)
    # V = V.view(N, d_model)
    d_k = d_model // h
    Q = torch.stack(torch.chunk(Q.view(N, d_model).to(torch.float32), h, dim=1))
    K = torch.stack(torch.chunk(K.view(N, d_model).to(torch.float32), h, dim=1))
    V = torch.stack(torch.chunk(V.view(N, d_model).to(torch.float32), h, dim=1))
    output[:] = torch.cat(torch.unbind(torch.bmm(torch.softmax(torch.bmm(Q, K.transpose(-1, -2)) / (float(d_k) ** 0.5), dim=-1), V), dim=0), dim=1)