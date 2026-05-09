import torch


# x, W_gate, W_up, W_down, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    torch.matmul(torch.nn.functional.silu(torch.matmul(x, W_gate)) * torch.matmul(x, W_up), W_down, out=output)
