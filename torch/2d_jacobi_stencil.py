import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    if rows <= 2 or cols <= 2:
        output.copy_(input)
        return 
    kernel = torch.tensor([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=input.dtype, device='cuda')
    output[1:-1, 1:-1] = torch.nn.functional.conv2d(input.view(1, 1, rows, cols), kernel.view(1, 1, 3, 3), stride=1, padding=0).squeeze() 
    output[0, :] = input[0, :]
    output[-1, :] = input[-1, :]
    output[:, 0] = input[:, 0]
    output[:, -1] = input[:, -1]
