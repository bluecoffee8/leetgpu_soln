import torch

# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    res = torch.nn.functional.conv1d(input.view(1,1,input_size), kernel.view(1,1,kernel_size), padding='valid')
    output.copy_(res.view(-1))
