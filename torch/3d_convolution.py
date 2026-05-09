import torch


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output[:] = torch.nn.functional.conv3d(input.view(1, 1, input_depth, input_rows, input_cols), kernel.view(1, 1, kernel_depth, kernel_rows, kernel_cols))
