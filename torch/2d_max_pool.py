import torch


# input, output are tensors on the GPU
def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    output[:] = torch.nn.functional.max_pool2d(input.view(N, C, H, W), kernel_size, stride, padding).flatten()
