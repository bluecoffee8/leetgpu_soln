import torch


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    image[:] = 255 - image[:]
    image[3::4] = 255 - image[3::4]
