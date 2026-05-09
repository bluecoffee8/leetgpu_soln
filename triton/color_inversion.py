import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    offset = pid * BLOCK_SIZE * 4 + 4 * tl.arange(0, BLOCK_SIZE)
    ptrs0 = image + offset 
    ptrs1 = image + offset + 1
    ptrs2 = image + offset + 2
    # ptrs3 = image + offset + 3
    tile0 = tl.load(ptrs0, mask=(offset < width * height * 4), other=0.0)
    tile1 = tl.load(ptrs1, mask=(offset + 1 < width * height * 4), other=0.0)
    tile2 = tl.load(ptrs2, mask=(offset + 2 < width * height * 4), other=0.0)
    # tile3 = tl.load(ptrs3, mask=(offset + 3 < width * height * 4), other=0.0)
    tl.store(ptrs0, 255 - tile0, mask=(offset < width * height * 4))
    tl.store(ptrs1, 255 - tile1, mask=(offset + 1 < width * height * 4))
    tl.store(ptrs2, 255 - tile2, mask=(offset + 2 < width * height * 4))

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)
