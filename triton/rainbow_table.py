import torch
import triton
import triton.language as tl


@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261

    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)

    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME

    return hash_val


@triton.jit
def fnv1a_hash_kernel(input, output, n_elements, n_rounds, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 
    mask = (offset < n_elements)
    x = tl.load(input + offset, mask=mask, other=0.0).to(tl.uint32)
    for i in range(n_rounds):
        x = fnv1a_hash(x)
    tl.store(output + offset, x, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](input, output, N, R, BLOCK_SIZE)
