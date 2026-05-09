import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_block = tl.load(a + offsets, mask=mask)
    b_block = tl.load(b + offsets, mask=mask)
    tl.store(c + offsets, a_block + b_block, mask=mask)

@gluon.jit 
def vec_add_gluon(a, b, c, N, BLOCK_SIZE: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * BLOCK_SIZE
    end = min(N, start + BLOCK_SIZE)
    for i in range(start, end):
        a_val = gl.load(a + i)
        b_val = gl.load(b + i) 
        gl.store(c + i, a_val + b_val)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
    # vec_add_gluon[grid](a, b, c, N, BLOCK_SIZE)
