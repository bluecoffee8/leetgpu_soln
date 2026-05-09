import torch
import triton
import triton.language as tl
import math

@triton.jit 
def moment(input, sos, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input + offset, mask=mask, other=0.0)
    tl.atomic_add(sos, tl.sum(x * x), sem="relaxed")

@triton.jit 
def rms_norm(input, sos, gamma, beta, output, N, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    sos_val = tl.load(sos)
    rms = tl.sqrt((sos_val / N) + eps)
    x = tl.load(input + offset, mask=mask, other=0.0)
    y = (x / rms) * gamma + beta 
    tl.store(output + offset, y, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 1024
    sos = torch.zeros((1, ), dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    moment[grid](input, sos, N, BLOCK_SIZE)
    rms_norm[grid](input, sos, gamma, beta, output, N, eps, BLOCK_SIZE)
