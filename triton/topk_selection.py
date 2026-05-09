import torch
import triton
import triton.language as tl

@triton.jit 
def topk(input, bk, N, k: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 
    mask = offset < N 
    x = tl.load(input + offset, mask=mask, other=-float("inf"))
    if k == 1: 
        tl.store(bk + pid, tl.max(x))
    else:
        tk = tl.topk(x, k) 
        tl.store(bk + pid * k + tl.arange(0, k), tk)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 1024 
    nb = triton.cdiv(N, BLOCK_SIZE)
    grid = (nb, )
    bk = torch.zeros((nb, triton.next_power_of_2(k)), device='cuda')
    topk[grid](input, bk, N, triton.next_power_of_2(k), BLOCK_SIZE)
    if nb > 1:
        solve(bk, output, bk.numel(), k)
    else:
        output[:k].copy_(bk[0, :k])