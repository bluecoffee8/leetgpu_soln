import torch
import triton
import triton.language as tl

@triton.jit 
def rec(a1, x1, a2, x2):
    return a2 * a1, a2 * x1 + x2

@triton.jit 
def lin_rec_kernel(a, x, h, L,
                   stride_a0, stride_a1, 
                   stride_x0, stride_x1,
                   stride_h0, stride_h1, 
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    a0 = a + pid * stride_a0 
    x0 = x + pid * stride_x0 
    h0 = h + pid * stride_h0 

    H = 0.0

    for i in range(0, L, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < L

        A = tl.load(a0 + offset * stride_a1, mask=mask, other=1.0)
        X = tl.load(x0 + offset * stride_x1, mask=mask, other=0.0)

        A_scan, X_scan = tl.associative_scan((A, X), axis=0, combine_fn=rec)

        h_true = A_scan * H + X_scan 

        tl.store(h0 + offset * stride_h1, h_true, mask=mask)
        lv = tl.arange(0, BLOCK_SIZE) == (BLOCK_SIZE-1)
        H = tl.sum(tl.where(lv, h_true, 0.0), axis=0)

# a, x, h are tensors on the GPU
def solve(a: torch.Tensor, x: torch.Tensor, h: torch.Tensor, B: int, L: int):
    BLOCK_SIZE = 4096
    grid = (B, )
    lin_rec_kernel[grid](a, x, h, L, 
                         a.stride(0), a.stride(1), 
                         x.stride(0), x.stride(1), 
                         h.stride(0), h.stride(1), 
                         BLOCK_SIZE=BLOCK_SIZE)
