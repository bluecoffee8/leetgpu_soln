import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 
    mask = (offset < N)
    a = tl.load(A_ptr + offset, mask=mask, other=0.0)
    b = tl.load(B_ptr + offset, mask=mask, other=0.0)
    c = tl.interleave(a, b) 
    offset_o = 2 * pid * BLOCK_SIZE + tl.arange(0, 2 * BLOCK_SIZE)
    tl.store(output_ptr + offset_o, c, mask=(offset_o < 2 * N))

# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)
