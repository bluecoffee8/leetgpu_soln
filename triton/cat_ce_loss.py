import torch
import triton
import triton.language as tl

@triton.jit 
def compute_sum(logits_ptr, sum_ptr, N, C, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr):
    pid0 = tl.program_id(axis=0) 
    pid1 = tl.program_id(axis=1) 

    mN = pid0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mC = pid1 * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offset = mN[:, None] * C + mC[None, :]
    mask = (mN[:, None] < N) & (mC[None, :] < C)
    l = tl.load(logits_ptr + offset, mask=mask, other=-float("inf"))
    tl.atomic_add(sum_ptr + mN, tl.sum(tl.exp(l), axis=1), mask=(mN < N))

@triton.jit 
def compute_label_sum(logits_ptr, true_labels, label_sum, N, C, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0) 

    mN = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl_z = tl.load(true_labels + mN, mask=(mN < N))
    z = tl.load(logits_ptr + mN * C + tl_z, mask=(mN < N), other=0.0)
    tl.atomic_add(label_sum, -tl.sum(z))

@triton.jit 
def compute_loss(logits_ptr, sum_ptr, loss, N, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0) 

    mN = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    s = tl.load(sum_ptr + mN, mask=(mN < N), other=1.0)
    tl.atomic_add(loss, tl.sum(tl.log(s)) / N)

# logits, true_labels, loss are tensors on the GPU
def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_C = 64
    sum_vector = torch.zeros((N, ), device='cuda')
    label_sum = torch.zeros((1, ), device='cuda')
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(C, BLOCK_SIZE_C))
    compute_sum[grid](logits, sum_vector, N, C, BLOCK_SIZE_N, BLOCK_SIZE_C)
    grid = (triton.cdiv(N, BLOCK_SIZE_N), )
    compute_label_sum[grid](logits, true_labels, label_sum, N, C, BLOCK_SIZE_N)
    loss[0] = label_sum[0] / N 
    compute_loss[grid](logits, sum_vector, loss, N, BLOCK_SIZE_N)

