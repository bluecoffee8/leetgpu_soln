import torch


# A, B, C are tensors on the GPU
def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
):
    A = A.view(M, K)
    B = B.view(K, N)
    C = C.view(M, N)
    
    A_c = (A - zero_point_A).float() 
    B_c = (B - zero_point_B).float() 

    acc = torch.matmul(A_c, B_c) 

    acc *= ((scale_A * scale_B) / scale_C)

    acc_r = torch.round(acc) + zero_point_C

    C[:] = torch.clamp(acc_r, -128, 127).to(torch.int8)