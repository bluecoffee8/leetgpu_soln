import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel_grouped(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_k >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_an > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_ck > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_block = a + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_block = b + (offs_n[:, None] * stride_bn + offs_bk[None, :] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a_val = tl.load(a_block, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)
        b_val = tl.load(b_block, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        accumulator += tl.dot(a_val, b_val, allow_tf32=False)
        a_block += BLOCK_SIZE_N * stride_an 
        b_block += BLOCK_SIZE_N * stride_bn
    c_val = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_block = c + stride_cm * offs_cm[:, None] + stride_ck * offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_block, c_val, mask=c_mask)

@triton.jit 
def matmul(a_ptr, b_ptr, c_ptr, M, N, K, 
           stride_am, stride_an,
           stride_bn, stride_bk, 
           stride_cm, stride_ck, 
           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1) 

    oM = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mM = (oM < M)
    oK = pid1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mK = (oK < K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, N, BLOCK_SIZE_N):
        oN = n + tl.arange(0, BLOCK_SIZE_N) 
        mN = oN < N 

        A = tl.load(a_ptr + oM[:, None] * stride_am + oN[None, :] * stride_an, mask=(mM[:, None] & mN[None, :]), other=0.0)
        B = tl.load(b_ptr + oN[:, None] * stride_bn + oK[None, :] * stride_bk, mask=(mN[:, None] & mK[None, :]), other=0.0)
        acc += tl.dot(A, B, input_precision="ieee")

    tl.store(c_ptr + oM[:, None] * stride_cm + oK[None, :] * stride_ck, acc, mask=(mM[:, None] & mK[None, :]))

@triton.jit 
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SM):
    gid = tile_id // num_pid_in_group
    first_pid_m = gid * GROUP_SIZE_M 
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) 
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_k = (tile_id % num_pid_in_group) // group_size_m 
    return pid_m, pid_k 

@triton.jit 
def matmul_persistent(a_ptr, b_ptr, c_ptr, M, N, K, 
                      stride_am, stride_an,
                      stride_bn, stride_bk, 
                      stride_cm, stride_ck, 
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                      BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
                      NUM_SM: tl.constexpr):
    pid = tl.program_id(axis=0) 
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_k
    num_pid_in_group = GROUP_SIZE_M * num_pid_k

    # tl.assume(pid_m >= 0)
    # tl.assume(pid_k >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_an > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_ck > 0)

    # tile_id_c = pid - NUM_SM 

    for tile_id in tl.range(pid, num_tiles, NUM_SM):
        pid0, pid1 = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SM)
        tl.assume(pid0 >= 0)
        tl.assume(pid1 >= 0)

        oM = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mM = (oM < M)
        oK = pid1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mK = (oK < K)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for n in range(0, N, BLOCK_SIZE_N):
            oN = n + tl.arange(0, BLOCK_SIZE_N) 
            mN = oN < N 

            A = tl.load(a_ptr + oM[:, None] * stride_am + oN[None, :] * stride_an, mask=(mM[:, None] & mN[None, :]), other=0.0)
            B = tl.load(b_ptr + oN[:, None] * stride_bn + oK[None, :] * stride_bk, mask=(mN[:, None] & mK[None, :]), other=0.0)
            acc += tl.dot(A, B, input_precision="ieee")

        tl.store(c_ptr + oM[:, None] * stride_cm + oK[None, :] * stride_ck, acc, mask=(mM[:, None] & mK[None, :]))        

@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        ):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([offs_am, offs_k])
        # tl.device_print("a: ", a)
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(a_desc.dtype)
    c_desc.store([offs_am, offs_bn], accumulator)

# a, b, c are tensors on the GPU
def solve(a_ptr: torch.Tensor, b_ptr: torch.Tensor, c_ptr: torch.Tensor, M: int, N: int, K: int):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), )
    # Leading dimensions must be multiples of 16-byte strides
    if M % 4 == 0 and N % 4 == 0 and K % 4 == 0:
        import ctypes
        cudart = ctypes.CDLL("libcudart.so")
        cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cudart.cudaMalloc.restype = ctypes.c_int
        from typing import Optional
        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            ptr = ctypes.c_void_p()
            err = cudart.cudaMalloc(ctypes.byref(ptr), size)
            if err != 0:
                raise RuntimeError(f"cudaMalloc failed, code {err}")
            return ptr.value
        triton.set_allocator(alloc_fn)
        matmul_kernel_make_tensor_desciptor[grid](
            a_ptr, b_ptr, c_ptr,
            M, K, N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_K=32,
            BLOCK_SIZE_N=32,
        )
    else:
        matrix_multiplication_kernel_grouped[grid](
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            N, 1,
            K, 1,
            K, 1,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_K=64,
            BLOCK_SIZE_N=64,
            GROUP_SIZE_M=8
        )

    # stride_am, stride_an = N, 1
    # stride_bn, stride_bk = K, 1
    # stride_cm, stride_ck = K, 1

    # stride_am, stride_an = a.stride()
    # stride_bn, stride_bk = b.stride()
    # stride_cm, stride_ck = c.stride()

    # BLOCK_SIZE_M = 64
    # BLOCK_SIZE_N = 64
    # BLOCK_SIZE_K = 64
    # GROUP_SIZE_M = 4 

    # TMA descriptors require a global memory allocation
    # def alloc_fn(size: int, alignment: int, stream: int | None):
    #     return torch.empty(size, device="cuda", dtype=torch.int8)

    # triton.set_allocator(alloc_fn)

    # grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), )

    # if M % 4 == 0 and N % 4 == 0 and K % 4 == 0:
    #     import ctypes
    #     cudart = ctypes.CDLL("libcudart.so")
    #     cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    #     cudart.cudaMalloc.restype = ctypes.c_int
    #     from typing import Optional
    #     # TMA descriptors require a global memory allocation
    #     def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    #         ptr = ctypes.c_void_p()
    #         err = cudart.cudaMalloc(ctypes.byref(ptr), size)
    #         if err != 0:
    #             raise RuntimeError(f"cudaMalloc failed, code {err}")
    #         return ptr.value
    #     triton.set_allocator(alloc_fn)
    #     matmul_kernel_make_tensor_desciptor[grid](
    #         a, b, c,
    #         M, N, K,
    #         BLOCK_SIZE_M=32,
    #         BLOCK_SIZE_K=32,
    #         BLOCK_SIZE_N=32,
    #     )
    # else:
    #     matrix_multiplication_kernel_grouped[grid](
    #         a, b, c, M, N, K, 
    #         N, 1, K, 1, K, 1, 
    #         BLOCK_SIZE_M=64,
    #         BLOCK_SIZE_K=64,
    #         BLOCK_SIZE_N=64,
    #         GROUP_SIZE_M=8
    #     )

    # NUM_SM = torch.cuda.get_device_properties("cuda").multi_processor_count

    # grid = (min(NUM_SM, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_K)), )
    # matmul_persistent[grid](a, b, c, M, N, K, 
    #                   stride_am, stride_an,
    #                   stride_bn, stride_bk, 
    #                   stride_cm, stride_ck, 
    #                   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_SM)

    # grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
    # matmul[grid](a, b, c, M, N, K,
    #             stride_am, stride_an,
    #             stride_bn, stride_bk, 
    #             stride_cm, stride_ck, 
    #             BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"]), )
    # matrix_multiplication_kernel[grid](
    #     a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    #     BLOCK_SIZE_M = 64,
    #     BLOCK_SIZE_N = 64,
    #     BLOCK_SIZE_K = 64,
    #     GROUP_SIZE_M = 4
    # )
