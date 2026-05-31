#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <utility>

__global__ __launch_bounds__(1024) void gemm_simple(half* A, half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0f;
    for (int n = 0; n < N; n++) {
        tmp += __half2float(A[m * N + n]) * __half2float(B[n * K + k]);
    }
    if (m < M && k < K) {
        C[m * K + k] = __float2half(alpha * tmp + beta * __half2float(C[m * K + k]));
    }
}

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

__device__ uint64_t make_smem_desc(half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
  }

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N> 
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

template<int BlockMajorSize, int BlockMinorSize> 
void create_tensor_map(CUtensorMap *tma_map, half* gmem_ptr, int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width, (uint64_t)BlockMajorSize * blocks_height, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(half), sizeof(half) * BlockMinorSize * blocks_width, 0, 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1}; 

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address, gmem_prob_shape, 
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    assert(result == CUDA_SUCCESS);
}

CUtensorMap *d_tma_map_A, *d_tma_map_B;

template<int BlockMajorSize, int BlockMinorSize> 
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(half* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d, tma_map_host; 
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d; 
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], half* sA, half* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]), desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<const int BM, const int BN, const int BK, const int WGMMA_M, const int WGMMA_N, const int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) gemm(int M, int N, int K, half* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB) {
    __shared__ alignas(128) half sA[BM * BK], sB[BK * BN];
    float d[WGMMA_N/16][8];
    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_blocks_k = K / BK; 
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);
    #pragma nv_diag_suppress static_var_with_dynamic_init

    __shared__ barrier barA, barB; 
    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads(); 

    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; block_k_iter++) {
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter * BK, num_block_m * BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter * BK, num_block_n * BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB)); 
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        warpgroup_arrive(); 
        wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    {
        int tid = threadIdx.x; 
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp * 16 + lane / 4;
        half *block_C = C + num_block_n * BN * M + num_block_m * BM;

        for (int m_it = 0; m_it < BM / WGMMA_M; m_it++) {
            for (int n_it = 0; n_it < BN / WGMMA_N; n_it++) {
                for (int w = 0; w < WGMMA_N/16; w++) {
                    int col = 16 * w + 2 * (tid % 4);
                    #define IDX(i, j) ((j + n_it * WGMMA_N) * M + ((i) + m_it * WGMMA_M))

                    block_C[IDX(row, col)] = d[w][0];
                    block_C[IDX(row, col+1)] = d[w][1];
                    block_C[IDX(row+8, col)] = d[w][2];
                    block_C[IDX(row+8, col+1)] = d[w][3]; 

                    block_C[IDX(row, col+8)] = d[w][4];
                    block_C[IDX(row, col+9)] = d[w][5];
                    block_C[IDX(row+8, col+8)] = d[w][6];
                    block_C[IDX(row+8, col+9)] = d[w][7];

                    #undef IDX
                }
            }
        }
    }
}

__global__ void transpose_kernel(half* __restrict__ out, const half* __restrict__ in, int K, int N) {
    __shared__ half tile[32][33]; // 33 to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < N && y < K)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];

    __syncthreads();

    // transposed output is N x K
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < K && y < N)
        out[y * K + x] = tile[threadIdx.x][threadIdx.y];
}

// A, B, and C are device pointers
extern "C" void solve(half* A, half* B_, half* C_, int M, int N, int K, float alpha,
                      float beta) {
    if (M % 64 == 0 && N % 64 == 0 && K % 64 == 0) {
        constexpr int BM = 64, BN = 64, BK = 64, NUM_THREADS = 128, WGMMA_M = 64, WGMMA_N = 64, WGMMA_K = 16;
        // tranpose matrix B
        half *B, *C; cudaMalloc(&B, N * K * sizeof(half)); cudaMalloc(&C, M * N * sizeof(half));
        dim3 transpose_block_B(32, 32), transpose_grid_B((N + 31) / 32, (K + 31) / 32);
        transpose_kernel<<<transpose_grid_B, transpose_block_B>>>(B, B_, K, N);
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
        dim3 gemm_grid((M / BM) * (N / BN));
        gemm<BM, BN, BK, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS><<<gemm_grid, NUM_THREADS>>>(M, N, K, C, d_tma_map_A, d_tma_map_B);
        dim3 transpose_block_C(32, 32), transpose_grid_C((M + 31) / 32, (N + 31) / 32);
        transpose_kernel<<<transpose_grid_C, transpose_block_C>>>(C_, C, N, M);
    } else {
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        gemm_simple<<<blocksPerGrid, threadsPerBlock>>>(A, B_, C_, M, N, K, alpha, beta);
        cudaDeviceSynchronize();
    }
}
