#include <cuda_runtime.h>

__global__ __launch_bounds__(1024) void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;
    for (int n = 0; n < N; n++) {
        tmp += A[m * N + n] * B[n * K + k];
    }
    if (m < M && k < K) {
        C[m * K + k] = tmp; 
    }
}

const int WARPSIZE = 32;
#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

namespace wt {
    template<const int BLOCK_SIZE, const int a_stride, const int b_stride> 
    __device__ void load_from_gmem(float* A, float* B, float* As, float* Bs, int a_m, int a_n, int b_n, int b_k, int N, int K) {
        #pragma unroll
        for (int i = 0; i + a_stride <= BLOCK_SIZE; i += a_stride) {
            const float4 tmp = FETCH_FLOAT4(A[OFFSET(a_m + i, a_n, N)]);
            As[OFFSET(a_n, a_m + i, BLOCK_SIZE)] = tmp.x;
            As[OFFSET(a_n + 1, a_m + i, BLOCK_SIZE)] = tmp.y;
            As[OFFSET(a_n + 2, a_m + i, BLOCK_SIZE)] = tmp.z;
            As[OFFSET(a_n + 3, a_m + i, BLOCK_SIZE)] = tmp.w;
        }
        #pragma unroll
        for (int i = 0; i + b_stride <= BLOCK_SIZE; i += b_stride) {
            FETCH_FLOAT4(Bs[OFFSET(b_n + i, b_k, BLOCK_SIZE)]) = FETCH_FLOAT4(B[OFFSET(b_n + i, b_k, K)]);
        }
    }

    template<const int BLOCK_SIZE, const int WARP_BLOCK_SIZE, const int WARP_M_ITER, const int WARP_K_ITER, const int WARP_SUB_M, const int WARP_SUB_K, const int T>
    __device__ void process_from_smem(float* As, float* Bs, float* reg_m, float* reg_k, float* tmp, 
                                      const int warp_m, const int warp_k, const int lane_m, const int lane_k, int n, int N) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (n + i < N) {
                #pragma unroll 
                for (int j = 0; j < WARP_M_ITER; j++) {
                    for (int l = 0; l < T; l++) {
                        reg_m[j * T + l] = As[i * BLOCK_SIZE + warp_m * WARP_BLOCK_SIZE + j * WARP_SUB_M + lane_m * T + l];
                    }
                }

                #pragma unroll 
                for (int j = 0; j < WARP_K_ITER; j++) {
                    for (int l = 0; l < T; l++) {
                        reg_k[j * T + l] = Bs[i * BLOCK_SIZE + warp_k * WARP_BLOCK_SIZE + j * WARP_SUB_K + lane_k * T + l];
                    }
                }

                #pragma unroll
                for (int x = 0; x < WARP_M_ITER; x++) {
                    for (int y = 0; y < WARP_K_ITER; y++) {
                        for (int i = 0; i < T; i++) {
                            for (int j = 0; j < T; j++) {
                                tmp[(x * T + i) * (WARP_K_ITER * T) + (y * T + j)]
                                    += reg_m[x * T + i] * reg_k[y * T + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

template<const int BLOCK_SIZE, const int WARP_BLOCK_SIZE, const int WARP_K_ITER, const int T, const int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void matmul_vectorized(float* A, float* B, float* C, int M, int N, int K) {
    int bm = blockIdx.y; 
    int bk = blockIdx.x;

    const int warp_idx = threadIdx.x / WARPSIZE; 
    const int warp_m = warp_idx / (BLOCK_SIZE / WARP_BLOCK_SIZE); 
    const int warp_k = warp_idx % (BLOCK_SIZE / WARP_BLOCK_SIZE);

    constexpr int WARP_M_ITER = (WARP_BLOCK_SIZE * WARP_BLOCK_SIZE) / (WARPSIZE * T * T * WARP_K_ITER);
    constexpr int WARP_SUB_M = WARP_BLOCK_SIZE / WARP_M_ITER;
    constexpr int WARP_SUB_K = WARP_BLOCK_SIZE / WARP_K_ITER; 

    const int lane_idx = threadIdx.x % WARPSIZE;
    const int lane_m = lane_idx / (WARP_SUB_K / T);
    const int lane_k = lane_idx % (WARP_SUB_K / T); 

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE], Bs[BLOCK_SIZE * BLOCK_SIZE]; 

    A += bm * BLOCK_SIZE * N;
    C += bm * BLOCK_SIZE * K + bk * BLOCK_SIZE; 

    int a_m = threadIdx.x / (BLOCK_SIZE / 4);
    const int a_n_offset = (threadIdx.x % (BLOCK_SIZE / 4)) * 4;
    const int a_stride = (NUM_THREADS * 4) / BLOCK_SIZE;

    int b_n = threadIdx.x / (BLOCK_SIZE / 4);
    int b_k = bk * BLOCK_SIZE + (threadIdx.x % (BLOCK_SIZE / 4)) * 4;
    const int b_stride = (NUM_THREADS) / (BLOCK_SIZE / 4); 

    float tmp[WARP_M_ITER * T * WARP_K_ITER * T] = {0.0f}; 
    float reg_m[WARP_M_ITER * T] = {0.0f};
    float reg_k[WARP_K_ITER * T] = {0.0f};

    #pragma unroll
    for (int n = 0; n < N; n += BLOCK_SIZE) {
        int a_n = n + a_n_offset;
        wt::load_from_gmem<BLOCK_SIZE, a_stride, b_stride>(A, B, As, Bs, a_m, a_n, b_n, b_k, N, K);
        __syncthreads();
        wt::process_from_smem<BLOCK_SIZE, WARP_BLOCK_SIZE, WARP_M_ITER, WARP_K_ITER, WARP_SUB_M, WARP_SUB_K, T>(
            As, Bs, reg_m, reg_k, tmp, warp_m, warp_k, lane_m, lane_k, n, N
        );
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * K;
        __syncthreads();
    }
    for (int x = 0; x < WARP_M_ITER; x++) {
        for (int y = 0; y < WARP_K_ITER; y++) {
            float* Ctmp = C + (x * WARP_SUB_M) * K + (y * WARP_SUB_K);
            for (int i = 0; i < min(T, M - bm * BLOCK_SIZE - x * WARP_SUB_M - lane_m * T); i += 1) {
                for (int j = 0; j < min(T, K - bk * BLOCK_SIZE - y * WARP_SUB_K - lane_k * T); j += 4) {
                    if (j + 3 < K - bk * BLOCK_SIZE - y * WARP_SUB_K - lane_k * T) {
                        FETCH_FLOAT4(Ctmp[OFFSET(lane_m * T + i, lane_k * T + j, K)]) = FETCH_FLOAT4(tmp[OFFSET(x * WARP_SUB_M + i, y * WARP_SUB_K + j, WARP_K_ITER * T)]);
                    } else {
                        for (int j_ = j; j_ < min(T, K - bk * BLOCK_SIZE - y * WARP_SUB_K - lane_k * T); j_++) {
                            Ctmp[OFFSET(lane_m * T + i, lane_k * T + j_, K)] = tmp[OFFSET(x * WARP_SUB_M + i, y * WARP_SUB_K + j_, WARP_K_ITER * T)];
                        }
                    }
                }
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* A, float* B, float* C, int M, int N, int K) {
    if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
        const int BLOCK_SIZE = 64, T = 4, WARP_BLOCK_SIZE = 32, WARP_K_ITER = 2, NUM_THREADS = 128; 
        dim3 threadsPerBlock(NUM_THREADS);
        dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_vectorized<BLOCK_SIZE, WARP_BLOCK_SIZE, WARP_K_ITER, T, NUM_THREADS><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    } else {
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    }
}
