#include <cuda_runtime.h>

template<const int BLOCK_SIZE, const int T>
__global__ __launch_bounds__(64) void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int bm = blockIdx.y; 
    int bk = blockIdx.x;

    int block_size = BLOCK_SIZE / T; 
    int thread_num = block_size * block_size; 

    // real located, strided 
    int tm = (threadIdx.x / block_size) * T;
    int tk = (threadIdx.x % block_size) * T; 

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE], Bs[BLOCK_SIZE * BLOCK_SIZE]; 

    A += bm * BLOCK_SIZE * N; 
    B += bk * BLOCK_SIZE; 
    C += bm * BLOCK_SIZE * K + bk * BLOCK_SIZE; 

    // start 

    int a_m = threadIdx.x / BLOCK_SIZE;
    int a_n = threadIdx.x % BLOCK_SIZE;
    int a_stride = thread_num / BLOCK_SIZE;

    int b_n = threadIdx.x / BLOCK_SIZE;
    int b_k = threadIdx.x % BLOCK_SIZE;
    int b_stride = thread_num / BLOCK_SIZE; 

    float tmp[T][T] = {0.0f}; 
    #pragma unroll
    for (int n = 0; n < N; n += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += a_stride) {
            As[(a_m + i) * BLOCK_SIZE + a_n] = A[(a_m + i) * N + a_n];
        }
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += b_stride) {
            Bs[(b_n + i) * BLOCK_SIZE + b_k] = B[(b_n + i) * K + b_k];
        }
        __syncthreads(); 
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * K; 
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (n + i < N) {
                #pragma unroll 
                for (int j = 0; j < T; j++) {
                    for (int l = 0; l < T; l++) {
                        tmp[j][l] += As[(tm + j) * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + (tk + l)];
                    }
                }
            }
        }
        __syncthreads(); 
    }
    #pragma unroll 
    for (int j = 0; j < T; j++) {
        for (int l = 0; l < T; l++) {
            int m = bm * BLOCK_SIZE + tm + j;
            int k = bk * BLOCK_SIZE + tk + l;
            if (m < M && k < K) {
                C[(tm + j) * K + (tk + l)] = tmp[j][l]; 
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* A, float* B, float* C, int M, int N, int K) {
    const int BLOCK_SIZE = 32, T = 4; 
    dim3 threadsPerBlock((BLOCK_SIZE / T) * (BLOCK_SIZE / T));
    dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul<BLOCK_SIZE, T><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
