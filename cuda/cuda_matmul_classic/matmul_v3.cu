#include <cuda_runtime.h>

template<const int BLOCK_SIZE>
__global__ __launch_bounds__(1024) void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int bm = blockIdx.y; 
    int bk = blockIdx.x;

    int tm = threadIdx.x / BLOCK_SIZE;
    int tk = threadIdx.x % BLOCK_SIZE; 

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE], Bs[BLOCK_SIZE * BLOCK_SIZE]; 

    A += bm * BLOCK_SIZE * N; 
    B += bk * BLOCK_SIZE; 
    C += bm * BLOCK_SIZE * K + bk * BLOCK_SIZE; 

    float tmp = 0.0;
    for (int n = 0; n < N; n += BLOCK_SIZE) {
        As[tm * BLOCK_SIZE + tk] = A[tm * N + tk];
        Bs[tm * BLOCK_SIZE + tk] = B[tm * K + tk]; 
        __syncthreads(); 
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * K; 
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (n + i < N) {
                tmp += As[tm * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + tk];
            }
        }
        __syncthreads(); 
    }
    int m = bm * BLOCK_SIZE + tm; 
    int k = bk * BLOCK_SIZE + tk;
    if (m < M && k < K) {
        C[tm * K + tk] = tmp; 
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* A, float* B, float* C, int M, int N, int K) {
    const int BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
