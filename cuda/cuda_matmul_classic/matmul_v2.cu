#include <cuda_runtime.h>

template<const int BLOCK_SIZE>
__global__ __launch_bounds__(1024) void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int m = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    int k = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    float tmp = 0.0;
    for (int n = 0; n < N; n++) {
        tmp += A[m * N + n] * B[n * K + k];
    }
    if (m < M && k < K) {
        C[m * K + k] = tmp; 
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
