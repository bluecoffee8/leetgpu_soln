#include <cuda_runtime.h>

__global__ __launch_bounds__(1024) void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
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

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
