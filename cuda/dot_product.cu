#include <cuda_runtime.h>

__global__ void dotproduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd(result, A[i] * B[i]); 
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    constexpr int BLOCK_SIZE = 256;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    dotproduct<<<gridDim, blockDim>>>(A, B, result, N);
}
