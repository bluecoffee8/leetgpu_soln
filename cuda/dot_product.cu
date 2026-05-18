#include <cuda_runtime.h>

__global__ void dotproduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N) {
    float s = 0.0f;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        s += A[i] * B[i];
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(result, s);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    constexpr int BLOCK_SIZE = 256;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    dotproduct<<<gridDim, blockDim>>>(A, B, result, N);
}
