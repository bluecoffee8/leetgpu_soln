#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff
#define BLOCK 32

__global__ void matvecmul(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N) {
    int r = blockIdx.x;
    int t = threadIdx.x; 
    float s = 0.0f;
    for (int i = t; i < N; i += BLOCK) {
        s += A[r*N+i] * x[i]; 
    }
    for (int i = BLOCK/2; i > 0; i >>= 1) {
        s += __shfl_down_sync(FULL_MASK, s, i); 
    }
    if (t == 0) {
        y[r] = s; 
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    matvecmul<<<M, BLOCK>>>(A, x, y, M, N);
}
