#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff
#define WARPS_PER_ROW 4
#define BLOCK (32 * WARPS_PER_ROW)

__global__ void matvecmul(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N) {
    __shared__ float smem[WARPS_PER_ROW];

    int r       = blockIdx.x;
    int t       = threadIdx.x;
    int warp_id = t / 32;
    int lane    = t % 32;

    float s = 0.0f;
    for (int i = t; i < N; i += BLOCK)
        s += A[r * N + i] * x[i];

    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(FULL_MASK, s, off);

    if (lane == 0) smem[warp_id] = s;
    __syncthreads();

    if (warp_id == 0) {
        s = (lane < WARPS_PER_ROW) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(FULL_MASK, s, off);
        if (lane == 0) y[r] = s;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    matvecmul<<<M, BLOCK>>>(A, x, y, M, N);
}
