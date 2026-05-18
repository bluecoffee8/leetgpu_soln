#include <cuda_runtime.h>

__global__ void subarray_sum_kernel(const int* __restrict__ input, int* __restrict__ output,
                                    int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    const int rows = E_ROW - S_ROW + 1;
    const int cols = E_COL - S_COL + 1;
    const int total = rows * cols;

    // Grid-stride accumulation; integer div/mod maps flat index to (row, col)
    int val = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        int r = S_ROW + i / cols;
        int c = S_COL + i % cols;
        val += input[r * M + c];
    }

    // Warp-level reduction via shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    // One slot per warp; blockDim.x=256 → 8 warps
    __shared__ int smem[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if (lane == 0)
        smem[warp] = val;
    __syncthreads();

    // First warp reduces all warp partials, then atomically adds to output
    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x >> 5)) ? smem[threadIdx.x] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x == 0)
            atomicAdd(output, val);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    const int total = (E_ROW - S_ROW + 1) * (E_COL - S_COL + 1);
    cudaMemset(output, 0, sizeof(int));
    if (total <= 0) return;

    const int blockSize = 256;
    const int gridSize = min((total + blockSize - 1) / blockSize, 1024);
    subarray_sum_kernel<<<gridSize, blockSize>>>(input, output, M, S_ROW, E_ROW, S_COL, E_COL);
}
