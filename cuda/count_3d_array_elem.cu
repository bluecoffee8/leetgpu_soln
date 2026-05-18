#include <cuda_runtime.h>

__global__ void count_kernel(const int* __restrict__ input, int* __restrict__ output,
                             int total, int P) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int partial = 0;

    for (int i = blockIdx.x * blockDim.x + tid; i < total; i += gridDim.x * blockDim.x)
        partial += (input[i] == P);

    smem[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        int val = smem[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (tid == 0) atomicAdd(output, val);
    }
}

extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    int total = N * M * K;
    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    if (grid > 1024) grid = 1024;

    cudaMemset(output, 0, sizeof(int));
    count_kernel<<<grid, BLOCK, BLOCK * sizeof(int)>>>(input, output, total, P);
}
