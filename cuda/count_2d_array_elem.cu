#include <cuda_runtime.h>

__global__ void count_kernel(const int* __restrict__ input, int* output, int total, int K) {
    int val = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x)
        val += (input[i] == K);

    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(output, val);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    cudaMemset(output, 0, sizeof(int));

    int total = N * M;
    int block_size = 256;
    int grid_size = min((total + block_size - 1) / block_size, 1024);

    count_kernel<<<grid_size, block_size>>>(input, output, total, K);
}
