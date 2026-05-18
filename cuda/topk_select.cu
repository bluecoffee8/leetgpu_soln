#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void copy_k(const float* __restrict__ src, float* __restrict__ dst, int k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    float* d_sorted;
    cudaMalloc(&d_sorted, N * sizeof(float));

    // Two-call CUB pattern: first to query temp storage size, then to sort
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_bytes, input, d_sorted, N);

    void* d_temp;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortKeysDescending(d_temp, temp_bytes, input, d_sorted, N);

    copy_k<<<(k + 255) / 256, 256>>>(d_sorted, output, k);

    cudaFree(d_temp);
    cudaFree(d_sorted);
}
