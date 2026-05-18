#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* __restrict__ A, float* __restrict__ B, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i4 = i * 4;
    if (i4 + 3 < total) {
        reinterpret_cast<float4*>(B)[i] = reinterpret_cast<const float4*>(A)[i];
    } else {
        for (int j = i4; j < total; j++)
            B[j] = A[j];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
