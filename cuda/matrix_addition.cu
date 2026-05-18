#include <cuda_runtime.h>

__global__ void matrix_add(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i4 = i * 4;
    if (i4 + 3 < total) {
        float4 a = reinterpret_cast<const float4*>(A)[i];
        float4 b = reinterpret_cast<const float4*>(B)[i];
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y;
        c.z = a.z + b.z; c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[i] = c;
    } else {
        for (int j = i4; j < total; j++)
            C[j] = A[j] + B[j];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, total);
    cudaDeviceSynchronize();
}
