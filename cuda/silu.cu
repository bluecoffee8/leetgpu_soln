#include <cuda_runtime.h>

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void silu_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i4 = i * 4;
    if (i4 + 3 < N) {
        float4 v = reinterpret_cast<const float4*>(input)[i];
        v.x = silu_f(v.x);
        v.y = silu_f(v.y);
        v.z = silu_f(v.z);
        v.w = silu_f(v.w);
        reinterpret_cast<float4*>(output)[i] = v;
    } else {
        for (int j = i4; j < N; j++)
            output[j] = silu_f(input[j]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
