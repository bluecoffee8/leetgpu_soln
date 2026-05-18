#include <cuda_runtime.h>

__device__ __forceinline__ float leaky_relu_f(float x) {
    return x >= 0.0f ? x : 0.01f * x;
}

__global__ void leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i4 = i * 4;
    if (i4 + 3 < N) {
        float4 v = reinterpret_cast<const float4*>(input)[i];
        v.x = leaky_relu_f(v.x);
        v.y = leaky_relu_f(v.y);
        v.z = leaky_relu_f(v.z);
        v.w = leaky_relu_f(v.w);
        reinterpret_cast<float4*>(output)[i] = v;
    } else {
        for (int j = i4; j < N; j++)
            output[j] = leaky_relu_f(input[j]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
