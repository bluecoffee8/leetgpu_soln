#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i4 = i * 4;
    if (i4 + 3 < N) {
        float4 v = reinterpret_cast<const float4*>(input)[i];
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        reinterpret_cast<float4*>(output)[i] = v;
    } else {
        for (int j = i4; j < N; j++)
            output[j] = fmaxf(input[j], 0.0f);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
