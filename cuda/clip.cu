#include <cuda_runtime.h>

__global__ void clip_kernel(const float* __restrict__ input, float* __restrict__ output, float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i4 = i * 4;
    if (i4 + 3 < N) {
        float4 v = reinterpret_cast<const float4*>(input)[i];
        v.x = fminf(fmaxf(v.x, lo), hi);
        v.y = fminf(fmaxf(v.y, lo), hi);
        v.z = fminf(fmaxf(v.z, lo), hi);
        v.w = fminf(fmaxf(v.w, lo), hi);
        reinterpret_cast<float4*>(output)[i] = v;
    } else {
        for (int j = i4; j < N; j++)
            output[j] = fminf(fmaxf(input[j], lo), hi);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
