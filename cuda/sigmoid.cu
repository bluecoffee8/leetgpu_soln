#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void sigmoid_kernel(const float* __restrict__ X, float* __restrict__ Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i4 = i * 4;
    if (i4 + 3 < N) {
        float4 v = reinterpret_cast<const float4*>(X)[i];
        v.x = sigmoid_f(v.x);
        v.y = sigmoid_f(v.y);
        v.z = sigmoid_f(v.z);
        v.w = sigmoid_f(v.w);
        reinterpret_cast<float4*>(Y)[i] = v;
    } else {
        for (int j = i4; j < N; j++)
            Y[j] = sigmoid_f(X[j]);
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
