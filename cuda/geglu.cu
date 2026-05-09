#include <cuda_runtime.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < halfN) {
        output[i] = input[i] * 0.5f * input[halfN + i] * (1.0f + erff(input[halfN + i] * 0.707106781f));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
