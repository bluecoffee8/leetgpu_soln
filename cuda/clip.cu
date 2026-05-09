#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < N) {
        output[i] = min(max(input[i], lo), hi); 
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
