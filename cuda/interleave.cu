#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    // if (i < N) {
    //     output[2 * i] = A[i];
    //     output[2 * i + 1] = B[i]; 
    // }
    if (i >= N) return;
    ((float2*)(output))[i] = make_float2(A[i], B[i]);
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
