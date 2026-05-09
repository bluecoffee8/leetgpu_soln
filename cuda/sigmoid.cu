#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* __restrict__ X, float* __restrict__ Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < N) {
        float val = X[i]; 
        Y[i] = 1.0 / (1.0 + __expf(-val));
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
