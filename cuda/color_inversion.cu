#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int W, int H) {
    const int z = blockIdx.x * blockDim.x + threadIdx.x; 
    if (z < W * H) {
        image[4*z] = 255 - image[4*z]; 
        image[4*z+1] = 255 - image[4*z+1]; 
        image[4*z+2] = 255 - image[4*z+2]; 
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int W, int H) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (W * H + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, W, H);
    cudaDeviceSynchronize();
}
