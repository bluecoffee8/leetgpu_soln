#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int total_pixels) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z < total_pixels) {
        uchar4 px = reinterpret_cast<uchar4*>(image)[z];
        px.x = 255 - px.x;
        px.y = 255 - px.y;
        px.z = 255 - px.z;
        // alpha channel (px.w) left unchanged
        reinterpret_cast<uchar4*>(image)[z] = px;
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int W, int H) {
    int total_pixels = W * H;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, total_pixels);
    cudaDeviceSynchronize();
}
