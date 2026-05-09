#include <cuda_runtime.h>

// __global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x; 

//     if (i < width * height) {
//         output[i] = 0.299 * input[3 * i] + 0.587 * input[3 * i + 1] + 0.114 * input[3 * i + 2];
//     }
// }

__global__ void rgb_to_grayscale_kernel(const float* __restrict__ input, float* __restrict__ output, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < total_pixels) {
        // output[i] = 0.299 * input[3 * i] + 0.587 * input[3 * i + 1] + 0.114 * input[3 * i + 2];
        float3 rgb = reinterpret_cast<const float3*>(input)[i];
        float gray = rgb.x * 0.299f + rgb.y * 0.587f + rgb.z * 0.114f;
        output[i] = gray; 
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, total_pixels);
    cudaDeviceSynchronize();
}
