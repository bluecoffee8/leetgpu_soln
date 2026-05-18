#include <cuda_runtime.h>
#include <cub/cub.cuh>

// data is device pointer
extern "C" void solve(float* data, int N) {
    float* d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, data, d_out, N);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, data, d_out, N);

    cudaMemcpy(data, d_out, N * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_out);
    cudaFree(d_temp);
}
