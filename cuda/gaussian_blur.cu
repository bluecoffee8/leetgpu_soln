#include <cuda_runtime.h>

__global__ void gaussian_blur_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_kernel,
    float* __restrict__ output,
    int input_rows, int input_cols,
    int kernel_rows, int kernel_cols
) {
    extern __shared__ float smem[];

    const int half_kr   = kernel_rows >> 1;
    const int half_kc   = kernel_cols >> 1;
    const int tile_cols = blockDim.x + kernel_cols - 1;
    const int tile_rows = blockDim.y + kernel_rows - 1;
    const int kernel_size = kernel_rows * kernel_cols;

    float* s_kernel = smem;
    float* s_input  = smem + kernel_size;

    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // Load convolution kernel weights into shared memory
    for (int i = tid; i < kernel_size; i += block_size)
        s_kernel[i] = conv_kernel[i];

    // Load input tile with halo into shared memory (zero-pad out-of-bounds)
    const int in_start_row = blockIdx.y * blockDim.y - half_kr;
    const int in_start_col = blockIdx.x * blockDim.x - half_kc;
    const int tile_size    = tile_rows * tile_cols;

    for (int i = tid; i < tile_size; i += block_size) {
        int r = i / tile_cols + in_start_row;
        int c = i % tile_cols + in_start_col;
        s_input[i] = (r >= 0 && r < input_rows && c >= 0 && c < input_cols)
                     ? input[r * input_cols + c] : 0.0f;
    }

    __syncthreads();

    const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row >= input_rows || out_col >= input_cols) return;

    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; kr++) {
        const float* krow  = s_kernel + kr * kernel_cols;
        const float* irow  = s_input + (threadIdx.y + kr) * tile_cols + threadIdx.x;
        for (int kc = 0; kc < kernel_cols; kc++)
            sum += krow[kc] * irow[kc];
    }

    output[out_row * input_cols + out_col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    dim3 block(16, 16);
    dim3 grid((input_cols + block.x - 1) / block.x,
              (input_rows  + block.y - 1) / block.y);

    int tile_cols = block.x + kernel_cols - 1;
    int tile_rows = block.y + kernel_rows - 1;
    size_t smem_size = (kernel_rows * kernel_cols + tile_rows * tile_cols) * sizeof(float);

    gaussian_blur_kernel<<<grid, block, smem_size>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols
    );
}
