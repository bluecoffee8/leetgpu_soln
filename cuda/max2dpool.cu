#include <cuda_runtime.h>
#include <float.h>

__global__ void max2dpool_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  int N, int C, int H, int W,
                                  int H_out, int W_out,
                                  int kernel_size, int stride, int padding) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    int n = nc / C;
    int c = nc % C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -FLT_MAX;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W) continue;
            float val = input[((n * C + c) * H + h_in) * W + w_in];
            if (val > max_val) max_val = val;
        }
    }

    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
}

extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N * C);

    max2dpool_kernel<<<grid, block>>>(input, output, N, C, H, W, H_out, W_out,
                                     kernel_size, stride, padding);
}
