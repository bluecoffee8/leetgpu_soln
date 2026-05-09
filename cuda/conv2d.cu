#include <cuda_runtime.h>

__constant__ float kernel_const[31 * 31];

#define BLOCK_SIZE 32 
#define KERNEL_SIZE 32

__global__ void conv2d(const float* __restrict__ input, float* __restrict__ output, int i_rows,
                      int i_cols, int k_rows, int k_cols, int o_rows, int o_cols) {
    __shared__ float smem[BLOCK_SIZE + KERNEL_SIZE][BLOCK_SIZE + KERNEL_SIZE]; 

    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    const int x = blockIdx.x * BLOCK_SIZE + tx;
    const int y = blockIdx.y * BLOCK_SIZE + ty;

    for (int dx = 0; tx+dx < BLOCK_SIZE+KERNEL_SIZE && x+dx < i_rows; dx += BLOCK_SIZE) {
        for (int dy = 0; ty+dy < BLOCK_SIZE+KERNEL_SIZE && y+dy < i_cols; dy += BLOCK_SIZE) {
            smem[tx+dx][ty+dy] = input[(x+dx) * i_cols + (y+dy)];
        }
    }
    __syncthreads(); 

    if (x < o_rows && y < o_cols) {
        float s = 0.0f;
        for (int dx = 0; dx < k_rows; dx++) {
            for (int dy = 0; dy < k_cols; dy++) {
                s += smem[tx+dx][ty+dy] * kernel_const[dx*k_cols+dy];
            }
        }
        output[x*o_cols+y] = s; 
    }
}

__global__ void conv2d_fast(const float* __restrict__ input, float* __restrict__ output, int i_rows,
                      int i_cols, int k_rows, int k_cols, int o_rows, int o_cols) {
    __shared__ float smem[BLOCK_SIZE + KERNEL_SIZE][BLOCK_SIZE + KERNEL_SIZE]; 

    const int ty = threadIdx.y; 
    const int tx = threadIdx.x;

    const int y = blockIdx.y * BLOCK_SIZE + ty;
    const int x = blockIdx.x * BLOCK_SIZE + tx;

    for (int dy = 0; ty + dy < BLOCK_SIZE+KERNEL_SIZE && y + dy < i_rows; dy += BLOCK_SIZE) {
        for (int dx = 0; tx + dx < BLOCK_SIZE+KERNEL_SIZE && x + dx < i_cols; dx += BLOCK_SIZE) {
            smem[ty+dy][tx+dx] = input[(y + dy) * i_cols + (x + dx)];
        }
    }
    __syncthreads(); 

    if (y < o_rows && x < o_cols) {
        float s = 0.0f;
        for (int dy = 0; dy < k_rows; dy++) {
            for (int dx = 0; dx < k_cols; dx++) {
                s += smem[ty+dy][tx+dx] * kernel_const[dy*k_cols+dx];
            }
        }
        output[y*o_cols+x] = s; 
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int i_rows,
                      int i_cols, int k_rows, int k_cols) {
    cudaMemcpyToSymbol(kernel_const, kernel, k_rows * k_cols * sizeof(float));

    int o_rows = i_rows - k_rows + 1;
    int o_cols = i_cols - k_cols + 1;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 gridDim((o_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, (o_cols + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // conv2d<<<gridDim, blockDim>>>(input, output, i_rows, i_cols, k_rows, k_cols, o_rows, o_cols);

    dim3 gridDim((o_cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (o_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    conv2d_fast<<<gridDim, blockDim>>>(input, output, i_rows, i_cols, k_rows, k_cols, o_rows, o_cols);
}
