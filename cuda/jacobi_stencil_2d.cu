#include <cuda_runtime.h>

#define TILE_W 32
#define TILE_H 16

__global__ void jacobi_2d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  int rows, int cols) {
    // +2 for one-cell halo on each side; TILE_W+2=34, no bank conflicts (34%32=2, shifts each row)
    __shared__ float smem[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_W + tx;
    int row = blockIdx.y * TILE_H + ty;

    int sx = tx + 1;
    int sy = ty + 1;

    smem[sy][sx] = (row < rows && col < cols) ? input[row * cols + col] : 0.0f;

    if (tx == 0)
        smem[sy][0] = (row < rows && col > 0) ? input[row * cols + col - 1] : 0.0f;
    if (tx == TILE_W - 1)
        smem[sy][TILE_W + 1] = (row < rows && col + 1 < cols) ? input[row * cols + col + 1] : 0.0f;
    if (ty == 0)
        smem[0][sx] = (row > 0 && col < cols) ? input[(row - 1) * cols + col] : 0.0f;
    if (ty == TILE_H - 1)
        smem[TILE_H + 1][sx] = (row + 1 < rows && col < cols) ? input[(row + 1) * cols + col] : 0.0f;

    __syncthreads();

    if (row >= rows || col >= cols) return;

    if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
        output[row * cols + col] = input[row * cols + col];
        return;
    }

    output[row * cols + col] = 0.25f * (smem[sy - 1][sx] + smem[sy + 1][sx] +
                                         smem[sy][sx - 1] + smem[sy][sx + 1]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE_W, TILE_H);
    dim3 grid((cols + TILE_W - 1) / TILE_W, (rows + TILE_H - 1) / TILE_H);
    jacobi_2d_kernel<<<grid, block>>>(input, output, rows, cols);
}
