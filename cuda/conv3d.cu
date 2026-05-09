#include <cuda_runtime.h>

#define D_TILE 2
#define R_TILE 16
#define C_TILE 32

#define KERNEL_SIZE 5

__constant__ float kernel_const[1024];

__global__ void conv3d(const float* __restrict__ input, float* __restrict__ output, int i_d,
                      int i_r, int i_c, int k_d, int k_r, int k_c, int o_d, int o_r, int o_c) {
    __shared__ float smem[D_TILE + KERNEL_SIZE][R_TILE + KERNEL_SIZE][C_TILE + KERNEL_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z; 

    const int x = blockIdx.x * C_TILE + tx;
    const int y = blockIdx.y * R_TILE + ty; 
    const int z = blockIdx.z * D_TILE + tz; 

    for (int dz = 0; tz + dz < D_TILE + KERNEL_SIZE && z + dz < i_d; dz += D_TILE) {
        for (int dy = 0; ty + dy < R_TILE + KERNEL_SIZE && y + dy < i_r; dy += R_TILE) {
            for (int dx = 0; tx + dx < C_TILE + KERNEL_SIZE && x + dx < i_c; dx += C_TILE) {
                smem[tz+dz][ty+dy][tx+dx] = input[(z+dz)*i_r*i_c + (y+dy)*i_c + (x+dx)];
            }
        }
    }
    __syncthreads(); 

    if (x < o_c && y < o_r && z < o_d) {
        float s = 0.0f;
        for (int dz = 0; dz < k_d; dz++) {
            for (int dy = 0; dy < k_r; dy++) {
                for (int dx = 0; dx < k_c; dx++) {
                    s += smem[tz+dz][ty+dy][tx+dx] * kernel_const[dz*k_r*k_c+dy*k_c+dx];
                }
            }
        }
        output[z*o_r*o_c+y*o_c+x] = s; 
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int i_d,
                      int i_r, int i_c, int k_d, int k_r, int k_c) {
    cudaMemcpyToSymbol(kernel_const, kernel, k_d * k_r * k_c * sizeof(float));

    int o_d = i_d - k_d + 1;
    int o_r = i_r - k_r + 1;
    int o_c = i_c - k_c + 1;
     
    dim3 blockDim(C_TILE, R_TILE, D_TILE);
    dim3 gridDim((o_c + C_TILE - 1) / C_TILE, (o_r + R_TILE - 1) / R_TILE, (o_d + D_TILE - 1) / D_TILE);
    conv3d<<<gridDim, blockDim>>>(input, output, i_d, i_r, i_c, k_d, k_r, k_c, o_d, o_r, o_c); 
}
