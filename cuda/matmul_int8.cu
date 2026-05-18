#include <cuda_runtime.h>
#include <stdint.h>

#define TILE 32

// Affine quantization: x_real = scale * (x_q - zero_point)
// C_q[m,n] = clamp(round(scale_A * scale_B / scale_C *
//             sum_k((A[m,k]-zp_A)*(B[k,n]-zp_B))) + zp_C, -128, 127)

__global__ void int8_matmul_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__ C,
    int M, int N, int K,
    float scale_ratio,
    int zp_A, int zp_B, int zp_C)
{
    __shared__ int8_t sA[TILE][TILE];
    __shared__ int8_t sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int32_t acc = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ak = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K)
            ? A[row * K + ak] : (int8_t)zp_A;

        int bk = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N)
            ? B[bk * N + col] : (int8_t)zp_B;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += (int32_t)(sA[threadIdx.y][k] - zp_A) *
                   (int32_t)(sB[k][threadIdx.x] - zp_B);

        __syncthreads();
    }

    if (row < M && col < N) {
        int val = __float2int_rn((float)acc * scale_ratio) + zp_C;
        C[row * N + col] = (int8_t)max(-128, min(127, val));
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    int8_matmul_kernel<<<grid, block>>>(
        A, B, C, M, N, K,
        scale_A * scale_B / scale_C,
        zero_point_A, zero_point_B, zero_point_C);
}
