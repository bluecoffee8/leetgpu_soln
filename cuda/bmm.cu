#include <cuda_runtime.h>

static constexpr int TILE = 32;

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int b   = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    const float* A_b = A + (long long)b * M * K;
    const float* B_b = B + (long long)b * K * N;
    float*       C_b = C + (long long)b * M * N;

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A_b[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B_b[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_b[row * N + col] = acc;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, BATCH);
    bmm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
