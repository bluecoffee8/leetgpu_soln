#include <cuda_runtime.h>

#define TILE 16

// C[M][N] = alpha * A[M][K] @ B[N][K]^T + beta * C[M][N]
__global__ void gemm_nt(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K,
                        float alpha, float beta) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // As[ty][tx] = A[row][t*TILE+tx] — coalesced across tx
        As[ty][tx] = (row < M && t * TILE + tx < K) ? A[row * K + t * TILE + tx] : 0.f;
        // Bs[ty][tx] = B[col][t*TILE+ty] — transposed load so Bs[k][tx] = B[col][t*TILE+k]
        Bs[ty][tx] = (col < N && t * TILE + ty < K) ? B[col * K + t * TILE + ty] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        float old = (beta != 0.f) ? C[row * N + col] : 0.f;
        C[row * N + col] = alpha * acc + beta * old;
    }
}

// x:      (batch, d_in)
// W:      (d_out, d_in)   — base weight
// A:      (rank,  d_in)   — LoRA down-projection
// B:      (d_out, rank)   — LoRA up-projection
// output = x @ W^T + lora_scale * (x @ A^T) @ B^T
extern "C" void solve(const float* x, const float* W, const float* A, const float* B, float* output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) {
    float* temp;
    cudaMalloc(&temp, (size_t)batch * rank * sizeof(float));

    dim3 block(TILE, TILE);

    // output = x @ W^T  (M=batch, N=d_out, K=d_in)
    dim3 grid1((d_out + TILE - 1) / TILE, (batch + TILE - 1) / TILE);
    gemm_nt<<<grid1, block>>>(x, W, output, batch, d_out, d_in, 1.f, 0.f);

    // temp = x @ A^T  (M=batch, N=rank, K=d_in)
    dim3 grid2((rank + TILE - 1) / TILE, (batch + TILE - 1) / TILE);
    gemm_nt<<<grid2, block>>>(x, A, temp, batch, rank, d_in, 1.f, 0.f);

    // output += lora_scale * temp @ B^T  (M=batch, N=d_out, K=rank)
    dim3 grid3((d_out + TILE - 1) / TILE, (batch + TILE - 1) / TILE);
    gemm_nt<<<grid3, block>>>(temp, B, output, batch, d_out, rank, lora_scale, 1.f);

    cudaFree(temp);
}
