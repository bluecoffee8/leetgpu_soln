#include <cuda_runtime.h>

// One block per channel; threads reduce over the N dimension.
// Strided access (N*C layout, fixed c) is unavoidable for the stats pass.
// Uses warp shuffles + shared memory to compute per-channel mean and inv_std.
__global__ void compute_stats(const float* __restrict__ input, float* __restrict__ mean,
                              float* __restrict__ inv_std, int N, int C, float eps) {
    extern __shared__ float smem[];

    int c      = blockIdx.x;
    int tid    = threadIdx.x;
    int nwarps = blockDim.x >> 5;

    float sum = 0.f, sum_sq = 0.f;
    for (int n = tid; n < N; n += blockDim.x) {
        float v = __ldg(&input[n * C + c]);
        sum    += v;
        sum_sq += v * v;
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum    += __shfl_down_sync(0xffffffff, sum,    offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) {
        smem[wid]          = sum;
        smem[wid + nwarps] = sum_sq;
    }
    __syncthreads();

    if (tid == 0) {
        sum = 0.f; sum_sq = 0.f;
        for (int w = 0; w < nwarps; w++) {
            sum    += smem[w];
            sum_sq += smem[w + nwarps];
        }
        float mu  = sum / N;
        float var = fmaxf(sum_sq / N - mu * mu, 0.f);
        mean[c]    = mu;
        inv_std[c] = rsqrtf(var + eps);
    }
}

// Threads iterate over C (last dim → coalesced), blocks iterate over N.
__global__ void apply_norm(const float* __restrict__ input, const float* __restrict__ gamma,
                           const float* __restrict__ beta,  const float* __restrict__ mean,
                           const float* __restrict__ inv_std, float* __restrict__ output,
                           int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (c < C) {
        int   idx = n * C + c;
        float x   = __ldg(&input[idx]);
        output[idx] = __ldg(&gamma[c]) * (x - __ldg(&mean[c])) * __ldg(&inv_std[c])
                    + __ldg(&beta[c]);
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    constexpr int BLOCK = 256;

    float *d_mean, *d_inv_std;
    cudaMalloc(&d_mean,    C * sizeof(float));
    cudaMalloc(&d_inv_std, C * sizeof(float));

    int nwarps  = BLOCK / 32;
    int smem_sz = 2 * nwarps * sizeof(float);
    compute_stats<<<C, BLOCK, smem_sz>>>(input, d_mean, d_inv_std, N, C, eps);

    dim3 block(BLOCK);
    dim3 grid((C + BLOCK - 1) / BLOCK, N);
    apply_norm<<<grid, block>>>(input, gamma, beta, d_mean, d_inv_std, output, N, C);

    cudaFree(d_mean);
    cudaFree(d_inv_std);
}
