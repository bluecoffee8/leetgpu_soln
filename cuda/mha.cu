#include <cuda_runtime.h>
#include <float.h>

// One block per (query position, head).
// Shared memory layout: scores[N] | reduce_buf[32]
__global__ void mha_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int N, int d_model, int h, int head_dim
) {
    int q  = blockIdx.x;   // query position [0, N)
    int hi = blockIdx.y;   // head index [0, h)
    int tid  = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];
    float* scores     = smem;       // [N]
    float* reduce_buf = smem + N;   // [32]

    const float scale  = rsqrtf((float)head_dim);
    const float* q_ptr = Q + q * d_model + hi * head_dim;

    // ── Step 1: scaled dot-product scores ────────────────────────────────────
    for (int k = tid; k < N; k += bdim) {
        const float* k_ptr = K + k * d_model + hi * head_dim;
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++)
            dot += q_ptr[d] * k_ptr[d];
        scores[k] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: softmax numerically-stable (find max) ─────────────────────────
    float lmax = -FLT_MAX;
    for (int k = tid; k < N; k += bdim)
        lmax = fmaxf(lmax, scores[k]);
    for (int s = 16; s > 0; s >>= 1)
        lmax = fmaxf(lmax, __shfl_down_sync(0xFFFFFFFF, lmax, s));
    if ((tid & 31) == 0) reduce_buf[tid >> 5] = lmax;
    __syncthreads();
    if (tid == 0) {
        float gmax = reduce_buf[0];
        int nw = bdim >> 5;
        for (int i = 1; i < nw; i++) gmax = fmaxf(gmax, reduce_buf[i]);
        reduce_buf[0] = gmax;
    }
    __syncthreads();
    float gmax = reduce_buf[0];

    // ── Step 3: exp + sum ────────────────────────────────────────────────────
    float lsum = 0.0f;
    for (int k = tid; k < N; k += bdim) {
        scores[k] = __expf(scores[k] - gmax);
        lsum += scores[k];
    }
    for (int s = 16; s > 0; s >>= 1)
        lsum += __shfl_down_sync(0xFFFFFFFF, lsum, s);
    if ((tid & 31) == 0) reduce_buf[tid >> 5] = lsum;
    __syncthreads();
    if (tid == 0) {
        float gsum = 0.0f;
        int nw = bdim >> 5;
        for (int i = 0; i < nw; i++) gsum += reduce_buf[i];
        reduce_buf[0] = __fdividef(1.0f, gsum);
    }
    __syncthreads();
    float inv_sum = reduce_buf[0];

    for (int k = tid; k < N; k += bdim)
        scores[k] *= inv_sum;
    __syncthreads();

    // ── Step 4: weighted sum over V ───────────────────────────────────────────
    float* out_ptr = output + q * d_model + hi * head_dim;
    for (int d = tid; d < head_dim; d += bdim) {
        float val = 0.0f;
        for (int k = 0; k < N; k++)
            val += scores[k] * V[k * d_model + hi * head_dim + d];
        out_ptr[d] = val;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h) {
    int head_dim = d_model / h;

    // Block size: multiple of 32, large enough to cover N scores efficiently
    int block_size = 256;
    if (N <= 64)  block_size = 64;
    else if (N <= 128) block_size = 128;

    // scores[N] + reduce_buf[32]
    size_t smem = (N + 32) * sizeof(float);

    dim3 grid(N, h);
    mha_kernel<<<grid, block_size, smem>>>(Q, K, V, output, N, d_model, h, head_dim);
}
