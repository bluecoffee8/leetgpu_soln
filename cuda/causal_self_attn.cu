#include <cuda_runtime.h>

// One block per query row i.
// smem layout: [reduce_buf(bdim) | out_acc(d) | q_cache(d)]
// Two-pass approach: pass 1 finds the global max score, pass 2 computes the
// weighted sum using that max directly. This avoids the cascaded scale_prev
// multiplications in the online softmax, where early V[j] contributions are
// rescaled once per subsequent key — compounding rounding error for long seqs.
__global__ void causal_self_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int d
) {
    extern __shared__ float smem[];
    const int bdim = blockDim.x;
    float* reduce_buf = smem;
    float* out_acc    = smem + bdim;
    float* q_cache    = smem + bdim + d;

    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    // sqrtf is correctly rounded (IEEE 754); rsqrtf is only ~2 ULP accurate.
    const float scale = 1.0f / sqrtf((float)d);

    for (int k = tid; k < d; k += bdim) {
        out_acc[k] = 0.0f;
        q_cache[k] = Q[i * d + k];
    }
    __syncthreads();

    // Pass 1: find global max score across all j = 0..i
    float m_final = -INFINITY;
    for (int j = 0; j <= i; j++) {
        float partial = 0.0f;
        for (int k = tid; k < d; k += bdim)
            partial = fmaf(q_cache[k], K[j * d + k], partial);

        reduce_buf[tid] = partial;
        __syncthreads();
        for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
            __syncthreads();
        }
        float s = reduce_buf[0] * scale;
        __syncthreads();
        m_final = fmaxf(m_final, s);
    }

    // Pass 2: weighted sum with a single expf per key — no cascaded rescaling.
    // Each w_j = exp(s_j - m_final) is computed directly; V[j] is multiplied
    // by exactly one exp value, matching the two-pass reference numerically.
    float l = 0.0f;
    for (int j = 0; j <= i; j++) {
        float partial = 0.0f;
        for (int k = tid; k < d; k += bdim)
            partial = fmaf(q_cache[k], K[j * d + k], partial);

        reduce_buf[tid] = partial;
        __syncthreads();
        for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
            __syncthreads();
        }
        float s = reduce_buf[0] * scale;
        __syncthreads();

        float w = expf(s - m_final);
        for (int k = tid; k < d; k += bdim)
            out_acc[k] = fmaf(w, V[j * d + k], out_acc[k]);
        l += w;
    }

    const float inv_l = 1.0f / l;
    for (int k = tid; k < d; k += bdim)
        output[i * d + k] = out_acc[k] * inv_l;
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    const int block_size = 128;
    const size_t smem_bytes = (block_size + 2 * d) * sizeof(float);
    causal_self_attn_kernel<<<M, block_size, smem_bytes>>>(Q, K, V, output, M, d);
    cudaDeviceSynchronize();
}
