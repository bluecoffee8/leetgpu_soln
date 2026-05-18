#include <cuda_runtime.h>
#include <float.h>

// Grid: (num_q_heads, seq_len) — each block handles one (q_head, q_pos) pair.
// Score computation: warp w owns keys k = w, w+num_warps, ... (no cross-warp sync in the k loop).
// Softmax: two block-level reductions (max, sum) via warp shuffle + smem.
// Output: thread d owns output dim d, loops over k with coalesced V access.
__global__ void gqa_kernel(
    const float* __restrict__ Q,    // [H_q,  S, D]
    const float* __restrict__ K,    // [H_kv, S, D]
    const float* __restrict__ V,    // [H_kv, S, D]
    float* __restrict__ output,     // [H_q,  S, D]
    int H_q, int H_kv, int S, int D
) {
    const int q_head   = blockIdx.x;
    const int q_pos    = blockIdx.y;
    const int kv_head  = (q_head * H_kv) / H_q;

    const int tid       = threadIdx.x;
    const int bsz       = blockDim.x;
    const int num_warps = bsz >> 5;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;

    // smem layout: [q_cache | scores | warp_buf]
    extern __shared__ float smem[];
    float* q_cache  = smem;
    float* scores   = smem + D;
    float* warp_buf = smem + D + S;

    const float scale = rsqrtf((float)D);

    // Load Q[q_head, q_pos, :] into shared memory
    const float* q_ptr = Q + ((long long)q_head * S + q_pos) * D;
    for (int d = tid; d < D; d += bsz)
        q_cache[d] = q_ptr[d];
    __syncthreads();

    // Compute attention scores: each warp handles its strided subset of key positions.
    // K access within a warp is coalesced (all lanes read consecutive head_dim elements).
    const float* k_base = K + (long long)kv_head * S * D;
    for (int k = warp_id; k < S; k += num_warps) {
        const float* k_ptr = k_base + (long long)k * D;
        float partial = 0.0f;
        for (int d = lane; d < D; d += 32)
            partial += q_cache[d] * k_ptr[d];
        for (int mask = 16; mask > 0; mask >>= 1)
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, mask);
        if (lane == 0)
            scores[k] = partial * scale;
    }
    __syncthreads();

    // Softmax — block-level max reduction
    float m = -FLT_MAX;
    for (int k = tid; k < S; k += bsz)
        m = fmaxf(m, scores[k]);
    for (int mask = 16; mask > 0; mask >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, mask));
    if (lane == 0) warp_buf[warp_id] = m;
    __syncthreads();
    if (warp_id == 0) {
        m = (lane < num_warps) ? warp_buf[lane] : -FLT_MAX;
        for (int mask = 16; mask > 0; mask >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, mask));
        if (lane == 0) warp_buf[0] = m;
    }
    __syncthreads();
    m = warp_buf[0];

    // Exp(score - max) + block-level sum reduction
    float s = 0.0f;
    for (int k = tid; k < S; k += bsz) {
        float e = expf(scores[k] - m);
        scores[k] = e;
        s += e;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        s += __shfl_xor_sync(0xFFFFFFFF, s, mask);
    if (lane == 0) warp_buf[warp_id] = s;
    __syncthreads();
    if (warp_id == 0) {
        s = (lane < num_warps) ? warp_buf[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, mask);
        if (lane == 0) warp_buf[0] = s;
    }
    __syncthreads();
    const float inv_s = 1.0f / warp_buf[0];

    for (int k = tid; k < S; k += bsz)
        scores[k] *= inv_s;
    __syncthreads();

    // Output = softmax @ V.
    // Thread d computes output dim d; inner loop over k is coalesced across threads
    // (all threads access V[k, tid..tid+bsz-1] simultaneously for each k).
    const float* v_base = V + (long long)kv_head * S * D;
    float* out_ptr = output + ((long long)q_head * S + q_pos) * D;
    for (int d = tid; d < D; d += bsz) {
        float acc = 0.0f;
        for (int k = 0; k < S; k++)
            acc += scores[k] * v_base[(long long)k * D + d];
        out_ptr[d] = acc;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    const int block_size = 128;
    const int num_warps  = block_size / 32;
    const size_t smem    = (size_t)(head_dim + seq_len + num_warps) * sizeof(float);

    cudaFuncSetAttribute(gqa_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    dim3 grid(num_q_heads, seq_len);
    gqa_kernel<<<grid, block_size, smem>>>(
        Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim
    );
}
