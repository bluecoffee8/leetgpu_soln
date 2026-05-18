#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void topk_moe_kernel(const float* __restrict__ logits,
                                 float* __restrict__ topk_weights,
                                 int* __restrict__ topk_indices,
                                 int M, int E, int k) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int nwarps = bsz >> 5;

    extern __shared__ float smem[];
    float* svals  = smem;                        // [E]    logit values
    float* wvals  = svals + E;                   // [nwarps] warp-level reduction values
    int*   widxs  = (int*)(wvals + nwarps);      // [nwarps] warp-level reduction indices
    float* ssel   = (float*)(widxs + nwarps);    // [k]    selected logit values
    int*   sidx   = (int*)(ssel + k);            // [k]    selected expert indices

    // Load this token's logits into shared memory
    for (int i = tid; i < E; i += bsz)
        svals[i] = logits[row * E + i];
    __syncthreads();

    // Iteratively extract top-k maxima via warp-shuffle + cross-warp reduction
    for (int ki = 0; ki < k; ki++) {
        // Each thread finds its local max over strided elements
        float lmax = -FLT_MAX;
        int   lmax_idx = 0;
        for (int i = tid; i < E; i += bsz) {
            if (svals[i] > lmax) { lmax = svals[i]; lmax_idx = i; }
        }

        // Warp-level reduction via shuffle
        unsigned mask = 0xFFFFFFFF;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float  other_val = __shfl_xor_sync(mask, lmax,     offset);
            int    other_idx = __shfl_xor_sync(mask, lmax_idx, offset);
            if (other_val > lmax) { lmax = other_val; lmax_idx = other_idx; }
        }

        // Lane 0 of each warp writes to shared memory
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        if (lane_id == 0) {
            wvals[warp_id] = lmax;
            widxs[warp_id] = lmax_idx;
        }
        __syncthreads();

        // First warp reduces across warps
        if (warp_id == 0) {
            float  v = (lane_id < nwarps) ? wvals[lane_id] : -FLT_MAX;
            int    x = (lane_id < nwarps) ? widxs[lane_id] : 0;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float  ov = __shfl_xor_sync(mask, v, offset);
                int    ox = __shfl_xor_sync(mask, x, offset);
                if (ov > v) { v = ov; x = ox; }
            }
            if (lane_id == 0) {
                ssel[ki]        = v;
                sidx[ki]        = x;
                svals[x]        = -FLT_MAX; // mark as taken
            }
        }
        __syncthreads();
    }

    // Thread 0 computes softmax over top-k and writes output
    if (tid == 0) {
        // Numerically stable softmax
        float max_v = ssel[0];
        for (int ki = 1; ki < k; ki++)
            if (ssel[ki] > max_v) max_v = ssel[ki];

        float sum = 0.0f;
        for (int ki = 0; ki < k; ki++) {
            ssel[ki] = expf(ssel[ki] - max_v);
            sum += ssel[ki];
        }
        float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < k; ki++) {
            topk_weights[row * k + ki]  = ssel[ki] * inv_sum;
            topk_indices[row * k + ki]  = sidx[ki];
        }
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    const int bsz    = 256;
    const int nwarps = bsz >> 5;
    // smem: svals[E] + wvals[nwarps] + widxs[nwarps] + ssel[k] + sidx[k]
    size_t smem_bytes = (size_t)(E + nwarps + k) * sizeof(float)
                      + (size_t)(nwarps + k)      * sizeof(int);
    topk_moe_kernel<<<M, bsz, smem_bytes>>>(logits, topk_weights, topk_indices, M, E, k);
}
