#include <cuda_runtime.h>

// h[t] = a[t]*h[t-1] + x[t], h[-1] = 0
// Maps to prefix scan: combine((a1,x1),(a2,x2)) = (a2*a1, a2*x1+x2)
// One block per batch; Hillis-Steele scan per tile, sequential carry across tiles.
__global__ void linear_recurrence_kernel(const float* __restrict__ a,
                                          const float* __restrict__ x,
                                          float* __restrict__ h,
                                          int B, int L) {
    extern __shared__ float smem[];
    float* s_a = smem;
    float* s_x = smem + blockDim.x;

    int b = blockIdx.x;
    if (b >= B) return;

    const float* a_b = a + (long long)b * L;
    const float* x_b = x + (long long)b * L;
    float* h_b       = h + (long long)b * L;

    // carry_x = h at the last element of the previous tile (h[-1] = 0)
    float carry_x = 0.0f;

    for (int tile_start = 0; tile_start < L; tile_start += (int)blockDim.x) {
        int t = tile_start + threadIdx.x;

        // Load tile; pad with identity (1, 0) for out-of-bounds threads
        s_a[threadIdx.x] = (t < L) ? a_b[t] : 1.0f;
        s_x[threadIdx.x] = (t < L) ? x_b[t] : 0.0f;
        __syncthreads();

        // Hillis-Steele inclusive prefix scan within tile
        for (int stride = 1; stride < (int)blockDim.x; stride <<= 1) {
            float a_left = (threadIdx.x >= stride) ? s_a[threadIdx.x - stride] : 1.0f;
            float x_left = (threadIdx.x >= stride) ? s_x[threadIdx.x - stride] : 0.0f;
            __syncthreads();
            if (threadIdx.x >= stride) {
                float cur_a = s_a[threadIdx.x];
                float cur_x = s_x[threadIdx.x];
                s_a[threadIdx.x] = cur_a * a_left;
                s_x[threadIdx.x] = cur_a * x_left + cur_x;
            }
            __syncthreads();
        }

        // Apply carry from prior tiles and write output
        if (t < L)
            h_b[t] = s_a[threadIdx.x] * carry_x + s_x[threadIdx.x];

        // Update carry: h at last real element of this tile
        int last = min((int)blockDim.x, L - tile_start) - 1;
        carry_x = s_a[last] * carry_x + s_x[last];
        __syncthreads();
    }
}

extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {
    const int block_size = 1024;
    size_t smem = 2 * block_size * sizeof(float);
    linear_recurrence_kernel<<<B, block_size, smem>>>(a, x, h, B, L);
}
