#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define NWARPS (BLOCK_SIZE / 32)

// Warp-level inclusive scan via shuffle
__device__ float warp_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset) val += tmp;
    }
    return val;
}

// Block-level inclusive scan; writes per-block totals to block_sums (may be nullptr)
__global__ void scan_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             float* __restrict__ block_sums,
                             int N) {
    __shared__ float smem[NWARPS];

    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int lane  = threadIdx.x & 31;
    int warp  = threadIdx.x >> 5;

    float val = tid < N ? in[tid] : 0.0f;
    val = warp_scan(val);

    if (lane == 31) smem[warp] = val;
    __syncthreads();

    // First warp scans the per-warp totals
    if (warp == 0) {
        float w = lane < NWARPS ? smem[lane] : 0.0f;
        w = warp_scan(w);
        if (lane < NWARPS) smem[lane] = w;
    }
    __syncthreads();

    if (warp > 0) val += smem[warp - 1];

    if (tid < N) out[tid] = val;

    // Thread blockDim.x-1 always holds the block total (padded slots are 0)
    if (block_sums && threadIdx.x == blockDim.x - 1)
        block_sums[blockIdx.x] = val;
}

// Add the inclusive prefix of block totals back into each block's output
__global__ void add_kernel(float* __restrict__ out,
                            const float* __restrict__ prefix_sums,
                            int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N && blockIdx.x > 0)
        out[tid] += prefix_sums[blockIdx.x - 1];
}

static void prefix_sum_device(const float* in, float* out, int N) {
    if (N == 0) return;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (nblocks == 1) {
        scan_kernel<<<1, BLOCK_SIZE>>>(in, out, nullptr, N);
        return;
    }

    float *block_sums, *scanned_sums;
    cudaMalloc(&block_sums,   nblocks * sizeof(float));
    cudaMalloc(&scanned_sums, nblocks * sizeof(float));

    scan_kernel<<<nblocks, BLOCK_SIZE>>>(in, out, block_sums, N);
    prefix_sum_device(block_sums, scanned_sums, nblocks);  // recurse on block totals
    add_kernel<<<nblocks, BLOCK_SIZE>>>(out, scanned_sums, N);

    cudaFree(block_sums);
    cudaFree(scanned_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    prefix_sum_device(input, output, N);
}
