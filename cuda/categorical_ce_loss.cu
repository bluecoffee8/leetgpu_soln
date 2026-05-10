#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int b = 16; b > 0; b >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, b));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int b = 16; b > 0; b >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, b);
    }
    return v;
}

__global__ void CE(const float* __restrict__ logits, const int* __restrict__ true_labels, float* loss, int N, int C) {
    int warpid = blockIdx.x;
    int lane = threadIdx.x; 

    int base_ptr = warpid * C;
    float s = 0.0f; 
    for (int i = lane; i < C; i += 32) {
        s += __expf(logits[base_ptr+i]);
    }
    s = warp_reduce_sum(s); 
    if (lane == 0) {
        float z = __logf(s) - logits[base_ptr+true_labels[warpid]];
        atomicAdd(loss, z/N); 
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    CE<<<N, 32>>>(logits, true_labels, loss, N, C);
}
