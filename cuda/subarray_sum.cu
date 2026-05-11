#include <cuda_runtime.h>

const int num_threads = 256;
const int warp_size = 32;
const int coarse_factor = 4; 

__global__ void subarray_sum(const int* __restrict__ input, int* __restrict__ output, int N, int S, int E) {
    __shared__ int sdata[num_threads / warp_size];

    int i = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x; 
    int w_id = threadIdx.x / warp_size;
    int l_id = threadIdx.x % warp_size; 

    int s = 0;
    for (int j = 0; j < coarse_factor; j++) {
        int i_ = i + j * num_threads;
        if (S + i_ <= E) s += input[S + i_];
    }
    for (int b = 16; b > 0; b >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, b); 
    }
    if (l_id == 0) sdata[w_id] = s;
    __syncthreads(); 

    if (w_id == 0) {
        s = (threadIdx.x < num_threads / warp_size) ? sdata[threadIdx.x] : 0.0f;
        for (int b = 16; b > 0; b >>= 1) {
            s += __shfl_down_sync(0xffffffff, s, b);
        }
        if (l_id == 0) {
            atomicAdd(output, s); 
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    int num_blocks = ((E-S+1) + (coarse_factor * num_threads) - 1) / (coarse_factor * num_threads);
    subarray_sum<<<num_blocks, num_threads>>>(input, output, N, S, E); 
}
