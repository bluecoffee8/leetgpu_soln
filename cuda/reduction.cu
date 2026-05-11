#include <cuda_runtime.h>

template<int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void reduce(const float* __restrict__ input, float* __restrict__ output, int N) {
    __shared__ float smem[BLOCK_SIZE];
    int base = blockIdx.x * BLOCK_SIZE * NUM_PER_THREAD; 
    int tid = threadIdx.x; 
    // for (int i = base + threadIdx.x; i < base + BLOCK_SIZE * NUM_PER_THREAD; i += BLOCK_SIZE) {
    //     if (i < N) {
    //         smem[threadIdx.x] += input[i]; 
    //     }
    // }
    float sum = 0; 
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        if (tid + base + i * BLOCK_SIZE < N) {
            sum += input[tid + base + i * BLOCK_SIZE];
        }
    }
    smem[tid] = sum; 
    __syncthreads();

    for (int B = BLOCK_SIZE/2; B > 0; B >>= 1) {
        if (tid < B) {
            smem[tid] += smem[tid + B];
        }
        __syncthreads(); 
    }
    if (tid == 0) {
        atomicAdd(output, smem[0]); 
    }
}

const int num_threads = 128;
const int warp_size = 32;
const int coarse_factor = 4; 

__global__ void warp_reduce(const float* __restrict__ input, float* __restrict__ output, int N) {
    __shared__ float sdata[num_threads / warp_size];

    int i = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x; 
    int w_id = threadIdx.x / warp_size;
    int l_id = threadIdx.x % warp_size; 

    float s = 0.0;
    for (int j = 0; j < coarse_factor; j++) {
        int i_ = i + j * num_threads;
        if (i_ < N) s += input[i_];
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

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // constexpr int BLOCK_SIZE = 256;
    // constexpr int NUM_PER_THREAD = 8; 

    // dim3 blockSize(BLOCK_SIZE);
    // dim3 gridSize((N + (BLOCK_SIZE * NUM_PER_THREAD) - 1) / (BLOCK_SIZE * NUM_PER_THREAD));
    // reduce<BLOCK_SIZE, NUM_PER_THREAD><<<gridSize, blockSize>>>(input, output, N); 
    int num_blocks = (N + (coarse_factor * num_threads) - 1) / (coarse_factor * num_threads);
    warp_reduce<<<num_blocks, num_threads>>>(input, output, N);
}
