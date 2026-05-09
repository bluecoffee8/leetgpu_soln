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

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int NUM_PER_THREAD = 8; 

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + (BLOCK_SIZE * NUM_PER_THREAD) - 1) / (BLOCK_SIZE * NUM_PER_THREAD));
    reduce<BLOCK_SIZE, NUM_PER_THREAD><<<gridSize, blockSize>>>(input, output, N); 
}
