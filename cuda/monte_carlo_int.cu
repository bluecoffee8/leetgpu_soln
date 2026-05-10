#include <cuda_runtime.h>

const int warp_size = 32;
const int threads = 128;
const int tw = 4;

__global__ void mc_kernel(const float* __restrict__ y_samples, float* __restrict__ result, float a, float b, int n_samples) {
    __shared__ float acc[threads / warp_size];

    int i = blockIdx.x * blockDim.x * tw + threadIdx.x;
    int w_id = threadIdx.x / warp_size;
    int l_id = threadIdx.x % warp_size; 

    float s = 0.0f;
    for (int j = 0; j < tw; j++) {
        int i_ = i + j * blockDim.x;
        if (i_ < n_samples) {
            s += y_samples[i_];
        }
    }
    for (int B = warp_size/2; B>0; B>>=1) {
        s += __shfl_down_sync(0xffffffff, s, B); 
    }
    if (l_id == 0) acc[w_id] = s; 
    __syncthreads(); 

    if (w_id == 0) {
        s = (threadIdx.x < threads / warp_size) ? acc[threadIdx.x] : 0.0f;
        for (int B = warp_size/2; B>0; B>>=1) {
            s += __shfl_down_sync(0xffffffff, s, B); 
        }
        if (l_id == 0) {
            atomicAdd(result, s * (b - a) / (float)n_samples); 
        }
    }
}

__global__ void mc_kernel_simple(const float* __restrict__ y_samples, float* __restrict__ result, int n_samples, float mul) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid; 

    smem[tid] = (i < n_samples) ? y_samples[i] : 0.0f;
    __syncthreads(); 

    for (int B = blockDim.x / 2; B > 0; B>>=1) {
        if (tid < B) {
            smem[tid] += smem[tid + B]; 
        }
        __syncthreads(); 
    }
    if (tid == 0) {
        atomicAdd(result, smem[0] * mul); 
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int num_blocks = (n_samples + (tw * threads) - 1) / (tw * threads);
    mc_kernel<<<num_blocks, threads>>>(y_samples, result, a, b, n_samples);
    // int BLOCK_SIZE = 256;
    // int NUM_BLOCKS = (n_samples + BLOCK_SIZE-1) / BLOCK_SIZE;
    // mc_kernel_simple<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(y_samples, result, n_samples, (b-a)/(float)n_samples);
    // cudaDeviceSynchronize(); 
}
