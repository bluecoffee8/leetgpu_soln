#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < input_size - kernel_size + 1) {
        #pragma unroll 
        for (int j = 0; j < kernel_size; j++) {
            output[i] += input[i+j] * kernel[j]; 
        }
    }
}

#define KERNEL_SIZE 2048

template<int BLOCK_SIZE>
__global__ void convolution_1d_kernel_v2(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    __shared__ float sdata[BLOCK_SIZE+KERNEL_SIZE];
    int base = blockIdx.x * blockDim.x;
    int ti = threadIdx.x; 
    int output_size = input_size - kernel_size + 1;

    if (base + ti < input_size) {
        sdata[ti] = input[base + ti]; 
    }

    if (ti == BLOCK_SIZE - 1) {
        for (int j = BLOCK_SIZE; j < BLOCK_SIZE+kernel_size-1; j++) {
            if (base + j < input_size) {
                sdata[j] = input[base + j];
            }
        }
    }

    __syncthreads(); 

    if (base + ti < output_size) {
        #pragma unroll 
        for (int j = 0; j < kernel_size; j++) {
            output[base + ti] += sdata[ti + j] * kernel[j]; 
        }
    }
}

#define MAX_KERNEL_SIZE 2047
#define ITEMS_PER_THREAD 4
__constant__ float c_kernel[MAX_KERNEL_SIZE]; 

__global__ void conv1d(const float* __restrict__ input, float* __restrict__ output, 
                        int input_size, int kernel_size, int output_size) {
    extern __shared__ float s_input[];

    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    int bdim = blockDim.x; 

    int items_per_block = bdim * ITEMS_PER_THREAD;
    int block_start = bid * items_per_block;
    int tile_size = items_per_block + kernel_size - 1;

    for (int t = tid; t < tile_size; t += bdim) {
        int global_idx = block_start + t;
        if (global_idx < input_size) {
            s_input[t] = input[global_idx];
        } else {
            s_input[t] = 0.0f; 
        }
    }

    __syncthreads();

    int thread_offset = tid * ITEMS_PER_THREAD; 
    int global_idx = block_start + thread_offset; 

    if (global_idx < output_size) {
        float sum[ITEMS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int j = 0; j < kernel_size; j++) {
            float k_val = c_kernel[j];

            #pragma unroll
            for (int k = 0; k < ITEMS_PER_THREAD; k++) {
                sum[k] += s_input[thread_offset + k + j] * k_val; 
            }
        }

        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; k++) {
            if (global_idx + k < output_size) {
                output[global_idx + k] = sum[k]; 
            }
        }
    }
}


// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    // int output_size = input_size - kernel_size + 1;
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
    //                                                           kernel_size);

    // constexpr int BLOCK_SIZE = 256; 
    // int blocksPerGrid = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // convolution_1d_kernel_v2<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(input, kernel, output, input_size,
                                                                    //   kernel_size);

    cudaMemcpyToSymbol(c_kernel, kernel, kernel_size * sizeof(float));

    int output_size = input_size - kernel_size + 1;
    int threads = 256;
    int blocks = (output_size + (threads * ITEMS_PER_THREAD) - 1) / (threads * ITEMS_PER_THREAD);
    int shared_mem_size = (kernel_size + (threads * ITEMS_PER_THREAD)) * sizeof(float);
    
    conv1d<<<blocks, threads, shared_mem_size>>>(input, output, input_size, kernel_size, output_size);
    cudaDeviceSynchronize();
}
