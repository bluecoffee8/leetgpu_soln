#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    extern __shared__ float smem[]; 
    float* max_arr = smem; 
    float* sum_arr = &smem[blockDim.x];

    int tid = threadIdx.x; 
    float local_max = -FLT_MAX; 
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    max_arr[tid] = local_max; 
    __syncthreads(); 

    for (int b = blockDim.x/2; b > 0; b>>=1) {
        if (tid < b) {
            max_arr[tid] = fmaxf(max_arr[tid], max_arr[tid + b]);
        }
        __syncthreads(); 
    }
    float max_val = max_arr[0];
    __syncthreads(); 

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += __expf(input[i] - max_val);
    }    
    sum_arr[tid] = local_sum;
    __syncthreads(); 

    for (int b = blockDim.x/2; b > 0; b>>=1) {
        if (tid < b) {
            sum_arr[tid] += sum_arr[tid + b];
        }
        __syncthreads(); 
    }
    float tot_sum = sum_arr[0];
    __syncthreads(); 

    if (tid + blockDim.x * blockIdx.x < N) {
        output[tid + blockDim.x * blockIdx.x ] = __expf(input[tid + blockDim.x * blockIdx.x ] - max_val) / tot_sum; 
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t smem_size = 2 * threadsPerBlock * sizeof(float); 

    softmax_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, N);
    cudaDeviceSynchronize();
}
