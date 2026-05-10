#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* __restrict__ input, int* __restrict__ output, int N, int num_bins) {
    extern __shared__ int shared_hist[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    const int4* int4_ptr = reinterpret_cast<const int4*>(input);
    if (threadIdx.x < num_bins) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads(); 
    if (idx * 4 + 4 <= N) {
        int4 i4 = int4_ptr[idx]; 
        atomicAdd(&shared_hist[i4.w],1);
        atomicAdd(&shared_hist[i4.x],1);
        atomicAdd(&shared_hist[i4.y],1);
        atomicAdd(&shared_hist[i4.z],1);
    } else {
        for (int i = idx * 4; i < N; i++) {
            atomicAdd(&shared_hist[input[i]],1);
        }
    }
    __syncthreads();
    if (threadIdx.x < num_bins) {
        atomicAdd(output + threadIdx.x, shared_hist[threadIdx.x]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int block_dim = 512;
    int grid_dim = (N + 4*block_dim-1)/(4*block_dim);
    cudaMemset(histogram, 0, sizeof(int)*num_bins);
    histogram_kernel<<<grid_dim, block_dim, sizeof(int)*num_bins>>>(input, histogram, N, num_bins);
}
