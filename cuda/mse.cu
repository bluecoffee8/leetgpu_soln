#include <cuda_runtime.h>

const int threads = 128;
const int warp_size = 32;
const int tw = 4;

__global__ void mse_kernel(const float* __restrict__ pred, const float* __restrict__ targets, float* mse, int N) {
    __shared__ float acc[threads / warp_size];

    int base = blockIdx.x * threads * tw; 
    int idx = base + threadIdx.x; 

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size; 

    float val = 0.0f;
    for (int i = 0; i < tw; i++) {
        int li = idx + i * blockDim.x;
        if (li < N) {
            float delta = pred[li] - targets[li];
            val += delta * delta; 
        }
    }

    for (int B = warp_size/2; B > 0; B>>=1) {
        val += __shfl_down_sync(0xffffffff, val, B);
    }
    if (lane_id == 0) acc[warp_id] = val; 
    __syncthreads(); 

    if (warp_id == 0) {
        val = (threadIdx.x < threads / warp_size) ? acc[threadIdx.x] : 0.0f; 
        for (int B = warp_size / 2; B > 0; B /= 2) {
            val += __shfl_down_sync(0xffffffff, val, B);
        }
        if (lane_id == 0) {
            atomicAdd(mse, val / N); 
        }
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* pred, const float* targets, float* mse, int N) {
    int blocks = (N + threads * tw - 1) / (threads * tw);
    mse_kernel<<<blocks, threads>>>(pred, targets, mse, N);
}
