#include <cuda_runtime.h>

__global__ void kernel(const int* input, int* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;

    const int4* input4 = reinterpret_cast<const int4*>(input);
    int num_int4 = N / 4;

    // Vectorized grid-stride loop: process 4 elements per iteration
    for (int i = idx; i < num_int4; i += blockDim.x * gridDim.x) {
        int4 val = input4[i];
        if (val.x == K) count++;
        if (val.y == K) count++;
        if (val.z == K) count++;
        if (val.w == K) count++;
    }

    // Handle remaining elements (if N is not a multiple of 4)
    int start_remainder = num_int4 * 4;
    for (int i = start_remainder + idx; i < N; i += blockDim.x * gridDim.x) {
        if (input[i] == K) {
            count++;
        }
    }

    // Reduce the count within each warp using shuffle primitives
    for (int offset = 16; offset > 0; offset /= 2) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    // Only the first thread of each warp (lane 0) performs the atomic update
    if ((threadIdx.x & 31) == 0 && count > 0) {
        atomicAdd(output, count);
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
}
