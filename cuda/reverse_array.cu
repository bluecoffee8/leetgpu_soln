#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lo = i * 2;
    int hi = N - 1 - lo;
    if (lo + 1 <= hi - 1) {
        // swap pair [lo, lo+1] with [hi, hi-1] (reversed order)
        float2 a = reinterpret_cast<float2*>(input)[i];
        float2 b = reinterpret_cast<float2*>(input)[(N - lo - 2) / 2];
        float2 ra = make_float2(b.y, b.x);
        float2 rb = make_float2(a.y, a.x);
        reinterpret_cast<float2*>(input)[i] = ra;
        reinterpret_cast<float2*>(input)[(N - lo - 2) / 2] = rb;
    } else if (lo < N / 2) {
        // scalar swap for the middle element when N is odd
        float tmp = input[lo];
        input[lo] = input[N - 1 - lo];
        input[N - 1 - lo] = tmp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
