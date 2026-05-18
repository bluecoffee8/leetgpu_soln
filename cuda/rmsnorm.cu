#include <cuda_runtime.h>

__global__ void sum_sq_kernel(const float* __restrict__ x, float* __restrict__ out, int N) {
    extern __shared__ float smem[];
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        acc += x[i] * x[i];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = acc;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    acc = (lane < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
        if (lane == 0) atomicAdd(out, acc);
    }
}

__global__ void norm_kernel(const float* __restrict__ x, float* __restrict__ y,
                             float scale, float beta, int N) {
    int N4 = N >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N4; i += gridDim.x * blockDim.x) {
        float4 v = x4[i];
        y4[i] = {v.x * scale + beta, v.y * scale + beta,
                 v.z * scale + beta, v.w * scale + beta};
    }
    int tail_start = N4 << 2;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        y[i] = x[i] * scale + beta;
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    float* d_ss;
    cudaMalloc(&d_ss, sizeof(float));
    cudaMemset(d_ss, 0, sizeof(float));

    const int block = 256;
    const int grid = min((N + block - 1) / block, 1024);
    const int smem = (block >> 5) * sizeof(float);

    sum_sq_kernel<<<grid, block, smem>>>(input, d_ss, N);

    float h_ss;
    cudaMemcpy(&h_ss, d_ss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ss);

    float scale = gamma / sqrtf(h_ss / N + eps);
    norm_kernel<<<grid, block>>>(input, output, scale, beta, N);
}
