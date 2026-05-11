/*
NVIDIA RTX 3070 

SM count: 46
Tensor cores: 184
L1 core: 128 kB per SM 
L2 cache: 4 MB

FP16: 20.31 TFLOPS
FP32: 20.31 TFLOPS
FP64: 317.4 GLOPS

Memory: 8 GB
Bandwidth: 448 GB/s
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -> " << cudaGetErrorString(err) << std::endl;   \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

__global__ void matmul(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < M && j < K) {
        float s = 0.0f;
        for (int k = 0; k < N; k++) {
            s += A[i * N + k] * B[k * K + j];
        }
        C[i * K + j] = s; 
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TK>
__global__ void matmul_fast(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    const unsigned int c_row = blockIdx.y;
    const unsigned int c_col = blockIdx.x; 


    const int t_col = threadIdx.x % (BK/TK);
    const int t_row = threadIdx.x / (BK/TK); 

    __shared__ float As[BM * BN];
    __shared__ float Bs[BN * BK];

    A += c_row * BM * N; 
    B += c_col * BK; 
    C += c_row * BM * K + c_col * BK; 

    const unsigned int a_col = threadIdx.x % BN; 
    const unsigned int a_row = threadIdx.x / BN; 
    const unsigned int stride_a = blockDim.x / BN; 

    const unsigned int b_col = threadIdx.x % BK; 
    const unsigned int b_row = threadIdx.x / BK; 
    const unsigned int stride_b = blockDim.x / BK; 

    float acc[TM * TK] = {0.0};
    float regM[TM] = {0.0};
    float regK[TK] = {0.0};

    for (unsigned int n = 0; n < N; n += BN) {
        for (unsigned int offset = 0; offset < BM; offset += stride_a) {
            As[(offset + a_row) * BN + a_col] = A[(offset + a_row) * N + a_col];
        }
        for (unsigned int offset = 0; offset < BN; offset += stride_b) {
            Bs[(offset + b_row) * BK + b_col] = B[(offset + b_row) * K + b_col]; 
        }
        __syncthreads(); 

        A += BN; 
        B += BN * K; 

        for (unsigned int n_ = 0; n_ < fmin(N-n, BN); n_++) {
            for (unsigned int i = 0; i < TM; i++) {
                regM[i] = (As[(t_row * TM + i) * BN + n_]);
            }
            for (unsigned int i = 0; i < TK; i++) {
                regK[i] = (Bs[n_ * BK + t_col * TK + i]); 
            }
            for (unsigned int i = 0; i < TM; i++) {
                if (c_row * BM + t_row * TM + i < M) {
                    for (unsigned int j = 0; j < TK; j++) {
                        if (c_col * BK + t_col * TK + j < K) {
                            acc[i * TK + j] += regM[i] * regK[j]; 
                        }
                    }
                }
            }
        }
        __syncthreads(); 
    }

    for (unsigned int i = 0; i < TM; i++) {
        if (c_row * BM + t_row * TM + i < M) {
            for (unsigned int j = 0; j < TK; j++) {
                if (c_col * BK + t_col * TK + j < K) {
                    C[(t_row * TM + i) * K + (t_col * TK + j)] = (alpha * acc[i * TK + j] + beta * (C[(t_row * TM + i) * K + (t_col * TK + j)])); 
                }
            }
        }
    }
}

__host__ void solve(const float* A, const float* B, float* C, int M, int K, int N, float alpha,
                      float beta) {
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
                        
    const int BM = 64; 
    const int BN = 8;
    const int BK = 64; 
    const int TM = 8;
    const int TK = 8; 

    dim3 gridDim(CEIL_DIV(K, BK), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BK) / (TM * TK)); 
    matmul_fast<BM, BN, BK, TM, TK><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

int main(int argc, char** argv)
{
    std::random_device rd;  
    std::mt19937 gen(rd()); // Mersenne Twister RNG

    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    const int M = 512, N = 512, K = 512; 

    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t c_bytes = M * N * sizeof(float); 

    std::cout << "Matrix size: " << M * K << " elements\n";
    std::cout << "Memory per matrix: "
              << (a_bytes / (1024.0 * 1024.0)) << " MB\n\n";

    // Host vectors
    float h_a[M * K], h_b[K * N], h_c[M * N];

    // Initialize input data
    // All are same values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; j++) {
            h_a[i * N + j] = dist(gen);
            h_b[i * N + j] = dist(gen);
            h_c[i * N + j] = dist(gen); 
        }
    }

    // Device pointers
    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));

    // Copy inputs to GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    // CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));

    solve(d_a, d_b, d_c, M, N, K, 1.0f, 0.5f); 

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));

    double seconds = milliseconds / 1000.0;
    double bandwidthGBs =
        (3.0 * a_bytes) / (seconds * 1e9); // read A + read B + write C

    std::cout << "Kernel time: " << milliseconds << " ms\n";
    std::cout << "Effective bandwidth: "
              << bandwidthGBs << " GB/s\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}