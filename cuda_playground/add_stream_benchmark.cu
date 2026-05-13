// vector_add_cuda.cu
//
// Simple CUDA vector addition benchmark.
// Builds two equally sized vectors, adds them on the GPU,
// and measures kernel execution time.
//
// Compile:
//   nvcc -O2 vector_add_cuda.cu -o vector_add
//
// Run:
//   ./vector_add
//   ./vector_add 100000000   // optional vector size

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

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -> " << cudaGetErrorString(err) << std::endl;   \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void vectorAdd(const float* a,
                          const float* b,
                          float* c,
                          int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv)
{
    // Default: 10 million elements
    int N = 10'000'000;

    // if (argc > 1) {
    //     N = std::atoi(argv[1]);
    // }

    size_t bytes = N * sizeof(float);

    std::cout << "Vector size: " << N << " elements\n";
    std::cout << "Memory per vector: "
              << (bytes / (1024.0 * 1024.0)) << " MB\n\n";

    // Host pointers for pinned memory
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost((void**)&h_a, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_b, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_c, bytes));

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 2.0f;
    }

    // Device pointers
    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Stream setup
    const int M = 10; // Number of chunks/streams
    int chunkSize = N / M;
    cudaStream_t streams[M];
    for (int i = 0; i < M; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Kernel launch config
    int threadsPerBlock = 256;

    // CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark: Timing the entire overlapped pipeline (H2D -> Kernel -> D2H)
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < M; ++i) {
        int offset = i * chunkSize;
        int currentChunkSize = (i == M - 1) ? (N - offset) : chunkSize;
        int blocks = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;

        // Asynchronous H2D copy
        CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset, currentChunkSize * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset, currentChunkSize * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[i]));

        // Kernel execution in stream
        vectorAdd<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset, currentChunkSize);

        // Asynchronous D2H copy
        CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c + offset, currentChunkSize * sizeof(float), 
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // Synchronize all streams
    for (int i = 0; i < M; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Verify correctness
    bool ok = true;

    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];

        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            ok = false;
            std::cerr << "Mismatch at " << i
                      << ": got " << h_c[i]
                      << ", expected " << expected << "\n";
            break;
        }
    }

    double seconds = milliseconds / 1000.0;
    // Overall throughput considering the full pipeline
    double effectiveBandwidth = (3.0 * bytes) / (seconds * 1e9); 

    std::cout << "Total pipeline time (Overlap): " << milliseconds << " ms\n";
    std::cout << "Effective pipeline bandwidth: " << effectiveBandwidth << " GB/s\n";
    std::cout << "Verification: "
              << (ok ? "PASSED" : "FAILED") << "\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 0; i < M; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    return ok ? 0 : 1;
}