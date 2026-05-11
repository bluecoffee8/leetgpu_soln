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

    // Host vectors
    std::vector<float> h_a(N);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
    }

    // Device pointers
    float *d_a;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));

    // CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));

    // Copy inputs to GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    double seconds = milliseconds / 1000.0;
    double bandwidthGBs =
        (1.0 * bytes) / (seconds * 1e9); 

    std::cout << "Copy time: " << milliseconds << " ms\n";
    std::cout << "Effective bandwidth: "
              << bandwidthGBs << " GB/s\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));

    return 0; 
}