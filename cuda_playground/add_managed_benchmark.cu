// vector_add_cuda_prefetch.cu
//
// Vector addition benchmark using CUDA Unified Memory
// with explicit prefetching for accurate timing.
//
// Compile:
//   nvcc -O3 vector_add_cuda_prefetch.cu -o vector_add
//
// Run:
//   ./vector_add
//   ./vector_add 100000000

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

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
                          size_t n)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv)
{
    // Default: 10 million elements
    size_t N = 10'000'000;

    // if (argc > 1) {
    //     N = std::strtoull(argv[1], nullptr, 10);
    // }

    size_t bytes = N * sizeof(float);

    std::cout << "Vector size: " << N << " elements\n";
    std::cout << "Memory per vector: "
              << (bytes / (1024.0 * 1024.0)) << " MB\n\n";

    // Unified Memory allocations
    float* a;
    float* b;
    float* c;

    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    // Initialize on CPU
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 2.0f;
    }

    // Get active GPU device
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    // Prefetch memory to GPU
    CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, device));
    CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, device));
    CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, device));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel configuration
    constexpr int threadsPerBlock = 256;
    int blocks =
        static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);

    // CUDA timing events
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up launch
    vectorAdd<<<blocks, threadsPerBlock>>>(a, b, c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));

    vectorAdd<<<blocks, threadsPerBlock>>>(a, b, c, N);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Prefetch results back to CPU before verification
    CUDA_CHECK(cudaMemPrefetchAsync(
        c,
        bytes,
        cudaCpuDeviceId));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify correctness
    bool ok = true;

    for (size_t i = 0; i < N; ++i) {
        float expected = a[i] + b[i];

        if (std::fabs(c[i] - expected) > 1e-5f) {
            ok = false;

            std::cerr << "Mismatch at " << i
                      << ": got " << c[i]
                      << ", expected " << expected
                      << "\n";

            break;
        }
    }

    double seconds = milliseconds / 1000.0;

    // Memory traffic:
    // read A + read B + write C
    double bandwidthGBs =
        (3.0 * static_cast<double>(bytes)) /
        (seconds * 1e9);

    std::cout << "Kernel time: "
              << milliseconds
              << " ms\n";

    std::cout << "Effective bandwidth: "
              << bandwidthGBs
              << " GB/s\n";

    std::cout << "Verification: "
              << (ok ? "PASSED" : "FAILED")
              << "\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    return ok ? 0 : 1;
}