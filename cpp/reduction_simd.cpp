/*
COMPILATION:
  Compile to object file (M1 macOS with ARM NEON):
    clang++ -O3 -march=armv8-a reduction_simd.cpp -c -o reduction_simd.o

  Or with g++/clang++ directly:
    g++ -O3 -march=armv8-a reduction_simd.cpp -c -o reduction_simd.o

  Compile a test program:
    clang++ -O3 -march=armv8-a reduction_simd.cpp -DTEST_SIMD -o test_reduction.o

  Run test:
    ./test_reduction.o

NOTE: -march=armv8-a enables ARM NEON intrinsics on Apple Silicon (M1/M2/M3).
      -O3 enables aggressive optimizations for maximum performance.
*/

#include <arm_neon.h>
#include <cstring>

// SIMD sum reduction for float32 using ARM NEON (M1)
float simd_sum_f32(const float* data, size_t n) {
    if (n == 0) return 0.0f;

    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Process 4 elements per iteration
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        acc = vaddq_f32(acc, v);
    }

    // Horizontal sum: add all 4 elements in the accumulator
    float sum = vaddvq_f32(acc);

    // Handle remaining elements (0-3)
    for (; i < n; ++i) {
        sum += data[i];
    }

    return sum;
}

// SIMD sum reduction for int32 using ARM NEON (M1)
int32_t simd_sum_i32(const int32_t* data, size_t n) {
    if (n == 0) return 0;

    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;

    // Process 4 elements per iteration
    for (; i + 4 <= n; i += 4) {
        int32x4_t v = vld1q_s32(&data[i]);
        acc = vaddq_s32(acc, v);
    }

    // Horizontal sum: add all 4 elements in the accumulator
    int32_t sum = vaddvq_s32(acc);

    // Handle remaining elements (0-3)
    for (; i < n; ++i) {
        sum += data[i];
    }

    return sum;
}

// SIMD sum reduction for int64 using ARM NEON (M1)
int64_t simd_sum_i64(const int64_t* data, size_t n) {
    if (n == 0) return 0;

    int64x2_t acc = vdupq_n_s64(0);
    size_t i = 0;

    // Process 2 elements per iteration (128-bit register holds 2x int64)
    for (; i + 2 <= n; i += 2) {
        int64x2_t v = vld1q_s64(&data[i]);
        acc = vaddq_s64(acc, v);
    }

    // Horizontal sum: add both elements in the accumulator
    int64_t sum = vaddvq_s64(acc);

    // Handle remaining element
    if (i < n) {
        sum += data[i];
    }

    return sum;
}

// SIMD sum reduction for float64 using ARM NEON (M1)
double simd_sum_f64(const double* data, size_t n) {
    if (n == 0) return 0.0;

    float64x2_t acc = vdupq_n_f64(0.0);
    size_t i = 0;

    // Process 2 elements per iteration (128-bit register holds 2x double)
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(&data[i]);
        acc = vaddq_f64(acc, v);
    }

    // Horizontal sum: add both elements in the accumulator
    double sum = vaddvq_f64(acc);

    // Handle remaining element
    if (i < n) {
        sum += data[i];
    }

    return sum;
}

#ifdef TEST_SIMD
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "Testing SIMD sum reduction functions...\n\n";

    // Test float32
    {
        float data[] = {1.5f, 2.5f, 3.0f, 4.0f, 5.5f};
        float result = simd_sum_f32(data, 5);
        float expected = 16.5f;
        assert(std::abs(result - expected) < 1e-5f);
        std::cout << "✓ simd_sum_f32: " << result << " (expected " << expected << ")\n";
    }

    // Test int32
    {
        int32_t data[] = {1, 2, 3, 4, 5};
        int32_t result = simd_sum_i32(data, 5);
        int32_t expected = 15;
        assert(result == expected);
        std::cout << "✓ simd_sum_i32: " << result << " (expected " << expected << ")\n";
    }

    // Test int64
    {
        int64_t data[] = {100000000LL, 200000000LL, 300000000LL};
        int64_t result = simd_sum_i64(data, 3);
        int64_t expected = 600000000LL;
        assert(result == expected);
        std::cout << "✓ simd_sum_i64: " << result << " (expected " << expected << ")\n";
    }

    // Test float64
    {
        double data[] = {1.5, 2.5, 3.0, 4.0};
        double result = simd_sum_f64(data, 4);
        double expected = 11.0;
        assert(std::abs(result - expected) < 1e-10);
        std::cout << "✓ simd_sum_f64: " << result << " (expected " << expected << ")\n";
    }

    // Test edge cases (single element)
    {
        float data[] = {42.0f};
        float result = simd_sum_f32(data, 1);
        assert(std::abs(result - 42.0f) < 1e-5f);
        std::cout << "✓ simd_sum_f32 (single element): " << result << "\n";
    }

    // Test edge case (empty)
    {
        float result = simd_sum_f32(nullptr, 0);
        assert(result == 0.0f);
        std::cout << "✓ simd_sum_f32 (empty): " << result << "\n";
    }

    std::cout << "\nAll tests passed!\n";
    return 0;
}
#endif
