/*
 * OpenMP Sum Reduction Demonstrations
 *
 * Compilation Instructions:
 * =======================
 *
 * macOS (with GCC-15 from Homebrew):
 * -----------------------------------
 * g++-15 -O3 -fopenmp -o reduction_openmp reduction_openmp.cpp
 * # Or use the full path if not in PATH:
 * $(brew --prefix gcc)/bin/g++-15 -O3 -fopenmp -o reduction_openmp reduction_openmp.cpp
 *
 * Linux (with g++):
 * -----------------
 * g++ -O3 -fopenmp -o reduction_openmp reduction_openmp.cpp
 *
 * Running:
 * ========
 * ./reduction_openmp [num_threads] [array_size]
 * Examples:
 *   ./reduction_openmp 4 1000000      # 4 threads, 1M elements
 *   ./reduction_openmp 8 10000000     # 8 threads, 10M elements
 *   ./reduction_openmp                # Uses default: 4 threads, 10M elements
 *
 * To compile to object file (.o):
 * ================================
 * macOS:
 *   g++-15 -O3 -fopenmp -c -o reduction_openmp.o reduction_openmp.cpp
 * Linux:
 *   g++ -O3 -fopenmp -c -o reduction_openmp.o reduction_openmp.cpp
 *
 * To link the object file with another program:
 * ==============================================
 * macOS:
 *   g++-15 -O3 -fopenmp main.cpp reduction_openmp.o -o my_program
 * Linux:
 *   g++ -O3 -fopenmp main.cpp reduction_openmp.o -o my_program
 *
 * Viewing the compiled object file info:
 * ======================================
 * file reduction_openmp.o              # Shows object file format
 * nm reduction_openmp.o                # Lists symbols in object file
 * objdump -t reduction_openmp.o        # Detailed symbol information
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cmath>

// ============================================================================
// 1. PARALLEL FOR WITH REDUCTION CLAUSE (Most common and efficient)
// ============================================================================
double reduction_parallel_for(const std::vector<double>& arr) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// ============================================================================
// 2. PARALLEL REGION WITH MANUAL WORK-SHARING (Demonstrates parallel region)
// ============================================================================
double reduction_parallel_region(const std::vector<double>& arr) {
    double sum = 0.0;
    int num_threads = omp_get_max_threads();
    std::vector<double> thread_sums(num_threads, 0.0);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();

        // Each thread sums its portion
#pragma omp for nowait
        for (size_t i = 0; i < arr.size(); ++i) {
            thread_sums[thread_id] += arr[i];
        }
    }

    // Combine results (single-threaded at this point)
    for (double val : thread_sums) {
        sum += val;
    }
    return sum;
}

// ============================================================================
// 3. CRITICAL SECTION (Safe but slower - serializes access)
// ============================================================================
double reduction_critical(const std::vector<double>& arr) {
    double sum = 0.0;

#pragma omp parallel
    {
        double local_sum = 0.0;
        // Each thread does local accumulation
#pragma omp for
        for (size_t i = 0; i < arr.size(); ++i) {
            local_sum += arr[i];
        }

        // Only one thread enters at a time
#pragma omp critical
        {
            sum += local_sum;
        }
    }
    return sum;
}

// ============================================================================
// 4. ATOMIC OPERATIONS (More efficient than critical for simple ops)
// ============================================================================
double reduction_atomic(const std::vector<double>& arr) {
    double sum = 0.0;

#pragma omp parallel
    {
        double local_sum = 0.0;
#pragma omp for
        for (size_t i = 0; i < arr.size(); ++i) {
            local_sum += arr[i];
        }

#pragma omp atomic
        sum += local_sum;
    }
    return sum;
}

// ============================================================================
// 5. TASKLOOP REDUCTION (OpenMP 4.5+, good for irregular workloads)
// ============================================================================
double reduction_taskloop(const std::vector<double>& arr) {
    double sum = 0.0;

#pragma omp parallel
#pragma omp single
#pragma omp taskloop reduction(+:sum)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }

    return sum;
}

// ============================================================================
// 6. LOCKS (Manual reduction with locks - demonstrates synchronization)
// ============================================================================
double reduction_with_locks(const std::vector<double>& arr) {
    double sum = 0.0;
    omp_lock_t lock;
    omp_init_lock(&lock);

#pragma omp parallel
    {
        double local_sum = 0.0;
#pragma omp for
        for (size_t i = 0; i < arr.size(); ++i) {
            local_sum += arr[i];
        }

        omp_set_lock(&lock);
        {
            sum += local_sum;
        }
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);
    return sum;
}

// ============================================================================
// 7. COLLAPSE CLAUSE (For nested loops)
// ============================================================================
double reduction_collapse(size_t n) {
    double sum = 0.0;
    std::vector<double> matrix(n * n);

    // Initialize
    for (size_t i = 0; i < n * n; ++i) {
        matrix[i] = i * 0.1;
    }

    // Reduction with collapsed loops
#pragma omp parallel for collapse(2) reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            sum += matrix[i * n + j];
        }
    }
    return sum;
}

// ============================================================================
// 8. SIMD REDUCTION (OpenMP 4.0+, vectorization-aware)
// ============================================================================
double reduction_simd(const std::vector<double>& arr) {
    double sum = 0.0;
#pragma omp parallel for simd reduction(+:sum)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// ============================================================================
// 9. COMBINE REDUCTION WITH SCHEDULE
// ============================================================================
double reduction_schedule_static(const std::vector<double>& arr) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

double reduction_schedule_dynamic(const std::vector<double>& arr) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum) schedule(dynamic, 1000)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

double get_expected_sum(size_t size) {
    return size * (size - 1) * 0.5 / (double)100;
}

struct BenchmarkResult {
    std::string name;
    double time_ms;
    double result;
    bool correct;
};

template<typename Func>
BenchmarkResult benchmark(const std::string& name, Func func,
                          const std::vector<double>& arr,
                          double expected_sum,
                          int iterations = 5) {
    // Warm up
    func();

    auto start = std::chrono::high_resolution_clock::now();
    double result = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        result = func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = elapsed / iterations;

    bool correct = std::abs(result - expected_sum) < 1e-6 * std::abs(expected_sum);

    return {name, avg_time, result, correct};
}

// ============================================================================
// MAIN BENCHMARKING SUITE
// ============================================================================

int main(int argc, char* argv[]) {
    int num_threads = (argc > 1) ? std::atoi(argv[1]) : 4;
    size_t array_size = (argc > 2) ? std::atol(argv[2]) : 10000000;

    omp_set_num_threads(num_threads);

    std::cout << "================================\n";
    std::cout << "OpenMP Reduction Demonstrations\n";
    std::cout << "================================\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Array Size: " << array_size << " elements\n";
    std::cout << "Array Memory: " << (array_size * sizeof(double) / 1e6) << " MB\n\n";

    // Initialize array with known values
    std::vector<double> arr(array_size);
    for (size_t i = 0; i < array_size; ++i) {
        arr[i] = i * 0.01;  // 0, 0.01, 0.02, ...
    }

    double expected_sum = 0.0;
    for (double v : arr) {
        expected_sum += v;
    }

    // Run benchmarks
    std::vector<BenchmarkResult> results;

    std::cout << "Running benchmarks...\n\n";

    results.push_back(benchmark("1. parallel for + reduction",
        [&arr]() { return reduction_parallel_for(arr); },
        arr, expected_sum));

    results.push_back(benchmark("2. parallel region (manual)",
        [&arr]() { return reduction_parallel_region(arr); },
        arr, expected_sum));

    results.push_back(benchmark("3. critical section",
        [&arr]() { return reduction_critical(arr); },
        arr, expected_sum));

    results.push_back(benchmark("4. atomic operation",
        [&arr]() { return reduction_atomic(arr); },
        arr, expected_sum));

    results.push_back(benchmark("5. taskloop reduction",
        [&arr]() { return reduction_taskloop(arr); },
        arr, expected_sum));

    results.push_back(benchmark("6. locks",
        [&arr]() { return reduction_with_locks(arr); },
        arr, expected_sum));

    results.push_back(benchmark("7. SIMD reduction",
        [&arr]() { return reduction_simd(arr); },
        arr, expected_sum));

    results.push_back(benchmark("8. schedule(static)",
        [&arr]() { return reduction_schedule_static(arr); },
        arr, expected_sum));

    results.push_back(benchmark("9. schedule(dynamic)",
        [&arr]() { return reduction_schedule_dynamic(arr); },
        arr, expected_sum));

    // Display results
    std::cout << std::left << std::setw(35) << "Method"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Result"
              << std::setw(10) << "Correct\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(35) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << r.time_ms
                  << std::setw(12) << std::scientific << std::setprecision(2) << r.result
                  << std::setw(10) << (r.correct ? "✓" : "✗") << "\n";
    }

    std::cout << "\nExpected Sum: " << std::scientific << std::setprecision(10) << expected_sum << "\n";

    // Performance ranking
    std::cout << "\n================================\n";
    std::cout << "Performance Ranking (fastest first):\n";
    std::cout << "================================\n";

    std::vector<size_t> indices(results.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&results](size_t a, size_t b) { return results[a].time_ms < results[b].time_ms; });

    for (size_t rank = 0; rank < indices.size(); ++rank) {
        const auto& r = results[indices[rank]];
        std::cout << (rank + 1) << ". " << r.name << " ("
                  << std::fixed << std::setprecision(4) << r.time_ms << " ms)\n";
    }

    return 0;
}
