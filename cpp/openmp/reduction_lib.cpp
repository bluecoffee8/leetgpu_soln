/*
 * OpenMP Reduction Library - Function implementations
 * Compile as: g++-15 -O3 -fopenmp -c -o reduction_lib.o reduction_lib.cpp
 */

#include "reduction_openmp.h"
#include <vector>
#include <omp.h>

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
