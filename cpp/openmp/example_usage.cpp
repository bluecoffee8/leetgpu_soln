/*
 * Example: Using reduction_openmp.o in another program
 *
 * Compilation:
 * ============
 * macOS:
 *   g++-15 -O3 -fopenmp example_usage.cpp reduction_openmp.o -o example_usage.o
 *
 * Linux:
 *   g++ -O3 -fopenmp example_usage.cpp reduction_openmp.o -o example_usage.o
 *
 * Running:
 * ========
 * ./example_usage.o [num_threads]
 */

#include <iostream>
#include <vector>
#include <omp.h>
#include "reduction_openmp.h"

int main(int argc, char* argv[]) {
    int num_threads = (argc > 1) ? std::atoi(argv[1]) : 4;
    omp_set_num_threads(num_threads);

    size_t size = 5000000;
    std::vector<double> arr(size);

    // Initialize test data
    for (size_t i = 0; i < size; ++i) {
        arr[i] = i * 0.001;
    }

    // Expected sum
    double expected = 0.0;
    for (double v : arr) {
        expected += v;
    }

    std::cout << "Testing reduction functions from reduction_openmp.o\n";
    std::cout << "Array size: " << size << " elements\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Expected sum: " << expected << "\n\n";

    // Test each reduction function
    std::cout << "Results:\n";
    std::cout << "  parallel_for:     " << reduction_parallel_for(arr) << "\n";
    std::cout << "  parallel_region:  " << reduction_parallel_region(arr) << "\n";
    std::cout << "  critical:         " << reduction_critical(arr) << "\n";
    std::cout << "  atomic:           " << reduction_atomic(arr) << "\n";
    std::cout << "  taskloop:         " << reduction_taskloop(arr) << "\n";
    std::cout << "  with_locks:       " << reduction_with_locks(arr) << "\n";
    std::cout << "  simd:             " << reduction_simd(arr) << "\n";
    std::cout << "  schedule_static:  " << reduction_schedule_static(arr) << "\n";
    std::cout << "  schedule_dynamic: " << reduction_schedule_dynamic(arr) << "\n";

    return 0;
}
