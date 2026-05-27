#ifndef REDUCTION_OPENMP_H
#define REDUCTION_OPENMP_H

#include <vector>

// Sum reduction functions - all demonstrate different OpenMP pragmas
double reduction_parallel_for(const std::vector<double>& arr);
double reduction_parallel_region(const std::vector<double>& arr);
double reduction_critical(const std::vector<double>& arr);
double reduction_atomic(const std::vector<double>& arr);
double reduction_taskloop(const std::vector<double>& arr);
double reduction_with_locks(const std::vector<double>& arr);
double reduction_collapse(size_t n);
double reduction_simd(const std::vector<double>& arr);
double reduction_schedule_static(const std::vector<double>& arr);
double reduction_schedule_dynamic(const std::vector<double>& arr);

#endif // REDUCTION_OPENMP_H
