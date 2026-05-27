# OpenMP Reduction - Quick Start Guide

## 30-Second Setup

```bash
# 1. Compile (macOS with GCC)
g++-15 -O3 -fopenmp -o reduction_openmp.o reduction_openmp.cpp

# 2. Run with benchmarks
./reduction_openmp.o 4 1000000
```

## One-Liner Commands

**Compile & run:**
```bash
g++-15 -O3 -fopenmp -o reduction_openmp.o reduction_openmp.cpp && ./reduction_openmp
```

**Compile to object file:**
```bash
g++-15 -O3 -fopenmp -c reduction_openmp.cpp
```

**Compile library + example:**
```bash
g++-15 -O3 -fopenmp -c reduction_lib.cpp && \
g++-15 -O3 -fopenmp example_usage.cpp reduction_lib.o -o example_usage && \
./example_usage 4
```

## What You Get

| File | Purpose |
|------|---------|
| `reduction_openmp.cpp` | Standalone program with 9 reduction implementations + benchmarks |
| `reduction_lib.cpp` | Library version (functions only, no main) |
| `reduction_openmp.h` | Header file for using functions |
| `example_usage.cpp` | Example showing how to use as a library |
| `README_REDUCTION.md` | Full documentation |

## Running Tests

```bash
# Default: 4 threads, 10 million elements
./reduction_openmp

# Custom: 8 threads, 100 million elements
./reduction_openmp 8 100000000

# Run example library usage
./example_usage 4
```

## 9 Reduction Methods Included

1. **parallel for + reduction** ⭐ Usually fastest
2. parallel region (manual)
3. critical section
4. atomic operation
5. taskloop reduction
6. locks
7. **SIMD reduction** ⭐ Often fastest on modern CPUs
8. schedule(static)
9. schedule(dynamic)

## Using as a Library

**header:** `reduction_openmp.h`

```cpp
#include "reduction_openmp.h"
#include <vector>

std::vector<double> arr = {1,2,3,4,5};
double sum = reduction_parallel_for(arr);  // Use any function
```

**Compile:**
```bash
g++-15 -O3 -fopenmp -c reduction_lib.cpp  # Create library
g++-15 -O3 -fopenmp mycode.cpp reduction_lib.o -o myapp  # Link
```

## Benchmark Output Format

```
Method                                Time (ms)      Result  Correct
----------------------------------------------------------------------
1. parallel for + reduction              0.3146    5.00e+09       ✓
...

Performance Ranking (fastest first):
1. 7. SIMD reduction (0.1712 ms)
2. 1. parallel for + reduction (0.3146 ms)
...
```

## Key Pragmas Demonstrated

```cpp
// 1. Most common - automatic reduction
#pragma omp parallel for reduction(+:sum)

// 2. Manual control - each thread accumulates locally  
#pragma omp parallel
#pragma omp for
    sum += arr[i];

// 3. Explicit synchronization - one thread at a time
#pragma omp critical
    sum += local_sum;

// 4. Atomic - safe without locking
#pragma omp atomic
    sum += local_sum;

// 5. SIMD-aware - hints to vectorize
#pragma omp parallel for simd reduction(+:sum)

// 6. Scheduling variants
#pragma omp parallel for reduction(+:sum) schedule(static)
#pragma omp parallel for reduction(+:sum) schedule(dynamic, 1000)
```

## Expected Performance

Typical results on modern CPUs with 8 threads:

| Method | Time | Notes |
|--------|------|-------|
| SIMD | 0.8ms | Auto-vectorized |
| parallel for | 1.6ms | Best balance |
| atomic | 1.6ms | Simple & efficient |
| critical | 1.7ms | Serializes |
| dynamic | 1.9ms | Scheduling overhead |

(Times vary by system and array size)

## Compilation Issues?

**Error: "omp.h not found"**
- Use GCC: `g++-15 -O3 -fopenmp ...`
- NOT Clang: `clang++ -fopenmp ...` (doesn't work on macOS)

**Error: "undefined reference to omp_"**
- Add `-fopenmp` flag to compiler AND linker

**Wrong architecture**
- Check: `file reduction_lib.o`
- Use: `g++-15` (not system `g++`)

## Files Generated After Compilation

```
reduction_openmp       # Executable (31KB)
reduction_openmp.o     # Object file with main (26KB)
reduction_lib.o        # Library object file (9.5KB)
example_usage          # Example executable (21KB)
```

## Test Everything

```bash
./test_all.sh  # Runs comprehensive test suite
```

---

**For full documentation:** See `README_REDUCTION.md`
