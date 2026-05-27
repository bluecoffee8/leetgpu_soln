# OpenMP Sum Reduction Demonstrations

A comprehensive collection of sum reduction functions demonstrating all different OpenMP pragmas and synchronization techniques, with full benchmarking and testing suite.

## Features

**9 Different Reduction Implementations:**
1. **parallel for + reduction** - Most common and efficient (using `#pragma omp parallel for reduction(+:sum)`)
2. **parallel region** - Manual work-sharing with thread-local accumulation
3. **critical section** - Demonstrates `#pragma omp critical` (slower, serializes access)
4. **atomic operation** - Demonstrates `#pragma omp atomic` (more efficient than critical)
5. **taskloop reduction** - OpenMP 4.5+ task-based reduction (good for irregular workloads)
6. **locks** - Manual reduction with explicit locks (`omp_lock_t`)
7. **SIMD reduction** - Vectorization-aware reduction (`#pragma omp parallel for simd`)
8. **schedule(static)** - Static work distribution
9. **schedule(dynamic)** - Dynamic work distribution with load balancing

## Files

- `reduction_openmp.cpp` - Standalone executable with integrated benchmarking
- `reduction_lib.cpp` - Function library (without main)
- `reduction_openmp.h` - Header file for using functions as a library
- `example_usage.cpp` - Example program showing how to link and use the library
- `reduction_openmp.o` - Compiled object file from `reduction_openmp.cpp`
- `reduction_lib.o` - Library object file from `reduction_lib.cpp`

## Compilation

### macOS (with GCC-15 from Homebrew)

**Ensure GCC is installed:**
```bash
brew install gcc
```

**Compile to executable:**
```bash
g++-15 -O3 -fopenmp -o reduction_openmp reduction_openmp.cpp
```

**Or using full path:**
```bash
$(brew --prefix gcc)/bin/g++-15 -O3 -fopenmp -o reduction_openmp reduction_openmp.cpp
```

**Compile to object file:**
```bash
g++-15 -O3 -fopenmp -c -o reduction_openmp.o reduction_openmp.cpp
```

**Compile library:**
```bash
g++-15 -O3 -fopenmp -c -o reduction_lib.o reduction_lib.cpp
```

### Linux (with standard g++)

**Compile to executable:**
```bash
g++ -O3 -fopenmp -o reduction_openmp reduction_openmp.cpp
```

**Compile to object file:**
```bash
g++ -O3 -fopenmp -c -o reduction_openmp.o reduction_openmp.cpp
```

**Compile library:**
```bash
g++ -O3 -fopenmp -c -o reduction_lib.o reduction_lib.cpp
```

## Running

**Basic usage (defaults: 4 threads, 10M elements):**
```bash
./reduction_openmp
```

**With custom parameters:**
```bash
./reduction_openmp 4 1000000        # 4 threads, 1 million elements
./reduction_openmp 8 10000000       # 8 threads, 10 million elements
./reduction_openmp 16 100000000     # 16 threads, 100 million elements
```

### Example Output

```
================================
OpenMP Reduction Demonstrations
================================
Threads: 4
Array Size: 10000000 elements
Array Memory: 80 MB

Running benchmarks...

Method                                Time (ms)      Result  Correct
----------------------------------------------------------------------
1. parallel for + reduction              1.6014    5.00e+11       ✓
2. parallel region (manual)              5.9080    5.00e+11       ✓
3. critical section                      1.7208    5.00e+11       ✓
4. atomic operation                      1.6194    5.00e+11       ✓
5. taskloop reduction                    1.7076    5.00e+11       ✓
6. locks                                 1.5926    5.00e+11       ✓
7. SIMD reduction                        0.8062    5.00e+11       ✓
8. schedule(static)                      1.7784    5.00e+11       ✓
9. schedule(dynamic)                     1.8712    5.00e+11       ✓

Expected Sum: 4.9999995000e+11

================================
Performance Ranking (fastest first):
================================
1. 7. SIMD reduction (0.8062 ms)
2. 6. locks (1.5926 ms)
3. 1. parallel for + reduction (1.6014 ms)
4. 4. atomic operation (1.6194 ms)
...
```

## Using as a Library

### Step 1: Compile the library
```bash
g++-15 -O3 -fopenmp -c -o reduction_lib.o reduction_lib.cpp
```

### Step 2: Use the header in your code
```cpp
#include "reduction_openmp.h"

int main() {
    std::vector<double> arr = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    double result = reduction_parallel_for(arr);
    std::cout << "Sum: " << result << std::endl;
    
    return 0;
}
```

### Step 3: Compile and link
```bash
g++-15 -O3 -fopenmp my_program.cpp reduction_lib.o -o my_program
```

### Example: Using example_usage.cpp
```bash
# Compile library
g++-15 -O3 -fopenmp -c -o reduction_lib.o reduction_lib.cpp

# Compile and link example
g++-15 -O3 -fopenmp example_usage.cpp reduction_lib.o -o example_usage

# Run with custom threads
./example_usage 8
```

## Inspecting Object Files

**View file format:**
```bash
file reduction_openmp.o
file reduction_lib.o
```

**List all symbols:**
```bash
nm reduction_openmp.o
nm reduction_lib.o
```

**List reduction function symbols only:**
```bash
nm reduction_openmp.o | grep reduction
nm reduction_lib.o | grep reduction
```

**Detailed symbol information:**
```bash
objdump -t reduction_openmp.o
objdump -d reduction_openmp.o     # Disassembly
```

**Check object file size:**
```bash
ls -lh reduction_openmp.o reduction_lib.o
```

## Key Observations

### Performance Rankings
From typical benchmark results (varies by system/data):

1. **SIMD reduction** - Fastest due to auto-vectorization
2. **parallel for + reduction** - Simplest and most efficient
3. **atomic operation** - More efficient than critical
4. **critical section** - Serializes access, slower
5. **parallel region** - Can be slower due to memory overhead
6. **dynamic scheduling** - Overhead from dynamic assignment

### When to Use Each

| Method | Use Case |
|--------|----------|
| parallel for + reduction | Default choice - simple, efficient, readable |
| SIMD reduction | When you want explicit vectorization hints |
| atomic | Few threads, simple operations |
| critical | Legacy code, complex reduction operations |
| taskloop | Irregular workloads, task-based parallelism |
| parallel region | Fine-grained control over thread work division |
| locks | Educational purposes, explicit synchronization |

## OpenMP Pragmas Demonstrated

- `#pragma omp parallel` - Create a parallel region
- `#pragma omp for` - Distribute loop iterations
- `#pragma omp parallel for` - Combined parallel + for
- `#pragma omp reduction(+:sum)` - Reduction clause
- `#pragma omp critical` - Critical section
- `#pragma omp atomic` - Atomic operation
- `#pragma omp taskloop` - Task-based loop
- `#pragma omp simd` - SIMD vectorization hint
- `#pragma omp collapse(2)` - Collapse nested loops
- `#pragma omp schedule(static)` - Static scheduling
- `#pragma omp schedule(dynamic, chunk)` - Dynamic scheduling

## Compilation Flags Explained

- `-O3` - Maximum optimization level
- `-fopenmp` - Enable OpenMP support
- `-c` - Compile only, don't link (generates .o file)
- `-o filename` - Output filename

## Troubleshooting

**"omp.h not found"** - Use GCC instead of Clang on macOS
```bash
g++-15 -O3 -fopenmp ...
# Not: clang++ -fopenmp ...
```

**Undefined reference to OpenMP functions** - Add `-fopenmp` flag
```bash
g++-15 -O3 -fopenmp mycode.cpp reduction_lib.o -o myprogram
```

**Wrong architecture error** - Ensure compiler and libraries match
```bash
g++-15 --version                    # Check GCC version
file reduction_lib.o                # Check object file arch
```

## Performance Tips

1. Use `schedule(static)` for balanced workloads
2. Use `schedule(dynamic)` for unbalanced workloads
3. Use SIMD hints when you know operations are vectorizable
4. Keep thread-local accumulation to minimize synchronization
5. Avoid critical sections when atomic operations suffice
6. Test with different numbers of threads to find optimal value

## References

- [OpenMP Official Site](https://www.openmp.org/)
- [GCC OpenMP Documentation](https://gcc.gnu.org/projects/gomp/)
- [OpenMP 5.1 Specification](https://www.openmp.org/spec-html/5.1/openmp.html)
