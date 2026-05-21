/*
 * CPU Matrix Multiplication — OpenMP Pragma Ablation Study
 * Base kernel: matmul_tiled_k (IKJ loop order + K-dimension tiling)
 *
 * ── Compilation ──────────────────────────────────────────────────────────────
 *
 * Linux / GCC:
 *   g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 \
 *       -o matmul_openmp.o matmul_openmp.cpp
 *
 * macOS / Clang (requires libomp: brew install libomp):
 *   clang++ -O3 -march=native -ffast-math -std=c++17 \
 *       -Xpreprocessor -fopenmp \
 *       -I$(brew --prefix libomp)/include \
 *       -L$(brew --prefix libomp)/lib -lomp \
 *       -o matmul_openmp.o matmul_openmp.cpp
 *
 * ── Ablations ────────────────────────────────────────────────────────────────
 *
 *   A. Baseline             — no OpenMP
 *   B. parallel for         — parallelise i; default (static) scheduling
 *   C. parallel for+simd    — separate #pragma omp simd on inner j loop
 *   D. parallel for simd    — combined directive on i (simd applies to i's chunk)
 *   E. collapse(2)          — collapse i × (n/TILE_K) into one parallel space
 *   F. schedule(static)     — explicit static schedule
 *   G. schedule(dynamic,1)  — one-row-at-a-time dynamic dispatch
 *   H. schedule(guided)     — decreasing chunk sizes
 *   I. simd only            — no threading; only SIMD hint on j loop
 *   J. single + task        — master spawns one task per row; pool steals them
 *   K. for nowait+barrier   — explicit barrier replaces implicit one after for
 *
 * ── Notes ────────────────────────────────────────────────────────────────────
 *
 *   TILE_K = 16 keeps ~(16+2)×1024×4 ≈ 74 KB in L1D per thread.
 *   Matrix sizes 256, 512, 1024 are all divisible by TILE_K.
 *   With -O3 -march=native the compiler already auto-vectorises the j loop;
 *   #pragma omp simd is an explicit contract that suppresses dependency
 *   analysis and may expose wider SIMD widths or prevent loop splitting.
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

static const int BENCH_RUNS = 3;
static const int TILE_K     = 16;

// ─── Kernels ──────────────────────────────────────────────────────────────────

// A. Baseline — matmul_tiled_k verbatim, zero OpenMP annotations.
void matmul_baseline(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// B. #pragma omp parallel for — distribute i rows across threads.
//    No schedule clause: implementation-defined (almost always static).
void matmul_parallel_for(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// C. #pragma omp parallel for + #pragma omp simd (separate directives).
//    Outer i is distributed across threads; inner j gets an explicit SIMD hint.
//    The simd pragma tells the compiler to vectorise j even if it cannot prove
//    absence of aliasing — here we know A/B/C don't alias, so it confirms that.
void matmul_parallel_plus_simd(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                #pragma omp simd
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// D. #pragma omp parallel for simd (combined directive on the i loop).
//    Threads get chunks of i; within each chunk the iterations are further
//    packed into SIMD lanes.  Because the i body is not a simple FMA the
//    compiler may downgrade the simd part — compare against C to see the delta.
void matmul_parallel_for_simd(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for simd
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// E. collapse(2) — collapse i × (n/TILE_K) tile iterations into one flat
//    range.  For n=1024, TILE_K=16: 1024 × 64 = 65536 items distributed
//    across threads.  Finer granularity can improve load balance at the cost
//    of more scheduling overhead per item.
void matmul_collapse(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// F. schedule(static) — each thread receives a contiguous block of n/T rows
//    assigned at compile-time.  Zero runtime overhead; optimal when all rows
//    take equal time (which they do here).
void matmul_schedule_static(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// G. schedule(dynamic, 1) — rows handed out one-at-a-time as threads become
//    free.  Maximises load balance but adds per-row scheduler overhead
//    (~100 ns/row).  Worth it when work per row varies; costly when uniform.
void matmul_schedule_dynamic(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// H. schedule(guided) — starts with large chunks, shrinks them as the loop
//    tail is reached.  Amortises scheduling cost while retaining tail balance.
void matmul_schedule_guided(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// I. #pragma omp simd only — no thread parallelism at all.
//    Isolates the vectorisation effect: confirms the j loop is SIMD-vectorised
//    and lets us subtract the threading contribution from kernel C.
void matmul_simd_only(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                #pragma omp simd
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

// J. #pragma omp parallel + single + task — one thread inside the parallel
//    region runs the single block and creates one task per row; the other
//    threads steal and execute tasks immediately.  task + firstprivate(i)
//    captures the loop variable by value so each task operates on its own row.
void matmul_task(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel
    {
        #pragma omp single
        for (int i = 0; i < n; i++) {
            #pragma omp task firstprivate(i)
            {
                for (int kt = 0; kt < n; kt += TILE_K)
                    for (int k = kt; k < kt + TILE_K; k++) {
                        float a_ik = A[i*n + k];
                        for (int j = 0; j < n; j++)
                            C[i*n + j] += a_ik * B[k*n + j];
                    }
            }
        }
        // Implicit barrier at the end of the parallel region waits for all tasks.
    }
}

// K. #pragma omp for nowait + explicit #pragma omp barrier.
//    nowait removes the implicit barrier that normally follows a worksharing
//    for loop, allowing threads to proceed to other independent work.  Here we
//    add an explicit barrier for correctness, showing the idiom at no net cost.
//    In real code the gain comes from interleaving this loop with independent
//    work before the barrier.
void matmul_nowait(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < n; i++)
            for (int kt = 0; kt < n; kt += TILE_K)
                for (int k = kt; k < kt + TILE_K; k++) {
                    float a_ik = A[i*n + k];
                    for (int j = 0; j < n; j++)
                        C[i*n + j] += a_ik * B[k*n + j];
                }
        // Explicit barrier replaces the removed implicit one.
        #pragma omp barrier
    }
}

// ─── Benchmark harness ────────────────────────────────────────────────────────

void fill_random(float* M, int size) {
    for (int i = 0; i < size; i++)
        M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// Relative + absolute tolerance check: |ref - out| <= tol*(|ref| + 1)
bool check_close(const float* ref, const float* out, int size, float tol = 1e-3f) {
    for (int i = 0; i < size; i++)
        if (std::fabs(ref[i] - out[i]) > tol * (std::fabs(ref[i]) + 1.0f))
            return false;
    return true;
}

// 2*N^3 FLOPs, reported in GFLOPS.
double to_gflops(int n, double ms) {
    return 2.0 * n * n * n / (ms * 1e6);
}

struct Result { std::string name; double ms; double gflops; bool correct; };

Result benchmark(
    const std::string& name,
    std::function<void(const float*, const float*, float*, int)> kernel,
    const float* A, const float* B, const float* ref, float* C, int n)
{
    // Warmup: bring matrices into cache and let branch predictors settle.
    std::fill(C, C + n*n, 0.0f);
    kernel(A, B, C, n);

    double total_ms = 0.0;
    for (int r = 0; r < BENCH_RUNS; r++) {
        std::fill(C, C + n*n, 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        kernel(A, B, C, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double avg_ms = total_ms / BENCH_RUNS;
    bool ok = check_close(ref, C, n*n);
    return {name, avg_ms, to_gflops(n, avg_ms), ok};
}

void print_table(const std::vector<Result>& results) {
    std::cout << std::left  << std::setw(30) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "Correct" << "\n";
    std::cout << std::string(64, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left  << std::setw(30) << r.name
                  << std::right
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.gflops
                  << std::setw(10) << (r.correct ? "ok" : "FAIL") << "\n";
    }
    std::cout << "\n";
}

int main() {
    srand(42);

    const int nthreads = omp_get_max_threads();

    using KernelFn = std::function<void(const float*, const float*, float*, int)>;
    const std::vector<std::pair<std::string, KernelFn>> kernels = {
        {"A. Baseline (no OMP)",      matmul_baseline},
        {"B. parallel for",           matmul_parallel_for},
        {"C. parallel for + simd",    matmul_parallel_plus_simd},
        {"D. parallel for simd",      matmul_parallel_for_simd},
        {"E. collapse(2)",            matmul_collapse},
        {"F. schedule(static)",       matmul_schedule_static},
        {"G. schedule(dynamic,1)",    matmul_schedule_dynamic},
        {"H. schedule(guided)",       matmul_schedule_guided},
        {"I. simd only",              matmul_simd_only},
        {"J. single + task",          matmul_task},
        {"K. for nowait + barrier",   matmul_nowait},
    };

    std::cout << "OpenMP matmul ablation  |  "
              << BENCH_RUNS << " runs averaged  |  "
              << nthreads   << " OMP threads\n\n";

    for (int N : {256, 512, 1024}) {
        float* A   = new float[N*N];
        float* B   = new float[N*N];
        float* C   = new float[N*N];
        float* ref = new float[N*N];

        fill_random(A, N*N);
        fill_random(B, N*N);

        std::fill(ref, ref + N*N, 0.0f);
        matmul_baseline(A, B, ref, N);  // reference: serial tiled-K

        std::cout << "── Matrix " << N << "×" << N << " ──\n";

        std::vector<Result> results;
        for (const auto& [name, fn] : kernels)
            results.push_back(benchmark(name, fn, A, B, ref, C, N));

        print_table(results);

        delete[] A; delete[] B; delete[] C; delete[] ref;
    }
}
