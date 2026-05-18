/*
 * CPU Matrix Multiplication — Progressive Optimization
 * Reference: https://siboehm.com/articles/22/Fast-MMM-on-CPU
 *
 * ── Compilation ──────────────────────────────────────────────────────────────
 *
 * Linux / GCC (x86-64):
 *   g++ -O3 -march=native -ffast-math -std=c++17 -o matmul.o matmul.cpp -lpthread
 *
 * macOS / Apple Clang (Apple Silicon or Intel):
 *   clang++ -O3 -march=native -ffast-math -std=c++17 -o matmul.o matmul.cpp
 *
 * ── Running ───────────────────────────────────────────────────────────────────
 *
 *   ./matmul.o
 *
 *   Thread count defaults to hardware_concurrency(). Override at compile time:
 *     clang++ ... -DNUM_THREADS=4 -o matmul.o matmul.cpp
 *
 * ── Why these flags matter ───────────────────────────────────────────────────
 *
 *   -O3          enables loop vectorisation, inlining, and instruction scheduling
 *   -march=native emits AVX2/AVX-512 (x86) or NEON/SVE (ARM) — without it the
 *                compiler targets a generic baseline and leaves SIMD on the table
 *   -ffast-math  allows FP reassociation and FMA contraction; required for the
 *                compiler to auto-vectorise the accumulation loop
 *   -std=c++17   needed for std::thread and hardware_concurrency
 *
 * ── Expected timings (1024×1024, 8-core x86 with AVX2) ──────────────────────
 *
 *   Kernel                 Time     GFLOPS
 *   ──────────────────────────────────────
 *   1. Naive IJK           ~1500ms    ~1.4
 *   2. Register accum      ~1000ms    ~2.1
 *   3. IKJ reorder           ~90ms   ~23
 *   4. IKJ + K-tiling        ~70ms   ~30
 *   5. 2D tile + threads     ~16ms  ~130  (8 threads)
 *   ──────────────────────────────────────
 *   NumPy/MKL reference       ~8ms  ~270  (reference)
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Matrix dimension (must be divisible by all tile sizes below for simplicity)
static const int N = 1024;

// Number of timed runs (after 1 warmup); average is reported.
static const int BENCH_RUNS = 3;

// Tile sizes from grid search in the blog post.
// TILE_K = 16  keeps ~(16+2)*1024*4 ≈ 74 KB in L1D.
// TILE_I/J = 256 partitions work evenly across cores for N=1024.
static const int TILE_K = 16;
static const int TILE_I = 256;
static const int TILE_J = 256;

// ─── Kernels ──────────────────────────────────────────────────────────────────

/*
 * Kernel 1 — Naive IJK
 *
 * The canonical textbook triple loop. The innermost loop varies k:
 *   A[i][k] : stride 1 (sequential) — cache-friendly
 *   B[k][j] : stride N (column walk) — every iteration is a cache miss at N=1024
 *   C[i][j] : constant — fine
 *
 * B's column-major access pattern thrashes the cache and prevents auto-vectorisation,
 * leaving the CPU waiting on memory instead of doing arithmetic.
 */
void matmul_naive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
}

/*
 * Kernel 2 — Register Accumulator
 *
 * Accumulate the dot product in a local variable `acc` before writing back.
 * Without this, the compiler cannot prove C[i*n+j] doesn't alias A or B and
 * must reload it from memory each iteration. The register hint removes that
 * load–store dependency from the hot path, yielding a small but measurable gain.
 *
 * Still uses IJK ordering → B still accessed with stride N → no vectorisation.
 */
void matmul_reg_acc(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++)
                acc += A[i*n + k] * B[k*n + j];
            C[i*n + j] = acc;
        }
}

/*
 * Kernel 3 — IKJ Loop Reorder (cache-friendly)
 *
 * Swap the j and k loops. The innermost loop now varies j:
 *   A[i][k] : constant (hoisted as scalar `a_ik`) — broadcast across SIMD lanes
 *   B[k][j] : stride 1 — sequential reads, vectorisable
 *   C[i][j] : stride 1 — sequential reads+writes, vectorisable
 *
 * With -O3 -march=native -ffast-math the compiler emits VFMADD231PS (x86 AVX2)
 * or FMLA (ARM NEON), processing 8 floats per instruction cycle.
 * This is the single biggest jump: ~18× over Kernel 2.
 */
void matmul_ikj(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a_ik = A[i*n + k];  // hoist scalar; compiler broadcasts to YMM register
            for (int j = 0; j < n; j++)
                C[i*n + j] += a_ik * B[k*n + j];
        }
}

/*
 * Kernel 4 — IKJ + K-dimension Tiling
 *
 * Add an outer tile loop over k (strip size TILE_K = 16).
 * Working set per tile of the inner loops:
 *   TILE_K rows of B  : 16 × 1024 × 4 B = 64 KB
 *   1 row of C        : 1  × 1024 × 4 B =  4 KB
 *   1 scalar per tile : negligible
 *   Total             : ~68 KB — fits comfortably in a 256 KB L1D cache
 *
 * Without tiling, iterating all k values before moving the i tile means B rows
 * evict C from L1 before C is fully accumulated; tiling keeps C hot.
 */
void matmul_tiled_k(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int kt = 0; kt < n; kt += TILE_K)
            for (int k = kt; k < kt + TILE_K; k++) {
                float a_ik = A[i*n + k];
                for (int j = 0; j < n; j++)
                    C[i*n + j] += a_ik * B[k*n + j];
            }
}

/*
 * Kernel 5 — 2D Tiling (I, J, K) + std::thread parallelism
 *
 * Partition the output matrix into TILE_I × TILE_J blocks (256×256) and
 * distribute them across worker threads. Within each block the same K-tiling
 * from Kernel 4 is applied, keeping B slices hot in L1.
 *
 * Thread dispatch: build a flat list of (it, jt) tile coordinates and assign
 * a contiguous stripe to each thread. For N=1024 and 256×256 tiles there are
 * 4×4=16 tiles; with 8 threads each gets 2 tiles.
 *
 * Effective memory hierarchy:
 *   L1 per core : TILE_K rows of B + 1 row of C (same as Kernel 4)
 *   L2 per core : the TILE_I×TILE_J C-tile stays warm across K-tiles
 */
#ifndef NUM_THREADS
#define NUM_THREADS 0  // 0 → detect at runtime via hardware_concurrency
#endif

void matmul_tiled_threads(const float* A, const float* B, float* C, int n) {
    const int nthreads = (NUM_THREADS > 0)
        ? NUM_THREADS
        : static_cast<int>(std::thread::hardware_concurrency());

    // Flat list of output tile origins (it, jt)
    std::vector<std::pair<int,int>> tiles;
    for (int it = 0; it < n; it += TILE_I)
        for (int jt = 0; jt < n; jt += TILE_J)
            tiles.emplace_back(it, jt);

    auto worker = [&](int tid) {
        // Each thread processes a contiguous stripe of the tile list.
        for (int t = tid; t < static_cast<int>(tiles.size()); t += nthreads) {
            auto [it, jt] = tiles[t];
            for (int kt = 0; kt < n; kt += TILE_K)
                for (int i = it; i < it + TILE_I; i++)
                    for (int k = kt; k < kt + TILE_K; k++) {
                        float a_ik = A[i*n + k];
                        for (int j = jt; j < jt + TILE_J; j++)
                            C[i*n + j] += a_ik * B[k*n + j];
                    }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; t++)
        threads.emplace_back(worker, t);
    for (auto& th : threads)
        th.join();
}

// ─── Benchmark harness ───────────────────────────────────────────────────────

void fill_random(float* M, int size) {
    for (int i = 0; i < size; i++)
        M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// Relative + absolute tolerance check: |ref - out| <= tol*(|ref|+1)
bool check_close(const float* ref, const float* out, int size, float tol = 1e-3f) {
    for (int i = 0; i < size; i++)
        if (std::fabs(ref[i] - out[i]) > tol * (std::fabs(ref[i]) + 1.0f))
            return false;
    return true;
}

// 2*N^3 FLOPs (N^3 multiplications + N^3 additions), reported in GFLOPS.
double to_gflops(int n, double ms) {
    return 2.0 * n * n * n / (ms * 1e6);
}

struct Result { std::string name; double ms; double gflops; bool correct; };

Result benchmark(
    const std::string& name,
    std::function<void(const float*, const float*, float*, int)> kernel,
    const float* A, const float* B, const float* ref, float* C, int n)
{
    // One warmup run to bring data into cache and trigger JIT/branch prediction.
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

int main() {
    srand(42);

    float* A   = new float[N*N];
    float* B   = new float[N*N];
    float* C   = new float[N*N];
    float* ref = new float[N*N];

    fill_random(A, N*N);
    fill_random(B, N*N);

    // Build reference with the fastest single-threaded kernel.
    // All other kernels are validated against this.
    std::fill(ref, ref + N*N, 0.0f);
    matmul_ikj(A, B, ref, N);

    std::vector<Result> results;

    // Note: Kernels 1 and 2 are slow (~1.5s each with -O3). They are included
    // to show baseline and the small gain from register accumulation before the
    // large jump from loop reordering.
    results.push_back(benchmark("1. Naive IJK",      matmul_naive,     A, B, ref, C, N));
    results.push_back(benchmark("2. Register accum", matmul_reg_acc,   A, B, ref, C, N));
    results.push_back(benchmark("3. IKJ reorder",    matmul_ikj,       A, B, ref, C, N));
    results.push_back(benchmark("4. IKJ + K-tile",   matmul_tiled_k,   A, B, ref, C, N));
    results.push_back(benchmark("5. 2D tile+threads", matmul_tiled_threads, A, B, ref, C, N));

    const int nthreads = (NUM_THREADS > 0)
        ? NUM_THREADS
        : static_cast<int>(std::thread::hardware_concurrency());

    std::cout << "\nMatrix: " << N << "×" << N
              << "  |  " << BENCH_RUNS << " runs averaged  |  "
              << nthreads << " threads\n\n";
    std::cout << std::left  << std::setw(24) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "Correct\n";
    std::cout << std::string(58, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left  << std::setw(24) << r.name
                  << std::right
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.gflops
                  << std::setw(10) << (r.correct ? "ok" : "FAIL") << "\n";
    }
    std::cout << "\n";

    delete[] A; delete[] B; delete[] C; delete[] ref;
}
