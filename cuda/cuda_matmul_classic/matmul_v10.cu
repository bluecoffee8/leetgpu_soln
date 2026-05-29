#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

__global__ __launch_bounds__(1024) void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0;
    for (int n = 0; n < N; n++) {
        tmp += A[m * N + n] * B[n * K + k];
    }
    if (m < M && k < K) {
        C[m * K + k] = tmp;
    }
}

#define OFFSET(row, col, ld) ((row)*(ld)+(col))

// TF32 tensor-core tile shape (Ampere+, SM_80). The wmma GEMM computes
// D[WMMA_M x WMMA_N] = A[WMMA_M x WMMA_K] * B[WMMA_K x WMMA_N]. Mapping our
// problem C[M][K] = A[M][N] * B[N][K] onto it: rows = m, cols = k, and the
// contraction WMMA_K runs along n.
static const int WMMA_M = 16;
static const int WMMA_N = 16;
static const int WMMA_K = 8;

// Tensor-core matmul: C[M][K] = A[M][N] * B[N][K] (all row-major).
//   BLOCK_M  - block output tile height (along m)
//   BLOCK_N  - block output tile width  (along k)
//   BLOCK_K  - contraction tile depth   (along n), must be a multiple of WMMA_K
//   WARP_ROWS/WARP_COLS - warp grid within the block (warps along m / k)
//   NUM_THREADS = WARP_ROWS * WARP_COLS * WARPSIZE
// A/B tiles are zero-padded in shared memory so out-of-range threads contribute
// nothing, which lets the kernel handle arbitrary M, N, K without alignment.
template<const int BLOCK_M, const int BLOCK_N, const int BLOCK_K,
         const int WARP_ROWS, const int WARP_COLS, const int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void matmul_vectorized(
        const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int N, int K) {
    // Number of 16x16 wmma tiles each warp owns along m / k.
    constexpr int WARP_M_TILES = BLOCK_M / (WARP_ROWS * WMMA_M);
    constexpr int WARP_N_TILES = BLOCK_N / (WARP_COLS * WMMA_N);

    const int block_row = blockIdx.y * BLOCK_M; // first output row (m)
    const int block_col = blockIdx.x * BLOCK_N; // first output col (k)

    const int warp_id  = threadIdx.x / warpSize;
    const int warp_row = warp_id / WARP_COLS;   // this warp's tile row (m)
    const int warp_col = warp_id % WARP_COLS;   // this warp's tile col (k)

    // As holds A[m][n] (BLOCK_M x BLOCK_K), Bs holds B[n][k] (BLOCK_K x BLOCK_N).
    // Cs stages the block's output (BLOCK_M x BLOCK_N) before the bounds-checked
    // write-back to global memory.
    __shared__ float As[BLOCK_M * BLOCK_K];
    __shared__ float Bs[BLOCK_K * BLOCK_N];
    __shared__ float Cs[BLOCK_M * BLOCK_N];

    // Each lane owns ACC_ELEMS of every 16x16 accumulator tile.
    constexpr int ACC_ELEMS = (WMMA_M * WMMA_N) / 32;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_M_TILES][WARP_N_TILES];
    // Kahan compensation (the low-order bits dropped by FP32 accumulation).
    float comp[WARP_M_TILES][WARP_N_TILES][ACC_ELEMS];
    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; i++)
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
            #pragma unroll
            for (int t = 0; t < ACC_ELEMS; t++) comp[i][j][t] = 0.0f;
        }

    for (int n0 = 0; n0 < N; n0 += BLOCK_K) {
        // Stage A tile (zero-padded outside the matrix).
        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
            int r = idx / BLOCK_K;          // m within tile
            int c = idx % BLOCK_K;          // n within tile
            int gm = block_row + r;
            int gn = n0 + c;
            As[idx] = (gm < M && gn < N) ? A[OFFSET(gm, gn, N)] : 0.0f;
        }
        // Stage B tile (zero-padded outside the matrix).
        for (int idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += NUM_THREADS) {
            int r = idx / BLOCK_N;          // n within tile
            int c = idx % BLOCK_N;          // k within tile
            int gn = n0 + r;
            int gk = block_col + c;
            Bs[idx] = (gn < N && gk < K) ? B[OFFSET(gn, gk, K)] : 0.0f;
        }
        __syncthreads();

        // Accumulate this contraction chunk into a fresh partial sum, then fold
        // it into the master accumulator with Kahan compensation below. Keeping
        // the per-chunk sum separate bounds the FP32 accumulation error to O(eps)
        // instead of O(N*eps), which is what dominates once inputs are 3xTF32.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
            psum[WARP_M_TILES][WARP_N_TILES];
        #pragma unroll
        for (int i = 0; i < WARP_M_TILES; i++)
            #pragma unroll
            for (int j = 0; j < WARP_N_TILES; j++)
                wmma::fill_fragment(psum[i][j], 0.0f);

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            // 3xTF32: split each FP32 value x into x_hi + x_lo, both TF32-rounded,
            // then accumulate the three highest-order products and drop the
            // negligible a_lo*b_lo term. This recovers ~20 mantissa bits.
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> a_hi[WARP_M_TILES], a_lo[WARP_M_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> b_hi[WARP_N_TILES], b_lo[WARP_N_TILES];

            #pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++) {
                int a_row = warp_row * WARP_M_TILES * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(a_hi[i], &As[OFFSET(a_row, kk, BLOCK_K)], BLOCK_K);
                #pragma unroll
                for (int t = 0; t < a_hi[i].num_elements; t++) {
                    float orig = a_hi[i].x[t];
                    float hi = wmma::__float_to_tf32(orig);
                    a_lo[i].x[t] = wmma::__float_to_tf32(orig - hi);
                    a_hi[i].x[t] = hi;
                }
            }
            #pragma unroll
            for (int j = 0; j < WARP_N_TILES; j++) {
                int b_col = warp_col * WARP_N_TILES * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(b_hi[j], &Bs[OFFSET(kk, b_col, BLOCK_N)], BLOCK_N);
                #pragma unroll
                for (int t = 0; t < b_hi[j].num_elements; t++) {
                    float orig = b_hi[j].x[t];
                    float hi = wmma::__float_to_tf32(orig);
                    b_lo[j].x[t] = wmma::__float_to_tf32(orig - hi);
                    b_hi[j].x[t] = hi;
                }
            }

            // Accumulate correction terms first, then the dominant a_hi*b_hi term.
            #pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++)
                #pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++) {
                    wmma::mma_sync(psum[i][j], a_lo[i], b_hi[j], psum[i][j]);
                    wmma::mma_sync(psum[i][j], a_hi[i], b_lo[j], psum[i][j]);
                    wmma::mma_sync(psum[i][j], a_hi[i], b_hi[j], psum[i][j]);
                }
        }

        // Kahan-fold the chunk partial into the master accumulator.
        #pragma unroll
        for (int i = 0; i < WARP_M_TILES; i++)
            #pragma unroll
            for (int j = 0; j < WARP_N_TILES; j++)
                #pragma unroll
                for (int t = 0; t < ACC_ELEMS; t++) {
                    float y = psum[i][j].x[t] - comp[i][j][t];
                    float s = acc[i][j].x[t] + y;
                    comp[i][j][t] = (s - acc[i][j].x[t]) - y;
                    acc[i][j].x[t] = s;
                }
        __syncthreads();
    }

    // Store each warp's accumulators into the shared output tile.
    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++) {
            int c_row = warp_row * WARP_M_TILES * WMMA_M + i * WMMA_M;
            int c_col = warp_col * WARP_N_TILES * WMMA_N + j * WMMA_N;
            wmma::store_matrix_sync(&Cs[OFFSET(c_row, c_col, BLOCK_N)], acc[i][j],
                                    BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Bounds-checked write-back from shared to global memory.
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += NUM_THREADS) {
        int r = idx / BLOCK_N;
        int c = idx % BLOCK_N;
        int gm = block_row + r;
        int gk = block_col + c;
        if (gm < M && gk < K)
            C[OFFSET(gm, gk, K)] = Cs[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* A, float* B, float* C, int M, int N, int K) {
    const int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 16;
    const int WARP_ROWS = 2, WARP_COLS = 2;
    const int NUM_THREADS = WARP_ROWS * WARP_COLS * 32;

    dim3 threadsPerBlock(NUM_THREADS);
    dim3 blocksPerGrid((K + BLOCK_N - 1) / BLOCK_N,
                       (M + BLOCK_M - 1) / BLOCK_M);

    matmul_vectorized<BLOCK_M, BLOCK_N, BLOCK_K, WARP_ROWS, WARP_COLS, NUM_THREADS>
        <<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
