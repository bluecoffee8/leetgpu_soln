#include <cuda_runtime.h>

// Block tile: each thread block computes a BM×BN output tile.
// K is consumed in BK-wide slices loaded into shared memory.
// Each thread owns a TM×TN register-level output tile (outer-product accumulation).
static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 16;
static constexpr int TM = 8;
static constexpr int TN = 8;
static constexpr int NTHREADS = (BM / TM) * (BN / TN);  // 16×16 = 256

__global__ void __launch_bounds__(NTHREADS)
matrix_multiplication_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // Which TM×TN sub-tile this thread owns within the block tile
    const int ty = tid / (BN / TN);   // [0, BM/TM)
    const int tx = tid % (BN / TN);   // [0, BN/TN)

    // +1 column padding eliminates bank conflicts when the inner-loop reads
    // sA[:][kk] (stride-1 in column for each row, 17-float rows break the
    // period-16 bank aliasing).  sB gets the same treatment.
    __shared__ float sA[BM][BK + 1];
    __shared__ float sB[BK][BN + 1];

    // Per-thread accumulator and register fragments
    float accum[TM][TN] = {};
    float regA[TM], regB[TN];

    // ── A-tile load mapping (BM×BK = 128×16 = 2048 floats, 256 threads → 8 each) ──
    // Consecutive threads pick consecutive columns → coalesced 128-byte transactions.
    const int aRow = tid / BK;                      // 0..15
    const int aCol = tid % BK;                      // 0..15
    static constexpr int aStride = NTHREADS / BK;  // 16 rows per load wave

    // ── B-tile load mapping (BK×BN = 16×128 = 2048 floats, 256 threads → 8 each) ──
    // Consecutive threads pick consecutive columns → coalesced.
    const int bRow = tid / BN;                      // 0..1
    const int bCol = tid % BN;                      // 0..127
    static constexpr int bStride = NTHREADS / BN;  // 2 rows per load wave

    const int num_tiles = (N + BK - 1) / BK;

    for (int t = 0; t < num_tiles; t++) {
        const int bk = t * BK;

        // Load A tile: sA[row][col] = A[(bm+row), (bk+col)]
        #pragma unroll
        for (int i = 0; i < BM; i += aStride) {
            const int gr = bm + aRow + i;
            const int gc = bk + aCol;
            sA[aRow + i][aCol] = (gr < M && gc < N) ? A[gr * N + gc] : 0.f;
        }

        // Load B tile: sB[row][col] = B[(bk+row), (bn+col)]
        #pragma unroll
        for (int i = 0; i < BK; i += bStride) {
            const int gr = bk + bRow + i;
            const int gc = bn + bCol;
            sB[bRow + i][bCol] = (gr < N && gc < K) ? B[gr * K + gc] : 0.f;
        }

        __syncthreads();

        // Outer-product accumulation over BK columns of A / rows of B.
        // Loading regA and regB into registers before the m×n loop avoids
        // re-issuing shared-memory reads in the innermost loop body.
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) regA[m] = sA[ty * TM + m][kk];
            #pragma unroll
            for (int n = 0; n < TN; n++) regB[n] = sB[kk][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    accum[m][n] = __fmaf_rn(regA[m], regB[n], accum[m][n]);
        }

        __syncthreads();
    }

    // Write the TM×TN result tile to C
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        const int gr = bm + ty * TM + m;
        if (gr < M) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const int gc = bn + tx * TN + n;
                if (gc < K) C[gr * K + gc] = accum[m][n];
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    const dim3 block(NTHREADS);
    const dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    matrix_multiplication_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
