#include <cuda_runtime.h>
#include <float.h>

// K/V tile height (rows per tile).
// kv_smem rows are padded to (d+1) to break the period-32 bank aliasing that
// would arise from stride-d access when d is a multiple of 32.
static constexpr int BN         = 64;
static constexpr int BLOCK_SIZE = 128;

// Flash-Attention forward pass — one thread block per query row.
//
// Shared-memory layout (floats):
//   q_smem  [d]           – current query row
//   kv_smem [BN*(d+1)]   – K tile then V tile (reused; +1 per row for padding)
//   s_smem  [BN]          – per-row scores then unnorm. softmax weights
//   O_smem  [d]           – running output accumulator
//
// Thread roles per tile:
//   • K/V loads   – all threads, strided to stay coalesced in global memory.
//   • Score QK^T  – threads split on j (rows of K tile); each thread owns its j.
//   • O update    – threads split on k (output dimension); each thread owns its k.
//   • m / l stats – all threads compute identically (O(BN) redundant SMEM reads,
//                   negligible vs. the O(BN*d) global-memory bound).
__global__ void __launch_bounds__(BLOCK_SIZE)
flash_attention_kernel(const float* __restrict__ Q,
                       const float* __restrict__ K,
                       const float* __restrict__ V,
                       float* __restrict__ O,
                       int M, int N, int d,
                       float scale)
{
    const int qi  = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* q_smem  = smem;
    float* kv_smem = q_smem  + d;
    float* s_smem  = kv_smem + BN * (d + 1);
    float* O_smem  = s_smem  + BN;

    // Cooperatively load Q[qi] and zero the output accumulator.
    for (int k = tid; k < d; k += BLOCK_SIZE) {
        q_smem[k] = Q[qi * d + k];
        O_smem[k] = 0.f;
    }
    __syncthreads();

    // Per-thread running softmax state (all threads maintain the same values).
    float m = -FLT_MAX;   // running max of scores
    float l = 0.f;        // running sum of exp(score - m)

    const int num_tiles = (N + BN - 1) / BN;

    for (int t = 0; t < num_tiles; ++t) {
        const int tile_start = t * BN;
        const int tile_size  = min(BN, N - tile_start);

        // ── 1. Load K tile (coalesced: consecutive threads → consecutive addrs) ──
        for (int idx = tid; idx < tile_size * d; idx += BLOCK_SIZE) {
            const int j = idx / d, k = idx % d;
            kv_smem[j * (d + 1) + k] = K[(tile_start + j) * d + k];
        }
        __syncthreads();

        // ── 2. Scaled dot-product scores: s[j] = scale * (q · k_j) ───────────
        // Thread j computes one score; loops when tile_size > BLOCK_SIZE.
        for (int j = tid; j < tile_size; j += BLOCK_SIZE) {
            float dot = 0.f;
            #pragma unroll 4
            for (int k = 0; k < d; ++k)
                dot += q_smem[k] * kv_smem[j * (d + 1) + k];
            s_smem[j] = dot * scale;
        }
        __syncthreads();

        // ── 3. Online-softmax: find tile max, compute correction factor ────────
        float m_tile = -FLT_MAX;
        for (int j = 0; j < tile_size; ++j)
            m_tile = fmaxf(m_tile, s_smem[j]);
        const float m_new   = fmaxf(m, m_tile);
        const float m_scale = __expf(m - m_new);   // rescales previous O contribution

        // ── 4. Scores → unnormalised weights; each thread owns its j slots ─────
        for (int j = tid; j < tile_size; j += BLOCK_SIZE)
            s_smem[j] = __expf(s_smem[j] - m_new);
        __syncthreads();

        // ── 5. Accumulate tile normaliser (redundant but cheap vs. global mem) ─
        float l_tile = 0.f;
        for (int j = 0; j < tile_size; ++j)
            l_tile += s_smem[j];

        // ── 6. Load V tile, reusing kv_smem (s_smem untouched) ───────────────
        for (int idx = tid; idx < tile_size * d; idx += BLOCK_SIZE) {
            const int j = idx / d, k = idx % d;
            kv_smem[j * (d + 1) + k] = V[(tile_start + j) * d + k];
        }
        __syncthreads();

        // ── 7. O = O * m_scale + weights @ V  (threads split on d-dimension) ──
        // warp reads s_smem[j] as a broadcast (1 txn); kv_smem[j*(d+1)+0..31]
        // spans all 32 distinct banks (no conflicts).
        for (int k = tid; k < d; k += BLOCK_SIZE) {
            float acc = O_smem[k] * m_scale;
            #pragma unroll 4
            for (int j = 0; j < tile_size; ++j)
                acc = __fmaf_rn(s_smem[j], kv_smem[j * (d + 1) + k], acc);
            O_smem[k] = acc;
        }

        m = m_new;
        l = m_scale * l + l_tile;
        __syncthreads();    // guard kv_smem before next tile's K load
    }

    // ── 8. Normalise and store ───────────────────────────────────────────────
    const float inv_l = __fdividef(1.f, l);
    for (int k = tid; k < d; k += BLOCK_SIZE)
        O[qi * d + k] = O_smem[k] * inv_l;
}

// Q has size (M, d), K has size (N, d), V has size (N, d), output has size (M, d).
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int N, int d)
{
    const float  scale      = rsqrtf((float)d);
    // smem: q[d] + kv[BN*(d+1)] + s[BN] + O[d]
    const size_t smem_bytes = (size_t)(2 * d + BN * (d + 1) + BN) * sizeof(float);

    flash_attention_kernel<<<M, BLOCK_SIZE, smem_bytes>>>(
        Q, K, V, output, M, N, d, scale);
}
