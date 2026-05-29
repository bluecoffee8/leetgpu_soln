#include <cuda_runtime.h>

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

const int WARPSIZE = 32;
#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// ---------------------------------------------------------------------------
// cp.async helpers (Ampere sm_80+). These issue asynchronous gmem->smem copies
// that bypass the register file, enabling software-pipelined double buffering.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gmem_ptr) {
    // 16-byte copy with .cg (cache-global, bypass L1) caching policy.
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async4(void* smem_ptr, const void* gmem_ptr) {
    // 4-byte (single float) copy with .ca caching policy.
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int Remaining>
__device__ __forceinline__ void cp_async_wait() {
    // Block until at most `Remaining` previously committed groups are in flight.
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Remaining));
}

namespace wt {
    template<const int BLOCK_SIZE, const int a_stride, const int b_stride>
    __device__ void load_from_gmem_async(float* A, float* B, float* As, float* Bs, int a_m, int a_n, int b_n, int b_k, int N, int K) {
        #pragma unroll
        for (int i = 0; i + a_stride <= BLOCK_SIZE; i += a_stride) {
            // A is stored transposed in smem, so each of the 4 contiguous source
            // floats lands in a different smem row -- cp.async cannot scatter a
            // vector, so issue 4 separate 4-byte async copies.
            const float* src = &A[OFFSET(a_m + i, a_n, N)];
            cp_async4(&As[OFFSET(a_n + 0, a_m + i, BLOCK_SIZE)], src + 0);
            cp_async4(&As[OFFSET(a_n + 1, a_m + i, BLOCK_SIZE)], src + 1);
            cp_async4(&As[OFFSET(a_n + 2, a_m + i, BLOCK_SIZE)], src + 2);
            cp_async4(&As[OFFSET(a_n + 3, a_m + i, BLOCK_SIZE)], src + 3);
        }
        #pragma unroll
        for (int i = 0; i + b_stride <= BLOCK_SIZE; i += b_stride) {
            // B is copied without transpose, so a single 16-byte async copy works.
            cp_async16(&Bs[OFFSET(b_n + i, b_k, BLOCK_SIZE)], &B[OFFSET(b_n + i, b_k, K)]);
        }
    }

    template<const int BLOCK_SIZE, const int WARP_BLOCK_SIZE, const int WARP_M_ITER, const int WARP_K_ITER, const int WARP_SUB_M, const int WARP_SUB_K, const int T>
    __device__ void process_from_smem(float* As, float* Bs, float* reg_m, float* reg_k, float* tmp, 
                                      const int warp_m, const int warp_k, const int lane_m, const int lane_k, int n, int N) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (n + i < N) {
                #pragma unroll 
                for (int j = 0; j < WARP_M_ITER; j++) {
                    for (int l = 0; l < T; l++) {
                        reg_m[j * T + l] = As[i * BLOCK_SIZE + warp_m * WARP_BLOCK_SIZE + j * WARP_SUB_M + lane_m * T + l];
                    }
                }

                #pragma unroll 
                for (int j = 0; j < WARP_K_ITER; j++) {
                    for (int l = 0; l < T; l++) {
                        reg_k[j * T + l] = Bs[i * BLOCK_SIZE + warp_k * WARP_BLOCK_SIZE + j * WARP_SUB_K + lane_k * T + l];
                    }
                }

                #pragma unroll
                for (int x = 0; x < WARP_M_ITER; x++) {
                    for (int y = 0; y < WARP_K_ITER; y++) {
                        for (int i = 0; i < T; i++) {
                            for (int j = 0; j < T; j++) {
                                tmp[(x * T + i) * (WARP_K_ITER * T) + (y * T + j)]
                                    += reg_m[x * T + i] * reg_k[y * T + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

template<const int BLOCK_SIZE, const int WARP_BLOCK_SIZE, const int WARP_K_ITER, const int T, const int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void matmul_vectorized(float* A, float* B, float* C, int M, int N, int K) {
    int bm = blockIdx.y; 
    int bk = blockIdx.x;

    const int warp_idx = threadIdx.x / WARPSIZE; 
    const int warp_m = warp_idx / (BLOCK_SIZE / WARP_BLOCK_SIZE); 
    const int warp_k = warp_idx % (BLOCK_SIZE / WARP_BLOCK_SIZE);

    constexpr int WARP_M_ITER = (WARP_BLOCK_SIZE * WARP_BLOCK_SIZE) / (WARPSIZE * T * T * WARP_K_ITER);
    constexpr int WARP_SUB_M = WARP_BLOCK_SIZE / WARP_M_ITER;
    constexpr int WARP_SUB_K = WARP_BLOCK_SIZE / WARP_K_ITER; 

    const int lane_idx = threadIdx.x % WARPSIZE;
    const int lane_m = lane_idx / (WARP_SUB_K / T);
    const int lane_k = lane_idx % (WARP_SUB_K / T); 

    // Dynamic shared memory: two buffers each of As and Bs for double buffering.
    // (Static smem caps at 48 KB; two 64x64 float buffers of each is 64 KB, so we
    // must use an opt-in dynamic allocation -- see cudaFuncSetAttribute in solve.)
    constexpr int TILE = BLOCK_SIZE * BLOCK_SIZE;
    extern __shared__ float smem[];
    float* As = smem;            // As[0..TILE) and As[TILE..2*TILE) are the two buffers
    float* Bs = smem + 2 * TILE; // Bs[0..TILE) and Bs[TILE..2*TILE) are the two buffers

    // Advance the block tiles via pointer arithmetic; shared-memory indices stay local.
    A += bm * BLOCK_SIZE * N;
    B += bk * BLOCK_SIZE;
    // C must be moved to this warp's output tile, not just the block's origin.
    C += (bm * BLOCK_SIZE + warp_m * WARP_BLOCK_SIZE) * K + bk * BLOCK_SIZE + warp_k * WARP_BLOCK_SIZE;

    int a_m = threadIdx.x / (BLOCK_SIZE / 4);
    const int a_n = (threadIdx.x % (BLOCK_SIZE / 4)) * 4;
    const int a_stride = (NUM_THREADS * 4) / BLOCK_SIZE;

    int b_n = threadIdx.x / (BLOCK_SIZE / 4);
    int b_k = (threadIdx.x % (BLOCK_SIZE / 4)) * 4;
    const int b_stride = (NUM_THREADS * 4) / BLOCK_SIZE;

    float tmp[WARP_M_ITER * T * WARP_K_ITER * T] = {0.0f}; 
    float reg_m[WARP_M_ITER * T] = {0.0f};
    float reg_k[WARP_K_ITER * T] = {0.0f};

    // --- Software-pipelined double buffering via cp.async ---
    // Preload the first tile into buffer 0, then advance the gmem pointers so the
    // in-loop prefetch always targets the *next* tile.
    wt::load_from_gmem_async<BLOCK_SIZE, a_stride, b_stride>(A, B, As, Bs, a_m, a_n, b_n, b_k, N, K);
    cp_async_commit();
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * K;

    int read_stage = 0;
    #pragma unroll
    for (int n = 0; n < N; n += BLOCK_SIZE) {
        const int write_stage = read_stage ^ 1;
        if (n + BLOCK_SIZE < N) {
            // Prefetch the next tile into the other buffer while we compute on the
            // current one; wait_group<1> keeps this load in flight but guarantees
            // the older (read_stage) load has landed.
            wt::load_from_gmem_async<BLOCK_SIZE, a_stride, b_stride>(
                A, B, As + write_stage * TILE, Bs + write_stage * TILE, a_m, a_n, b_n, b_k, N, K);
            cp_async_commit();
            A += BLOCK_SIZE;
            B += BLOCK_SIZE * K;
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        wt::process_from_smem<BLOCK_SIZE, WARP_BLOCK_SIZE, WARP_M_ITER, WARP_K_ITER, WARP_SUB_M, WARP_SUB_K, T>(
            As + read_stage * TILE, Bs + read_stage * TILE, reg_m, reg_k, tmp, warp_m, warp_k, lane_m, lane_k, n, N
        );
        __syncthreads();
        read_stage = write_stage;
    }
    for (int x = 0; x < WARP_M_ITER; x++) {
        for (int y = 0; y < WARP_K_ITER; y++) {
            // C already points at this warp's tile; move to the current warp subtile.
            float* Ctmp = C + (x * WARP_SUB_M) * K + (y * WARP_SUB_K);
            for (int i = 0; i < T; i++) {
                int row = bm * BLOCK_SIZE + warp_m * WARP_BLOCK_SIZE + x * WARP_SUB_M + lane_m * T + i;
                if (row >= M) break;
                for (int j = 0; j < T; j += 4) {
                    int col = bk * BLOCK_SIZE + warp_k * WARP_BLOCK_SIZE + y * WARP_SUB_K + lane_k * T + j;
                    // tmp index must match the accumulation layout in process_from_smem.
                    int idx = (x * T + i) * (WARP_K_ITER * T) + y * T + j;
                    if (col + 3 < K) {
                        FETCH_FLOAT4(Ctmp[OFFSET(lane_m * T + i, lane_k * T + j, K)]) = FETCH_FLOAT4(tmp[idx]);
                    } else {
                        for (int j_ = j; j_ < T && col + (j_ - j) < K; j_++) {
                            Ctmp[OFFSET(lane_m * T + i, lane_k * T + j_, K)] = tmp[(x * T + i) * (WARP_K_ITER * T) + y * T + j_];
                        }
                    }
                }
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* A, float* B, float* C, int M, int N, int K) {
    if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
        const int BLOCK_SIZE = 64, T = 4, WARP_BLOCK_SIZE = 32, WARP_K_ITER = 2, NUM_THREADS = 128;
        dim3 threadsPerBlock(NUM_THREADS);
        dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Two buffers each of As and Bs (double buffering) => 4 * BLOCK_SIZE^2 floats.
        // This exceeds the 48 KB static smem cap, so request it as dynamic smem.
        const int smem_bytes = 4 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
        auto kernel = matmul_vectorized<BLOCK_SIZE, WARP_BLOCK_SIZE, WARP_K_ITER, T, NUM_THREADS>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        kernel<<<blocksPerGrid, threadsPerBlock, smem_bytes>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    } else {
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    }
}
