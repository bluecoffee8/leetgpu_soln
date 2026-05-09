#include <cuda_runtime.h>

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

__global__ void matmul_v1(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < M && j < K) {
        float s = 0.0f;
        for (int k = 0; k < N; k++) {
            s += A[i * N + k] * B[k * K + j];
        }
        C[i * K + j] = s; 
    }
}

template<int BLOCK_SIZE>
__global__ void matmul_v2(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE); 

    if (x < M && y < K) {
        float s = 0.0f;
        for (int i = 0; i < N; i++) {
            s += A[x * N + i] * B[i * K + y];
        }
        C[x * K + y] = s; 
    }
}

template<int BLOCK_SIZE>
__global__ void matmul_v3(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    const int tid = threadIdx.x; 
    const int i = (tid / BLOCK_SIZE);
    const int j = (tid % BLOCK_SIZE);

    const int x = blockIdx.x * BLOCK_SIZE + i;
    const int y = blockIdx.y * BLOCK_SIZE + j; 

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE]; 

    float s = 0.0f; 

    for (int n = 0; n < N; n += BLOCK_SIZE) {
        int y_ = n + j; 
        As[tid] = A[x * N + y_]; 
        int x_ = n + i; 
        Bs[tid] = B[x_ * K + y];

        __syncthreads();

        for (int n_ = 0; n + n_ < min(N, n + BLOCK_SIZE); n_++) {
            s += As[i * BLOCK_SIZE + n_] * Bs[n_ * BLOCK_SIZE + j];
        }

        __syncthreads(); 
    }
    
    if (x < M && y < K) {
        C[x * K + y] = s; 
    }
}

template<const int BM, const int BN, const int BK, const int TM>
__global__ void matmul_v4(const float* A, const float* B, float* C, int M, int N, int K) {
    const unsigned int c_row = blockIdx.y;
    const unsigned int c_col = blockIdx.x; 

    const int t_col = threadIdx.x % BK; 
    const int t_row = threadIdx.x / BK; 

    __shared__ float As[BM * BN];
    __shared__ float Bs[BN * BK];

    A += c_row * BM * N; 
    B += c_col * BK; 
    C += c_row * BM * K + c_col * BK; 

    // gmem coalesce access

    // from A, load (BM, BN)
    // from B, load (BN, BK)
    const unsigned int a_col = threadIdx.x % BN; 
    const unsigned int a_row = threadIdx.x / BN; 
    const unsigned int b_col = threadIdx.x % BK; 
    const unsigned int b_row = threadIdx.x / BK; 

    float acc[TM] = {0.0};

    for (unsigned int n = 0; n < N; n += BN) {
        As[a_row * BN + a_col] = A[a_row * N + a_col];
        Bs[b_row * BK + b_col] = B[b_row * K + b_col];
        __syncthreads(); 

        A += BN;
        B += BN * K; 

        for (unsigned int n_ = 0; n + n_ < min(N, n + BN); n_++) {
            if (t_col < K) {
                float tb = Bs[n_ * BK + t_col]; 
                for (unsigned int acc_i = 0; acc_i < TM; acc_i++) {
                    if (c_row * BM + t_row * TM + acc_i < M) {
                        acc[acc_i] += As[(t_row * TM + acc_i) * BN + n_] * tb; 
                    }
                }
            }
        }
        __syncthreads(); 
    }

    for (unsigned int acc_i = 0; acc_i < TM; acc_i++) {
        if ((t_row * TM + acc_i) < M && t_col < K) {
            C[(t_row * TM + acc_i) * K + t_col] = acc[acc_i];    
        }
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TK>
__global__ void matmul_v5(const float* A, const float* B, float* C, int M, int N, int K) {
    const unsigned int c_row = blockIdx.y;
    const unsigned int c_col = blockIdx.x; 

    // const unsigned int num_thread = blockDim.x; 

    // compute value (BM, BK)
    // tiled by (TM, TK)

    // num threads = (BM/TM) * (BK/TK)

    const int t_col = threadIdx.x % (BK/TK);
    const int t_row = threadIdx.x / (BK/TK); 

    __shared__ float As[BM * BN];
    __shared__ float Bs[BN * BK];

    A += c_row * BM * N; 
    B += c_col * BK; 
    C += c_row * BM * K + c_col * BK; 

    const unsigned int a_col = threadIdx.x % BN; 
    const unsigned int a_row = threadIdx.x / BN; 
    const unsigned int stride_a = blockDim.x / BN; 

    const unsigned int b_col = threadIdx.x % BK; 
    const unsigned int b_row = threadIdx.x / BK; 
    const unsigned int stride_b = blockDim.x / BK; 

    float acc[TM * TK] = {0.0};
    float regM[TM] = {0.0};
    float regK[TK] = {0.0};

    for (unsigned int n = 0; n < N; n += BN) {
        for (unsigned int offset = 0; offset < BM; offset += stride_a) {
            As[(offset + a_row) * BN + a_col] = A[(offset + a_row) * N + a_col];
        }
        for (unsigned int offset = 0; offset < BN; offset += stride_b) {
            Bs[(offset + b_row) * BK + b_col] = B[(offset + b_row) * K + b_col]; 
        }
        __syncthreads(); 

        A += BN; 
        B += BN * K; 

        for (unsigned int n_ = 0; n_ < min(N-n, BN); n_++) {
            for (unsigned int i = 0; i < TM; i++) {
                regM[i] = As[(t_row * TM + i) * BN + n_];
            }
            for (unsigned int i = 0; i < TK; i++) {
                regK[i] = Bs[n_ * BK + t_col * TK + i]; 
            }
            for (unsigned int i = 0; i < TM; i++) {
                if (c_row * BM + t_row * TM + i < M) {
                    for (unsigned int j = 0; j < TK; j++) {
                        if (c_col * BK + t_col * TK + j < K) {
                            acc[i * TK + j] += regM[i] * regK[j]; 
                        }
                    }
                }
            }
        }
        __syncthreads(); 
    }

    for (unsigned int i = 0; i < TM; i++) {
        if (c_row * BM + t_row * TM + i < M) {
            for (unsigned int j = 0; j < TK; j++) {
                if (c_col * BK + t_col * TK + j < K) {
                    C[(t_row * TM + i) * K + (t_col * TK + j)] = acc[i * TK + j]; 
                }
            }
        }
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TK>
__global__ void matmul_v6(const float* A, const float* B, float* C, int M, int N, int K) {
    const unsigned int c_row = blockIdx.y;
    const unsigned int c_col = blockIdx.x;

    const int t_col = threadIdx.x % (BK / TK);
    const int t_row = threadIdx.x / (BK / TK);

    __shared__ float As[BM * BN];
    __shared__ float Bs[BN * BK];

    A += c_row * BM * N;
    B += c_col * BK;
    C += c_row * BM * K + c_col * BK;

    // vectorized load indexing (float4)
    const unsigned int a_col = (threadIdx.x % (BN / 4)) * 4;
    const unsigned int a_row = threadIdx.x / (BN / 4);
    const unsigned int stride_a = blockDim.x / (BN / 4);

    const unsigned int b_col = (threadIdx.x % (BK / 4)) * 4;
    const unsigned int b_row = threadIdx.x / (BK / 4);
    const unsigned int stride_b = blockDim.x / (BK / 4);

    float acc[TM * TK] = {0.0};
    float regM[TM] = {0.0};
    float regK[TK] = {0.0};

    for (unsigned int n = 0; n < N; n += BN) {

        // vectorized A -> As
        #pragma unroll
        for (unsigned int offset = 0; offset < BM; offset += stride_a) {
            float4 tmp = *reinterpret_cast<const float4*>(
                &A[(offset + a_row) * N + a_col]);

            *reinterpret_cast<float4*>(
                &As[(offset + a_row) * BN + a_col]) = tmp;
        }

        // vectorized B -> Bs
        #pragma unroll
        for (unsigned int offset = 0; offset < BN; offset += stride_b) {
            float4 tmp = *reinterpret_cast<const float4*>(
                &B[(offset + b_row) * K + b_col]);

            *reinterpret_cast<float4*>(
                &Bs[(offset + b_row) * BK + b_col]) = tmp;
        }

        __syncthreads();

        A += BN;
        B += BN * K;

        for (unsigned int n_ = 0; n_ < min(N - n, BN); n_++) {

            #pragma unroll
            for (unsigned int i = 0; i < TM; i++) {
                regM[i] = As[(t_row * TM + i) * BN + n_];
            }

            #pragma unroll
            for (unsigned int i = 0; i < TK; i++) {
                regK[i] = Bs[n_ * BK + t_col * TK + i];
            }

            #pragma unroll
            for (unsigned int i = 0; i < TM; i++) {
                if (c_row * BM + t_row * TM + i < M) {
                    #pragma unroll
                    for (unsigned int j = 0; j < TK; j++) {
                        if (c_col * BK + t_col * TK + j < K) {
                            acc[i * TK + j] += regM[i] * regK[j];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (unsigned int i = 0; i < TM; i++) {
        if (c_row * BM + t_row * TM + i < M) {
            #pragma unroll
            for (unsigned int j = 0; j < TK; j++) {
                if (c_col * BK + t_col * TK + j < K) {
                    C[(t_row * TM + i) * K + (t_col * TK + j)] =
                        acc[i * TK + j];
                }
            }
        }
    }
}

// from: https://siboehm.com/articles/22/CUDA-MMM
// warptiling

const int WARPSIZE = 32;

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    // sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
    //                 float beta, float *C) {
    sgemmWarptiling(int M, int N, int K, const float *A, const float *B, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
        //   tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
        //   tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
        //   tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
        //   tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          tmp.x = threadResults[i + 0];
          tmp.y = threadResults[i + 1];
          tmp.z = threadResults[i + 2];
          tmp.w = threadResults[i + 3];
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // matmul_v1<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    // const int BLOCK_SIZE = 32; 
    // dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
    // dim3 blocksPerGrid(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(K, BLOCK_SIZE));
    // matmul_v3<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    // const int BM = 64; 
    // const int BN = 8;
    // const int BK = 64; 
    // const int TM = 8;

    // dim3 gridDim(CEIL_DIV(K, BK), CEIL_DIV(M, BM));
    // dim3 blockDim((BM * BK) / TM); 
    // matmul_v4<BM, BN, BK, TM><<<gridDim, blockDim>>>(A, B, C, M, N, K);

    // const int BM = 64; 
    // const int BN = 8;
    // const int BK = 64; 
    // const int TM = 8;
    // const int TN = 8; 

    // dim3 gridDim(CEIL_DIV(K, BK), CEIL_DIV(M, BM));
    // dim3 blockDim((BM * BK) / (TM * TN)); 
    // // matmul_v5<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    // if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
    //     matmul_v6<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    // } else {
    //     matmul_v5<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    // }

    const unsigned int NUM_THREADS = 128; 
    const unsigned int BK = 128;
    const unsigned int BM = 128; 
    const unsigned int BN = 16;
    const unsigned int WK = 64;
    const unsigned int WM = 64;
    const unsigned int WKITER = 4;
    const unsigned int TK = 4;
    const unsigned int TM = 8;

    dim3 gridDim(CEIL_DIV(K, BK), CEIL_DIV(M, BM));

    if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
        dim3 blockDim(NUM_THREADS); 
        sgemmWarptiling<BM, BK, BN, WM, WK, WKITER, TM, TK, NUM_THREADS><<<gridDim, blockDim>>>(M, K, N, A, B, C);
    } else {
        dim3 blockDim((BM * BK) / (TM * TK)); 
        matmul_v5<BM, BN, BK, TM, TK><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }

    cudaDeviceSynchronize();
}
