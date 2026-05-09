#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) ((a) + (b) - 1) / (b)

template<const int BM, const int BN, const int BK, const int TM, const int TK>
__global__ void matmul_v5(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const unsigned int c_row = blockIdx.y;
    const unsigned int c_col = blockIdx.x; 


    const int t_col = threadIdx.x % (BK/TK);
    const int t_row = threadIdx.x / (BK/TK); 

    __shared__ half As[BM * BN];
    __shared__ half Bs[BN * BK];

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
                regM[i] = __half2float(As[(t_row * TM + i) * BN + n_]);
            }
            for (unsigned int i = 0; i < TK; i++) {
                regK[i] = __half2float(Bs[n_ * BK + t_col * TK + i]); 
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
                    C[(t_row * TM + i) * K + (t_col * TK + j)] = __float2half(alpha * acc[i * TK + j] + beta * __half2float(C[(t_row * TM + i) * K + (t_col * TK + j)])); 
                }
            }
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int K, int N, float alpha,
                      float beta) {
    const int BM = 64; 
    const int BN = 8;
    const int BK = 64; 
    const int TM = 8;
    const int TK = 8; 

    dim3 gridDim(CEIL_DIV(K, BK), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BK) / (TM * TK)); 
    matmul_v5<BM, BN, BK, TM, TK><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}
