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

// Morton encode
__device__ __forceinline__
unsigned int part1by1(unsigned int x)
{
    x &= 0x0000ffff;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

__device__ __forceinline__
unsigned int morton2D(unsigned int x, unsigned int y)
{
    return part1by1(x) | (part1by1(y) << 1);
}

// Morton decode
__device__ __forceinline__
unsigned int compact1by1(unsigned int x)
{
    x &= 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

__device__ __forceinline__
void mortonDecode2D(unsigned int code,
                    unsigned int &x,
                    unsigned int &y)
{
    x = compact1by1(code);
    y = compact1by1(code >> 1);
}

template<const int BM, const int BN, const int BK, const int TM, const int TK>
__global__ void matmul_v6_morton(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // const unsigned int c_row = blockIdx.y;
    // const unsigned int c_col = blockIdx.x; 
    unsigned int L = blockIdx.y * gridDim.x + blockIdx.x; 
    unsigned int c_row, c_col;
    mortonDecode2D(L, c_row, c_col); 

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

// matmul_v7_morton: fixes the 4-way bank conflict in As compute reads.
// For half (2 bytes/element), each bank holds 2 elements, so the row stride
// must shift by a non-zero number of banks. With BN=8 and TM=8:
//   t_row * TM * BN = t_row * 64 half = t_row * 128 bytes → t_row * 32 banks ≡ 0.
// Padding by +2 half elements per row (4 bytes = 1 bank) gives:
//   t_row * TM * (BN+2) = t_row * 80 half = t_row * 160 bytes → t_row * 40 banks.
//   40 mod 32 = 8, so each successive t_row shifts by 8 banks → all 4 t_row
//   values land on distinct banks (0, 8, 16, 24 relative to i and n_).
template<const int BM, const int BN, const int BK, const int TM, const int TK>
__global__ void matmul_v7_morton(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int BN_PAD = BN + 2;
    constexpr int BK_PAD = BK + 2;

    unsigned int L = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int c_row, c_col;
    mortonDecode2D(L, c_row, c_col);

    const int t_col = threadIdx.x % (BK / TK);
    const int t_row = threadIdx.x / (BK / TK);

    __shared__ half As[BM * BN_PAD];
    __shared__ half Bs[BN * BK_PAD];

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
            As[(offset + a_row) * BN_PAD + a_col] = A[(offset + a_row) * N + a_col];
        }
        for (unsigned int offset = 0; offset < BN; offset += stride_b) {
            Bs[(offset + b_row) * BK_PAD + b_col] = B[(offset + b_row) * K + b_col];
        }
        __syncthreads();

        A += BN;
        B += BN * K;

        for (unsigned int n_ = 0; n_ < min(N - n, BN); n_++) {
            for (unsigned int i = 0; i < TM; i++) {
                regM[i] = __half2float(As[(t_row * TM + i) * BN_PAD + n_]);
            }
            for (unsigned int i = 0; i < TK; i++) {
                regK[i] = __half2float(Bs[n_ * BK_PAD + t_col * TK + i]);
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
                    C[(t_row * TM + i) * K + (t_col * TK + j)] = __float2half(
                        alpha * acc[i * TK + j] +
                        beta * __half2float(C[(t_row * TM + i) * K + (t_col * TK + j)]));
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
    matmul_v7_morton<BM, BN, BK, TM, TK><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}
