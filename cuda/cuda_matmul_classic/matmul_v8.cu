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

#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<const int BLOCK_SIZE, const int T>
__global__ __launch_bounds__(64) void matmul_vectorized(float* A, float* B, float* C, int M, int N, int K) {
    int bm = blockIdx.y; 
    int bk = blockIdx.x;

    const int block_size = BLOCK_SIZE / T; 
    const int thread_num = block_size * block_size; 

    int tm = (threadIdx.x / block_size) * T;
    int tk = (threadIdx.x % block_size) * T; 

    __shared__ float As[2][BLOCK_SIZE * BLOCK_SIZE], Bs[2][BLOCK_SIZE * BLOCK_SIZE]; 

    A += bm * BLOCK_SIZE * N; 
    B += bk * BLOCK_SIZE; 
    C += bm * BLOCK_SIZE * K + bk * BLOCK_SIZE; 

    const int ldg_a_num = BLOCK_SIZE * BLOCK_SIZE / (4 * thread_num);
    const int ldg_b_num = BLOCK_SIZE * BLOCK_SIZE / (4 * thread_num);

    int a_m = threadIdx.x / (BLOCK_SIZE / 4);
    int a_n = (threadIdx.x % (BLOCK_SIZE / 4)) * 4;
    int a_stride = BLOCK_SIZE / ldg_a_num;

    int b_n = threadIdx.x / (BLOCK_SIZE / 4);
    int b_k = (threadIdx.x % (BLOCK_SIZE / 4)) * 4;
    int b_stride = BLOCK_SIZE / ldg_b_num; 

    float ldg_a_reg[4 * ldg_a_num] = {0.0f};
    float ldg_b_reg[4 * ldg_b_num] = {0.0f};

    float tmp[T][T] = {0.0f}; 
    float a_frag[2][T];
    float b_frag[2][T];

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i += a_stride) {
        int ldg_index = (i / a_stride) * 4; 
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_m + i, a_n, N)]);
        As[0][OFFSET(a_n, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_n + 1, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_n + 2, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_n + 3, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index + 3];
    }
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i += b_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_n + i, b_k, BLOCK_SIZE)]) = FETCH_FLOAT4(B[OFFSET(b_n + i, b_k, K)]);
    }
    __syncthreads(); 

    #pragma unroll 
    for (int j = 0; j < T; j += 4) {
        FETCH_FLOAT4(a_frag[0][j]) = FETCH_FLOAT4(As[0][OFFSET(0, tm + j, BLOCK_SIZE)]);
    }
    #pragma unroll
    for (int l = 0; l < T; l += 4) {
        FETCH_FLOAT4(b_frag[0][l]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tk + l, BLOCK_SIZE)]);
    }

    int write_index = 1, load_index = 0;
    int n = 0; 
    do {
        n += BLOCK_SIZE;

        if (n < N) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i += a_stride) {
                int ldg_index = (i / a_stride) * 4; 
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_m + i, a_n + n, N)]);
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i += b_stride) {
                int ldg_index = (i / b_stride) * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(b_n + i + n, b_k, K)]);
            }
        }

        load_index = write_index ^ 1;

        #pragma unroll 
        for (int i = 0; i < BLOCK_SIZE-1; i++) {
            if (n + i + 1 - BLOCK_SIZE < N) {
                #pragma unroll 
                for (int j = 0; j < T; j += 4) {
                    FETCH_FLOAT4(a_frag[(i + 1) % 2][j]) = FETCH_FLOAT4(As[load_index][OFFSET(i + 1, tm + j, BLOCK_SIZE)]);
                }
                #pragma unroll
                for (int l = 0; l < T; l += 4) {
                    FETCH_FLOAT4(b_frag[(i + 1) % 2][l]) = FETCH_FLOAT4(Bs[load_index][OFFSET(i + 1, tk + l, BLOCK_SIZE)]);
                }
            }
            if (n + i - BLOCK_SIZE < N) {
                #pragma unroll 
                for (int j = 0; j < T; j++) {
                    for (int l = 0; l < T; l++) {
                        tmp[j][l] += a_frag[i % 2][j] * b_frag[i % 2][l];
                    }
                }
            }
        }

        if (n < N) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i += a_stride) {
                int ldg_index = (i / a_stride) * 4; 
                As[write_index][OFFSET(a_n, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_n + 1, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_n + 2, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_n + 3, a_m + i, BLOCK_SIZE)] = ldg_a_reg[ldg_index + 3];
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; i += b_stride) {
                int ldg_index = (i / b_stride) * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_n + i, b_k, BLOCK_SIZE)]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }

            __syncthreads(); 

            #pragma unroll 
            for (int j = 0; j < T; j += 4) {
                FETCH_FLOAT4(a_frag[0][j]) = FETCH_FLOAT4(As[write_index][OFFSET(0, tm + j, BLOCK_SIZE)]);
            }
            #pragma unroll
            for (int l = 0; l < T; l += 4) {
                FETCH_FLOAT4(b_frag[0][l]) = FETCH_FLOAT4(Bs[write_index][OFFSET(0, tk + l, BLOCK_SIZE)]);
            }

            write_index ^= 1;
        }

        if (n - 1 < N) {
            #pragma unroll 
            for (int j = 0; j < T; j++) {
                for (int l = 0; l < T; l++) {
                    tmp[j][l] += a_frag[(BLOCK_SIZE-1) % 2][j] * b_frag[(BLOCK_SIZE-1) % 2][l];
                }
            }
        }
    } while (n < N);

    #pragma unroll 
    for (int j = 0; j < min(T, M - (bm * BLOCK_SIZE + tm)); j++) {
        for (int l = 0; l < min(T, K - (bk * BLOCK_SIZE + tk)); l += 4) {
            if (l + 3 < min(T, K - (bk * BLOCK_SIZE + tk))) {
                FETCH_FLOAT4(C[OFFSET(tm + j, tk + l, K)]) = FETCH_FLOAT4(tmp[j][l]);
            } else {
                for (int l_ = l; l_ < T && bk * BLOCK_SIZE + tk + l_ < K; l_++) {
                    C[(tm + j) * K + (tk + l_)] = tmp[j][l_]; 
                }
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* A, float* B, float* C, int M, int N, int K) {
    if (M % 4 == 0 && N % 4 == 0 && K % 4 == 0) {
        const int BLOCK_SIZE = 32, T = 4; 
        dim3 threadsPerBlock((BLOCK_SIZE / T) * (BLOCK_SIZE / T));
        dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_vectorized<BLOCK_SIZE, T><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    } else {
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    }
}
