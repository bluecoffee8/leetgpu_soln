#include <cuda_runtime.h>

#define FETCH_CFLOAT4(p) (reinterpret_cast<const float4*>(&(p))[0])
#define FETCH_FLOAT4(p) (reinterpret_cast<float4*>(&(p))[0])

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (col < cols && row < rows) {
        output[col * rows + row] = input[row * cols + col]; 
    }
}

template<int BLOCK_SIZE>
__global__ void matrix_transpose_kernel_v1(const float* input, float* output, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y; 

    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE]; 

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;
    
    if (x < N && y < M) {
        sdata[ty][tx] = input[y * N + x];
    }
    __syncthreads();

    x = by * BLOCK_SIZE + tx;
    y = bx * BLOCK_SIZE + ty;
    if (y < N && x < M) {
        output[y * M + x] = sdata[tx][ty];
    }
}

template<int BLOCK_SIZE>
__global__ void matrix_transpose_kernel_v2(const float* input, float* output, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y; 

    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE+1]; 

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;
    
    if (x < N && y < M) {
        sdata[ty][tx] = input[y * N + x];
    }
    __syncthreads();

    x = by * BLOCK_SIZE + tx;
    y = bx * BLOCK_SIZE + ty;
    if (y < N && x < M) {
        output[y * M + x] = sdata[tx][ty];
    }
}

template<int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void matrix_transpose_kernel_v3(const float* input, float* output, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y; 

    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE]; 

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;

    constexpr int ROW_STRIDE = BLOCK_SIZE / NUM_PER_THREAD;
    
    if (x < N) {
        // sdata[ty][tx] = input[y * N + x];
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
            if (y + y_off < M) {
                sdata[ty + y_off][tx] = input[(y + y_off) * N + x]; 
            }
        }
    }
    __syncthreads();

    x = by * BLOCK_SIZE + tx;
    y = bx * BLOCK_SIZE + ty;
    if (x < M) {
        for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
            if (y + y_off < N) {
                output[(y + y_off) * M + x] = sdata[tx][ty + y_off]; 
            }
        }
        // output[y * M + x] = sdata[tx][ty];
    }
}

template<int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void matrix_transpose_kernel_v4(const float* input, float* output, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y; 

    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE]; 

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;

    constexpr int ROW_STRIDE = BLOCK_SIZE / NUM_PER_THREAD;
    
    if (x < N) {
        if (y + BLOCK_SIZE <= M) {
            #pragma unroll
            for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
                sdata[ty + y_off][tx] = input[(y + y_off) * N + x]; 
            }
        } else {
            for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
                if (y + y_off < M) {
                    sdata[ty + y_off][tx] = input[(y + y_off) * N + x]; 
                }
            }
        }
    }
    __syncthreads();

    x = by * BLOCK_SIZE + tx;
    y = bx * BLOCK_SIZE + ty;
    if (x < M) {
        if (y + BLOCK_SIZE <= N) {
            #pragma unroll
            for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
                output[(y + y_off) * M + x] = sdata[tx][ty + y_off]; 
            }
        } else {
            for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
                if (y + y_off < N) {
                    output[(y + y_off) * M + x] = sdata[tx][ty + y_off]; 
                }
            }
        }
    }
}

// see https://zhuanlan.zhihu.com/p/692010210

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int M, int N) {
    // constexpr int BLOCK_SIZE = 16;
    // dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    // matrix_transpose_kernel_v2<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N);
    constexpr int BLOCK_SIZE = 32; 
    constexpr int NUM_PER_THREAD = 4;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE / NUM_PER_THREAD);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_transpose_kernel_v4<BLOCK_SIZE, NUM_PER_THREAD><<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N); 

    cudaDeviceSynchronize();
}
