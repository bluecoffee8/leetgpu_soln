#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Returns the number of elements from A that appear in the first k elements of the merged output.
// Binary search over the "merge path diagonal": finds i s.t. A[0..i-1] and B[0..k-i-1]
// form the first k merged elements, i.e. A[i-1] <= B[k-i] and B[k-i-1] <= A[i].
__device__ int co_rank(int k, const float* A, int M, const float* B, int N) {
    int i_lo = max(0, k - N);
    int i_hi = min(k, M);
    while (i_lo < i_hi) {
        int i = i_lo + (i_hi - i_lo) / 2;
        int j = k - i;
        if (i < M && j > 0 && A[i] < B[j - 1])
            i_lo = i + 1;
        else
            i_hi = i;
    }
    return i_lo;
}

__global__ void merge_kernel(const float* A, const float* B, float* C, int M, int N) {
    extern __shared__ float smem[];
    float* sA = smem;
    float* sB = smem + BLOCK_SIZE;

    int total      = M + N;
    int chunk_start = blockIdx.x * BLOCK_SIZE;
    int chunk_end   = min(chunk_start + BLOCK_SIZE, total);
    int chunk_size  = chunk_end - chunk_start;

    // Find where this block's output chunk begins in A and B.
    int a_start = co_rank(chunk_start, A, M, B, N);
    int a_end   = co_rank(chunk_end,   A, M, B, N);
    int b_start = chunk_start - a_start;
    int a_len   = a_end - a_start;
    int b_len   = chunk_size - a_len;

    // Cooperatively load A and B sections into shared memory.
    for (int i = threadIdx.x; i < a_len; i += blockDim.x)
        sA[i] = A[a_start + i];
    for (int i = threadIdx.x; i < b_len; i += blockDim.x)
        sB[i] = B[b_start + i];
    __syncthreads();

    // Each thread resolves one output element via a second co-rank on shared memory.
    int tid = threadIdx.x;
    if (tid < chunk_size) {
        int i = co_rank(tid, sA, a_len, sB, b_len);
        int j = tid - i;
        C[chunk_start + tid] = (i < a_len && (j >= b_len || sA[i] <= sB[j])) ? sA[i] : sB[j];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int total      = M + N;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem    = 2 * BLOCK_SIZE * sizeof(float);
    merge_kernel<<<num_blocks, BLOCK_SIZE, smem>>>(A, B, C, M, N);
}
