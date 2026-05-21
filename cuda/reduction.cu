#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void reduce(const float* __restrict__ input, float* __restrict__ output, int N) {
    __shared__ float smem[BLOCK_SIZE];
    int base = blockIdx.x * BLOCK_SIZE * NUM_PER_THREAD; 
    int tid = threadIdx.x; 
    // for (int i = base + threadIdx.x; i < base + BLOCK_SIZE * NUM_PER_THREAD; i += BLOCK_SIZE) {
    //     if (i < N) {
    //         smem[threadIdx.x] += input[i]; 
    //     }
    // }
    float sum = 0; 
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        if (tid + base + i * BLOCK_SIZE < N) {
            sum += input[tid + base + i * BLOCK_SIZE];
        }
    }
    smem[tid] = sum; 
    __syncthreads();

    for (int B = BLOCK_SIZE/2; B > 0; B >>= 1) {
        if (tid < B) {
            smem[tid] += smem[tid + B];
        }
        __syncthreads(); 
    }
    if (tid == 0) {
        atomicAdd(output, smem[0]); 
    }
}

const int num_threads = 128;
const int warp_size = 32;
const int coarse_factor = 4; 

__global__ void warp_reduce(const float* __restrict__ input, float* __restrict__ output, int N) {
    __shared__ float sdata[num_threads / warp_size];

    int i = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x; 
    int w_id = threadIdx.x / warp_size;
    int l_id = threadIdx.x % warp_size; 

    float s = 0.0;
    for (int j = 0; j < coarse_factor; j++) {
        int i_ = i + j * num_threads;
        if (i_ < N) s += input[i_];
    }
    for (int b = 16; b > 0; b >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, b); 
    }
    if (l_id == 0) sdata[w_id] = s;
    __syncthreads(); 

    if (w_id == 0) {
        s = (threadIdx.x < num_threads / warp_size) ? sdata[threadIdx.x] : 0.0f;
        for (int b = 16; b > 0; b >>= 1) {
            s += __shfl_down_sync(0xffffffff, s, b);
        }
        if (l_id == 0) {
            atomicAdd(output, s); 
        }
    }
}

__global__ void coop_groups_reduce(const float* __restrict__ input, float* __restrict__ output, int N) {
    // Get the current thread block as a cooperative group
    auto block = cg::this_thread_block();
    __shared__ float smem[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Phase 1: Global load - each thread accumulates its share of input data
    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }

    // Store partial sums in shared memory
    smem[threadIdx.x] = sum;
    // Synchronize all threads in the block using cooperative group sync
    block.sync();

    // Phase 2: Reduce within 32-thread tiles (warps) using tiled partition and shuffles
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    float tile_sum = sum;

    // Perform warp-level reduction using cooperative group shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_sum += tile32.shfl_down(tile_sum, offset);
    }

    // Store warp-reduced results back to shared memory (one per warp)
    if (tile32.thread_rank() == 0) {
        smem[tile32.meta_group_rank()] = tile_sum;
    }
    block.sync();

    // Phase 3: Final reduction - use first warp to reduce all warp results
    if (threadIdx.x < (blockDim.x + 31) / 32) {
        sum = smem[threadIdx.x];
        cg::thread_block_tile<32> final_tile = cg::tiled_partition<32>(block);

        // Reduce across first warp using cooperative group shuffles
        float final_sum = sum;
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += final_tile.shfl_down(final_sum, offset);
        }

        // Only the first thread in the first tile writes to global memory
        if (final_tile.thread_rank() == 0) {
            atomicAdd(output, final_sum);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // constexpr int BLOCK_SIZE = 256;
    // constexpr int NUM_PER_THREAD = 8; 

    // dim3 blockSize(BLOCK_SIZE);
    // dim3 gridSize((N + (BLOCK_SIZE * NUM_PER_THREAD) - 1) / (BLOCK_SIZE * NUM_PER_THREAD));
    // reduce<BLOCK_SIZE, NUM_PER_THREAD><<<gridSize, blockSize>>>(input, output, N); 
    int num_blocks = (N + (coarse_factor * num_threads) - 1) / (coarse_factor * num_threads);
    // warp_reduce<<<num_blocks, num_threads>>>(input, output, N);
    coop_groups_reduce<<<num_blocks, num_threads>>>(input, output, N);
}
