#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_DIM 32

__global__ void fw_unified_kernel(
    unsigned int* __restrict__ dist,
    unsigned int* __restrict__ path,
    const size_t N,
    const size_t kb,
    const size_t num_tiles)
{
    const size_t kb_tile = kb / TILE_DIM;
    
    size_t tile_i, tile_j;
    
    if (blockIdx.x == 0) {
        tile_i = kb;
        tile_j = kb;
    } else if (blockIdx.x < 2 * num_tiles - 1) {
        size_t idx = blockIdx.x - 1;
        if (idx < num_tiles - 1) {
            size_t jt = idx;
            if (jt >= kb_tile) jt++;
            tile_i = kb;
            tile_j = jt * TILE_DIM;
        } else {
            size_t it = idx - (num_tiles - 1);
            if (it >= kb_tile) it++;
            tile_i = it * TILE_DIM;
            tile_j = kb;
        }
    } else {
        size_t idx = blockIdx.x - (2 * num_tiles - 1);
        size_t it = idx / (num_tiles - 1);
        size_t jt = idx % (num_tiles - 1);
        if (it >= kb_tile) it++;
        if (jt >= kb_tile) jt++;
        tile_i = it * TILE_DIM;
        tile_j = jt * TILE_DIM;
    }
    
    const size_t i = tile_i + threadIdx.y;
    const size_t j = tile_j + threadIdx.x;

    if (i >= N || j >= N) return;

    __shared__ unsigned int s_ik[TILE_DIM];
    __shared__ unsigned int s_kj[TILE_DIM];

    const size_t k_end = min(kb + TILE_DIM, N);
    for (size_t k = kb; k < k_end; ++k) {
        if (threadIdx.x == 0) s_ik[threadIdx.y] = dist[i * N + k];
        if (threadIdx.y == 0) s_kj[threadIdx.x] = dist[k * N + j];
        __syncthreads();

        const unsigned int newDist = s_ik[threadIdx.y] + s_kj[threadIdx.x];
        const size_t eidx = i * N + j;
        if (newDist < dist[eidx]) {
            dist[eidx] = newDist;
            path[eidx] = (unsigned int)k;
        }
        __syncthreads();
    }
}

int main() {
    size_t numNodes = 512;
    std::vector<unsigned int> dist_row(numNodes * numNodes);
    std::vector<unsigned int> path_row(numNodes * numNodes);
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist_row[i] = i % 100 + 1;
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist_row[i * numNodes + i] = 0;
    }
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        path_row[i] = i % numNodes;
    }
    
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    const size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_dist, matrixBytes));
    CUDA_CHECK(cudaMalloc(&d_path, matrixBytes));

    // Time memory transfer H2D
    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_dist, dist_row.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path_row.data(), matrixBytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto h2d_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("H2D transfer time: %ld ms\n", h2d_time.count());

    const size_t num_tiles = (numNodes + TILE_DIM - 1) / TILE_DIM;
    dim3 tile_block(TILE_DIM, TILE_DIM);
    size_t total_blocks = 1 + 2 * (num_tiles - 1) + (num_tiles - 1) * (num_tiles - 1);

    // Time kernel execution
    start = std::chrono::high_resolution_clock::now();
    for (size_t kb = 0; kb < numNodes; kb += TILE_DIM) {
        fw_unified_kernel<<<total_blocks, tile_block>>>(d_dist, d_path, numNodes, kb, num_tiles);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    auto kernel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Kernel execution time: %ld ms\n", kernel_time.count());

    // Time memory transfer D2H
    start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(dist_row.data(), d_dist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path_row.data(), d_path, matrixBytes, cudaMemcpyDeviceToHost));
    end = std::chrono::high_resolution_clock::now();
    auto d2h_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("D2H transfer time: %ld ms\n", d2h_time.count());

    printf("Total GPU time: %ld ms\n", h2d_time.count() + kernel_time.count() + d2h_time.count());

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));

    return 0;
}
