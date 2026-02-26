#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for Floyd-Warshall computation
// Computes shortest paths through intermediate node k for assigned rows
__global__ void floydWarshall_kernel(unsigned int* dist, size_t numNodes, 
                                     size_t row_start, size_t row_count, size_t k) {
    // Thread indices
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds check
    if (col >= numNodes || row_idx >= row_count) return;
    
    size_t i = row_start + row_idx;
    size_t j = col;
    
    // Row-major indexing: row * numNodes + col
    // Compute shortest path through k
    unsigned int distIJ = dist[i * numNodes + j];
    unsigned int distIK = dist[i * numNodes + k];
    unsigned int distKJ = dist[k * numNodes + j];
    
    unsigned int newDist = distIK + distKJ;
    
    if (newDist < distIJ) {
        dist[i * numNodes + j] = newDist;
    }
}

// Host-callable wrapper for CUDA kernel
extern "C" {
    void floydWarshall_cuda_kernel(unsigned int* dist_dev, unsigned int* dist_row_k_dev, 
                                   unsigned int* dist_col_k_dev, size_t numNodes, 
                                   size_t row_start, size_t row_count, size_t k) {
        // Grid and block dimensions
        dim3 blockSize(32, 32);
        dim3 gridSize((numNodes + blockSize.x - 1) / blockSize.x,
                      (row_count + blockSize.y - 1) / blockSize.y);
        
        // Launch kernel
        floydWarshall_kernel<<<gridSize, blockSize>>>(dist_dev, numNodes, row_start, row_count, k);
        
        // Synchronize device
        cudaDeviceSynchronize();
    }
}
