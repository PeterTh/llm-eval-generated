// CUDA kernels for QT Clustering

#include <cuda_runtime.h>

struct Point {
    double x, y;
};

// CUDA kernel: Compute max distance from each candidate to all cluster members
__global__ void computeMaxDistancesKernel(const Point* points, const int* members,
                                          int member_count, const int* candidates,
                                          int num_candidates, double* results,
                                          double threshold) {
    int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate_idx >= num_candidates) return;
    
    int candidate = candidates[candidate_idx];
    double max_dist = 0.0;
    
    // Compute max distance to all members
    for (int i = 0; i < member_count; ++i) {
        int member = members[i];
        double dx = points[candidate].x - points[member].x;
        double dy = points[candidate].y - points[member].y;
        double dist = sqrt(dx * dx + dy * dy);
        max_dist = fmax(max_dist, dist);
    }
    
    results[candidate_idx] = max_dist;
}

// Host function to launch the kernel
extern "C" void computeMaxDistances_cuda(const Point* d_points, const int* d_members,
                                         int member_count, const int* d_candidates,
                                         int num_candidates, double* d_results,
                                         double threshold) {
    int blockSize = 256;
    int numBlocks = (num_candidates + blockSize - 1) / blockSize;
    
    computeMaxDistancesKernel<<<numBlocks, blockSize>>>(
        d_points, d_members, member_count, d_candidates,
        num_candidates, d_results, threshold);
    
    cudaDeviceSynchronize();
}
