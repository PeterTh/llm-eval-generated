// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <climits>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
        // Make sure we don't make more points than we need
        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }
            
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Keep all greedy decisions on the GPU. A block owns one seed and its
// threshold-neighbor list; threads cooperate to select each successive member.
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

constexpr int BLOCK = 128;

// Include the seed in the allocation, even if the threshold is NaN (the
// original algorithm then produces singleton clusters).
__global__ void neighborCounts(const Point* points, int n, double threshold,
                               size_t* offsets) {
    const int seed = blockIdx.x;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        count += j == seed || gpuDistance(points[seed], points[j]) < threshold;
    __shared__ int sums[BLOCK];
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) {
        offsets[seed] = sums[0];
        if (!seed) offsets[n] = 0;
    }
}

__global__ void buildNeighbors(const Point* points, int n, double threshold,
                               const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x;
    __shared__ int used;
    if (!threadIdx.x) used = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == seed || gpuDistance(points[seed], points[j]) < threshold) {
            const int slot = atomicAdd(&used, 1);
            neighbors[offsets[seed] + slot] = j;
        }
    }
}

// Lexicographic reduction makes results independent of neighbor storage order,
// GPU scheduling, and reduction order, including exact distance ties.
__device__ int closest(double value, int index, double* warpValues, int* warpIds) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int shift = 16; shift; shift /= 2) {
        const double other = __shfl_down_sync(0xffffffff, value, shift);
        const int id = __shfl_down_sync(0xffffffff, index, shift);
        if (other < value || (other == value && id < index)) {
            value = other;
            index = id;
        }
    }
    if (!lane) { warpValues[warp] = value; warpIds[warp] = index; }
    __syncthreads();
    if (!warp) {
        value = lane < BLOCK / 32 ? warpValues[lane] : CUDART_INF;
        index = lane < BLOCK / 32 ? warpIds[lane] : INT_MAX;
        for (int shift = 16; shift; shift /= 2) {
            const double other = __shfl_down_sync(0xffffffff, value, shift);
            const int id = __shfl_down_sync(0xffffffff, index, shift);
            if (other < value || (other == value && id < index)) {
                value = other;
                index = id;
            }
        }
        if (!lane) warpIds[0] = index;
    }
    __syncthreads();
    return warpIds[0];
}

__global__ void candidates(const Point* points, int n, double threshold,
                           const size_t* offsets, const int* neighbors,
                           const unsigned char* assigned, double* maxima,
                           int* members, int* sizes) {
    const int seed = blockIdx.x;
    if (assigned[seed]) { if (!threadIdx.x) sizes[seed] = 0; return; }
    const size_t begin = offsets[seed], end = offsets[seed + 1];
    __shared__ int dirty;
    __shared__ double warpValues[BLOCK / 32];
    __shared__ int warpIds[BLOCK / 32];
    if (!threadIdx.x) dirty = sizes[seed] == 0;
    __syncthreads();
    // If none of the previous greedy choices was removed, the complete
    // sequence remains valid and optimal for this seed. Reuse it exactly.
    for (int j = threadIdx.x; j < sizes[seed]; j += BLOCK)
        if (assigned[members[begin + j]]) atomicExch(&dirty, 1);
    __syncthreads();
    if (!dirty) return;
    for (size_t j = begin + threadIdx.x; j < end; j += BLOCK)
        maxima[j] = assigned[neighbors[j]] || neighbors[j] == seed ? CUDART_INF : 0.0;
    if (!threadIdx.x) members[begin] = seed;
    int last = seed, count = 1;
    while (count < n) {
        double best = CUDART_INF;
        int bestId = INT_MAX;
        const Point added = points[last];
        for (size_t j = begin + threadIdx.x; j < end; j += BLOCK) {
            double value = maxima[j];
            const int id = neighbors[j];
            if (id == last) value = CUDART_INF;
            if (value != CUDART_INF) {
                value = fmax(value, gpuDistance(points[id], added));
                if (!(value < threshold)) value = CUDART_INF;
            }
            maxima[j] = value;
            if (value < best || (value == best && value != CUDART_INF && id < bestId)) {
                best = value;
                bestId = id;
            }
        }
        last = closest(best, bestId, warpValues, warpIds);
        if (last == INT_MAX) break;
        if (!threadIdx.x) members[begin + count] = last;
        ++count;
        // All threads must read the reduction result before the next reduction.
        __syncthreads();
    }
    if (!threadIdx.x) sizes[seed] = count;
}

__global__ void selectCluster(int n, const size_t* offsets, const int* members,
                              const int* sizes, unsigned char* assigned,
                              int* output, int outputOffset, int* result) {
    __shared__ double warpValues[BLOCK / 32];
    __shared__ int warpIds[BLOCK / 32];
    int bestSize = 0, seed = INT_MAX;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        if (sizes[i] > bestSize) { bestSize = sizes[i]; seed = i; }
    }
    const int winner = closest(-double(bestSize), seed, warpValues, warpIds);
    const int count = sizes[winner];
    for (int i = threadIdx.x; i < count; i += BLOCK) {
        const int member = members[offsets[winner] + i];
        assigned[member] = 1;
        output[outputOffset + i] = member;
    }
    if (!threadIdx.x) { result[0] = winner; result[1] = count; }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    cudaCheck(cudaFree(nullptr)); // CUDA is required; there is no CPU fallback.
    if (!n) return {};
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<size_t> offsets(size_t(n) + 1);
    DeviceBuffer<unsigned char> assigned(n);
    DeviceBuffer<int> sizes(n), output(n), result(2);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(assigned.data, 0, n));
    cudaCheck(cudaMemset(sizes.data, 0, n * sizeof(int)));
    neighborCounts<<<n, BLOCK>>>(devicePoints.data, n, threshold, offsets.data);
    cudaCheck(cudaGetLastError());
    thrust::device_ptr<size_t> first(offsets.data);
    thrust::exclusive_scan(first, first + size_t(n) + 1, first);
    size_t entries = 0;
    cudaCheck(cudaMemcpy(&entries, offsets.data + n, sizeof(size_t), cudaMemcpyDeviceToHost));
    DeviceBuffer<int> neighbors(entries), members(entries);
    DeviceBuffer<double> maxima(entries);
    buildNeighbors<<<n, BLOCK>>>(devicePoints.data, n, threshold, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    std::vector<Cluster> clusters;
    int total = 0;
    while (total < n) {
        candidates<<<n, BLOCK>>>(devicePoints.data, n, threshold, offsets.data,
            neighbors.data, assigned.data, maxima.data, members.data, sizes.data);
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, BLOCK>>>(n, offsets.data, members.data, sizes.data,
            assigned.data, output.data, total, result.data);
        cudaCheck(cudaGetLastError());
        int selected[2];
        cudaCheck(cudaMemcpy(selected, result.data, sizeof(selected), cudaMemcpyDeviceToHost));
        clusters.push_back({std::vector<int>(selected[1]), selected[0]});
        total += selected[1];
    }
    std::vector<int> ordered(n);
    cudaCheck(cudaMemcpy(ordered.data(), output.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    size_t position = 0;
    for (auto& cluster : clusters) {
        std::copy_n(ordered.begin() + position, cluster.members.size(), cluster.members.begin());
        position += cluster.members.size();
    }
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        // Check diameter (max distance between any two points)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    
    // Count clustered points
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }
    
    const double avg_cluster_size = clusters.empty() ? 0.0 : 
        static_cast<double>(total_clustered) / clusters.size();
    
    printf("Points clustered: %d / %d (%.1f%%)\n", 
           total_clustered, num_points, 
           100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);
    
    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", 
           clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults) {
        // Serialize cluster membership for hashing
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                membership[clusters[c].members[i]] = static_cast<int>(c);
            }
        }
        for (int m : membership) {
            membershipData.push_back(static_cast<double>(m));
        }
        print_results(membershipData, "ClusterMembership");
    }
    
    // Validation
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
