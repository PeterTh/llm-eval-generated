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
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>

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

// CUDA failures are fatal: clustering always runs on the GPU.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
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

__device__ double pointDistance(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Only neighbors of the seed can ever join its candidate. Build a CSR graph
// without allocating a dense N-by-N distance matrix.
__global__ void countNeighbors(const Point* points, int n, double threshold, int* degrees) {
    int seed = blockIdx.x;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        count += (j == seed || pointDistance(points[seed], points[j]) < threshold);
    __shared__ int counts[128];
    counts[threadIdx.x] = count;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) degrees[seed] = counts[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors) {
    int seed = blockIdx.x;
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == seed || pointDistance(points[seed], points[j]) < threshold) {
            int slot = atomicAdd(&count, 1);
            neighbors[offsets[seed] + slot] = j;
        }
    }
}

__device__ bool better(double a, int ai, double b, int bi) {
    return a < b || (a == b && ai < bi);
}

__global__ void growCandidates(const Point* points, double threshold,
                               const size_t* offsets, const int* neighbors,
                               const int* clustered, int* sizes, int* members,
                               double* scratch, int sharedCapacity) {
    int seed = blockIdx.x;
    if (clustered[seed]) {
        if (threadIdx.x == 0) sizes[seed] = 0;
        return;
    }
    size_t base = offsets[seed];
    int degree = static_cast<int>(offsets[seed + 1] - base);
    // Deleting points outside a cached candidate cannot change any of its
    // greedy choices. Recompute only candidates touched by the last winners.
    int invalid = sizes[seed] == 0;
    for (int k = threadIdx.x; k < sizes[seed]; k += blockDim.x)
        invalid |= clustered[members[base + k]];
    if (!__syncthreads_or(invalid)) return;

    extern __shared__ double sharedMax[];
    double* maxima = degree <= sharedCapacity ? sharedMax : scratch + base;
    for (int k = threadIdx.x; k < degree; k += blockDim.x) {
        int id = neighbors[base + k];
        maxima[k] = clustered[id] || id == seed ? -1.0 : 0.0;
    }
    __shared__ double warpDistances[4];
    __shared__ int warpIds[4];
    __shared__ int next;
    int last = seed, count = 1;
    if (threadIdx.x == 0) members[base] = seed;
    __syncthreads();
    while (true) {
        double best = CUDART_INF;
        int bestId = INT_MAX;
        Point newest = points[last];
        for (int k = threadIdx.x; k < degree; k += blockDim.x) {
            double value = maxima[k];
            if (value < 0.0) continue;
            int id = neighbors[base + k];
            if (id == last) { maxima[k] = -1.0; continue; }
            value = fmax(value, pointDistance(points[id], newest));
            // A rejected point can never become eligible as the cluster grows.
            maxima[k] = value < threshold ? value : -1.0;
            if (value < threshold && better(value, id, best, bestId)) {
                best = value;
                bestId = id;
            }
        }
        for (int delta = 16; delta; delta >>= 1) {
            double other = __shfl_down_sync(0xffffffff, best, delta);
            int otherId = __shfl_down_sync(0xffffffff, bestId, delta);
            if (better(other, otherId, best, bestId)) { best = other; bestId = otherId; }
        }
        if ((threadIdx.x & 31) == 0) {
            warpDistances[threadIdx.x / 32] = best;
            warpIds[threadIdx.x / 32] = bestId;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < 4; ++w)
                if (better(warpDistances[w], warpIds[w], best, bestId)) {
                    best = warpDistances[w]; bestId = warpIds[w];
                }
            next = bestId;
            if (bestId != INT_MAX) members[base + count] = bestId;
        }
        __syncthreads();
        if (next == INT_MAX) break;
        last = next;
        ++count;
    }
    if (threadIdx.x == 0) sizes[seed] = count;
}

__global__ void selectCluster(int n, const int* sizes, const size_t* offsets,
                              const int* members, int* clustered, int* output) {
    // Largest cardinality, then smallest seed, exactly as in the serial loop.
    unsigned long long best = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned long long key = (static_cast<unsigned long long>(sizes[i]) << 32)
                               | static_cast<unsigned int>(n - i);
        if (key > best) best = key;
    }
    __shared__ unsigned long long keys[128];
    keys[threadIdx.x] = best;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (threadIdx.x < stride && keys[threadIdx.x + stride] > keys[threadIdx.x])
            keys[threadIdx.x] = keys[threadIdx.x + stride];
        __syncthreads();
    }
    int seed = n - static_cast<unsigned int>(keys[0]);
    int count = static_cast<int>(keys[0] >> 32);
    if (threadIdx.x == 0) { output[0] = seed; output[1] = count; }
    for (int k = threadIdx.x; k < count; k += blockDim.x) {
        int member = members[offsets[seed] + k];
        clustered[member] = 1;
        output[k + 2] = member;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    cudaCheck(cudaFree(nullptr));
    std::vector<Cluster> clusters;
    if (!n) return clusters;
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> degrees(n), sizes(n), clustered(n), output(size_t(n) + 2);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    countNeighbors<<<n, 128>>>(devicePoints.data, n, threshold, degrees.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> hostDegrees(n);
    cudaCheck(cudaMemcpy(hostDegrees.data(), degrees.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> offsets(size_t(n) + 1, 0);
    int maxDegree = 0;
    for (int i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + hostDegrees[i];
        maxDegree = std::max(maxDegree, hostDegrees[i]);
    }
    DeviceBuffer<size_t> deviceOffsets(offsets.size());
    DeviceBuffer<int> neighbors(offsets.back()), members(offsets.back());
    // Shared memory handles typical sparse neighborhoods; large rows use
    // global scratch so dense inputs do not exceed per-block resource limits.
    const int sharedCapacity = std::min(maxDegree, 4096);
    DeviceBuffer<double> scratch(maxDegree > sharedCapacity ? offsets.back() : 0);
    cudaCheck(cudaMemcpy(deviceOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(sizes.data, 0, n * sizeof(int)));
    cudaCheck(cudaMemset(clustered.data, 0, n * sizeof(int)));
    fillNeighbors<<<n, 128>>>(devicePoints.data, n, threshold, deviceOffsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> result(size_t(n) + 2);
    int remaining = n;
    while (remaining) {
        growCandidates<<<n, 128, sharedCapacity * sizeof(double)>>>(
            devicePoints.data, threshold, deviceOffsets.data, neighbors.data,
            clustered.data, sizes.data, members.data, scratch.data, sharedCapacity);
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, 128>>>(n, sizes.data, deviceOffsets.data, members.data,
                                 clustered.data, output.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(result.data(), output.data, result.size() * sizeof(int), cudaMemcpyDeviceToHost));
        int count = result[1];
        if (count <= 0 || count > remaining) {
            fprintf(stderr, "CUDA clustering produced an invalid cardinality\n");
            std::exit(EXIT_FAILURE);
        }
        clusters.push_back({std::vector<int>(result.begin() + 2, result.begin() + 2 + count), result[0]});
        remaining -= count;
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
    
    // Initialize the CUDA runtime before timing the clustering work.
    cudaCheck(cudaFree(nullptr));

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
