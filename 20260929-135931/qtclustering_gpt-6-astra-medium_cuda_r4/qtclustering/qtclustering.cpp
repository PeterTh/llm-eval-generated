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
#include <cfloat>
#include <cuda_runtime.h>

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

// All CUDA work uses the default stream; kernel boundaries order cluster removal
// before the next round of candidate generation. No CPU clustering fallback.
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

constexpr int BLOCK_SIZE = 128;

__device__ double gpuDistance(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// A seed can only accept points strictly within threshold of itself. Store
// these neighborhoods in CSR form, avoiding a quadratic distance matrix.
__global__ void countNeighbors(const Point* points, int n, double threshold, int* counts) {
    int seed = blockIdx.x;
    int count = 0;
    Point p = points[seed];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        count += i != seed && gpuDistance(p, points[i]) < threshold;
    __shared__ int partial[BLOCK_SIZE];
    partial[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) counts[seed] = partial[0] + 1; // reserve seed slot
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors) {
    int seed = blockIdx.x;
    __shared__ int used;
    if (!threadIdx.x) {
        used = 1;
        neighbors[offsets[seed]] = seed;
    }
    __syncthreads();
    Point p = points[seed];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i != seed && gpuDistance(p, points[i]) < threshold) {
            int slot = atomicAdd(&used, 1);
            neighbors[offsets[seed] + slot] = i;
        }
    }
}

// Lexicographic reductions preserve the sequential first-index tie rules,
// independently of the order in which neighbors were stored.
__device__ void minimum(double& value, int& index) {
    __shared__ double values[BLOCK_SIZE / 32];
    __shared__ int indices[BLOCK_SIZE / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int delta = 16; delta; delta /= 2) {
        double other = __shfl_down_sync(0xffffffff, value, delta);
        int oi = __shfl_down_sync(0xffffffff, index, delta);
        if (other < value || (other == value && oi < index)) {
            value = other; index = oi;
        }
    }
    if (!lane) { values[warp] = value; indices[warp] = index; }
    __syncthreads();
    if (!warp) {
        value = lane < BLOCK_SIZE / 32 ? values[lane] : DBL_MAX;
        index = lane < BLOCK_SIZE / 32 ? indices[lane] : INT_MAX;
        for (int delta = 16; delta; delta /= 2) {
            double other = __shfl_down_sync(0xffffffff, value, delta);
            int oi = __shfl_down_sync(0xffffffff, index, delta);
            if (other < value || (other == value && oi < index)) {
                value = other; index = oi;
            }
        }
        if (!lane) { values[0] = value; indices[0] = index; }
    }
    __syncthreads();
    value = values[0]; index = indices[0];
    // Readers must finish before the next invocation overwrites shared state.
    __syncthreads();
}

__global__ void buildCandidates(const Point* points, double threshold,
                                const size_t* offsets, const int* neighbors,
                                const int* assigned, int* sizes, int* members,
                                double* workspace, bool sharedWorkspace) {
    int seed = blockIdx.x;
    if (assigned[seed]) return;
    size_t begin = offsets[seed], degree = offsets[seed + 1] - begin;
    int oldSize = sizes[seed];
    bool dirty = oldSize == 0;
    for (int i = threadIdx.x; i < oldSize; i += blockDim.x)
        dirty |= assigned[members[begin + i]] != 0;
    // Removing a point that was never selected cannot change this greedy
    // candidate. Reuse its exact membership and order until a member is lost.
    if (!__syncthreads_or(dirty)) return;

    extern __shared__ double sharedMaxima[];
    double* maxima = sharedWorkspace ? sharedMaxima : workspace + begin;
    for (size_t i = threadIdx.x; i < degree; i += blockDim.x)
        maxima[i] = assigned[neighbors[begin + i]] ? DBL_MAX : 0.0;
    int last = seed, count = 1;
    if (!threadIdx.x) members[begin] = seed;
    while (true) {
        double best = DBL_MAX;
        int bestIndex = INT_MAX;
        Point p = points[last];
        for (size_t i = threadIdx.x; i < degree; i += blockDim.x) {
            int candidate = neighbors[begin + i];
            double d = maxima[i];
            if (candidate == last) d = DBL_MAX;
            if (d != DBL_MAX) {
                d = fmax(d, gpuDistance(p, points[candidate]));
                if (!(d < threshold)) d = DBL_MAX;
            }
            maxima[i] = d;
            if (d < best || (d == best && d != DBL_MAX && candidate < bestIndex)) {
                best = d; bestIndex = candidate;
            }
        }
        minimum(best, bestIndex);
        if (bestIndex == INT_MAX) break;
        if (!threadIdx.x) members[begin + count] = bestIndex;
        ++count;
        last = bestIndex;
    }
    if (!threadIdx.x) sizes[seed] = count;
}

__global__ void selectCluster(int n, const size_t* offsets, const int* sizes,
                              const int* members, int* assigned, int* output,
                              int outputOffset, int* result) {
    double score = DBL_MAX;
    int seed = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (!assigned[i] && (-double(sizes[i]) < score ||
            (-double(sizes[i]) == score && i < seed))) {
            score = -double(sizes[i]); seed = i;
        }
    }
    minimum(score, seed);
    int count = sizes[seed];
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int member = members[offsets[seed] + i];
        assigned[member] = 1;
        output[outputOffset + i] = member;
    }
    if (!threadIdx.x) { result[0] = seed; result[1] = count; }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int n = static_cast<int>(points.size());
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> counts(n), assigned(n), sizes(n), output(n), result(2);
    DeviceBuffer<size_t> offsets(n + 1);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(assigned.data, 0, n * sizeof(int)));
    cudaCheck(cudaMemset(sizes.data, 0, n * sizeof(int)));
    countNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, counts.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> hostCounts(n);
    cudaCheck(cudaMemcpy(hostCounts.data(), counts.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> hostOffsets(n + 1, 0);
    int maxDegree = 0;
    for (int i = 0; i < n; ++i) {
        hostOffsets[i + 1] = hostOffsets[i] + hostCounts[i];
        maxDegree = std::max(maxDegree, hostCounts[i]);
    }
    cudaCheck(cudaMemcpy(offsets.data, hostOffsets.data(), (n + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    DeviceBuffer<int> neighbors(hostOffsets[n]), members(hostOffsets[n]);
    // Cap per-block shared memory to keep several seed blocks resident per SM.
    bool sharedWorkspace = maxDegree <= 2048;
    size_t sharedBytes = sharedWorkspace ? maxDegree * sizeof(double) : 0;
    DeviceBuffer<double> workspace(sharedWorkspace ? 0 : hostOffsets[n]);
    fillNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    std::vector<Cluster> clusters;
    int used = 0;
    while (used < n) {
        buildCandidates<<<n, BLOCK_SIZE, sharedBytes>>>(devicePoints.data, threshold,
            offsets.data, neighbors.data, assigned.data, sizes.data, members.data,
            workspace.data, sharedWorkspace);
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, BLOCK_SIZE>>>(n, offsets.data, sizes.data, members.data,
            assigned.data, output.data, used, result.data);
        cudaCheck(cudaGetLastError());
        int winner[2];
        cudaCheck(cudaMemcpy(winner, result.data, sizeof(winner), cudaMemcpyDeviceToHost));
        clusters.push_back({std::vector<int>(winner[1]), winner[0]});
        used += winner[1];
    }
    std::vector<int> hostMembers(n);
    cudaCheck(cudaMemcpy(hostMembers.data(), output.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    used = 0;
    for (auto& cluster : clusters) {
        std::copy_n(hostMembers.begin() + used, cluster.members.size(), cluster.members.begin());
        used += static_cast<int>(cluster.members.size());
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
