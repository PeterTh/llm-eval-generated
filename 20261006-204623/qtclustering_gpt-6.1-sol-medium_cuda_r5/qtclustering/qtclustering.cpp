// QT Clustering Benchmark - CUDA implementation
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
#include <stdexcept>
#include <cuda_runtime.h>
#include <math_constants.h>

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

// Every candidate is a threshold-neighbor of its seed. Store these sparse
// rows instead of a dense distance matrix; row offsets use 64-bit addressing.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("CUDA allocation size overflow");
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void countNeighbors(const Point* points, int n, double threshold, int* counts) {
    const int seed = blockIdx.x;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        count += (j == seed || gpuDistance(points[seed], points[j]) < threshold);
    __shared__ int partial[256];
    partial[threadIdx.x] = count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[seed] = partial[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x;
    __shared__ int cursor;
    if (threadIdx.x == 0) cursor = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == seed || gpuDistance(points[seed], points[j]) < threshold) {
            const int slot = atomicAdd(&cursor, 1);
            neighbors[offsets[seed] + slot] = j;
        }
    }
}

// Lexicographic minimum: the sequential loop chooses the smallest point
// index whenever distances tie, irrespective of the neighbor row ordering.
__device__ void warpMinimum(double& value, int& index) {
    for (int delta = 16; delta; delta /= 2) {
        const double other = __shfl_down_sync(0xffffffff, value, delta);
        const int otherIndex = __shfl_down_sync(0xffffffff, index, delta);
        if (other < value || (other == value && otherIndex < index)) {
            value = other;
            index = otherIndex;
        }
    }
}

template<int Threads>
__global__ void generateCandidates(const Point* points, const unsigned char* clustered,
                                   double threshold, const size_t* offsets,
                                   const int* neighbors, double* maxima,
                                   int* members, int* cardinalities) {
    const int seed = blockIdx.x;
    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinalities[seed] = 0;
        return;
    }
    const size_t begin = offsets[seed];
    const size_t length = offsets[seed + 1] - begin;
    const int previousSize = cardinalities[seed];
    bool dirty = previousSize == 0;
    for (int i = threadIdx.x; i < previousSize; i += Threads)
        dirty |= clustered[members[begin + i]] != 0;
    // Removing points that were never selected cannot alter the greedy path.
    if (!__syncthreads_or(dirty)) return;

    for (size_t i = threadIdx.x; i < length; i += Threads) {
        const int point = neighbors[begin + i];
        maxima[begin + i] = (point == seed || clustered[point]) ? CUDART_INF : 0.0;
    }
    if (threadIdx.x == 0) members[begin] = seed;
    __syncthreads();
    __shared__ double warpValues[Threads / 32];
    __shared__ int warpIndices[Threads / 32];
    __shared__ int chosen;
    int latest = seed;
    int size = 1;
    while (true) {
        double best = CUDART_INF;
        int bestIndex = INT_MAX;
        const Point newest = points[latest];
        for (size_t i = threadIdx.x; i < length; i += Threads) {
            const int point = neighbors[begin + i];
            double score = maxima[begin + i];
            if (point == latest) score = CUDART_INF;
            if (score < threshold) {
                score = fmax(score, gpuDistance(points[point], newest));
                if (!(score < threshold)) score = CUDART_INF;
            }
            maxima[begin + i] = score;
            if (score < best || (score == best && point < bestIndex)) {
                best = score;
                bestIndex = point;
            }
        }
        warpMinimum(best, bestIndex);
        if constexpr (Threads == 32) {
            latest = __shfl_sync(0xffffffff, best < threshold ? bestIndex : -1, 0);
        } else {
            if ((threadIdx.x & 31) == 0) {
                warpValues[threadIdx.x / 32] = best;
                warpIndices[threadIdx.x / 32] = bestIndex;
            }
            __syncthreads();
            if (threadIdx.x < 32) {
                best = threadIdx.x < Threads / 32 ? warpValues[threadIdx.x] : CUDART_INF;
                bestIndex = threadIdx.x < Threads / 32 ? warpIndices[threadIdx.x] : INT_MAX;
                warpMinimum(best, bestIndex);
                if (threadIdx.x == 0) chosen = best < threshold ? bestIndex : -1;
            }
            __syncthreads();
            latest = chosen;
        }
        if (latest < 0) break;
        if (threadIdx.x == 0) members[begin + size] = latest;
        ++size;
        // Only the owning thread accesses each maximum; members are read on
        // subsequent launches, so no additional barrier is needed here.
    }
    if (threadIdx.x == 0) cardinalities[seed] = size;
}

// Pick the largest cluster, then the lowest seed, and retain the complete
// ordered result on the device. state = {number of emitted points, clusters}.
__global__ void selectCluster(int n, const size_t* offsets, const int* members,
                             const int* cardinalities, unsigned char* clustered,
                             int* output, int* seeds, int* ends, int* state) {
    __shared__ unsigned long long keys[256];
    unsigned long long key = 0;
    for (int seed = threadIdx.x; seed < n; seed += blockDim.x) {
        const unsigned long long candidate =
            (static_cast<unsigned long long>(cardinalities[seed]) << 32) |
            static_cast<unsigned int>(n - seed);
        if (cardinalities[seed] > 0 && candidate > key) key = candidate;
    }
    keys[threadIdx.x] = key;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride && keys[threadIdx.x + stride] > keys[threadIdx.x])
            keys[threadIdx.x] = keys[threadIdx.x + stride];
        __syncthreads();
    }
    const int size = static_cast<int>(keys[0] >> 32);
    if (size <= 1) {
        // The remaining clusters are all singletons, emitted in seed order.
        if (threadIdx.x == 0) {
            for (int seed = 0; seed < n; ++seed) {
                if (!clustered[seed]) {
                    clustered[seed] = 1;
                    output[state[0]++] = seed;
                    seeds[state[1]] = seed;
                    ends[++state[1]] = state[0];
                }
            }
        }
        return;
    }
    const int seed = n - static_cast<int>(static_cast<unsigned int>(keys[0]));
    const int start = state[0];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const int point = members[offsets[seed] + i];
        output[start + i] = point;
        clustered[point] = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        seeds[state[1]] = seed;
        state[0] += size;
        ends[++state[1]] = state[0];
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    if (n == 0) return {};
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> counts(n);
    DeviceBuffer<size_t> offsets(n + size_t(1));
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    countNeighbors<<<n, 256>>>(devicePoints.data, n, threshold, counts.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> hostCounts(n);
    cudaCheck(cudaMemcpy(hostCounts.data(), counts.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> hostOffsets(n + size_t(1), 0);
    for (int i = 0; i < n; ++i) {
        if (hostOffsets[i] > std::numeric_limits<size_t>::max() - hostCounts[i])
            throw std::runtime_error("Neighbor count overflow");
        hostOffsets[i + 1] = hostOffsets[i] + hostCounts[i];
    }
    cudaCheck(cudaMemcpy(offsets.data, hostOffsets.data(), hostOffsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    const size_t entries = hostOffsets.back();
    DeviceBuffer<int> neighbors(entries), members(entries);
    DeviceBuffer<double> maxima(entries);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<int> output(n), seeds(n), ends(n + size_t(1)), state(2);
    cudaCheck(cudaMemset(clustered.data, 0, n));
    cudaCheck(cudaMemset(counts.data, 0, n * sizeof(int)));
    cudaCheck(cudaMemset(state.data, 0, 2 * sizeof(int)));
    cudaCheck(cudaMemset(ends.data, 0, sizeof(int)));
    fillNeighbors<<<n, 256>>>(devicePoints.data, n, threshold, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    int hostState[2] = {0, 0};
    while (hostState[0] < n) {
        if (entries / n <= 128) {
            generateCandidates<32><<<n, 32>>>(devicePoints.data, clustered.data, threshold,
                offsets.data, neighbors.data, maxima.data, members.data, counts.data);
        } else {
            generateCandidates<128><<<n, 128>>>(devicePoints.data, clustered.data, threshold,
                offsets.data, neighbors.data, maxima.data, members.data, counts.data);
        }
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, 256>>>(n, offsets.data, members.data, counts.data, clustered.data,
                                output.data, seeds.data, ends.data, state.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(hostState, state.data, sizeof(hostState), cudaMemcpyDeviceToHost));
    }
    std::vector<int> hostMembers(n), hostSeeds(hostState[1]), hostEnds(hostState[1] + 1);
    cudaCheck(cudaMemcpy(hostMembers.data(), output.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hostSeeds.data(), seeds.data, hostSeeds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hostEnds.data(), ends.data, hostEnds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Cluster> clusters;
    clusters.reserve(hostSeeds.size());
    for (size_t c = 0; c < hostSeeds.size(); ++c)
        clusters.push_back({std::vector<int>(hostMembers.begin() + hostEnds[c],
                                           hostMembers.begin() + hostEnds[c + 1]), hostSeeds[c]});
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
    
    std::vector<Cluster> clusters;
    try {
        clusters = qtClustering(points, threshold);
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA clustering failed: %s\n", error.what());
        return 1;
    }
    
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
