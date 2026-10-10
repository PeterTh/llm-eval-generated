// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
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

// All clustering work runs on CUDA. Fail explicitly rather than falling back to CPU.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data),
                             std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK_SIZE = 128;
struct Choice { double value; int index; };

__device__ Choice better(Choice a, Choice b) {
    return (b.value < a.value || (b.value == a.value && b.index < a.index)) ? b : a;
}

// A deterministic reduction: equal distances/cardinalities choose the lowest index.
__device__ Choice blockMinimum(Choice v) {
    __shared__ double values[BLOCK_SIZE / 32];
    __shared__ int indices[BLOCK_SIZE / 32];
    __shared__ Choice result;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    for (int delta = 16; delta; delta /= 2) {
        Choice other{__shfl_down_sync(0xffffffff, v.value, delta),
                     __shfl_down_sync(0xffffffff, v.index, delta)};
        v = better(v, other);
    }
    if (lane == 0) { values[warp] = v.value; indices[warp] = v.index; }
    __syncthreads();
    if (warp == 0) {
        v = lane < BLOCK_SIZE / 32 ? Choice{values[lane], indices[lane]}
                                  : Choice{DBL_MAX, INT_MAX};
        for (int delta = 16; delta; delta /= 2) {
            Choice other{__shfl_down_sync(0xffffffff, v.value, delta),
                         __shfl_down_sync(0xffffffff, v.index, delta)};
            v = better(v, other);
        }
        if (lane == 0) result = v;
    }
    __syncthreads();
    return result;
}

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Only neighbors of the seed can ever enter its cluster. Store them in ascending
// point order in a compact adjacency list instead of allocating an N x N matrix.
__global__ void countNeighbors(const Point* points, int n, double threshold, int* counts) {
    const int seed = blockIdx.x;
    int count = 0;
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE)
        count += i != seed && pointDistance(points[seed], points[i]) < threshold;
    __shared__ int sums[BLOCK_SIZE];
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int step = BLOCK_SIZE / 2; step; step /= 2) {
        if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[seed] = sums[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors, double* initial) {
    const int seed = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ int warpCounts[BLOCK_SIZE / 32];
    size_t base = offsets[seed];
    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int i = tile + threadIdx.x;
        const double d = i < n ? pointDistance(points[seed], points[i]) : DBL_MAX;
        const bool include = i < n && i != seed && d < threshold;
        const unsigned mask = __ballot_sync(0xffffffff, include);
        if (lane == 0) warpCounts[warp] = __popc(mask);
        __syncthreads();
        int prefix = 0, total = 0;
        for (int w = 0; w < BLOCK_SIZE / 32; ++w) {
            if (w < warp) prefix += warpCounts[w];
            total += warpCounts[w];
        }
        if (include) {
            const size_t pos = base + prefix + __popc(mask & ((1u << lane) - 1));
            neighbors[pos] = i;
            initial[pos] = d;
        }
        base += total;
        __syncthreads();
    }
}

__global__ void candidateClusters(const Point* points, int n, double threshold,
                                  const size_t* offsets, const int* neighbors,
                                  const double* initial, double* maxima,
                                  const unsigned char* clustered, int* cardinalities,
                                  int* members, bool sharedScratch) {
    extern __shared__ double localMaxima[];
    const int seed = blockIdx.x;
    if (clustered[seed]) return;
    const size_t begin = offsets[seed], end = offsets[seed + 1];
    double* scores = sharedScratch ? localMaxima : maxima + begin;
    // Each seed gets room for its neighbors plus the seed itself.
    int* output = members + begin + seed;
    const int oldCount = cardinalities[seed];
    if (oldCount > 0) {
        int invalid = 0;
        for (int i = threadIdx.x; i < oldCount; i += BLOCK_SIZE)
            invalid |= clustered[output[i]];
        if (!__syncthreads_or(invalid)) return;
    }
    for (size_t i = begin + threadIdx.x; i < end; i += BLOCK_SIZE)
        scores[i - begin] = clustered[neighbors[i]] ? DBL_MAX : initial[i];
    if (threadIdx.x == 0) output[0] = seed;
    __syncthreads();
    int count = 1;
    while (count < n) {
        Choice best{DBL_MAX, INT_MAX};
        for (size_t i = begin + threadIdx.x; i < end; i += BLOCK_SIZE)
            if (scores[i - begin] != DBL_MAX)
                best = better(best, Choice{scores[i - begin], neighbors[i]});
        best = blockMinimum(best);
        if (best.index == INT_MAX || !(best.value < threshold)) break;
        const int selected = best.index;
        if (threadIdx.x == 0) output[count] = selected;
        ++count;
        // The maximum is monotonic. Candidates rejected by the threshold never
        // become eligible again, and selected points are removed with a sentinel.
        for (size_t i = begin + threadIdx.x; i < end; i += BLOCK_SIZE) {
            double value = scores[i - begin];
            if (neighbors[i] == selected) value = DBL_MAX;
            else if (value != DBL_MAX && value < threshold) {
                value = fmax(value, pointDistance(points[neighbors[i]], points[selected]));
                if (!(value < threshold)) value = DBL_MAX;
            }
            scores[i - begin] = value;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) cardinalities[seed] = count;
}

// state = {remaining points, number of output clusters, output membership count}.
// Keep the complete result on the device; only a completion counter is read per round.
__global__ void selectCluster(int n, const size_t* offsets, const int* cardinalities,
                              const int* candidates, unsigned char* clustered,
                              int* state, int* seeds, int* boundaries, int* members) {
    Choice best{DBL_MAX, INT_MAX};
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE)
        if (!clustered[i]) best = better(best, Choice{-double(cardinalities[i]), i});
    best = blockMinimum(best);
    const int seed = best.index, count = int(-best.value);
    if (count == 1) {
        // All remaining clusters are singletons, in ascending seed order.
        if (threadIdx.x == 0) {
            for (int i = 0; i < n; ++i) if (!clustered[i]) {
                seeds[state[1]] = i;
                boundaries[state[1]++] = state[2];
                members[state[2]++] = i;
                clustered[i] = 1;
            }
            state[0] = 0;
            boundaries[state[1]] = state[2];
        }
        return;
    }
    const int out = state[2], cluster = state[1];
    const int* selected = candidates + offsets[seed] + seed;
    for (int i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        const int point = selected[i];
        members[out + i] = point;
        clustered[point] = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        seeds[cluster] = seed;
        boundaries[cluster] = out;
        boundaries[cluster + 1] = out + count;
        state[0] -= count;
        state[1] = cluster + 1;
        state[2] = out + count;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> counts(n), cardinalities(n), state(3), seeds(n), boundaries(n + size_t(1)), result(n);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<size_t> deviceOffsets(n + size_t(1));
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    countNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, counts.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> hostCounts(n);
    cudaCheck(cudaMemcpy(hostCounts.data(), counts.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> offsets(n + size_t(1), 0);
    for (int i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + hostCounts[i];
    const size_t edges = offsets.back();
    const int maxDegree = *std::max_element(hostCounts.begin(), hostCounts.end());
    const bool sharedScratch = maxDegree <= 512;
    const size_t scratchBytes = sharedScratch ? maxDegree * sizeof(double) : 0;
    DeviceBuffer<int> neighbors(edges), candidates(edges + n);
    DeviceBuffer<double> initial(edges), maxima(sharedScratch ? 0 : edges);
    cudaCheck(cudaMemcpy(deviceOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    fillNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, deviceOffsets.data, neighbors.data, initial.data);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemset(clustered.data, 0, n));
    cudaCheck(cudaMemset(cardinalities.data, 0, n * sizeof(int)));
    int hostState[3] = {n, 0, 0};
    cudaCheck(cudaMemcpy(state.data, hostState, sizeof(hostState), cudaMemcpyHostToDevice));
    while (hostState[0] > 0) {
        candidateClusters<<<n, BLOCK_SIZE, scratchBytes>>>(devicePoints.data, n, threshold, deviceOffsets.data,
            neighbors.data, initial.data, maxima.data, clustered.data, cardinalities.data, candidates.data, sharedScratch);
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, BLOCK_SIZE>>>(n, deviceOffsets.data, cardinalities.data, candidates.data,
            clustered.data, state.data, seeds.data, boundaries.data, result.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(hostState, state.data, sizeof(hostState), cudaMemcpyDeviceToHost));
    }
    std::vector<int> hostSeeds(hostState[1]), hostBoundaries(hostState[1] + 1), hostMembers(n);
    cudaCheck(cudaMemcpy(hostSeeds.data(), seeds.data, hostSeeds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hostBoundaries.data(), boundaries.data, hostBoundaries.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hostMembers.data(), result.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Cluster> clusters;
    clusters.reserve(hostSeeds.size());
    for (size_t i = 0; i < hostSeeds.size(); ++i)
        clusters.push_back({std::vector<int>(hostMembers.begin() + hostBoundaries[i],
                                            hostMembers.begin() + hostBoundaries[i + 1]), hostSeeds[i]});
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
    
    // Initialize the CUDA context before timing the clustering algorithm.
    cudaCheck(cudaFree(nullptr));
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
