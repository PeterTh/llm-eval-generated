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
#include <cfloat>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>
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

// Keep double arithmetic and disable contraction in CMake: threshold decisions and
// equal-distance tie breaks must agree with the sequential implementation.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count = 0) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Only points within the threshold of a seed can ever join its cluster. Build
// this sparse graph once. Atomic insertion need not sort the neighbors: every
// reduction explicitly breaks ties by the original point index.
__global__ void countNeighbors(const Point* points, int n, double threshold, int* counts) {
    const int seed = blockIdx.x;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        count += j != seed && pointDistance(points[seed], points[j]) < threshold;
    __shared__ int sums[128];
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) counts[seed] = sums[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors, double* distances) {
    const int seed = blockIdx.x;
    __shared__ int count;
    if (!threadIdx.x) count = 0;
    __syncthreads();
    const Point origin = points[seed];
    const int lane = threadIdx.x & 31;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        double d = DBL_MAX;
        if (j < n && j != seed) d = pointDistance(origin, points[j]);
        const bool eligible = j < n && j != seed && d < threshold;
        const unsigned mask = __ballot_sync(0xffffffff, eligible);
        int start = 0;
        // One atomic reservation per warp gives neighboring lanes contiguous
        // entries and avoids contention for dense neighborhoods.
        if (!lane && mask) start = atomicAdd(&count, __popc(mask));
        start = __shfl_sync(0xffffffff, start, 0);
        if (eligible) {
            const size_t pos = offsets[seed] + start
                             + __popc(mask & ((1U << lane) - 1U));
            neighbors[pos] = j;
            distances[pos] = d;
        }
    }
}

__device__ bool closer(double a, int ai, double b, int bi) {
    return a < b || (a == b && ai < bi);
}

// Each block grows one seed's cluster. Scores are the running maximum distance
// to its members, so adding a member requires just one distance per candidate.
// Seeds are independent; the iterations within a seed retain their greedy order.
// A packed atomic maximum selects largest cardinality, then lowest seed index.
// The winner is regenerated into a single output buffer, avoiding O(E) member
// lists and host transfers of every losing candidate.
template <int Threads, bool Output>
__global__ void growCandidates(const Point* points, int n, double threshold,
                               const unsigned char* clustered, const size_t* offsets,
                               const int* neighbors, const double* distances,
                               double* scores, size_t scoreBase, int firstSeed,
                               unsigned long long* winner, int* output) {
    const int seed = Output ? n - static_cast<int>(*winner & 0xffffffffULL)
                            : firstSeed + blockIdx.x;
    if (clustered[seed]) return;
    const bool sparse = offsets != nullptr;
    const size_t begin = sparse ? offsets[seed] : static_cast<size_t>(seed) * n;
    const int degree = sparse ? static_cast<int>(offsets[seed + 1] - begin) : n;
    double* state = scores + (begin - scoreBase);
    const Point origin = points[seed];
    for (int j = threadIdx.x; j < degree; j += Threads) {
        const int id = sparse ? neighbors[begin + j] : j;
        double d = DBL_MAX;
        if (id != seed && !clustered[id]) {
            d = sparse ? distances[begin + j] : pointDistance(origin, points[id]);
            if (!(d < threshold)) d = DBL_MAX;
        }
        state[j] = d;
    }
    __shared__ double warpDistances[Threads / 32];
    __shared__ int warpIndices[Threads / 32];
    __shared__ int warpCounts[Threads / 32];
    __shared__ int selected;
    __shared__ bool pruned;
    int cardinality = 1;
    if (Output && !threadIdx.x) output[0] = seed;
    __syncthreads();
    while (true) {
        double bestDistance = DBL_MAX;
        int bestIndex = INT_MAX;
        int available = 0;
        for (int j = threadIdx.x; j < degree; j += Threads) {
            const double d = state[j];
            const int id = sparse ? neighbors[begin + j] : j;
            available += d < DBL_MAX;
            if (d < DBL_MAX && closer(d, id, bestDistance, bestIndex)) {
                bestDistance = d;
                bestIndex = id;
            }
        }
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x / 32;
        for (int delta = 16; delta; delta /= 2) {
            const double d = __shfl_down_sync(0xffffffff, bestDistance, delta);
            const int id = __shfl_down_sync(0xffffffff, bestIndex, delta);
            available += __shfl_down_sync(0xffffffff, available, delta);
            if (closer(d, id, bestDistance, bestIndex)) {
                bestDistance = d;
                bestIndex = id;
            }
        }
        if (!lane) {
            warpDistances[warp] = bestDistance;
            warpIndices[warp] = bestIndex;
            warpCounts[warp] = available;
        }
        __syncthreads();
        if (!threadIdx.x) {
            double d = warpDistances[0];
            int id = warpIndices[0];
            int left = warpCounts[0];
            for (int w = 1; w < Threads / 32; ++w) {
                left += warpCounts[w];
                if (closer(warpDistances[w], warpIndices[w], d, id)) {
                    d = warpDistances[w];
                    id = warpIndices[w];
                }
            }
            selected = id;
            pruned = false;
            if (!Output) {
                // Even accepting every remaining neighbor cannot beat this
                // bound. The atomic winner only improves, so pruning is safe
                // regardless of block scheduling (including cardinality ties).
                const unsigned long long bound =
                    (static_cast<unsigned long long>(cardinality + left) << 32)
                    | static_cast<unsigned int>(n - seed);
                pruned = bound <= atomicAdd(winner, 0ULL);
            }
            if (Output && id != INT_MAX) output[cardinality] = id;
        }
        __syncthreads();
        if (pruned) return;
        if (selected == INT_MAX) break;
        ++cardinality;
        const Point member = points[selected];
        for (int j = threadIdx.x; j < degree; j += Threads) {
            if (state[j] == DBL_MAX) continue;
            const int id = sparse ? neighbors[begin + j] : j;
            if (id == selected) {
                state[j] = DBL_MAX;
            } else {
                const double d = pointDistance(points[id], member);
                const double maximum = fmax(state[j], d);
                state[j] = maximum < threshold ? maximum : DBL_MAX;
            }
        }
        __syncthreads();
    }
    if (!Output && !threadIdx.x) {
        const unsigned long long key = (static_cast<unsigned long long>(cardinality) << 32)
                                    | static_cast<unsigned int>(n - seed);
        atomicMax(winner, key);
    }
}

__global__ void markCluster(unsigned char* clustered, const int* members,
                            const unsigned long long* winner) {
    const int count = static_cast<int>(*winner >> 32);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x)
        clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (!n) return clusters;
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<int> counts(n), members(n);
    DeviceBuffer<unsigned long long> winner(1);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(clustered.data, 0, n));
    countNeighbors<<<n, 128>>>(devicePoints.data, n, threshold, counts.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> hostCounts(n);
    cudaCheck(cudaMemcpy(hostCounts.data(), counts.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> hostOffsets(n + 1, 0);
    int maxDegree = 0;
    for (int i = 0; i < n; ++i) {
        hostOffsets[i + 1] = hostOffsets[i] + hostCounts[i];
        maxDegree = std::max(maxDegree, hostCounts[i]);
    }
    size_t freeBytes, totalBytes;
    cudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Leave headroom for the runtime and other users of the device. If the
    // neighbor graph itself is too large, compute distances directly on the
    // GPU instead. Scratch space is bounded by batching consecutive seeds.
    const size_t budget = freeBytes * 3 / 4;
    const size_t edges = hostOffsets[n];
    const bool sparse = edges <= budget / (sizeof(int) + 2 * sizeof(double));
    DeviceBuffer<size_t> offsets(sparse ? n + 1 : 0);
    DeviceBuffer<int> neighbors(sparse ? edges : 0);
    DeviceBuffer<double> distances(sparse ? edges : 0);
    size_t scratchCount;
    if (sparse) {
        cudaCheck(cudaMemcpy(offsets.data, hostOffsets.data(), (n + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
        fillNeighbors<<<n, 128>>>(devicePoints.data, n, threshold, offsets.data,
                                  neighbors.data, distances.data);
        cudaCheck(cudaGetLastError());
        scratchCount = std::max<size_t>(edges, 1);
    } else {
        for (int i = 0; i <= n; ++i) hostOffsets[i] = static_cast<size_t>(i) * n;
        maxDegree = n;
        scratchCount = std::min(hostOffsets[n], budget / sizeof(double));
        if (scratchCount < static_cast<size_t>(n))
            throw std::runtime_error("CUDA: insufficient GPU memory for one candidate cluster");
    }
    DeviceBuffer<double> scores(scratchCount);
    const int threads = maxDegree <= 64 ? 32 : 128;
    int remaining = n;
    std::vector<unsigned char> assigned(n, 0);
    while (remaining) {
        cudaCheck(cudaMemset(winner.data, 0, sizeof(unsigned long long)));
        for (int first = 0; first < n;) {
            int end = static_cast<int>(std::upper_bound(hostOffsets.begin() + first + 1,
                hostOffsets.end(), hostOffsets[first] + scratchCount) - hostOffsets.begin()) - 1;
            end = std::max(end, first + 1);
            if (threads == 32)
                growCandidates<32, false><<<end - first, 32>>>(devicePoints.data, n, threshold,
                    clustered.data, offsets.data, neighbors.data, distances.data, scores.data,
                    hostOffsets[first], first, winner.data, members.data);
            else
                growCandidates<128, false><<<end - first, 128>>>(devicePoints.data, n, threshold,
                    clustered.data, offsets.data, neighbors.data, distances.data, scores.data,
                    hostOffsets[first], first, winner.data, members.data);
            cudaCheck(cudaGetLastError());
            first = end;
        }
        unsigned long long best;
        cudaCheck(cudaMemcpy(&best, winner.data, sizeof(best), cudaMemcpyDeviceToHost));
        const int seed = n - static_cast<int>(best & 0xffffffffULL);
        const int count = static_cast<int>(best >> 32);
        if (count == 1) {
            // The GPU has proved that no remaining pair can form a cluster.
            // All subsequent greedy choices are singletons in point order.
            for (int i = 0; i < n; ++i)
                if (!assigned[i]) clusters.push_back(Cluster{{i}, i});
            break;
        }
        if (threads == 32)
            growCandidates<32, true><<<1, 32>>>(devicePoints.data, n, threshold,
                clustered.data, offsets.data, neighbors.data, distances.data, scores.data,
                hostOffsets[seed], seed, winner.data, members.data);
        else
            growCandidates<128, true><<<1, 128>>>(devicePoints.data, n, threshold,
                clustered.data, offsets.data, neighbors.data, distances.data, scores.data,
                hostOffsets[seed], seed, winner.data, members.data);
        cudaCheck(cudaGetLastError());
        markCluster<<<(count + 127) / 128, 128>>>(clustered.data, members.data, winner.data);
        cudaCheck(cudaGetLastError());
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(count);
        cudaCheck(cudaMemcpy(cluster.members.data(), members.data, count * sizeof(int), cudaMemcpyDeviceToHost));
        for (int id : cluster.members) assigned[id] = 1;
        clusters.push_back(std::move(cluster));
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
    
    // Initialize the CUDA runtime before timing the clustering computation.
    try {
        cudaCheck(cudaFree(nullptr));
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    std::vector<Cluster> clusters;
    try {
        clusters = qtClustering(points, threshold);
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
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
