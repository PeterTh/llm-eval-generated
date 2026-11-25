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
#include <numeric>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <vector>

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
        // Keep group generation productive even for inputs of 30 points or fewer.
        int group_cnt = static_cast<int>(frand() * (N <= 30 ? 2.0 : N / 30.0));
        
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

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double pointDistance(Point a, Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(fma(dx, dx, dy * dy));
}

constexpr unsigned warpMask = 0xffffffffu;
constexpr int blockSize = 128;
constexpr int warpsPerBlock = blockSize / 32;

// Count, then compact, the threshold neighborhood of each seed. The x-sorted
// index limits the search to a vertical strip. Include the seed's own slot
// so the same row layout has room for every possible cluster member.
template<bool write>
__global__ void buildNeighborhoods(const Point* points, const int* order,
                                   const int2* ranges, int first, int end, double threshold,
                                   int* counts, const size_t* offsets, int* neighbors) {
    int seed = first + blockIdx.x * warpsPerBlock + threadIdx.x / 32;
    if (seed >= end) return;
    int lane = threadIdx.x % 32;
    int2 range = ranges[seed];
    int count = 0;
    for (int base = range.x; base < range.y; base += 32) {
        int id = base + lane < range.y ? order[base + lane] : -1;
        bool keep = id == seed ||
                    (id >= 0 && pointDistance(points[seed], points[id]) < threshold);
        unsigned mask = __ballot_sync(warpMask, keep);
        if (write && keep) {
            int rank = __popc(mask & ((1u << lane) - 1));
            neighbors[offsets[seed] - offsets[first] + count + rank] = id;
        }
        count += __popc(mask);
    }
    if (!write && lane == 0) counts[seed] = count;
}

__device__ bool better(double d, int id, double best, int bestId) {
    return d < best || (d == best && id < bestId);
}

// A warp grows a candidate in greedy order. Small neighborhoods stay in
// registers. Larger ones use a disjoint scratch row in device memory.
template<int slots>
__device__ int growCandidate(const Point* points, double threshold, int seed,
                             const int* neighbors, int degree, const int* removed,
                             double* scratch, int* members) {
    int lane = threadIdx.x % 32;
    double localMax[slots ? slots : 1];
    int localId[slots ? slots : 1];
    int rounds = slots ? slots : (degree + 31) / 32;
    for (int i = 0; i < rounds; ++i) {
        int pos = i * 32 + lane;
        int id = pos < degree ? neighbors[pos] : seed;
        double initial = (id == seed || removed[id]) ? CUDART_INF : 0.0;
        if (slots) {
            localId[i] = id;
            localMax[i] = initial;
        } else if (pos < degree) {
            scratch[pos] = initial;
        }
    }
    int last = seed;
    int size = 1;
    if (lane == 0) members[0] = seed;
    while (size < degree) {
        Point p = points[last];
        double best = CUDART_INF;
        int bestId = INT_MAX;
        for (int i = 0; i < rounds; ++i) {
            int pos = i * 32 + lane;
            if (!slots && pos >= degree) break;
            double d = slots ? localMax[i] : scratch[pos];
            if (d == CUDART_INF) continue;
            int id = slots ? localId[i] : neighbors[pos];
            d = id == last ? CUDART_INF : fmax(d, pointDistance(p, points[id]));
            if (!(d < threshold)) d = CUDART_INF;
            if (slots) localMax[i] = d;
            else scratch[pos] = d;
            if (d < CUDART_INF && better(d, id, best, bestId)) {
                best = d;
                bestId = id;
            }
        }
        for (int step = 16; step; step /= 2) {
            double d = __shfl_down_sync(warpMask, best, step);
            int id = __shfl_down_sync(warpMask, bestId, step);
            if (better(d, id, best, bestId)) {
                best = d;
                bestId = id;
            }
        }
        last = __shfl_sync(warpMask, bestId, 0);
        if (last == INT_MAX) break;
        if (lane == 0) members[size] = last;
        ++size;
    }
    return size;
}

__global__ void evaluateSeeds(const Point* points, double threshold, int first, int end, bool cached,
                              const size_t* offsets, const int* neighbors,
                              const int* removed, int* sizes, int* members,
                              double* scratch, unsigned long long* winner) {
    int seed = first + blockIdx.x * warpsPerBlock + threadIdx.x / 32;
    if (seed >= end || removed[seed]) return;
    int lane = threadIdx.x % 32;
    size_t begin = offsets[seed] - offsets[first];
    int degree = static_cast<int>(offsets[seed + 1] - offsets[seed]);
    int size = cached ? sizes[seed] : 0;
    int* row = members + begin;
    // Removing points outside the old greedy sequence cannot change any
    // of its choices. Only rebuild if that sequence lost a member.
    bool rebuild = size == 0;
    for (int i = lane; i < size; i += 32)
        rebuild |= removed[row[i]] != 0;
    if (__any_sync(warpMask, rebuild)) {
        const int* ids = neighbors + begin;
        if (degree <= 32)
            size = growCandidate<1>(points, threshold, seed, ids, degree, removed, nullptr, row);
        else if (degree <= 128)
            size = growCandidate<4>(points, threshold, seed, ids, degree, removed, nullptr, row);
        else if (degree <= 512)
            size = growCandidate<16>(points, threshold, seed, ids, degree, removed, nullptr, row);
        else
            size = growCandidate<0>(points, threshold, seed, ids, degree, removed, scratch + begin, row);
        if (lane == 0) sizes[seed] = size;
    }
    if (lane == 0) {
        unsigned long long key = (static_cast<unsigned long long>(size) << 32) |
                                 static_cast<unsigned>(INT_MAX - seed);
        atomicMax(winner, key);
    }
}

// Preserve a batch winner before the next batch reuses its member storage.
__global__ void saveWinner(const unsigned long long* winner, int first, int end,
                           const size_t* offsets, const int* members, int* saved) {
    int seed = INT_MAX - static_cast<int>(static_cast<unsigned>(*winner));
    if (seed < first || seed >= end) return;
    int size = static_cast<int>(*winner >> 32);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        saved[i] = members[offsets[seed] - offsets[first] + i];
}

__global__ void commitCluster(const int* members, int size, int* removed, int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int id = members[i];
        removed[id] = 1;
        output[i] = id;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return points[a].x < points[b].x; });
    std::vector<int2> ranges(n);
    int left = 0, right = 0;
    for (int seed : order) {
        double x = points[seed].x;
        while (left < n && x - points[order[left]].x > threshold) ++left;
        while (right < n && !(points[order[right]].x - x > threshold)) ++right;
        ranges[seed] = make_int2(left, right);
    }

    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> deviceOrder(n), counts(n);
    DeviceBuffer<int2> deviceRanges(n);
    checkCuda(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceOrder.data, order.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceRanges.data, ranges.data(), n * sizeof(int2), cudaMemcpyHostToDevice));
    int blocks = (n + warpsPerBlock - 1) / warpsPerBlock;
    buildNeighborhoods<false><<<blocks, blockSize>>>(devicePoints.data, deviceOrder.data,
        deviceRanges.data, 0, n, threshold, counts.data, nullptr, nullptr);
    std::vector<int> degrees(n);
    checkCuda(cudaMemcpy(degrees.data(), counts.data, n * sizeof(int), cudaMemcpyDeviceToHost));

    // If every pair fits, every seed wins the same cardinality.
    int rows = std::all_of(degrees.begin(), degrees.end(), [n](int d) { return d == n; }) ? 1 : n;
    blocks = (rows + warpsPerBlock - 1) / warpsPerBlock;
    std::vector<size_t> offsets(rows + 1);
    for (int i = 0; i < rows; ++i) offsets[i + 1] = offsets[i] + degrees[i];
    DeviceBuffer<size_t> deviceOffsets(rows + 1);
    checkCuda(cudaMemcpy(deviceOffsets.data, offsets.data(), offsets.size() * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    DeviceBuffer<int> removed(n), sizes(rows), output(n);
    DeviceBuffer<unsigned long long> winner(1);
    checkCuda(cudaMemset(removed.data, 0, n * sizeof(int)));
    checkCuda(cudaMemset(sizes.data, 0, rows * sizeof(int)));

    // Keep all candidate rows resident when possible. Otherwise fit as many
    // complete rows per batch as available memory permits and reuse the arena.
    size_t freeBytes, totalBytes;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));
    bool needsScratch = *std::max_element(degrees.begin(), degrees.end()) > 512;
    size_t rowCapacity = (freeBytes - freeBytes / 8) /
                         (2 * sizeof(int) + (needsScratch ? sizeof(double) : 0));
    std::vector<int> batchEnds;
    size_t arenaSize = 0;
    for (int first = 0; first < rows;) {
        auto limit = std::upper_bound(offsets.begin() + first + 1, offsets.end(),
                                      offsets[first] + rowCapacity);
        int end = std::max(first + 1, static_cast<int>(limit - offsets.begin() - 1));
        batchEnds.push_back(end);
        arenaSize = std::max(arenaSize, offsets[end] - offsets[first]);
        first = end;
    }
    bool cached = batchEnds.size() == 1;
    DeviceBuffer<int> neighbors(arenaSize), members(arenaSize), saved(cached ? 0 : n);
    DeviceBuffer<double> scratch(needsScratch ? arenaSize : 0);
    auto buildRows = [&](int first, int end, int grid) {
        buildNeighborhoods<true><<<grid, blockSize>>>(devicePoints.data, deviceOrder.data,
            deviceRanges.data, first, end, threshold, nullptr, deviceOffsets.data, neighbors.data);
        checkCuda(cudaGetLastError());
    };
    if (cached) buildRows(0, rows, blocks);

    std::vector<Cluster> clusters;
    int assigned = 0;
    while (assigned < n) {
        checkCuda(cudaMemset(winner.data, 0, sizeof(unsigned long long)));
        int first = 0;
        for (int end : batchEnds) {
            int grid = (end - first + warpsPerBlock - 1) / warpsPerBlock;
            if (!cached) buildRows(first, end, grid);
            evaluateSeeds<<<grid, blockSize>>>(devicePoints.data, threshold, first, end, cached,
                deviceOffsets.data, neighbors.data, removed.data, sizes.data, members.data,
                scratch.data, winner.data);
            checkCuda(cudaGetLastError());
            if (!cached) {
                saveWinner<<<1, blockSize>>>(winner.data, first, end, deviceOffsets.data,
                                             members.data, saved.data);
                checkCuda(cudaGetLastError());
            }
            first = end;
        }
        unsigned long long key;
        checkCuda(cudaMemcpy(&key, winner.data, sizeof(key), cudaMemcpyDeviceToHost));
        int size = static_cast<int>(key >> 32);
        if (size == 1) break; // All remaining clusters are singletons in index order.
        int seed = INT_MAX - static_cast<int>(static_cast<unsigned>(key));
        clusters.push_back({std::vector<int>(size), seed});
        commitCluster<<<(size + blockSize - 1) / blockSize, blockSize>>>(
            cached ? members.data + offsets[seed] : saved.data, size, removed.data, output.data + assigned);
        checkCuda(cudaGetLastError());
        assigned += size;
    }
    std::vector<int> packed(assigned);
    if (assigned) checkCuda(cudaMemcpy(packed.data(), output.data, assigned * sizeof(int), cudaMemcpyDeviceToHost));
    auto next = packed.begin();
    for (Cluster& cluster : clusters) {
        std::copy_n(next, cluster.members.size(), cluster.members.begin());
        next += cluster.members.size();
    }
    if (assigned < n) {
        std::vector<bool> used(n, false);
        for (int id : packed) used[id] = true;
        for (int id = 0; id < n; ++id)
            if (!used[id]) clusters.push_back({{id}, id});
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
