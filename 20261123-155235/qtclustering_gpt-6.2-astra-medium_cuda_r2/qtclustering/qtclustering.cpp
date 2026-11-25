// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (N <= 30) group_cnt = 1; // Keep tiny inputs from stalling on zero-sized groups.
        
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

namespace {

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) {
        if (count) checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

constexpr int threads = 128;
constexpr int sharedLimit = 1024;
constexpr unsigned fullWarp = 0xffffffffu;

__device__ double pointDistance(Point a, Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(fma(dx, dx, __dmul_rn(dy, dy)));
}

// Each warp counts or writes one sorted neighborhood. The dense distance
// cache is optional so its quadratic size does not limit supported inputs.
template<bool fill>
__global__ void neighborhoods(const Point* points, int n, double threshold,
                              double* distances, int* degrees,
                              const size_t* offsets, int* neighbors, int first, int end) {
    int seed = first + blockIdx.x * (threads / 32) + threadIdx.x / 32;
    if (seed >= end) return;
    int lane = threadIdx.x % 32;
    int count = 0;
    Point center = points[seed];
    for (long long base = 0; base < n; base += 32) {
        long long id = base + lane;
        bool eligible = false;
        if (id < n) {
            size_t index = size_t(seed) * n + id;
            double d;
            if (fill && distances) {
                d = distances[index];
            } else {
                d = pointDistance(center, points[id]);
                if (!fill && distances) distances[index] = d;
            }
            eligible = id != seed && d < threshold;
        }
        unsigned mask = __ballot_sync(fullWarp, eligible);
        if (fill && eligible) {
            neighbors[(offsets[seed] - offsets[first]) + count + __popc(mask & ((1u << lane) - 1))] = id;
        }
        count += __popc(mask);
    }
    if (!fill && lane == 0) degrees[seed] = count;
}

struct Choice {
    double diameter;
    int id;
};

__device__ Choice warpMin(Choice best) {
    for (int delta = 16; delta; delta /= 2) {
        Choice other{__shfl_down_sync(fullWarp, best.diameter, delta),
                     __shfl_down_sync(fullWarp, best.id, delta)};
        if (other.diameter < best.diameter ||
            (other.diameter == best.diameter && other.id < best.id)) best = other;
    }
    return best;
}

__global__ void evaluate(const Point* points, int n, double threshold,
                         const double* distances, const size_t* offsets,
                         const int* neighbors, const unsigned char* removed,
                         int* members, int* sizes, double* scratch,
                         unsigned long long* winner, int first,
                         bool reuse) {
    int seed = first + blockIdx.x;
    if (removed[seed]) return;
    int tid = threadIdx.x;
    size_t start = offsets[seed] - offsets[first];
    int degree = offsets[seed + 1] - offsets[seed] - 1;
    int size = reuse ? sizes[seed] : 0;
    bool dirty = size == 0;
    for (int i = tid; i < size && !dirty; i += threads)
        dirty = removed[members[start + i]];

    if (__syncthreads_or(dirty)) {
        extern __shared__ double localMax[];
        double* maxima = degree <= sharedLimit ? localMax : scratch + start;
        __shared__ Choice warpBest[threads / 32];
        __shared__ int chosen;
        int last = seed;
        size = 1;
        if (tid == 0) members[start] = seed;
        while (size <= degree) {
            Choice best{CUDART_INF, INT_MAX};
            for (int i = tid; i < degree; i += threads) {
                int id = neighbors[start + i];
                double previous = size == 1 ? 0.0 : maxima[i];
                if (size == 1 ? removed[id] != 0 : (id == last || previous >= threshold)) {
                    maxima[i] = CUDART_INF;
                    continue;
                }
                double d = distances ? distances[size_t(last) * n + id]
                                     : pointDistance(points[last], points[id]);
                double diameter = fmax(previous, d);
                maxima[i] = diameter;
                // Neighbors are sorted, so a lane encounters smaller IDs first.
                if (diameter < threshold && diameter < best.diameter) best = {diameter, id};
            }
            best = warpMin(best);
            if ((tid & 31) == 0) warpBest[tid / 32] = best;
            __syncthreads();
            if (tid < 32) {
                best = tid < threads / 32 ? warpBest[tid] : Choice{CUDART_INF, INT_MAX};
                best = warpMin(best);
                if (tid == 0) chosen = best.id;
            }
            __syncthreads();
            last = chosen;
            if (last == INT_MAX) break;
            if (tid == 0) members[start + size] = last;
            ++size;
        }
        if (tid == 0) sizes[seed] = size;
    }
    if (tid == 0) {
        unsigned long long key = (static_cast<unsigned long long>(size) << 32) |
                                 static_cast<unsigned>(INT_MAX - seed);
        atomicMax(winner, key);
    }
}

__global__ void markRemoved(const int* members, int count, unsigned char* removed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) removed[members[i]] = 1;
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int n = static_cast<int>(points.size());
    DeviceArray<Point> coords(n);
    DeviceArray<int> degrees(n), sizes(n), selected(n);
    DeviceArray<size_t> offsets(size_t(n) + 1);
    DeviceArray<unsigned char> removed(n);
    DeviceArray<unsigned long long> winner(1);
    checkCuda(cudaMemcpy(coords.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));

    size_t freeBytes, totalBytes;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));
    size_t pairs = size_t(n) * n;
    DeviceArray<double> distances(pairs <= freeBytes / (4 * sizeof(double)) ? pairs : 0);
    int blocks = 1 + (n - 1) / (threads / 32);
    neighborhoods<false><<<blocks, threads>>>(coords.data, n, threshold, distances.data,
                                               degrees.data, nullptr, nullptr, 0, n);
    checkCuda(cudaGetLastError());
    std::vector<int> hostDegrees(n);
    checkCuda(cudaMemcpy(hostDegrees.data(), degrees.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> hostOffsets(size_t(n) + 1);
    int maxDegree = 0;
    for (int i = 0; i < n; ++i) {
        // One additional slot per row holds the seed in the membership list.
        hostOffsets[i + 1] = hostOffsets[i] + hostDegrees[i] + 1;
        maxDegree = std::max(maxDegree, hostDegrees[i]);
    }
    // If all pairs are eligible, every seed produces the full cluster. Only
    // the first seed is needed to recover the exact greedy membership order.
    int seedCount = hostOffsets.back() == pairs ? 1 : n;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));
    size_t bytesPerEntry = 2 * sizeof(int) + (maxDegree > sharedLimit ? sizeof(double) : 0);
    size_t capacity = std::min(hostOffsets[seedCount], (freeBytes - freeBytes / 8) / bytesPerEntry);
    if (capacity < size_t(maxDegree) + 1) {
        std::fprintf(stderr, "Insufficient GPU memory for one candidate neighborhood\n");
        std::exit(EXIT_FAILURE);
    }
    DeviceArray<int> neighbors(capacity), members(capacity);
    DeviceArray<double> scratch(maxDegree > sharedLimit ? capacity : 0);
    checkCuda(cudaMemcpy(offsets.data, hostOffsets.data(), hostOffsets.size() * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(sizes.data, 0, n * sizeof(int)));
    checkCuda(cudaMemset(removed.data, 0, n));
    size_t sharedBytes = std::min(maxDegree, sharedLimit) * sizeof(double);
    bool reuse = capacity == hostOffsets[seedCount];
    std::vector<Cluster> clusters;
    std::vector<unsigned char> assigned(n);
    int remaining = n;
    while (remaining) {
        unsigned long long bestKey = 0;
        for (int first = 0; first < seedCount;) {
            auto endOffset = std::upper_bound(hostOffsets.begin() + first + 1,
                                              hostOffsets.begin() + seedCount + 1,
                                              hostOffsets[first] + capacity);
            int end = static_cast<int>(endOffset - hostOffsets.begin() - 1);
            int batchSize = end - first;
            if (!reuse || remaining == n) {
                int batchBlocks = 1 + (batchSize - 1) / (threads / 32);
                neighborhoods<true><<<batchBlocks, threads>>>(coords.data, n, threshold, distances.data,
                    degrees.data, offsets.data, neighbors.data, first, end);
                checkCuda(cudaGetLastError());
            }
            checkCuda(cudaMemset(winner.data, 0, sizeof(unsigned long long)));
            evaluate<<<batchSize, threads, sharedBytes>>>(coords.data, n, threshold, distances.data,
                offsets.data, neighbors.data, removed.data, members.data, sizes.data,
                scratch.data, winner.data, first, reuse);
            checkCuda(cudaGetLastError());
            unsigned long long key;
            checkCuda(cudaMemcpy(&key, winner.data, sizeof(key), cudaMemcpyDeviceToHost));
            if (key > bestKey) {
                bestKey = key;
                int seed = INT_MAX - static_cast<int>(key & 0xffffffffu);
                const int* chosen = members.data + (hostOffsets[seed] - hostOffsets[first]);
                if (!reuse) {
                    checkCuda(cudaMemcpy(selected.data, chosen, (key >> 32) * sizeof(int),
                                         cudaMemcpyDeviceToDevice));
                }
            }
            first = end;
        }
        int size = bestKey >> 32;
        if (size == 1) {
            for (int i = 0; i < n; ++i)
                if (!assigned[i]) clusters.push_back({{i}, i});
            break;
        }
        int seed = INT_MAX - static_cast<int>(bestKey & 0xffffffffu);
        const int* chosen = reuse ? members.data + hostOffsets[seed] : selected.data;
        clusters.push_back({std::vector<int>(size), seed});
        auto& ids = clusters.back().members;
        checkCuda(cudaMemcpy(ids.data(), chosen, size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int id : ids) assigned[id] = 1;
        markRemoved<<<1 + (size - 1) / threads, threads>>>(chosen, size, removed.data);
        checkCuda(cudaGetLastError());
        remaining -= size;
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
