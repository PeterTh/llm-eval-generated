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
#include <climits>
#include <stdexcept>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
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

// Keep double precision and separate multiply/add rounding, just as in the
// reference. In particular, do not compare squared distances: sqrt rounding
// can affect both strict threshold comparisons and equal-distance ties.
__device__ inline double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> class DeviceBuffer {
    T* data_ = nullptr;
public:
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("GPU allocation size overflow");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_),
                             std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
};

constexpr int BLOCK_SIZE = 128;
using CountReduce = cub::BlockReduce<unsigned long long, BLOCK_SIZE>;
using IntScan = cub::BlockScan<int, BLOCK_SIZE>;

// The seed's threshold neighborhood is an exact superset of every possible
// candidate cluster from that seed. Store it in CSR rather than an N*N matrix.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               unsigned long long* offsets) {
    __shared__ CountReduce::TempStorage storage;
    const int seed = blockIdx.x;
    const Point p = points[seed];
    unsigned long long count = 0;
    for (int j = threadIdx.x; j < n; j += BLOCK_SIZE)
        if (j != seed && deviceDistance(p, points[j]) < threshold) ++count;
    const auto total = CountReduce(storage).Sum(count);
    if (threadIdx.x == 0) offsets[seed] = total;
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const unsigned long long* offsets, int* neighbors) {
    __shared__ IntScan::TempStorage storage;
    __shared__ unsigned long long cursor;
    const int seed = blockIdx.x;
    const Point p = points[seed];
    if (threadIdx.x == 0) cursor = offsets[seed];
    __syncthreads();
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        const int keep = j < n && j != seed && deviceDistance(p, points[j]) < threshold;
        int prefix, total;
        IntScan(storage).ExclusiveSum(keep, prefix, total);
        if (keep) neighbors[cursor + prefix] = j;
        __syncthreads();
        if (threadIdx.x == 0) cursor += total;
        __syncthreads();
    }
}

struct Closest {
    double diameter;
    int point;
    unsigned long long slot;
};
struct Minimum {
    __device__ Closest operator()(Closest a, Closest b) const {
        return b.diameter < a.diameter ||
               (b.diameter == a.diameter && b.point < a.point) ? b : a;
    }
};
using ClosestReduce = cub::BlockReduce<Closest, BLOCK_SIZE>;

// One block per seed; threads cooperate on the candidate search. Maintaining
// max distance incrementally removes the reference's repeated scans over all
// previously selected members. Each seed owns its scratch and member storage.
__global__ void buildCandidates(const Point* points, double threshold,
                                const unsigned long long* offsets,
                                const int* neighbors, const unsigned char* clustered,
                                double* maxima, int* members, int* cardinalities) {
    const int seed = blockIdx.x;
    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinalities[seed] = 0;
        return;
    }
    const unsigned long long begin = offsets[seed], end = offsets[seed + 1];
    const int old_count = cardinalities[seed];
    // A cached candidate remains exactly valid until one of its members is
    // removed. Removing other points cannot change any of its greedy choices.
    bool dirty = old_count == 0;
    for (int i = threadIdx.x; i < old_count - 1; i += BLOCK_SIZE)
        dirty |= clustered[members[begin + i]] != 0;
    if (!__syncthreads_or(dirty)) return;

    __shared__ ClosestReduce::TempStorage storage;
    __shared__ Closest selected;
    __shared__ int count;
    __shared__ int last;
    for (auto slot = begin + threadIdx.x; slot < end; slot += BLOCK_SIZE)
        maxima[slot] = clustered[neighbors[slot]] ? CUDART_INF : 0.0;
    if (threadIdx.x == 0) { count = 1; last = seed; }
    __syncthreads();
    for (;;) {
        const Point added = points[last];
        Closest best = {CUDART_INF, INT_MAX, 0};
        for (auto slot = begin + threadIdx.x; slot < end; slot += BLOCK_SIZE) {
            double diameter = maxima[slot];
            if (diameter == CUDART_INF) continue;
            const int candidate = neighbors[slot];
            diameter = fmax(diameter, deviceDistance(added, points[candidate]));
            if (!(diameter < threshold)) diameter = CUDART_INF;
            maxima[slot] = diameter;
            if (diameter < CUDART_INF)
                best = Minimum()(best, Closest{diameter, candidate, slot});
        }
        const Closest winner = ClosestReduce(storage).Reduce(best, Minimum());
        if (threadIdx.x == 0) {
            selected = winner;
            if (winner.point != INT_MAX) {
                members[begin + count - 1] = winner.point;
                ++count;
                last = winner.point;
                maxima[winner.slot] = CUDART_INF;
            }
        }
        __syncthreads();
        if (selected.point == INT_MAX) break;
    }
    if (threadIdx.x == 0) cardinalities[seed] = count;
}

struct Largest {
    int count;
    int seed;
};
struct Maximum {
    __device__ Largest operator()(Largest a, Largest b) const {
        return b.count > a.count || (b.count == a.count && b.seed < a.seed) ? b : a;
    }
};

// Select the first largest cluster, publish its ordered members, and mark them
// on the device. Only the winning cluster is transferred back to the host.
__global__ void selectCluster(int n, const unsigned long long* offsets,
                              const int* members, const int* cardinalities,
                              unsigned char* clustered, int* result) {
    __shared__ cub::BlockReduce<Largest, BLOCK_SIZE>::TempStorage storage;
    __shared__ Largest selected;
    Largest best = {0, INT_MAX};
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE)
        best = Maximum()(best, Largest{cardinalities[i], i});
    const Largest winner = cub::BlockReduce<Largest, BLOCK_SIZE>(storage).Reduce(best, Maximum());
    if (threadIdx.x == 0) {
        selected = winner;
        result[0] = winner.seed;
        result[1] = winner.count;
        result[2] = winner.seed;
        clustered[winner.seed] = 1;
    }
    __syncthreads();
    const auto begin = offsets[selected.seed];
    for (int i = threadIdx.x; i < selected.count - 1; i += BLOCK_SIZE) {
        const int point = members[begin + i];
        result[i + 3] = point;
        clustered[point] = 1;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (n == 0) return clusters;
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned long long> d_offsets(size_t(n) + 1);
    DeviceBuffer<unsigned char> d_clustered(n);
    DeviceBuffer<int> d_cardinalities(n), d_result(size_t(n) + 2);
    cudaCheck(cudaMemcpy(d_points.get(), points.data(), size_t(n) * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_clustered.get(), 0, n));
    cudaCheck(cudaMemset(d_cardinalities.get(), 0, size_t(n) * sizeof(int)));
    cudaCheck(cudaMemset(d_offsets.get() + n, 0, sizeof(unsigned long long)));
    countNeighbors<<<n, BLOCK_SIZE>>>(d_points.get(), n, threshold, d_offsets.get());
    cudaCheck(cudaGetLastError());
    auto offsets = thrust::device_pointer_cast(d_offsets.get());
    thrust::exclusive_scan(offsets, offsets + size_t(n) + 1, offsets);
    unsigned long long edge_count;
    cudaCheck(cudaMemcpy(&edge_count, d_offsets.get() + n, sizeof(edge_count), cudaMemcpyDeviceToHost));
    DeviceBuffer<int> d_neighbors(edge_count), d_members(edge_count);
    DeviceBuffer<double> d_maxima(edge_count);
    fillNeighbors<<<n, BLOCK_SIZE>>>(d_points.get(), n, threshold, d_offsets.get(), d_neighbors.get());
    cudaCheck(cudaGetLastError());
    int remaining = n;
    while (remaining > 0) {
        buildCandidates<<<n, BLOCK_SIZE>>>(d_points.get(), threshold, d_offsets.get(),
            d_neighbors.get(), d_clustered.get(), d_maxima.get(), d_members.get(), d_cardinalities.get());
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, BLOCK_SIZE>>>(n, d_offsets.get(), d_members.get(),
            d_cardinalities.get(), d_clustered.get(), d_result.get());
        cudaCheck(cudaGetLastError());
        int header[2];
        cudaCheck(cudaMemcpy(header, d_result.get(), sizeof(header), cudaMemcpyDeviceToHost));
        // If the largest candidate is a singleton, every remaining point is
        // isolated. Emit the final singleton clusters in ascending seed order
        // without another kernel launch and host synchronization per point.
        if (header[1] == 1) {
            std::vector<int> cardinalities(n);
            cudaCheck(cudaMemcpy(cardinalities.data(), d_cardinalities.get(),
                size_t(n) * sizeof(int), cudaMemcpyDeviceToHost));
            for (int seed = 0; seed < n; ++seed)
                if (cardinalities[seed] > 0) clusters.push_back(Cluster{{seed}, seed});
            break;
        }
        Cluster cluster;
        cluster.seed_point = header[0];
        cluster.members.resize(header[1]);
        cudaCheck(cudaMemcpy(cluster.members.data(), d_result.get() + 2,
            cluster.members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        remaining -= header[1];
        clusters.push_back(std::move(cluster));
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
    
    // Initialize the required CUDA runtime before timing the clustering work.
    try {
        cudaCheck(cudaFree(nullptr));
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA initialization failed: %s\n", error.what());
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
