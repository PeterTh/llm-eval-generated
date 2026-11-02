// QT Clustering Benchmark - CUDA version
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

// Each seed is evaluated by one block.  The running maximum distance to
// its members is maintained for every candidate, so adding a member requires
// only one new distance per candidate instead of revisiting all members.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr int BLOCK_SIZE = 128;

__global__ void calculateDistances(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         k < total; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(k / n);
        const int j = static_cast<int>(k % n);
        const double dx = points[i].x - points[j].x;
        const double dy = points[i].y - points[j].y;
        // Match the host's double-precision multiply and fused multiply-add.
        distances[k] = sqrt(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
    }
}

// Reduce by distance, then by index, as in the original ascending scan.
__device__ __forceinline__ bool better(double a, int ai, double b, int bi) {
    return a < b || (a == b && ai < bi);
}

__device__ int selectClosest(const double* scores, int n, double threshold,
                             double* warp_values, int* warp_indices) {
    double value = threshold;
    int index = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = scores[i];
        if (v < threshold && better(v, i, value, index)) {
            value = v;
            index = i;
        }
    }
    for (int offset = 16; offset; offset >>= 1) {
        const double v = __shfl_down_sync(0xffffffff, value, offset);
        const int i = __shfl_down_sync(0xffffffff, index, offset);
        if (better(v, i, value, index)) { value = v; index = i; }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_values[warp] = value; warp_indices[warp] = index; }
    __syncthreads();
    if (warp == 0) {
        value = lane < BLOCK_SIZE / 32 ? warp_values[lane] : threshold;
        index = lane < BLOCK_SIZE / 32 ? warp_indices[lane] : n;
        for (int offset = 16; offset; offset >>= 1) {
            const double v = __shfl_down_sync(0xffffffff, value, offset);
            const int i = __shfl_down_sync(0xffffffff, index, offset);
            if (better(v, i, value, index)) { value = v; index = i; }
        }
        if (lane == 0) warp_indices[0] = index;
    }
    __syncthreads();
    return warp_indices[0];
}

// With Record set, this block rebuilds the winner and records its members.
// Otherwise each block computes only its seed's cardinality.
template <bool Record>
__global__ void evaluateSeeds(const double* distances, int n, double threshold,
                              int* clustered, int* cardinalities, double* workspace,
                              int* result) {
    extern __shared__ double shared_scores[];
    __shared__ double warp_values[BLOCK_SIZE / 32];
    __shared__ int warp_indices[BLOCK_SIZE / 32];
    __shared__ int chosen_seed;
    int seed;
    if constexpr (Record) {
        // Find the greatest cardinality; a lower seed wins ties.
        int count = -1, index = n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int c = cardinalities[i];
            if (c > count || (c == count && i < index)) { count = c; index = i; }
        }
        for (int offset = 16; offset; offset >>= 1) {
            int c = __shfl_down_sync(0xffffffff, count, offset);
            int i = __shfl_down_sync(0xffffffff, index, offset);
            if (c > count || (c == count && i < index)) { count = c; index = i; }
        }
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) { warp_values[warp] = count; warp_indices[warp] = index; }
        __syncthreads();
        if (warp == 0) {
            count = lane < BLOCK_SIZE / 32 ? static_cast<int>(warp_values[lane]) : -1;
            index = lane < BLOCK_SIZE / 32 ? warp_indices[lane] : n;
            for (int offset = 16; offset; offset >>= 1) {
                int c = __shfl_down_sync(0xffffffff, count, offset);
                int i = __shfl_down_sync(0xffffffff, index, offset);
                if (c > count || (c == count && i < index)) { count = c; index = i; }
            }
            if (lane == 0) chosen_seed = index;
        }
        __syncthreads();
        seed = chosen_seed;
    } else {
        seed = blockIdx.x;
        if (clustered[seed]) {
            if (threadIdx.x == 0) cardinalities[seed] = 0;
            return;
        }
    }

    double* scores = workspace ? workspace + static_cast<size_t>(seed) * n : shared_scores;
    const double* row = distances + static_cast<size_t>(seed) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        scores[i] = (clustered[i] || i == seed) ? CUDART_INF : row[i];
    __syncthreads();
    if constexpr (Record) {
        if (threadIdx.x == 0) { result[0] = seed; result[2] = seed; clustered[seed] = 1; }
    }
    int count = 1;
    while (count < n) {
        const int closest = selectClosest(scores, n, threshold, warp_values, warp_indices);
        if (closest == n) break;
        if constexpr (Record) {
            if (threadIdx.x == 0) { result[count + 2] = closest; clustered[closest] = 1; }
        }
        ++count;
        row = distances + static_cast<size_t>(closest) * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (i == closest) scores[i] = CUDART_INF;
            else if (scores[i] < threshold && row[i] > scores[i]) scores[i] = row[i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if constexpr (Record) result[1] = count;
        else cardinalities[seed] = count;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t matrix_elements = static_cast<size_t>(n) * n;
    if (matrix_elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Point count is too large for the distance matrix\n");
        std::exit(EXIT_FAILURE);
    }
    Point* d_points = nullptr;
    double *d_distances = nullptr, *d_workspace = nullptr;
    int *d_clustered = nullptr, *d_counts = nullptr, *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_distances, matrix_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, (static_cast<size_t>(n) + 2) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, static_cast<size_t>(n) * sizeof(int)));
    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    // Reserve room for the reduction's static shared arrays.
    const size_t shared_bytes = static_cast<size_t>(n) * sizeof(double);
    const bool use_shared = shared_bytes + 1024 <= properties.sharedMemPerBlock;
    if (!use_shared) CUDA_CHECK(cudaMalloc(&d_workspace, matrix_elements * sizeof(double)));
    const size_t dynamic_shared = use_shared ? shared_bytes : 0;
    int blocks = static_cast<int>(std::min<size_t>((matrix_elements + 255) / 256, 65535));
    calculateDistances<<<blocks, 256>>>(d_points, d_distances, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<Cluster> clusters;
    std::vector<int> result(static_cast<size_t>(n) + 2);
    int remaining = n;
    while (remaining > 0) {
        evaluateSeeds<false><<<n, BLOCK_SIZE, dynamic_shared>>>(d_distances, n, threshold,
            d_clustered, d_counts, d_workspace, d_result);
        CUDA_CHECK(cudaGetLastError());
        evaluateSeeds<true><<<1, BLOCK_SIZE, dynamic_shared>>>(d_distances, n, threshold,
            d_clustered, d_counts, d_workspace, d_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(result.data(), d_result, result.size() * sizeof(int), cudaMemcpyDeviceToHost));
        Cluster cluster;
        cluster.seed_point = result[0];
        cluster.members.assign(result.begin() + 2, result.begin() + 2 + result[1]);
        remaining -= result[1];
        clusters.push_back(std::move(cluster));
        // If the largest candidate is a singleton, all remaining candidates
        // are singletons too, and the original seed-order tie break fixes
        // their order without further candidate evaluations.
        if (result[1] == 1 && remaining > 0) {
            std::vector<int> clustered(n);
            CUDA_CHECK(cudaMemcpy(clustered.data(), d_clustered,
                                  static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                if (!clustered[i]) clusters.push_back(Cluster{{i}, i});
            }
            break;
        }
    }
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_points));
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
