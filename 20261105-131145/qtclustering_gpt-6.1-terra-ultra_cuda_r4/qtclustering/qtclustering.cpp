// QT Clustering Benchmark - Simplified Sequential Version
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
#include <cstdint>
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

// Each block grows one seed's candidate cluster.  The running maximum for
// each point is updated only when a member is added, which is equivalent to
// recomputing the maximum over the entire cluster on every iteration.
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

struct Choice {
    double distance;
    int index;
};

__device__ __forceinline__ Choice betterChoice(Choice a, Choice b) {
    return (b.distance < a.distance ||
            (b.distance == a.distance && b.index < a.index)) ? b : a;
}

__device__ __forceinline__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Match the host's native optimized dx*dx + dy*dy evaluation.
#if defined(__FMA__)
    return sqrt(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
#else
    return sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
#endif
}

__global__ void buildDistances(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        distances[i] = pointDistance(points[i / n], points[i % n]);
    }
}

// A block-wide minimum, with the same lowest-index tie break as the CPU loop.
__device__ __forceinline__ Choice blockMinimum(Choice value, Choice* warp_choices) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        Choice other{__shfl_down_sync(0xffffffff, value.distance, offset),
                     __shfl_down_sync(0xffffffff, value.index, offset)};
        value = betterChoice(value, other);
    }
    if (lane == 0) warp_choices[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < blockDim.x / 32 ? warp_choices[lane] : Choice{DBL_MAX, INT_MAX};
        for (int offset = 16; offset > 0; offset >>= 1) {
            Choice other{__shfl_down_sync(0xffffffff, value.distance, offset),
                         __shfl_down_sync(0xffffffff, value.index, offset)};
            value = betterChoice(value, other);
        }
        if (lane == 0) warp_choices[0] = value;
    }
    __syncthreads();
    return warp_choices[0];
}

// When writing the winner, the result is [cardinality, seed, members...].
// Otherwise each block writes just the cardinality of its seed.
__global__ void growClusters(const Point* points, const double* distances,
                             const unsigned char* clustered, int n, double threshold,
                             int first_seed, int* cardinalities, int* result,
                             double* global_work, unsigned int* memberships, int words, bool write_winner) {
    extern __shared__ double shared_work[];
    __shared__ Choice warp_choices[8];
    const int seed = write_winner ? result[1] : first_seed + blockIdx.x;
    if (seed >= n || clustered[seed]) {
        if (!write_winner && threadIdx.x == 0 && seed < n) cardinalities[seed] = 0;
        return;
    }
    // Unchanged candidate clusters can be reused until an accepted cluster
    // removes one of their members.
    if (!write_winner && memberships && cardinalities[seed] >= 0) return;
    unsigned int* membership = memberships ? memberships + static_cast<size_t>(seed) * words : nullptr;
    if (!write_winner && membership) {
        for (int i = threadIdx.x; i < words; i += blockDim.x) membership[i] = 0;
        __syncthreads();
        if (threadIdx.x == 0) membership[seed / 32] |= 1u << (seed % 32);
    }
    double* work = global_work ? global_work + static_cast<size_t>(blockIdx.x) * n : shared_work;
    for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
        if (candidate == seed || clustered[candidate]) {
            work[candidate] = DBL_MAX;
        } else {
            const double d = distances ? distances[static_cast<size_t>(seed) * n + candidate]
                                       : pointDistance(points[seed], points[candidate]);
            // std::max(0.0, d) leaves a NaN distance at zero.
            work[candidate] = d > 0.0 ? d : 0.0;
        }
    }
    if (write_winner && threadIdx.x == 0) result[2] = seed;
    __syncthreads();
    int count = 1;
    while (count < n) {
        Choice local{DBL_MAX, INT_MAX};
        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            double d = work[candidate];
            if (d < threshold && d < DBL_MAX) {
                local = betterChoice(local, Choice{d, candidate});
            }
        }
        Choice best = blockMinimum(local, warp_choices);
        if (best.index == INT_MAX) break;
        if (threadIdx.x == 0) {
            if (write_winner) result[2 + count] = best.index;
            else if (membership) membership[best.index / 32] |= 1u << (best.index % 32);
        }
        ++count;
        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            if (candidate == best.index) {
                work[candidate] = DBL_MAX;
            } else {
                double current = work[candidate];
                if (current < threshold && current < DBL_MAX) {
                    const double d = distances ? distances[static_cast<size_t>(best.index) * n + candidate]
                                               : pointDistance(points[best.index], points[candidate]);
                    if (d > current) work[candidate] = d;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (write_winner) result[0] = count;
        else cardinalities[seed] = count;
    }
}

__global__ void chooseSeed(const int* cardinalities, int n, int* result) {
    __shared__ Choice warp_choices[8];
    Choice local{0.0, INT_MAX};
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (cardinalities[i] > 0)
            local = betterChoice(local, Choice{-static_cast<double>(cardinalities[i]), i});
    }
    Choice best = blockMinimum(local, warp_choices);
    if (threadIdx.x == 0) {
        result[0] = static_cast<int>(-best.distance);
        result[1] = best.index;
    }
}

// A cached candidate changes only if an accepted cluster removes one of
// its members. Candidates outside the chosen cluster never affect its greedy
// sequence, so disjoint candidate clusters retain their exact cardinality.
__global__ void invalidateCandidates(const unsigned int* memberships, int words,
                                     const unsigned char* clustered, int n,
                                     const int* result, int* cardinalities) {
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= n) return;
    if (clustered[seed]) {
        cardinalities[seed] = 0;
        return;
    }
    const unsigned int* own = memberships + static_cast<size_t>(seed) * words;
    const unsigned int* winner = memberships + static_cast<size_t>(result[1]) * words;
    for (int i = 0; i < words; ++i) {
        if (own[i] & winner[i]) {
            cardinalities[seed] = -1;
            return;
        }
    }
}

__global__ void markCluster(unsigned char* clustered, const int* result) {
    for (int i = threadIdx.x; i < result[0]; i += blockDim.x)
        clustered[result[2 + i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    Point* d_points = nullptr;
    double* d_distances = nullptr;
    double* d_work = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_cardinalities = nullptr;
    int* d_result = nullptr;
    unsigned int* d_memberships = nullptr;
    cudaCheck(cudaMalloc(&d_points, sizeof(Point) * static_cast<size_t>(n)), "allocate points");
    cudaCheck(cudaMalloc(&d_clustered, static_cast<size_t>(n)), "allocate membership");
    cudaCheck(cudaMalloc(&d_cardinalities, sizeof(int) * static_cast<size_t>(n)), "allocate cardinalities");
    cudaCheck(cudaMalloc(&d_result, sizeof(int) * (static_cast<size_t>(n) + 2)), "allocate result");
    cudaCheck(cudaMemcpy(d_points, points.data(), sizeof(Point) * static_cast<size_t>(n), cudaMemcpyHostToDevice), "copy points");
    cudaCheck(cudaMemset(d_clustered, 0, static_cast<size_t>(n)), "initialize membership");

    // The distance table is an optional cache; large problems compute distances
    // on demand to keep device memory bounded.
    size_t free_bytes = 0, total_bytes = 0;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "query memory");
    const size_t row_bytes = sizeof(double) * static_cast<size_t>(n);
    if (row_bytes <= SIZE_MAX / static_cast<size_t>(n)) {
        const size_t matrix_bytes = row_bytes * static_cast<size_t>(n);
        if (matrix_bytes <= (size_t{512} << 20) && matrix_bytes <= free_bytes / 3) {
            cudaCheck(cudaMalloc(&d_distances, matrix_bytes), "allocate distance cache");
            int blocks = static_cast<int>(std::min<size_t>((static_cast<size_t>(n) * n + 255) / 256, 65535));
            buildDistances<<<blocks, 256>>>(d_points, d_distances, n);
            cudaCheck(cudaGetLastError(), "launch distance cache");
        }
    }

    // Cache candidate memberships when the bit matrix fits comfortably.
    // This avoids rebuilding clusters unaffected by each winning cluster.
    const int words = (n - 1) / 32 + 1;
    const size_t membership_row = sizeof(unsigned int) * static_cast<size_t>(words);
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "query cache memory");
    if (membership_row <= SIZE_MAX / static_cast<size_t>(n)) {
        size_t bytes = membership_row * static_cast<size_t>(n);
        if (bytes <= (size_t{256} << 20) && bytes <= free_bytes / 4)
            cudaCheck(cudaMalloc(&d_memberships, bytes), "allocate candidate cache");
    }
    cudaCheck(cudaMemset(d_cardinalities, 0xff, sizeof(int) * static_cast<size_t>(n)), "initialize cardinalities");

    int device = 0;
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDevice(&device), "get device");
    cudaCheck(cudaGetDeviceProperties(&properties, device), "get device properties");
    // Leave room for the static reduction buffer and avoid opt-in shared memory.
    const bool use_shared = row_bytes + 8 * sizeof(Choice) <= properties.sharedMemPerBlock;
    size_t shared_bytes = use_shared ? row_bytes : 0;
    int batch_size = n;
    if (!use_shared) {
        cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "query work memory");
        size_t budget = std::min<size_t>(size_t{256} << 20, free_bytes / 2);
        size_t rows = std::max<size_t>(1, budget / row_bytes);
        batch_size = static_cast<int>(std::min<size_t>(static_cast<size_t>(n), rows));
        cudaCheck(cudaMalloc(&d_work, row_bytes * static_cast<size_t>(batch_size)), "allocate cluster workspace");
    }
    std::vector<int> host_result(static_cast<size_t>(n) + 2);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        for (int first = 0; first < n; first += batch_size) {
            int count = std::min(batch_size, n - first);
            growClusters<<<count, 256, shared_bytes>>>(d_points, d_distances, d_clustered, n, threshold,
                                                      first, d_cardinalities, d_result, d_work, d_memberships, words, false);
            cudaCheck(cudaGetLastError(), "launch candidate clusters");
        }
        chooseSeed<<<1, 256>>>(d_cardinalities, n, d_result);
        cudaCheck(cudaGetLastError(), "launch seed selection");
        growClusters<<<1, 256, shared_bytes>>>(d_points, d_distances, d_clustered, n, threshold,
                                              0, d_cardinalities, d_result, d_work, d_memberships, words, true);
        cudaCheck(cudaGetLastError(), "launch winning cluster");
        markCluster<<<1, 256>>>(d_clustered, d_result);
        cudaCheck(cudaGetLastError(), "launch cluster marking");
        if (d_memberships) {
            invalidateCandidates<<<(n - 1) / 256 + 1, 256>>>(d_memberships, words, d_clustered,
                                                           n, d_result, d_cardinalities);
            cudaCheck(cudaGetLastError(), "launch candidate invalidation");
        }
        cudaCheck(cudaMemcpy(host_result.data(), d_result, sizeof(int) * (static_cast<size_t>(n) + 2),
                             cudaMemcpyDeviceToHost), "copy cluster result");
        const int count = host_result[0];
        Cluster cluster;
        cluster.seed_point = host_result[1];
        cluster.members.assign(host_result.begin() + 2, host_result.begin() + 2 + count);
        clusters.push_back(std::move(cluster));
        remaining -= count;
    }
    cudaFree(d_memberships);
    cudaFree(d_work);
    cudaFree(d_distances);
    cudaFree(d_result);
    cudaFree(d_cardinalities);
    cudaFree(d_clustered);
    cudaFree(d_points);
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
    
    // Initialize the CUDA context before timing the clustering work.
    cudaCheck(cudaFree(nullptr), "initialize CUDA");

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
