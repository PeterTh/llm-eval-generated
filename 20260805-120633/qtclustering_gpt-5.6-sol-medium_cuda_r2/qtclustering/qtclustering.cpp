// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error_ = (call);                                       \
        if (error_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(error_));                   \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

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
        // N <= 30 made the original generator spin forever because truncation
        // always produced zero points.
        if (N <= 30 && group_cnt == 0) group_cnt = 1;
        
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
inline double pointDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

namespace {

constexpr int CUDA_BLOCK_SIZE = 256;

// The matrix is stored transposed (member-major), so all threads in a warp read
// consecutive candidate distances while growing a cluster.
__global__ void buildDistanceMatrix(const Point* points, double* distances,
                                    const int point_count) {
    for (int member = blockIdx.y * blockDim.y + threadIdx.y;
         member < point_count; member += blockDim.y * gridDim.y) {
        const Point member_point = points[member];
        for (int candidate = blockIdx.x * blockDim.x + threadIdx.x;
             candidate < point_count; candidate += blockDim.x * gridDim.x) {
            const double dx = points[candidate].x - member_point.x;
            const double dy = points[candidate].y - member_point.y;
            distances[static_cast<size_t>(member) * point_count + candidate] =
                sqrt(dx * dx + dy * dy);
        }
    }
}

__device__ __forceinline__ double deviceDistance(const Point* points,
                                                  const double* distances,
                                                  const int point_count,
                                                  const int candidate,
                                                  const int member) {
    if (distances != nullptr) {
        return distances[static_cast<size_t>(member) * point_count + candidate];
    }
    const double dx = points[candidate].x - points[member].x;
    const double dy = points[candidate].y - points[member].y;
    return sqrt(dx * dx + dy * dy);
}

// One block builds one seed's greedy candidate. Threads parallelize the scan
// over candidate points; blocks parallelize independent seeds. max_diameters
// incrementally caches max(distance(candidate, each member)), changing the
// original O(k^2*N) seed construction into O(k*N) without changing choices.
__global__ void generateCandidates(const Point* points, const double* distances,
                                   const unsigned char* clustered,
                                   double* max_diameters, const int point_count,
                                   const double threshold, const int seed_begin,
                                   unsigned long long* best_key,
                                   int* output_members) {
    const int row = blockIdx.x;
    const int seed = seed_begin + row;
    if (seed >= point_count || clustered[seed]) {
        return;
    }

    double* const state = max_diameters + static_cast<size_t>(row) * point_count;
    __shared__ double block_dist[CUDA_BLOCK_SIZE];
    __shared__ int block_index[CUDA_BLOCK_SIZE];
    __shared__ int current_member;
    __shared__ int member_count;

    if (threadIdx.x == 0) {
        current_member = seed;
        member_count = 1;
        if (output_members != nullptr) output_members[0] = seed;
    }
    for (int candidate = threadIdx.x; candidate < point_count;
         candidate += blockDim.x) {
        state[candidate] = (clustered[candidate] || candidate == seed)
                               ? DBL_MAX
                               : 0.0;
    }
    __syncthreads();

    while (true) {
        double thread_best = DBL_MAX;
        int thread_index = -1;

        for (int candidate = threadIdx.x; candidate < point_count;
             candidate += blockDim.x) {
            double max_distance = state[candidate];
            if (max_distance != DBL_MAX) {
                const double d = deviceDistance(points, distances, point_count,
                                                candidate, current_member);
                if (d > max_distance) max_distance = d;
                state[candidate] = max_distance;
                if (max_distance < threshold &&
                    (max_distance < thread_best ||
                     (max_distance == thread_best && candidate < thread_index))) {
                    thread_best = max_distance;
                    thread_index = candidate;
                }
            }
        }

        block_dist[threadIdx.x] = thread_best;
        block_index[threadIdx.x] = thread_index;
        __syncthreads();

        for (int offset = CUDA_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                const double other_dist = block_dist[threadIdx.x + offset];
                const int other_index = block_index[threadIdx.x + offset];
                if (other_index >= 0 &&
                    (block_index[threadIdx.x] < 0 ||
                     other_dist < block_dist[threadIdx.x] ||
                     (other_dist == block_dist[threadIdx.x] &&
                      other_index < block_index[threadIdx.x]))) {
                    block_dist[threadIdx.x] = other_dist;
                    block_index[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }

        const int selected = block_index[0];
        if (selected < 0) break;

        if (threadIdx.x == 0) {
            state[selected] = DBL_MAX;
            current_member = selected;
            if (output_members != nullptr) output_members[member_count] = selected;
            ++member_count;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && best_key != nullptr) {
        // Higher cardinality wins; for a tie, the lower seed index wins.
        const unsigned long long key =
            (static_cast<unsigned long long>(member_count) << 32) |
            static_cast<unsigned int>(0xffffffffu - static_cast<unsigned int>(seed));
        atomicMax(best_key, key);
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members,
                              const int member_count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < member_count;
         i += blockDim.x * gridDim.x) {
        clustered[members[i]] = 1;
    }
}

} // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    double* device_distances = nullptr;
    double* device_state = nullptr;
    int* device_members = nullptr;
    unsigned long long* device_best = nullptr;

    CUDA_CHECK(cudaMalloc(&device_points, static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&device_clustered, static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&device_members, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_best, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(),
                          static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered, 0, static_cast<size_t>(N)));

    if (static_cast<size_t>(N) >
        std::numeric_limits<size_t>::max() / sizeof(double) / static_cast<size_t>(N)) {
        std::fprintf(stderr, "Problem size is too large\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t matrix_bytes =
        static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(double);

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // Cache all pair distances whenever it leaves ample room for candidate
    // state. For very large inputs the same CUDA kernel computes distances on
    // demand, so GPU execution and bounded-memory seed batching remain intact.
    if (matrix_bytes <= free_bytes * 45 / 100) {
        const cudaError_t allocation = cudaMalloc(&device_distances, matrix_bytes);
        if (allocation == cudaSuccess) {
            const dim3 block(32, 8);
            const dim3 grid(std::min((N + 31) / 32, 65535),
                            std::min((N + 7) / 8, 65535));
            buildDistanceMatrix<<<grid, block>>>(device_points, device_distances, N);
            CUDA_CHECK(cudaGetLastError());
        } else {
            device_distances = nullptr;
            (void)cudaGetLastError();
        }
    }

    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t row_bytes = static_cast<size_t>(N) * sizeof(double);
    size_t batch_size = std::min(static_cast<size_t>(N),
                                 (free_bytes * 70 / 100) / row_bytes);
    if (batch_size == 0) batch_size = 1;
    CUDA_CHECK(cudaMalloc(&device_state, batch_size * row_bytes));

    int remaining = N;
    while (remaining > 0) {
        CUDA_CHECK(cudaMemset(device_best, 0, sizeof(unsigned long long)));
        for (int seed_begin = 0; seed_begin < N;
             seed_begin += static_cast<int>(batch_size)) {
            const int seeds = std::min(static_cast<int>(batch_size), N - seed_begin);
            generateCandidates<<<seeds, CUDA_BLOCK_SIZE>>>(
                device_points, device_distances, device_clustered, device_state,
                N, threshold, seed_begin, device_best, nullptr);
            CUDA_CHECK(cudaGetLastError());
        }

        unsigned long long best_key = 0;
        CUDA_CHECK(cudaMemcpy(&best_key, device_best, sizeof(best_key),
                              cudaMemcpyDeviceToHost));
        const int max_cardinality = static_cast<int>(best_key >> 32);
        const int best_seed = static_cast<int>(
            0xffffffffu - static_cast<unsigned int>(best_key));
        if (max_cardinality <= 0 || best_seed < 0 || best_seed >= N) break;

        // Re-run only the winning seed to recover its deterministic member
        // order. This avoids an N-by-N integer output matrix for all losers.
        generateCandidates<<<1, CUDA_BLOCK_SIZE>>>(
            device_points, device_distances, device_clustered, device_state, N,
            threshold, best_seed, nullptr, device_members);
        CUDA_CHECK(cudaGetLastError());

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(max_cardinality);
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), device_members,
                              static_cast<size_t>(max_cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        clusters.push_back(std::move(cluster));

        const int mark_blocks = std::min((max_cardinality + CUDA_BLOCK_SIZE - 1) /
                                             CUDA_BLOCK_SIZE,
                                         65535);
        markClustered<<<mark_blocks, CUDA_BLOCK_SIZE>>>(
            device_clustered, device_members, max_cardinality);
        CUDA_CHECK(cudaGetLastError());
        remaining -= max_cardinality;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_best));
    CUDA_CHECK(cudaFree(device_members));
    CUDA_CHECK(cudaFree(device_state));
    if (device_distances != nullptr) CUDA_CHECK(cudaFree(device_distances));
    CUDA_CHECK(cudaFree(device_clustered));
    CUDA_CHECK(cudaFree(device_points));
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
                const double dist = pointDistance(points[cluster.members[i]],
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
