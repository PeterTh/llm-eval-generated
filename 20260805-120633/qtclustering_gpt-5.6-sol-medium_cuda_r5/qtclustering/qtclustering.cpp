// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
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

namespace {

constexpr int CUDA_BLOCK_SIZE = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr)

__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances,
                                    int point_count,
                                    size_t pair_count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pair_count) return;
    const int member = static_cast<int>(index / point_count);
    const int candidate = static_cast<int>(index - static_cast<size_t>(member) * point_count);
    const double dx = points[candidate].x - points[member].x;
    const double dy = points[candidate].y - points[member].y;
    distances[index] = sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ bool betterCandidate(double lhs_distance, int lhs_index,
                                                double rhs_distance, int rhs_index) {
    return lhs_distance < rhs_distance ||
           (lhs_distance == rhs_distance && lhs_index < rhs_index);
}

// Each block independently constructs one seed's greedy candidate cluster. Threads
// cooperate over candidates; blocks expose parallelism over all currently live seeds.
// max_distance is updated only for the newly inserted member, which is algebraically
// identical to rescanning all members on every iteration.
__global__ void constructCandidateClusters(const Point* __restrict__ points,
                                           const double* __restrict__ distances,
                                           const unsigned char* __restrict__ clustered,
                                           const int* __restrict__ seeds,
                                           int seed_offset,
                                           int point_count,
                                           double threshold,
                                           double* __restrict__ workspace,
                                           int* __restrict__ cardinalities,
                                           int* __restrict__ stored_members) {
    __shared__ double reduction_distance[CUDA_BLOCK_SIZE];
    __shared__ int reduction_index[CUDA_BLOCK_SIZE];
    __shared__ int current_member;
    __shared__ int member_count;

    const int tid = threadIdx.x;
    const int seed_position = seed_offset + static_cast<int>(blockIdx.x);
    const int seed = seeds[seed_position];
    double* const max_distance = workspace + static_cast<size_t>(blockIdx.x) * point_count;

    for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
        max_distance[candidate] = (clustered[candidate] || candidate == seed) ? -1.0 : 0.0;
    }
    if (tid == 0) {
        current_member = seed;
        member_count = 1;
        if (stored_members) stored_members[0] = seed;
    }
    __syncthreads();

    while (current_member >= 0) {
        const int member = current_member;
        double local_distance = DBL_MAX;
        int local_index = INT_MAX;

        for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
            double candidate_distance = max_distance[candidate];
            if (candidate_distance < 0.0) continue;

            double new_distance;
            if (distances) {
                new_distance = distances[static_cast<size_t>(member) * point_count + candidate];
            } else {
                const double dx = points[candidate].x - points[member].x;
                const double dy = points[candidate].y - points[member].y;
                new_distance = sqrt(dx * dx + dy * dy);
            }
            candidate_distance = fmax(candidate_distance, new_distance);
            max_distance[candidate] = candidate_distance;
            if (candidate_distance < threshold &&
                betterCandidate(candidate_distance, candidate, local_distance, local_index)) {
                local_distance = candidate_distance;
                local_index = candidate;
            }
        }

        reduction_distance[tid] = local_distance;
        reduction_index[tid] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride &&
                betterCandidate(reduction_distance[tid + stride], reduction_index[tid + stride],
                                reduction_distance[tid], reduction_index[tid])) {
                reduction_distance[tid] = reduction_distance[tid + stride];
                reduction_index[tid] = reduction_index[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int selected = reduction_index[0] == INT_MAX ? -1 : reduction_index[0];
            if (selected >= 0) {
                max_distance[selected] = -1.0;
                if (stored_members) stored_members[member_count] = selected;
                ++member_count;
            }
            current_member = selected;
            if (selected < 0) cardinalities[seed_position] = member_count;
        }
        __syncthreads();
    }
}

} // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_seeds = nullptr;
    int* device_cardinalities = nullptr;
    int* device_members = nullptr;
    double* device_distances = nullptr;
    double* device_workspace = nullptr;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points),
                          static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered),
                          static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_seeds),
                          static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cardinalities),
                          static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_members),
                          static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(), static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));

    const size_t pair_count = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (N != 0 && pair_count / static_cast<size_t>(N) != static_cast<size_t>(N)) {
        std::fprintf(stderr, "Point count is too large for CUDA indexing\n");
        std::exit(EXIT_FAILURE);
    }

    // Cache the all-pairs matrix whenever it fits comfortably. Large problems still
    // remain fully GPU-parallel and compute distances on demand instead of failing.
    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    (void)total_memory;
    const bool matrix_size_valid =
        pair_count <= std::numeric_limits<size_t>::max() / sizeof(double);
    const size_t distance_bytes = matrix_size_valid
                                      ? pair_count * sizeof(double)
                                      : std::numeric_limits<size_t>::max();
    const size_t one_row_bytes = static_cast<size_t>(N) * sizeof(double);
    const size_t distance_budget = (free_memory / 10) * 7;
    if (matrix_size_valid && distance_bytes <= distance_budget &&
        distance_bytes + one_row_bytes >= distance_bytes) {
        const cudaError_t allocation_status =
            cudaMalloc(reinterpret_cast<void**>(&device_distances), distance_bytes);
        if (allocation_status == cudaSuccess) {
            const size_t blocks = (pair_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            buildDistanceMatrix<<<static_cast<unsigned int>(blocks), CUDA_BLOCK_SIZE>>>(
                device_points, device_distances, N, pair_count);
            CUDA_CHECK(cudaGetLastError());
        } else if (allocation_status == cudaErrorMemoryAllocation) {
            // Fragmentation can make the optimistic memory query stale. The kernel's
            // on-demand distance path has identical behavior and needs no matrix.
            (void)cudaGetLastError();
            device_distances = nullptr;
        } else {
            CUDA_CHECK(allocation_status);
        }
    }

    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    size_t workspace_rows = std::min(static_cast<size_t>(N),
                                     ((free_memory / 10) * 7) / one_row_bytes);
    workspace_rows = std::max<size_t>(workspace_rows, 1);
    while (true) {
        const cudaError_t allocation_status = cudaMalloc(
            reinterpret_cast<void**>(&device_workspace), workspace_rows * one_row_bytes);
        if (allocation_status == cudaSuccess) break;
        if (allocation_status != cudaErrorMemoryAllocation || workspace_rows == 1) {
            CUDA_CHECK(allocation_status);
        }
        (void)cudaGetLastError();
        workspace_rows = std::max<size_t>(workspace_rows / 2, 1);
    }

    std::vector<int> cardinalities(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        size_t best_seed_position = 0;

        CUDA_CHECK(cudaMemcpy(device_clustered, clustered.data(),
                              static_cast<size_t>(N) * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_seeds, unclustered_indices.data(),
                              unclustered_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

        for (size_t offset = 0; offset < unclustered_indices.size(); offset += workspace_rows) {
            const size_t batch_size = std::min(workspace_rows, unclustered_indices.size() - offset);
            constructCandidateClusters<<<static_cast<unsigned int>(batch_size), CUDA_BLOCK_SIZE>>>(
                device_points, device_distances, device_clustered, device_seeds,
                static_cast<int>(offset), N, threshold, device_workspace,
                device_cardinalities, nullptr);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), device_cardinalities,
                              unclustered_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));

        // A strict comparison in ascending seed order reproduces the CPU tie-break.
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_seed_position = i;
            }
        }

        // If every candidate is a singleton, no pair of remaining points is within
        // the threshold. Subsequent iterations can only emit the same ascending
        // sequence of singleton clusters, so finish it without redundant launches.
        if (max_cardinality == 1) {
            for (const int seed : unclustered_indices) {
                Cluster cluster;
                cluster.seed_point = seed;
                cluster.members.push_back(seed);
                clusters.push_back(std::move(cluster));
            }
            break;
        }

        constructCandidateClusters<<<1, CUDA_BLOCK_SIZE>>>(
            device_points, device_distances, device_clustered, device_seeds,
            static_cast<int>(best_seed_position), N, threshold, device_workspace,
            device_cardinalities, device_members);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> best_cluster_members(static_cast<size_t>(max_cardinality));
        CUDA_CHECK(cudaMemcpy(best_cluster_members.data(), device_members,
                              best_cluster_members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        const int best_seed = unclustered_indices[best_seed_position];
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = 1;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaFree(device_workspace));
    if (device_distances) CUDA_CHECK(cudaFree(device_distances));
    CUDA_CHECK(cudaFree(device_members));
    CUDA_CHECK(cudaFree(device_cardinalities));
    CUDA_CHECK(cudaFree(device_seeds));
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
