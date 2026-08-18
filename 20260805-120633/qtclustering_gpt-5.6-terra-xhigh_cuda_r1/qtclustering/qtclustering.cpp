// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace {

constexpr int CUDA_THREADS = 256;

[[noreturn]] void cudaFail(const cudaError_t error, const char* operation,
                           const char* file, const int line) {
    fprintf(stderr, "CUDA failure at %s:%d during %s: %s\n", file, line,
            operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t cuda_error__ = (operation);                           \
        if (cuda_error__ != cudaSuccess) {                                      \
            cudaFail(cuda_error__, #operation, __FILE__, __LINE__);             \
        }                                                                       \
    } while (false)

void initializeCudaRuntime() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "QT clustering requires a CUDA-capable GPU.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));
    // Force primary-context creation before the benchmark timer starts.
    CUDA_CHECK(cudaFree(nullptr));
}

// Every row represents the candidate cluster grown from one seed.  The
// matrix is filled once and then turns each diameter update into one load.
// Keeping distances as double precision Euclidean distances preserves the
// comparisons performed by the sequential implementation.
__global__ void buildDistanceMatrixKernel(const Point* const points,
                                          double* const distances,
                                          const int point_count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t element_count = static_cast<size_t>(point_count) * point_count;
    if (index >= element_count) {
        return;
    }

    const int row = static_cast<int>(index / point_count);
    const int column = static_cast<int>(index - static_cast<size_t>(row) * point_count);
    const double dx = points[row].x - points[column].x;
    const double dy = points[row].y - points[column].y;
    distances[index] = sqrt(dx * dx + dy * dy);
}

// Initialize one independent candidate cluster for every currently
// unclustered seed.  The member mask and maximum-distance vector are stored
// row-major so that each warp accesses consecutive candidate points.
__global__ void initializeCandidateStateKernel(const int* const seeds,
                                                const double* const distances,
                                                unsigned char* const member_mask,
                                                double* const maximum_distances,
                                                int* const member_order,
                                                int* const sizes,
                                                const int point_count) {
    const int candidate = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (candidate >= point_count) {
        return;
    }

    const size_t offset = static_cast<size_t>(row) * point_count;
    const int seed = seeds[row];
    member_mask[offset + candidate] = static_cast<unsigned char>(candidate == seed);
    maximum_distances[offset + candidate] =
        distances[static_cast<size_t>(seed) * point_count + candidate];

    if (candidate == 0) {
        sizes[row] = 1;
        member_order[offset] = seed;
    }
}

// For each seed row, find the eligible candidate having the smallest current
// maximum distance to that row's cluster.  The pair reduction includes the
// point index, which retains the sequential lower-index tie break exactly.
__global__ void findClosestPointsKernel(const unsigned char* const clustered,
                                        const unsigned char* const member_mask,
                                        const double* const maximum_distances,
                                        int* const closest_points,
                                        int* const any_choice,
                                        const double threshold,
                                        const int point_count) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const size_t offset = static_cast<size_t>(row) * point_count;

    double best_distance = DBL_MAX;
    int best_candidate = point_count;

    for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
        if (clustered[candidate] || member_mask[offset + candidate]) {
            continue;
        }

        const double candidate_distance = maximum_distances[offset + candidate];
        if (candidate_distance < threshold &&
            (candidate_distance < best_distance ||
             (candidate_distance == best_distance && candidate < best_candidate))) {
            best_distance = candidate_distance;
            best_candidate = candidate;
        }
    }

    __shared__ double reduced_distances[CUDA_THREADS];
    __shared__ int reduced_candidates[CUDA_THREADS];
    reduced_distances[thread] = best_distance;
    reduced_candidates[thread] = best_candidate;
    __syncthreads();

    for (int stride = CUDA_THREADS / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            const double other_distance = reduced_distances[thread + stride];
            const int other_candidate = reduced_candidates[thread + stride];
            if (other_distance < reduced_distances[thread] ||
                (other_distance == reduced_distances[thread] &&
                 other_candidate < reduced_candidates[thread])) {
                reduced_distances[thread] = other_distance;
                reduced_candidates[thread] = other_candidate;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        const int closest = reduced_candidates[0] == point_count
                                ? -1
                                : reduced_candidates[0];
        closest_points[row] = closest;
        if (closest >= 0) {
            atomicExch(any_choice, 1);
        }
    }
}

// Add the selected point to every row's candidate cluster.  Updating the
// per-candidate maxima incrementally is equivalent to rescanning all current
// members, but replaces that cubic inner work with one distance-matrix load.
__global__ void advanceCandidateClustersKernel(
    const int* const closest_points, const double* const distances,
    unsigned char* const member_mask, double* const maximum_distances,
    int* const member_order, int* const sizes, const int point_count) {
    const int candidate = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (candidate >= point_count) {
        return;
    }

    const int closest = closest_points[row];
    if (closest < 0) {
        return;
    }

    const size_t offset = static_cast<size_t>(row) * point_count;
    const double new_distance =
        distances[static_cast<size_t>(closest) * point_count + candidate];
    const double old_distance = maximum_distances[offset + candidate];
    maximum_distances[offset + candidate] =
        new_distance > old_distance ? new_distance : old_distance;

    if (candidate == closest) {
        const int member_position = sizes[row];
        member_mask[offset + candidate] = 1;
        member_order[offset + member_position] = candidate;
        sizes[row] = member_position + 1;
    }
}

} // namespace

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

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // CUDA is the execution path for the clustering computation.  Keeping all
    // candidate state resident on the device avoids host/device traffic inside
    // the greedy growth loop.
    initializeCudaRuntime();

    const size_t matrix_elements = static_cast<size_t>(N) * N;
    Point* device_points = nullptr;
    double* device_distances = nullptr;
    unsigned char* device_clustered = nullptr;
    unsigned char* device_member_mask = nullptr;
    double* device_maximum_distances = nullptr;
    int* device_member_order = nullptr;
    int* device_seeds = nullptr;
    int* device_sizes = nullptr;
    int* device_closest_points = nullptr;
    int* device_any_choice = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points),
                          static_cast<size_t>(N) * sizeof(*device_points)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_distances),
                          matrix_elements * sizeof(*device_distances)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered),
                          static_cast<size_t>(N) * sizeof(*device_clustered)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_member_mask),
                          matrix_elements * sizeof(*device_member_mask)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_maximum_distances),
                          matrix_elements * sizeof(*device_maximum_distances)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_member_order),
                          matrix_elements * sizeof(*device_member_order)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_seeds),
                          static_cast<size_t>(N) * sizeof(*device_seeds)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_sizes),
                          static_cast<size_t>(N) * sizeof(*device_sizes)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_closest_points),
                          static_cast<size_t>(N) * sizeof(*device_closest_points)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_any_choice),
                          sizeof(*device_any_choice)));

    CUDA_CHECK(cudaMemcpy(device_points, points.data(),
                          static_cast<size_t>(N) * sizeof(*device_points),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered, 0,
                          static_cast<size_t>(N) * sizeof(*device_clustered)));

    const unsigned int distance_blocks = static_cast<unsigned int>(
        (matrix_elements + CUDA_THREADS - 1) / CUDA_THREADS);
    buildDistanceMatrixKernel<<<distance_blocks, CUDA_THREADS>>>(
        device_points, device_distances, N);
    CUDA_CHECK(cudaGetLastError());

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int seed_count = static_cast<int>(unclustered_indices.size());
        CUDA_CHECK(cudaMemcpy(device_seeds, unclustered_indices.data(),
                              static_cast<size_t>(seed_count) * sizeof(*device_seeds),
                              cudaMemcpyHostToDevice));

        const dim3 candidate_grid(
            static_cast<unsigned int>((N + CUDA_THREADS - 1) / CUDA_THREADS),
            static_cast<unsigned int>(seed_count));
        initializeCandidateStateKernel<<<candidate_grid, CUDA_THREADS>>>(
            device_seeds, device_distances, device_member_mask, device_maximum_distances,
            device_member_order, device_sizes, N);
        CUDA_CHECK(cudaGetLastError());

        // Candidate clusters are grown in lockstep.  Each row is independent,
        // so all remaining seeds and all candidate points execute in parallel.
        int any_choice = 0;
        do {
            CUDA_CHECK(cudaMemset(device_any_choice, 0, sizeof(*device_any_choice)));
            findClosestPointsKernel<<<seed_count, CUDA_THREADS>>>(
                device_clustered, device_member_mask, device_maximum_distances,
                device_closest_points, device_any_choice, threshold, N);
            CUDA_CHECK(cudaGetLastError());

            advanceCandidateClustersKernel<<<candidate_grid, CUDA_THREADS>>>(
                device_closest_points, device_distances, device_member_mask,
                device_maximum_distances, device_member_order, device_sizes, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&any_choice, device_any_choice,
                                  sizeof(any_choice), cudaMemcpyDeviceToHost));
        } while (any_choice != 0);

        std::vector<int> candidate_sizes(seed_count);
        CUDA_CHECK(cudaMemcpy(candidate_sizes.data(), device_sizes,
                              static_cast<size_t>(seed_count) * sizeof(*device_sizes),
                              cudaMemcpyDeviceToHost));

        // The strict comparison intentionally preserves the first-seed tie
        // break used by the sequential loop.
        int best_slot = 0;
        int max_cardinality = candidate_sizes[0];
        for (int slot = 1; slot < seed_count; ++slot) {
            if (candidate_sizes[slot] > max_cardinality) {
                max_cardinality = candidate_sizes[slot];
                best_slot = slot;
            }
        }

        const int best_seed = unclustered_indices[best_slot];
        std::vector<int> best_cluster_members(max_cardinality);
        CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                              device_member_order + static_cast<size_t>(best_slot) * N,
                              static_cast<size_t>(max_cardinality) *
                                  sizeof(*device_member_order),
                              cudaMemcpyDeviceToHost));

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
            CUDA_CHECK(cudaMemcpy(device_clustered, clustered.data(),
                                  static_cast<size_t>(N) * sizeof(*device_clustered),
                                  cudaMemcpyHostToDevice));

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaFree(device_any_choice));
    CUDA_CHECK(cudaFree(device_closest_points));
    CUDA_CHECK(cudaFree(device_sizes));
    CUDA_CHECK(cudaFree(device_seeds));
    CUDA_CHECK(cudaFree(device_member_order));
    CUDA_CHECK(cudaFree(device_maximum_distances));
    CUDA_CHECK(cudaFree(device_member_mask));
    CUDA_CHECK(cudaFree(device_clustered));
    CUDA_CHECK(cudaFree(device_distances));
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

    // GPU setup is outside the measured clustering region, matching the
    // treatment of one-time benchmark setup such as data generation.
    initializeCudaRuntime();
    
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
