// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

namespace {

constexpr int CUDA_THREADS = 256;
constexpr double CUDA_INFINITY = HUGE_VAL;

struct BestCandidate {
    int cardinality;
    int seed;
};

inline void checkCuda(const cudaError_t result, const char* expression,
                      const char* file, const int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                     expression, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ bool betterMinimum(const double candidate_distance,
                                                const int candidate_index,
                                                const double best_distance,
                                                const int best_index) {
    return candidate_distance < best_distance ||
           (candidate_distance == best_distance && candidate_index < best_index);
}

__device__ __forceinline__ bool betterCluster(const int candidate_cardinality,
                                               const int candidate_seed,
                                               const int best_cardinality,
                                               const int best_seed) {
    return candidate_cardinality > best_cardinality ||
           (candidate_cardinality == best_cardinality && candidate_seed < best_seed);
}

// One block constructs one candidate cluster.  The running maximum distance
// for every possible member is retained in global memory.  When a member is
// accepted, only its contribution has to be folded into those maxima.  This is
// exactly the same maximum evaluated by the sequential code, without
// repeatedly scanning the already selected cluster.
__global__ void evaluateCandidateClusters(const Point* __restrict__ points,
                                          const int* __restrict__ cluster_ids,
                                          double* __restrict__ workspace,
                                          int* __restrict__ cardinalities,
                                          const int first_seed,
                                          const int seed_count,
                                          const int point_count,
                                          const double threshold) {
    const int batch_seed = static_cast<int>(blockIdx.x);
    if (batch_seed >= seed_count) {
        return;
    }

    const int seed = first_seed + batch_seed;
    const int thread = static_cast<int>(threadIdx.x);

    if (cluster_ids[seed] >= 0) {
        if (thread == 0) {
            cardinalities[seed] = 0;
        }
        return;
    }

    double* const maximum_distances =
        workspace + static_cast<size_t>(batch_seed) * point_count;

    for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
        maximum_distances[candidate] =
            (candidate == seed || cluster_ids[candidate] >= 0)
                ? CUDA_INFINITY
                : distance(points[candidate], points[seed]);
    }
    __syncthreads();

    __shared__ double reduction_distances[CUDA_THREADS];
    __shared__ int reduction_indices[CUDA_THREADS];
    __shared__ int selected_point;

    int cardinality = 1;
    while (true) {
        double best_distance = CUDA_INFINITY;
        int best_index = INT_MAX;
        for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
            const double candidate_distance = maximum_distances[candidate];
            if (candidate_distance < threshold &&
                betterMinimum(candidate_distance, candidate, best_distance, best_index)) {
                best_distance = candidate_distance;
                best_index = candidate;
            }
        }

        reduction_distances[thread] = best_distance;
        reduction_indices[thread] = best_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (thread < offset &&
                betterMinimum(reduction_distances[thread + offset],
                              reduction_indices[thread + offset],
                              reduction_distances[thread], reduction_indices[thread])) {
                reduction_distances[thread] = reduction_distances[thread + offset];
                reduction_indices[thread] = reduction_indices[thread + offset];
            }
            __syncthreads();
        }

        if (thread == 0) {
            selected_point = reduction_indices[0];
        }
        __syncthreads();
        const int selected = selected_point;
        if (selected == INT_MAX) {
            break;
        }

        if (thread == 0) {
            maximum_distances[selected] = CUDA_INFINITY;
            ++cardinality;
        }
        __syncthreads();

        for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
            const double previous_maximum = maximum_distances[candidate];
            if (previous_maximum != CUDA_INFINITY) {
                const double new_distance = distance(points[candidate], points[selected]);
                if (new_distance > previous_maximum) {
                    maximum_distances[candidate] = new_distance;
                }
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        cardinalities[seed] = cardinality;
    }
}

// The serial outer loop needs the largest candidate cluster.  This reduction
// keeps the original first-seed-wins rule for equal cardinalities.
__global__ void selectBestCandidate(const int* __restrict__ cardinalities,
                                    const int point_count,
                                    BestCandidate* __restrict__ best_result) {
    const int thread = static_cast<int>(threadIdx.x);
    int best_cardinality = 0;
    int best_seed = INT_MAX;

    for (int seed = thread; seed < point_count; seed += blockDim.x) {
        const int cardinality = cardinalities[seed];
        if (betterCluster(cardinality, seed, best_cardinality, best_seed)) {
            best_cardinality = cardinality;
            best_seed = seed;
        }
    }

    __shared__ int reduction_cardinalities[CUDA_THREADS];
    __shared__ int reduction_seeds[CUDA_THREADS];
    reduction_cardinalities[thread] = best_cardinality;
    reduction_seeds[thread] = best_seed;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread < offset &&
            betterCluster(reduction_cardinalities[thread + offset], reduction_seeds[thread + offset],
                          reduction_cardinalities[thread], reduction_seeds[thread])) {
            reduction_cardinalities[thread] = reduction_cardinalities[thread + offset];
            reduction_seeds[thread] = reduction_seeds[thread + offset];
        }
        __syncthreads();
    }

    if (thread == 0) {
        *best_result = {reduction_cardinalities[0], reduction_seeds[0]};
    }
}

// Rebuild only the winning cluster.  Candidate construction is deterministic,
// so this produces the same members as the winning evaluation kernel while
// avoiding an N-by-N membership buffer.
__global__ void buildBestCluster(const Point* __restrict__ points,
                                 const int* __restrict__ cluster_ids,
                                 double* __restrict__ workspace,
                                 int* __restrict__ members,
                                 const int seed,
                                 const int point_count,
                                 const double threshold) {
    const int thread = static_cast<int>(threadIdx.x);
    double* const maximum_distances = workspace;

    for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
        maximum_distances[candidate] =
            (candidate == seed || cluster_ids[candidate] >= 0)
                ? CUDA_INFINITY
                : distance(points[candidate], points[seed]);
    }
    __syncthreads();

    __shared__ double reduction_distances[CUDA_THREADS];
    __shared__ int reduction_indices[CUDA_THREADS];
    __shared__ int selected_point;

    int cardinality = 1;
    if (thread == 0) {
        members[0] = seed;
    }

    while (true) {
        double best_distance = CUDA_INFINITY;
        int best_index = INT_MAX;
        for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
            const double candidate_distance = maximum_distances[candidate];
            if (candidate_distance < threshold &&
                betterMinimum(candidate_distance, candidate, best_distance, best_index)) {
                best_distance = candidate_distance;
                best_index = candidate;
            }
        }

        reduction_distances[thread] = best_distance;
        reduction_indices[thread] = best_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (thread < offset &&
                betterMinimum(reduction_distances[thread + offset],
                              reduction_indices[thread + offset],
                              reduction_distances[thread], reduction_indices[thread])) {
                reduction_distances[thread] = reduction_distances[thread + offset];
                reduction_indices[thread] = reduction_indices[thread + offset];
            }
            __syncthreads();
        }

        if (thread == 0) {
            selected_point = reduction_indices[0];
        }
        __syncthreads();
        const int selected = selected_point;
        if (selected == INT_MAX) {
            break;
        }

        if (thread == 0) {
            maximum_distances[selected] = CUDA_INFINITY;
            members[cardinality] = selected;
            ++cardinality;
        }
        __syncthreads();

        for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
            const double previous_maximum = maximum_distances[candidate];
            if (previous_maximum != CUDA_INFINITY) {
                const double new_distance = distance(points[candidate], points[selected]);
                if (new_distance > previous_maximum) {
                    maximum_distances[candidate] = new_distance;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void assignClusterMembers(const int* __restrict__ members,
                                     const int member_count,
                                     const int cluster_id,
                                     int* __restrict__ cluster_ids) {
    const int member = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (member < member_count) {
        cluster_ids[members[member]] = cluster_id;
    }
}

int candidateBatchSize(const int point_count) {
    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    const size_t bytes_per_seed = static_cast<size_t>(point_count) * sizeof(double);
    const size_t usable_workspace = (free_memory * 3) / 4;
    const size_t seeds_that_fit = usable_workspace / bytes_per_seed;
    if (seeds_that_fit == 0) {
        std::fprintf(stderr, "CUDA error: insufficient device memory for one QT candidate workspace\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(std::min(static_cast<size_t>(point_count), seeds_that_fit));
}

}  // namespace

// Main QT clustering algorithm.  All candidate construction and all distance
// work run on CUDA.  The small host loop only serializes the QT rule that a
// completed winning cluster is removed before the next round starts.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    if (N == 0) {
        return clusters;
    }

    Point* device_points = nullptr;
    int* device_cluster_ids = nullptr;
    int* device_cardinalities = nullptr;
    int* device_members = nullptr;
    double* device_workspace = nullptr;
    BestCandidate* device_best = nullptr;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&device_points, static_cast<size_t>(N) * sizeof(*device_points)));
    CUDA_CHECK(cudaMalloc(&device_cluster_ids,
                          static_cast<size_t>(N) * sizeof(*device_cluster_ids)));
    CUDA_CHECK(cudaMalloc(&device_cardinalities,
                          static_cast<size_t>(N) * sizeof(*device_cardinalities)));
    CUDA_CHECK(cudaMalloc(&device_members, static_cast<size_t>(N) * sizeof(*device_members)));
    CUDA_CHECK(cudaMalloc(&device_best, sizeof(*device_best)));

    int batch_size = candidateBatchSize(N);
    size_t workspace_elements = static_cast<size_t>(batch_size) * N;
    cudaError_t allocation = cudaMalloc(&device_workspace, workspace_elements * sizeof(*device_workspace));
    while (allocation == cudaErrorMemoryAllocation && batch_size > 1) {
        batch_size = std::max(1, batch_size / 2);
        workspace_elements = static_cast<size_t>(batch_size) * N;
        allocation = cudaMalloc(&device_workspace, workspace_elements * sizeof(*device_workspace));
    }
    CUDA_CHECK(allocation);

    CUDA_CHECK(cudaMemcpy(device_points, points.data(), static_cast<size_t>(N) * sizeof(*device_points),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_cluster_ids, 0xff,
                          static_cast<size_t>(N) * sizeof(*device_cluster_ids)));

    std::vector<int> cluster_seeds;
    cluster_seeds.reserve(N);
    int assigned_points = 0;

    for (int cluster_id = 0; cluster_id < N; ++cluster_id) {
        for (int first_seed = 0; first_seed < N; first_seed += batch_size) {
            const int seed_count = std::min(batch_size, N - first_seed);
            evaluateCandidateClusters<<<seed_count, CUDA_THREADS>>>(
                device_points, device_cluster_ids, device_workspace, device_cardinalities,
                first_seed, seed_count, N, threshold);
            CUDA_CHECK(cudaGetLastError());
        }

        selectBestCandidate<<<1, CUDA_THREADS>>>(device_cardinalities, N, device_best);
        CUDA_CHECK(cudaGetLastError());

        BestCandidate best{};
        CUDA_CHECK(cudaMemcpy(&best, device_best, sizeof(best), cudaMemcpyDeviceToHost));
        if (best.cardinality <= 0 || best.seed < 0 || best.seed >= N) {
            std::fprintf(stderr,
                         "CUDA error: QT candidate selection produced no cluster "
                         "(cardinality=%d, seed=%d)\n",
                         best.cardinality, best.seed);
            std::exit(EXIT_FAILURE);
        }

        buildBestCluster<<<1, CUDA_THREADS>>>(device_points, device_cluster_ids,
                                               device_workspace, device_members,
                                               best.seed, N, threshold);
        CUDA_CHECK(cudaGetLastError());
        assignClusterMembers<<<(best.cardinality + CUDA_THREADS - 1) / CUDA_THREADS,
                               CUDA_THREADS>>>(device_members, best.cardinality, cluster_id,
                                                device_cluster_ids);
        CUDA_CHECK(cudaGetLastError());
        cluster_seeds.push_back(best.seed);
        assigned_points += best.cardinality;
        if (assigned_points == N) {
            break;
        }
    }

    std::vector<int> cluster_ids(N);
    CUDA_CHECK(cudaMemcpy(cluster_ids.data(), device_cluster_ids,
                          static_cast<size_t>(N) * sizeof(*device_cluster_ids),
                          cudaMemcpyDeviceToHost));

    clusters.resize(cluster_seeds.size());
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        clusters[cluster_id].seed_point = cluster_seeds[cluster_id];
    }
    for (int point = 0; point < N; ++point) {
        const int cluster_id = cluster_ids[point];
        if (cluster_id < 0 || static_cast<size_t>(cluster_id) >= clusters.size()) {
            std::fprintf(stderr, "CUDA error: QT clustering did not assign point %d\n", point);
            std::exit(EXIT_FAILURE);
        }
        clusters[cluster_id].members.push_back(point);
    }

    CUDA_CHECK(cudaFree(device_best));
    CUDA_CHECK(cudaFree(device_workspace));
    CUDA_CHECK(cudaFree(device_members));
    CUDA_CHECK(cudaFree(device_cardinalities));
    CUDA_CHECK(cudaFree(device_cluster_ids));
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

    // CUDA context creation is one-time process setup, not clustering work.
    // Establish it before timing the algorithm, as the CPU version likewise
    // starts its timer after all runtime setup has completed.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

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
