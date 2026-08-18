// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <cfloat>
#include <algorithm>
#include <chrono>
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
constexpr double CUDA_MAX_DOUBLE = 1.79769313486231570815e+308;

inline void checkCuda(const cudaError_t error, const char* expression,
                      const char* file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line,
                     expression, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ double deviceDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Build the complete pairwise distance matrix once. The matrix is reused by
// every candidate seed and by every growth step of every candidate cluster.
__global__ void buildDistanceMatrix(const Point* points, double* distances,
                                    const int point_count) {
    const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int member = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (candidate >= point_count || member >= point_count) return;

    distances[static_cast<size_t>(candidate) * point_count + member] =
        deviceDistance(points[candidate], points[member]);
}

// Each row is the independent candidate cluster generated from one seed.
// max_distances stores the current diameter contribution for each candidate;
// it is updated incrementally when a member is added.
__global__ void initializeCandidateState(
    const double* distances, const unsigned char* clustered,
    unsigned char* in_cluster, double* max_distances, int* member_order,
    int* cardinalities, unsigned char* active, const int point_count) {
    const size_t flat_index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    const size_t matrix_size = static_cast<size_t>(point_count) * point_count;
    if (flat_index >= matrix_size) return;

    const int seed = static_cast<int>(flat_index / point_count);
    const int candidate = static_cast<int>(flat_index % point_count);
    const bool available_seed = clustered[seed] == 0;

    if (!available_seed) {
        if (candidate == 0) {
            cardinalities[seed] = 0;
            active[seed] = 0;
        }
        return;
    }

    const bool is_seed = candidate == seed;
    in_cluster[flat_index] = available_seed && is_seed;
    member_order[flat_index] = -1;
    max_distances[flat_index] = is_seed
        ? CUDA_MAX_DOUBLE
        : distances[static_cast<size_t>(candidate) * point_count + seed];

    if (candidate == 0) {
        cardinalities[seed] = 1;
        active[seed] = point_count > 1;
        member_order[flat_index] = seed;
    }
}

__device__ __forceinline__ bool isBetterCandidate(
    const double candidate_distance, const int candidate,
    const double current_distance, const int current_candidate) {
    return current_candidate < 0 || candidate_distance < current_distance ||
           (candidate_distance == current_distance && candidate < current_candidate);
}

// One block handles one seed. Threads cooperatively scan candidates, reduce
// to the lowest-index candidate among equal distances, append that candidate,
// and update all remaining diameter contributions in parallel.
__global__ void growCandidateClusters(
    const double* distances, const unsigned char* clustered,
    unsigned char* in_cluster, double* max_distances, int* member_order,
    int* cardinalities, unsigned char* active, int* active_count,
    const double threshold, const int point_count) {
    const int seed = static_cast<int>(blockIdx.x);
    if (seed >= point_count || active[seed] == 0) return;

    __shared__ double best_distances[CUDA_BLOCK_SIZE];
    __shared__ int best_candidates[CUDA_BLOCK_SIZE];
    __shared__ int chosen_candidate;
    __shared__ int update_scores;

    const int thread = static_cast<int>(threadIdx.x);
    const size_t row_offset = static_cast<size_t>(seed) * point_count;
    double local_best_distance = CUDA_MAX_DOUBLE;
    int local_best_candidate = -1;

    for (int candidate = thread; candidate < point_count;
         candidate += CUDA_BLOCK_SIZE) {
        if (clustered[candidate] || in_cluster[row_offset + candidate]) continue;

        const double candidate_distance = max_distances[row_offset + candidate];
        if (candidate_distance < threshold &&
            isBetterCandidate(candidate_distance, candidate,
                              local_best_distance, local_best_candidate)) {
            local_best_distance = candidate_distance;
            local_best_candidate = candidate;
        }
    }

    best_distances[thread] = local_best_distance;
    best_candidates[thread] = local_best_candidate;
    __syncthreads();

    for (int offset = CUDA_BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (thread < offset &&
            isBetterCandidate(best_distances[thread + offset],
                              best_candidates[thread + offset],
                              best_distances[thread], best_candidates[thread])) {
            best_distances[thread] = best_distances[thread + offset];
            best_candidates[thread] = best_candidates[thread + offset];
        }
        __syncthreads();
    }

    if (thread == 0) {
        chosen_candidate = best_candidates[0];
        update_scores = 0;

        if (chosen_candidate < 0) {
            active[seed] = 0;
            atomicSub(active_count, 1);
        } else {
            const int old_cardinality = cardinalities[seed];
            in_cluster[row_offset + chosen_candidate] = 1;
            member_order[row_offset + old_cardinality] = chosen_candidate;
            cardinalities[seed] = old_cardinality + 1;

            if (old_cardinality + 1 >= point_count) {
                active[seed] = 0;
                atomicSub(active_count, 1);
            } else {
                update_scores = 1;
            }
        }
    }
    __syncthreads();

    if (update_scores != 0) {
        for (int candidate = thread; candidate < point_count;
             candidate += CUDA_BLOCK_SIZE) {
            if (clustered[candidate] || in_cluster[row_offset + candidate]) continue;

            const double new_distance = distances[
                static_cast<size_t>(candidate) * point_count + chosen_candidate];
            double& current_distance = max_distances[row_offset + candidate];
            if (new_distance > current_distance) current_distance = new_distance;
        }
    }
}

struct BestSeed {
    int seed;
    int cardinality;
};

__global__ void chooseBestSeed(const unsigned char* clustered,
                               const int* cardinalities, BestSeed* result,
                               const int point_count) {
    __shared__ int best_cardinalities[CUDA_BLOCK_SIZE];
    __shared__ int best_seeds[CUDA_BLOCK_SIZE];

    const int thread = static_cast<int>(threadIdx.x);
    int local_cardinality = -1;
    int local_seed = -1;
    for (int seed = thread; seed < point_count; seed += CUDA_BLOCK_SIZE) {
        if (clustered[seed] != 0) continue;
        const int cardinality = cardinalities[seed];
        if (cardinality > local_cardinality ||
            (cardinality == local_cardinality &&
             (local_seed < 0 || seed < local_seed))) {
            local_cardinality = cardinality;
            local_seed = seed;
        }
    }

    best_cardinalities[thread] = local_cardinality;
    best_seeds[thread] = local_seed;
    __syncthreads();

    for (int offset = CUDA_BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (thread < offset) {
            const int other_cardinality = best_cardinalities[thread + offset];
            const int other_seed = best_seeds[thread + offset];
            if (other_cardinality > best_cardinalities[thread] ||
                (other_cardinality == best_cardinalities[thread] &&
                 other_seed >= 0 &&
                 (best_seeds[thread] < 0 || other_seed < best_seeds[thread]))) {
                best_cardinalities[thread] = other_cardinality;
                best_seeds[thread] = other_seed;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        result->seed = best_seeds[0];
        result->cardinality = best_cardinalities[0];
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members,
                              const int member_count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < member_count) clustered[members[index]] = 1;
}

// Main QT clustering algorithm. Candidate clusters for all currently
// unclustered seeds are generated concurrently on the GPU. The outer choice
// of the largest cluster is reduced deterministically to retain the original
// seed and tie-breaking semantics.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const size_t matrix_elements = static_cast<size_t>(N) * N;
    if (matrix_elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Point matrix is too large for the available address space\n");
        std::exit(EXIT_FAILURE);
    }

    const size_t matrix_bytes = matrix_elements * sizeof(double);
    const size_t byte_matrix_bytes = matrix_elements * sizeof(unsigned char);
    const size_t int_matrix_bytes = matrix_elements * sizeof(int);

    Point* device_points = nullptr;
    double* device_distances = nullptr;
    unsigned char* device_clustered = nullptr;
    unsigned char* device_in_cluster = nullptr;
    double* device_max_distances = nullptr;
    int* device_member_order = nullptr;
    int* device_cardinalities = nullptr;
    unsigned char* device_active = nullptr;
    int* device_active_count = nullptr;
    BestSeed* device_best_seed = nullptr;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points),
                          static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_distances), matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered),
                          static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_in_cluster), byte_matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_max_distances), matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_member_order), int_matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cardinalities),
                          static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_active),
                          static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_active_count), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_best_seed), sizeof(BestSeed)));

    std::vector<unsigned char> host_clustered(N, 0);
    CUDA_CHECK(cudaMemcpy(device_points, points.data(),
                          static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_clustered, host_clustered.data(),
                          static_cast<size_t>(N) * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    const dim3 distance_block(32, 8);
    const dim3 distance_grid(
        static_cast<unsigned int>((N + distance_block.x - 1) / distance_block.x),
        static_cast<unsigned int>((N + distance_block.y - 1) / distance_block.y));
    buildDistanceMatrix<<<distance_grid, distance_block>>>(
        device_points, device_distances, N);
    CUDA_CHECK(cudaGetLastError());

    std::vector<Cluster> clusters;
    int unclustered_count = N;
    while (unclustered_count > 0) {
        CUDA_CHECK(cudaMemcpy(device_active_count, &unclustered_count,
                              sizeof(int), cudaMemcpyHostToDevice));

        const size_t state_threads = 256;
        const size_t state_blocks =
            (matrix_elements + state_threads - 1) / state_threads;
        initializeCandidateState<<<static_cast<unsigned int>(state_blocks),
                                   static_cast<unsigned int>(state_threads)>>>(
            device_distances, device_clustered, device_in_cluster,
            device_max_distances, device_member_order, device_cardinalities,
            device_active, N);
        CUDA_CHECK(cudaGetLastError());

        int active_count = N > 1 ? unclustered_count : 0;
        while (active_count > 0) {
            growCandidateClusters<<<static_cast<unsigned int>(N), CUDA_BLOCK_SIZE>>>(
                device_distances, device_clustered, device_in_cluster,
                device_max_distances, device_member_order, device_cardinalities,
                device_active, device_active_count, threshold, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&active_count, device_active_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
        }

        chooseBestSeed<<<1, CUDA_BLOCK_SIZE>>>(
            device_clustered, device_cardinalities, device_best_seed, N);
        CUDA_CHECK(cudaGetLastError());

        BestSeed best;
        CUDA_CHECK(cudaMemcpy(&best, device_best_seed, sizeof(BestSeed),
                              cudaMemcpyDeviceToHost));
        if (best.seed < 0 || best.cardinality <= 0) break;

        std::vector<int> members(static_cast<size_t>(best.cardinality));
        const size_t best_row = static_cast<size_t>(best.seed) * N;
        CUDA_CHECK(cudaMemcpy(members.data(), device_member_order + best_row,
                              static_cast<size_t>(best.cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best.seed;
        cluster.members = std::move(members);
        clusters.push_back(std::move(cluster));

        markClustered<<<
            static_cast<unsigned int>((best.cardinality + CUDA_BLOCK_SIZE - 1) /
                                      CUDA_BLOCK_SIZE),
            CUDA_BLOCK_SIZE>>>(device_clustered, device_member_order + best_row,
                               best.cardinality);
        CUDA_CHECK(cudaGetLastError());
        unclustered_count -= best.cardinality;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_best_seed));
    CUDA_CHECK(cudaFree(device_active_count));
    CUDA_CHECK(cudaFree(device_active));
    CUDA_CHECK(cudaFree(device_cardinalities));
    CUDA_CHECK(cudaFree(device_member_order));
    CUDA_CHECK(cudaFree(device_max_distances));
    CUDA_CHECK(cudaFree(device_in_cluster));
    CUDA_CHECK(cudaFree(device_clustered));
    CUDA_CHECK(cudaFree(device_distances));
    CUDA_CHECK(cudaFree(device_points));

    return clusters;
}

} // namespace

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
