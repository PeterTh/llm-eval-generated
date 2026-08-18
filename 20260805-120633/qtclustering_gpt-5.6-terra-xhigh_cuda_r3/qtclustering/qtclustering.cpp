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

constexpr int CUDA_THREADS_PER_BLOCK = 256;
constexpr size_t CUDA_ALLOCATION_RESERVE = 64ULL * 1024ULL * 1024ULL;

void checkCuda(const cudaError_t error, const char* expression, const char* file,
               const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file,
                     line, expression, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

// This comparison implements the sequential scan's behavior exactly: the
// lower index wins when two candidates have the same diameter.
__device__ __forceinline__ bool isBetterCandidate(const double distance_a,
                                                   const int index_a,
                                                   const double distance_b,
                                                   const int index_b) {
    return distance_a < distance_b ||
           (distance_a == distance_b && index_a < index_b);
}

__device__ __forceinline__ double deviceDistance(const Point& p1,
                                                  const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void precomputeDistances(const Point* __restrict__ points,
                                    const int point_count,
                                    double* __restrict__ distances) {
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < point_count && column < point_count) {
        distances[static_cast<size_t>(row) * point_count + column] =
            deviceDistance(points[row], points[column]);
    }
}

// Each block constructs one candidate cluster.  Candidate points are divided
// among the block's threads and reduced deterministically after every added
// member.  The member rows are retained only for the current seed batch.
__global__ void buildCandidateClusters(const Point* __restrict__ points,
                                       const double* __restrict__ distances,
                                       const unsigned char* __restrict__ clustered,
                                       const int* __restrict__ seeds,
                                       const int point_count,
                                       const double threshold,
                                       int* __restrict__ candidate_members,
                                       int* __restrict__ cardinalities) {
    const int row = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int seed = seeds[row];

    __shared__ double best_distances[CUDA_THREADS_PER_BLOCK];
    __shared__ int best_indices[CUDA_THREADS_PER_BLOCK];
    __shared__ int member_count;

    if (clustered[seed]) {
        if (thread == 0) {
            cardinalities[row] = 0;
        }
        return;
    }

    int* const members = candidate_members + static_cast<size_t>(row) * point_count;
    if (thread == 0) {
        members[0] = seed;
        member_count = 1;
    }
    __syncthreads();

    while (member_count < point_count) {
        const int current_member_count = member_count;
        double thread_best_distance = DBL_MAX;
        int thread_best_index = INT_MAX;

        for (int candidate = thread; candidate < point_count;
             candidate += static_cast<int>(blockDim.x)) {
            if (clustered[candidate]) {
                continue;
            }

            bool already_a_member = false;
            double max_distance = 0.0;
            for (int i = 0; i < current_member_count; ++i) {
                const int member = members[i];
                if (candidate == member) {
                    already_a_member = true;
                    break;
                }

                const double candidate_distance = distances
                    ? distances[static_cast<size_t>(candidate) * point_count + member]
                    : deviceDistance(points[candidate], points[member]);
                if (candidate_distance > max_distance) {
                    max_distance = candidate_distance;
                }

                // A candidate that has reached the threshold cannot be
                // selected, so its remaining distances are irrelevant.
                if (max_distance >= threshold) {
                    break;
                }
            }

            if (!already_a_member && max_distance < threshold &&
                isBetterCandidate(max_distance, candidate, thread_best_distance,
                                  thread_best_index)) {
                thread_best_distance = max_distance;
                thread_best_index = candidate;
            }
        }

        best_distances[thread] = thread_best_distance;
        best_indices[thread] = thread_best_index;
        __syncthreads();

        for (int offset = CUDA_THREADS_PER_BLOCK / 2; offset > 0; offset /= 2) {
            if (thread < offset &&
                isBetterCandidate(best_distances[thread + offset],
                                  best_indices[thread + offset], best_distances[thread],
                                  best_indices[thread])) {
                best_distances[thread] = best_distances[thread + offset];
                best_indices[thread] = best_indices[thread + offset];
            }
            __syncthreads();
        }

        if (best_indices[0] == INT_MAX) {
            break;
        }

        if (thread == 0) {
            members[member_count] = best_indices[0];
            ++member_count;
        }
        __syncthreads();
    }

    if (thread == 0) {
        cardinalities[row] = member_count;
    }
}

}  // namespace

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
    unclustered_indices.reserve(N);

    Point* device_points = nullptr;
    double* device_distances = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_seeds = nullptr;
    int* device_candidate_members = nullptr;
    int* device_cardinalities = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points),
                          static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered),
                          static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(),
                          static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered, 0,
                          static_cast<size_t>(N) * sizeof(unsigned char)));

    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    // The distance matrix removes the repeated FP64 square roots in the QT
    // growth loop.  It is used whenever it leaves enough room for a useful
    // candidate batch; otherwise the same CUDA kernel computes distances
    // directly and still needs only O(N) persistent device storage.
    const size_t point_count_squared = static_cast<size_t>(N) * N;
    const size_t distance_matrix_bytes = point_count_squared * sizeof(double);
    const size_t matrix_budget = free_memory > CUDA_ALLOCATION_RESERVE
                                     ? (free_memory - CUDA_ALLOCATION_RESERVE) * 2 / 3
                                     : 0;
    if (distance_matrix_bytes <= matrix_budget) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_distances),
                              distance_matrix_bytes));
        constexpr int DISTANCE_TILE = 16;
        const dim3 distance_block(DISTANCE_TILE, DISTANCE_TILE);
        const dim3 distance_grid((N + DISTANCE_TILE - 1) / DISTANCE_TILE,
                                 (N + DISTANCE_TILE - 1) / DISTANCE_TILE);
        precomputeDistances<<<distance_grid, distance_block>>>(
            device_points, N, device_distances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    }

    // Retaining every candidate cluster is unnecessary.  Use as much of the
    // available device memory as possible for seed rows, and process larger
    // inputs in batches without changing the seed-order tie break.
    const size_t row_bytes = static_cast<size_t>(N) * sizeof(int);
    const size_t usable_memory = free_memory > CUDA_ALLOCATION_RESERVE
                                     ? free_memory - CUDA_ALLOCATION_RESERVE
                                     : free_memory;
    const size_t candidate_rows = (usable_memory * 7 / 8) / row_bytes;
    const int batch_size = static_cast<int>(
        std::min(static_cast<size_t>(N), std::max(static_cast<size_t>(1), candidate_rows)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_seeds),
                          static_cast<size_t>(batch_size) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cardinalities),
                          static_cast<size_t>(batch_size) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_candidate_members),
                          static_cast<size_t>(batch_size) * row_bytes));

    std::vector<int> cardinalities(batch_size);

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Build candidate clusters for an entire seed batch concurrently. The
        // batches stay in ascending seed order, preserving the original first
        // maximum-cardinality tie break even when all seeds cannot fit at once.
        for (size_t batch_begin = 0; batch_begin < unclustered_indices.size();
             batch_begin += batch_size) {
            const int count = static_cast<int>(std::min(
                static_cast<size_t>(batch_size), unclustered_indices.size() - batch_begin));
            CUDA_CHECK(cudaMemcpy(device_seeds,
                                  unclustered_indices.data() + batch_begin,
                                  static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyHostToDevice));

            buildCandidateClusters<<<count, CUDA_THREADS_PER_BLOCK>>>(
                device_points, device_distances, device_clustered, device_seeds, N, threshold,
                device_candidate_members, device_cardinalities);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(cardinalities.data(), device_cardinalities,
                                  static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            int batch_best = -1;
            for (int i = 0; i < count; ++i) {
                if (cardinalities[i] > max_cardinality) {
                    max_cardinality = cardinalities[i];
                    best_seed = unclustered_indices[batch_begin + i];
                    batch_best = i;
                }
            }

            if (batch_best >= 0) {
                best_cluster_members.resize(max_cardinality);
                CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                                      device_candidate_members +
                                          static_cast<size_t>(batch_best) * N,
                                      static_cast<size_t>(max_cardinality) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }
        }
        
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
            CUDA_CHECK(cudaMemcpy(device_clustered, clustered.data(),
                                  static_cast<size_t>(N) * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice));
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaFree(device_cardinalities));
    CUDA_CHECK(cudaFree(device_candidate_members));
    CUDA_CHECK(cudaFree(device_seeds));
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

    // Create the CUDA context outside the benchmark interval.  The clustering
    // implementation itself remains entirely GPU based.
    CUDA_CHECK(cudaFree(0));
    
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
