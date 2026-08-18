// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <chrono>
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

constexpr int CUDA_BLOCK_SIZE = 256;

[[noreturn]] void cudaError(const cudaError_t error, const char* expression,
                            const char* file, const int line) {
    fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
            expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t error = (expression); \
        if (error != cudaSuccess) cudaError(error, #expression, __FILE__, __LINE__); \
    } while (false)

__host__ __device__ inline double pointDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Store distances transposed: the row is the point already in a candidate
// cluster and the column is the point being considered.  This makes the
// column-wise candidate work performed by a CUDA block coalesced.
__global__ void buildDistanceMatrix(const Point* points, double* distances,
                                    const size_t point_count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = point_count * point_count;
    if (index >= elements) return;

    const int member = static_cast<int>(index / point_count);
    const int candidate = static_cast<int>(index -
                                           static_cast<size_t>(member) * point_count);
    distances[index] = pointDistance(points[candidate], points[member]);
}

__device__ inline bool isBetterCandidate(const double candidate_distance,
                                         const int candidate,
                                         const double best_distance,
                                         const int best_candidate) {
    return candidate_distance < best_distance ||
           (candidate_distance == best_distance && candidate < best_candidate);
}

// Generate one complete candidate cluster per seed.  Each CUDA block owns one
// seed and its row in candidate_membership/candidate_members.  The greedy
// additions are inherently sequential within one seed, but all seeds and all
// candidate points are evaluated concurrently.  candidate_max_dist is updated
// incrementally after each addition, so a new iteration costs O(N) rather than
// recomputing every candidate's distance to the whole current cluster.
__global__ void generateCandidateClusters(
    const double* distances, const unsigned char* clustered,
    double* candidate_max_dist, unsigned int* candidate_membership,
    int* candidate_members, int* cardinality, const int point_count,
    const double threshold) {
    const int seed = static_cast<int>(blockIdx.x);
    if (seed >= point_count) return;

    __shared__ double warp_best_distances[CUDA_BLOCK_SIZE / 32];
    __shared__ int warp_best_candidates[CUDA_BLOCK_SIZE / 32];
    __shared__ int current_count;
    __shared__ int chosen_candidate;
    __shared__ int stop;

    const int lane = static_cast<int>(threadIdx.x);
    const size_t row = static_cast<size_t>(seed) * point_count;
    const size_t membership_words =
        (static_cast<size_t>(point_count) + 31u) / 32u;
    const size_t membership_row = static_cast<size_t>(seed) * membership_words;

    // Clustered seeds do not participate in this generation.  All threads in
    // a block observe the same immutable snapshot, so this early return is
    // uniform and avoids resetting rows that can no longer be selected.
    if (clustered[seed] != 0) {
        if (lane == 0) cardinality[seed] = 0;
        return;
    }

    // Reset this seed's state.  This is done by the same block that consumes
    // it, avoiding a separate O(N^2) initialization kernel launch.
    for (size_t word = static_cast<size_t>(lane); word < membership_words;
         word += CUDA_BLOCK_SIZE) {
        candidate_membership[membership_row + word] = 0u;
    }
    for (int candidate = lane; candidate < point_count;
         candidate += CUDA_BLOCK_SIZE) {
        candidate_max_dist[row + candidate] =
            distances[static_cast<size_t>(seed) * point_count + candidate];
    }

    if (lane == 0) {
        cardinality[seed] = 1;
        current_count = 1;
        candidate_members[row] = seed;
    }
    __syncthreads();

    if (lane == 0) {
        candidate_membership[membership_row +
                            static_cast<size_t>(seed >> 5)] |=
            1u << (seed & 31);
    }
    __syncthreads();

    while (true) {
        const int count = current_count;
        if (count >= point_count) break;

        double local_best_distance = DBL_MAX;
        int local_best_candidate = -1;

        // Each thread evaluates a strided set of candidates.  The current
        // maximum distance is already available and is updated below after
        // the selected candidate is appended.
        for (int candidate = lane; candidate < point_count;
             candidate += CUDA_BLOCK_SIZE) {
            const unsigned int membership =
                candidate_membership[membership_row +
                                     static_cast<size_t>(candidate >> 5)];
            if (clustered[candidate] != 0 ||
                (membership & (1u << (candidate & 31))) != 0u) {
                continue;
            }

            const double max_distance = candidate_max_dist[row + candidate];
            if (max_distance < threshold &&
                isBetterCandidate(max_distance, candidate,
                                  local_best_distance, local_best_candidate)) {
                local_best_distance = max_distance;
                local_best_candidate = candidate;
            }
        }

        // First reduce within each warp.  The comparison is a deterministic
        // min-by-(distance,index), so the reduction order does not change QT's
        // first-candidate tie rule.
        const int warp_lane = lane & 31;
        const int warp = lane >> 5;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other_distance =
                __shfl_down_sync(0xffffffffu, local_best_distance, offset);
            const int other_candidate =
                __shfl_down_sync(0xffffffffu, local_best_candidate, offset);
            if (isBetterCandidate(other_distance, other_candidate,
                                  local_best_distance, local_best_candidate)) {
                local_best_distance = other_distance;
                local_best_candidate = other_candidate;
            }
        }

        if (warp_lane == 0) {
            warp_best_distances[warp] = local_best_distance;
            warp_best_candidates[warp] = local_best_candidate;
        }
        __syncthreads();

        // The first warp reduces the eight warp winners.
        if (warp == 0) {
            local_best_distance = (lane < CUDA_BLOCK_SIZE / 32)
                                      ? warp_best_distances[lane]
                                      : DBL_MAX;
            local_best_candidate = (lane < CUDA_BLOCK_SIZE / 32)
                                       ? warp_best_candidates[lane]
                                       : -1;
            for (int offset = 16; offset > 0; offset >>= 1) {
                const double other_distance =
                    __shfl_down_sync(0xffffffffu, local_best_distance, offset);
                const int other_candidate =
                    __shfl_down_sync(0xffffffffu, local_best_candidate, offset);
                if (isBetterCandidate(other_distance, other_candidate,
                                      local_best_distance,
                                      local_best_candidate)) {
                    local_best_distance = other_distance;
                    local_best_candidate = other_candidate;
                }
            }
            if (lane == 0) {
                chosen_candidate = local_best_candidate;
                stop = (chosen_candidate < 0) ? 1 : 0;
                if (stop == 0) {
                    candidate_members[row + count] = chosen_candidate;
                    candidate_membership[
                        membership_row +
                        static_cast<size_t>(chosen_candidate >> 5)] |=
                        1u << (chosen_candidate & 31);
                    current_count = count + 1;
                    cardinality[seed] = current_count;
                }
            }
        }
        __syncthreads();

        if (stop != 0) break;

        // Incorporate the new member into every candidate's maximum distance.
        // Threads access one contiguous candidate row at a time.
        const size_t new_member_row =
            static_cast<size_t>(chosen_candidate) * point_count;
        for (int candidate = lane; candidate < point_count;
             candidate += CUDA_BLOCK_SIZE) {
            const unsigned int membership =
                candidate_membership[membership_row +
                                     static_cast<size_t>(candidate >> 5)];
            if (clustered[candidate] != 0 ||
                (membership & (1u << (candidate & 31))) != 0u) {
                continue;
            }
            const double new_distance = distances[new_member_row + candidate];
            const size_t candidate_index = row + candidate;
            candidate_max_dist[candidate_index] =
                fmax(candidate_max_dist[candidate_index], new_distance);
        }
        __syncthreads();
    }
}

__global__ void markClustered(const int* members, unsigned char* clustered,
                              const int cluster_size) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < cluster_size) clustered[members[index]] = 1;
}

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

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const size_t point_count = static_cast<size_t>(N);
    const size_t matrix_elements = point_count * point_count;
    const size_t membership_words = (point_count + 31u) / 32u;
    const size_t membership_elements = point_count * membership_words;

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> cardinalities(N, 0);
    std::vector<Cluster> clusters;

    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    double* device_distances = nullptr;
    double* device_candidate_max_dist = nullptr;
    unsigned int* device_candidate_membership = nullptr;
    int* device_candidate_members = nullptr;
    int* device_cardinalities = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points),
                          point_count * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered),
                          point_count * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_distances),
                          matrix_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_candidate_max_dist),
                          matrix_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_candidate_membership),
                          membership_elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_candidate_members),
                          matrix_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cardinalities),
                          point_count * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(device_points, points.data(), point_count * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered, 0,
                          point_count * sizeof(unsigned char)));

    const size_t matrix_threads = 256;
    const size_t matrix_blocks =
        (matrix_elements + matrix_threads - 1) / matrix_threads;
    buildDistanceMatrix<<<static_cast<unsigned int>(matrix_blocks),
                          static_cast<unsigned int>(matrix_threads)>>>(
        device_points, device_distances, point_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Every outer iteration uses the same current clustered set for all seed
    // candidates, exactly as the sequential algorithm does.
    int clustered_count = 0;
    while (clustered_count < N) {
        generateCandidateClusters<<<static_cast<unsigned int>(N),
                                    CUDA_BLOCK_SIZE>>>(
            device_distances, device_clustered, device_candidate_max_dist,
            device_candidate_membership, device_candidate_members,
            device_cardinalities, N, threshold);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), device_cardinalities,
                              point_count * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_seed = -1;
        for (int seed = 0; seed < N; ++seed) {
            if (clustered[seed] != 0) continue;
            const int cardinality = cardinalities[seed];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            std::vector<int> best_cluster_members(max_cardinality);
            CUDA_CHECK(cudaMemcpy(
                best_cluster_members.data(),
                device_candidate_members +
                    static_cast<size_t>(best_seed) * point_count,
                static_cast<size_t>(max_cardinality) * sizeof(int),
                cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (const int member : best_cluster_members) {
                clustered[member] = 1;
                ++clustered_count;
            }

            markClustered<<<(max_cardinality + CUDA_BLOCK_SIZE - 1) /
                                CUDA_BLOCK_SIZE,
                            CUDA_BLOCK_SIZE>>>(
                device_candidate_members +
                    static_cast<size_t>(best_seed) * point_count,
                device_clustered, max_cardinality);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // No more clusters can be formed
            break;
        }
    }

    // The final markClustered launch can be asynchronous because no following
    // host-to-device operation is required after the last cluster.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_cardinalities));
    CUDA_CHECK(cudaFree(device_candidate_members));
    CUDA_CHECK(cudaFree(device_candidate_membership));
    CUDA_CHECK(cudaFree(device_candidate_max_dist));
    CUDA_CHECK(cudaFree(device_distances));
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
