// QT Clustering Benchmark - CUDA version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
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

namespace {

constexpr int THREADS_PER_SEED = 256;

// CUDA errors are fatal.  CUDA is deliberately not optional for this
// benchmark: silently running a CPU implementation would make its reported
// performance meaningless.
void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

void checkKernel(const char* operation) {
    checkCuda(cudaGetLastError(), operation);
}

// The matrix is laid out as distance[member * N + candidate].  Keeping it
// resident avoids recalculating the same pairwise distance for every seed and
// every QT growth iteration.  The 2-D launch gives coalesced writes along the
// candidate dimension, which is also the access pattern used by score updates.
__global__ void buildDistanceMatrix(const Point* points, double* distances,
                                     const int point_count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const int member = blockIdx.y * blockDim.y + threadIdx.y;
    if (candidate >= point_count || member >= point_count) return;

    const double dx = points[candidate].x - points[member].x;
    const double dy = points[candidate].y - points[member].y;
    distances[static_cast<size_t>(member) * point_count + candidate] =
        sqrt(dx * dx + dy * dy);
}

// Scores use two negative sentinels.  -1 means that the point is already in
// this seed's candidate cluster, while -2 means that it was assigned by a
// previously selected global cluster.  Non-negative values are the current
// maximum distance from the candidate to this candidate cluster.
__global__ void initializeCandidateState(const double* distances,
                                          const unsigned char* clustered,
                                          double* scores,
                                          const int point_count) {
    const int seed = blockIdx.y * blockDim.y + threadIdx.y;
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= point_count || candidate >= point_count) return;

    const size_t offset = static_cast<size_t>(seed) * point_count + candidate;
    if (clustered[candidate] != 0) {
        scores[offset] = -2.0;
    } else if (seed == candidate) {
        scores[offset] = -1.0;
    } else {
        // This is distance(points[candidate], points[seed]), matching the
        // order used by the original candidate-to-member calculation.
        scores[offset] = distances[static_cast<size_t>(seed) * point_count + candidate];
    }
}

__global__ void initializeSeedState(const unsigned char* clustered,
                                    unsigned char* active, int* member_counts,
                                    const int point_count) {
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= point_count) return;

    const bool is_active = clustered[seed] == 0;
    active[seed] = is_active ? 1 : 0;
    member_counts[seed] = is_active ? 1 : 0;
}

__device__ __forceinline__ bool isBetter(const double score, const int candidate,
                                          const double best_score,
                                          const int best_candidate) {
    return score < best_score ||
           (score == best_score && candidate < best_candidate);
}

// One block owns one seed.  Each thread scans a strided portion of all
// candidates and the block reduction retains the lowest candidate index on an
// exact score tie, just like the original ascending CPU loop.
__global__ void selectClosestCandidates(const double* scores,
                                        const unsigned char* active,
                                        int* best_candidates,
                                        const double threshold,
                                        const int point_count) {
    const int seed = blockIdx.x;
    if (seed >= point_count) return;

    const size_t row = static_cast<size_t>(seed) * point_count;
    double best_score = DBL_MAX;
    int best_candidate = point_count;

    if (active[seed] != 0) {
        for (int candidate = threadIdx.x; candidate < point_count;
             candidate += blockDim.x) {
            const double score = scores[row + candidate];
            if (score >= 0.0 && score < threshold &&
                isBetter(score, candidate, best_score, best_candidate)) {
                best_score = score;
                best_candidate = candidate;
            }
        }
    }

    __shared__ double shared_scores[THREADS_PER_SEED];
    __shared__ int shared_candidates[THREADS_PER_SEED];
    shared_scores[threadIdx.x] = best_score;
    shared_candidates[threadIdx.x] = best_candidate;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            isBetter(shared_scores[threadIdx.x + stride],
                     shared_candidates[threadIdx.x + stride],
                     shared_scores[threadIdx.x],
                     shared_candidates[threadIdx.x])) {
            shared_scores[threadIdx.x] = shared_scores[threadIdx.x + stride];
            shared_candidates[threadIdx.x] = shared_candidates[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        best_candidates[seed] = shared_candidates[0] == point_count
                                    ? -1
                                    : shared_candidates[0];
    }
}

// Add the chosen point to every seed's state in parallel.  Once a point is
// selected, the score update is just one lookup into the resident distance
// matrix for each still-eligible candidate.
__global__ void updateCandidateScores(const double* distances,
                                      const int* best_candidates,
                                      double* scores, const int point_count) {
    const int seed = blockIdx.y * blockDim.y + threadIdx.y;
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= point_count || candidate >= point_count) return;

    const int best = best_candidates[seed];
    if (best < 0) return;

    const size_t offset = static_cast<size_t>(seed) * point_count + candidate;
    if (candidate == best) {
        scores[offset] = -1.0;
        return;
    }

    const double old_score = scores[offset];
    if (old_score >= 0.0) {
        const double new_distance =
            distances[static_cast<size_t>(best) * point_count + candidate];
        scores[offset] = old_score > new_distance ? old_score : new_distance;
    }
}

__global__ void commitCandidates(const int* best_candidates,
                                  unsigned char* active, int* member_counts,
                                  int* added_count, const int point_count) {
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= point_count) return;

    if (active[seed] == 0) return;

    if (best_candidates[seed] < 0) {
        active[seed] = 0;
    } else {
        ++member_counts[seed];
        atomicAdd(added_count, 1);
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
    std::vector<Cluster> clusters;

    if (N == 0) return clusters;

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "CUDA error: no CUDA device is available\n");
        std::exit(EXIT_FAILURE);
    }
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    const size_t matrix_elements = static_cast<size_t>(N) * N;
    const size_t matrix_bytes = matrix_elements * sizeof(double);

    Point* device_points = nullptr;
    double* device_distances = nullptr;
    double* device_scores = nullptr;
    unsigned char* device_clustered = nullptr;
    unsigned char* device_active = nullptr;
    int* device_best_candidates = nullptr;
    int* device_member_counts = nullptr;
    int* device_added_count = nullptr;

    checkCuda(cudaMalloc(&device_points, static_cast<size_t>(N) * sizeof(Point)),
              "cudaMalloc(points)");
    checkCuda(cudaMalloc(&device_distances, matrix_bytes),
              "cudaMalloc(distances)");
    checkCuda(cudaMalloc(&device_scores, matrix_bytes), "cudaMalloc(scores)");
    checkCuda(cudaMalloc(&device_clustered, static_cast<size_t>(N) * sizeof(unsigned char)),
              "cudaMalloc(clustered)");
    checkCuda(cudaMalloc(&device_active, static_cast<size_t>(N) * sizeof(unsigned char)),
              "cudaMalloc(active)");
    checkCuda(cudaMalloc(&device_best_candidates, static_cast<size_t>(N) * sizeof(int)),
              "cudaMalloc(best candidates)");
    checkCuda(cudaMalloc(&device_member_counts, static_cast<size_t>(N) * sizeof(int)),
              "cudaMalloc(member counts)");
    checkCuda(cudaMalloc(&device_added_count, sizeof(int)),
              "cudaMalloc(added count)");

    checkCuda(cudaMemcpy(device_points, points.data(),
                         static_cast<size_t>(N) * sizeof(Point),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(points)");
    checkCuda(cudaMemcpy(device_clustered, clustered.data(),
                         static_cast<size_t>(N) * sizeof(unsigned char),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(clustered)");

    constexpr int TILE_X = 32;
    constexpr int TILE_Y = 8;
    const dim3 matrix_block(TILE_X, TILE_Y);
    const dim3 matrix_grid((N + TILE_X - 1) / TILE_X,
                           (N + TILE_Y - 1) / TILE_Y);
    buildDistanceMatrix<<<matrix_grid, matrix_block>>>(
        device_points, device_distances, N);
    checkKernel("buildDistanceMatrix");

    const dim3 state_block(TILE_X, TILE_Y);
    const dim3 state_grid((N + TILE_X - 1) / TILE_X,
                          (N + TILE_Y - 1) / TILE_Y);
    const int seed_grid = (N + THREADS_PER_SEED - 1) / THREADS_PER_SEED;
    const dim3 seed_block(THREADS_PER_SEED);

    std::vector<int> member_counts(N, 0);
    std::vector<double> best_scores(N);
    int remaining = N;

    while (remaining > 0) {
        initializeCandidateState<<<state_grid, state_block>>>(
            device_distances, device_clustered, device_scores, N);
        checkKernel("initializeCandidateState");
        initializeSeedState<<<seed_grid, seed_block>>>(
            device_clustered, device_active, device_member_counts, N);
        checkKernel("initializeSeedState");

        int added_count = 0;
        do {
            selectClosestCandidates<<<N, THREADS_PER_SEED>>>(
                device_scores, device_active, device_best_candidates,
                threshold, N);
            checkKernel("selectClosestCandidates");
            updateCandidateScores<<<state_grid, state_block>>>(
                device_distances, device_best_candidates, device_scores, N);
            checkKernel("updateCandidateScores");
            checkCuda(cudaMemset(device_added_count, 0, sizeof(int)),
                      "cudaMemset(added count)");
            commitCandidates<<<seed_grid, seed_block>>>(
                device_best_candidates, device_active, device_member_counts,
                device_added_count, N);
            checkKernel("commitCandidates");
            checkCuda(cudaMemcpy(&added_count, device_added_count, sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(added count)");
        } while (added_count != 0);

        checkCuda(cudaMemcpy(member_counts.data(), device_member_counts,
                             static_cast<size_t>(N) * sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(member counts)");

        // This is the one intentionally sequential part of QT: the first
        // seed with the largest candidate cardinality wins the round.
        int max_cardinality = -1;
        int best_seed = -1;
        for (int seed = 0; seed < N; ++seed) {
            if (clustered[seed] == 0 && member_counts[seed] > max_cardinality) {
                max_cardinality = member_counts[seed];
                best_seed = seed;
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        checkCuda(cudaMemcpy(best_scores.data(),
                             device_scores + static_cast<size_t>(best_seed) * N,
                             static_cast<size_t>(N) * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(best cluster state)");

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.reserve(static_cast<size_t>(max_cardinality));
        for (int candidate = 0; candidate < N; ++candidate) {
            if (best_scores[candidate] == -1.0) {
                cluster.members.push_back(candidate);
            }
        }

        // A selected point is encoded exactly as -1.  Keeping this check
        // protects the host-side bookkeeping if a future kernel changes the
        // state representation.
        if (static_cast<int>(cluster.members.size()) != max_cardinality) {
            std::fprintf(stderr, "CUDA error: candidate cardinality mismatch\n");
            std::exit(EXIT_FAILURE);
        }

        clusters.push_back(cluster);
        for (const int member : cluster.members) {
            clustered[member] = 1;
            --remaining;
        }
        checkCuda(cudaMemcpy(device_clustered, clustered.data(),
                             static_cast<size_t>(N) * sizeof(unsigned char),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(clustered update)");
    }

    checkCuda(cudaFree(device_added_count), "cudaFree(added count)");
    checkCuda(cudaFree(device_member_counts), "cudaFree(member counts)");
    checkCuda(cudaFree(device_best_candidates), "cudaFree(best candidates)");
    checkCuda(cudaFree(device_active), "cudaFree(active)");
    checkCuda(cudaFree(device_clustered), "cudaFree(clustered)");
    checkCuda(cudaFree(device_scores), "cudaFree(scores)");
    checkCuda(cudaFree(device_distances), "cudaFree(distances)");
    checkCuda(cudaFree(device_points), "cudaFree(points)");

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
