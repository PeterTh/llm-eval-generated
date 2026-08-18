// QT Clustering Benchmark - Simplified Sequential Version
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

namespace {

constexpr int CUDA_THREADS_PER_BLOCK = 256;

// Select the same candidate as the sequential loop: the smallest diameter wins,
// and an equal diameter keeps the first (smallest-index) candidate.
__device__ __forceinline__ bool isBetterCandidate(const double candidate_distance,
                                                   const int candidate,
                                                   const double best_distance,
                                                   const int best_candidate) {
    return candidate_distance < best_distance ||
           (candidate_distance == best_distance &&
            (best_candidate < 0 || candidate < best_candidate));
}

// Build every candidate cluster for one QT iteration. One block owns one seed.
// Candidate diameters are incrementally updated when a member is appended, so
// each iteration only needs one distance evaluation per seed/candidate pair.
__global__ void generateCandidateClustersKernel(
    const Point* __restrict__ points,
    const unsigned char* __restrict__ clustered,
    unsigned char* __restrict__ in_cluster,
    double* __restrict__ candidate_diameters,
    int* __restrict__ candidate_members,
    int* __restrict__ candidate_sizes,
    const int point_count,
    const double threshold) {
    const int seed = static_cast<int>(blockIdx.x);
    if (seed >= point_count) return;

    const int lane = static_cast<int>(threadIdx.x);
    const size_t row = static_cast<size_t>(seed) * point_count;

    // Clustered seeds do not participate in this iteration. Keeping their size
    // at zero makes the host-side cardinality reduction deterministic.
    if (clustered[seed] != 0) {
        if (lane == 0) candidate_sizes[seed] = 0;
        return;
    }

    // The seed is the first member. Initialize each candidate's current
    // diameter to its distance from that seed.
    for (int candidate = lane; candidate < point_count;
         candidate += static_cast<int>(blockDim.x)) {
        in_cluster[row + candidate] = 0;
        candidate_diameters[row + candidate] =
            distance(points[seed], points[candidate]);
    }

    // A seed can be initialized by a lane other than lane zero when the
    // point count exceeds the block width. Complete that initialization
    // before lane zero marks the seed as an existing member.
    __syncthreads();
    if (lane == 0) {
        in_cluster[row + seed] = 1;
        candidate_members[row] = seed;
        candidate_sizes[seed] = 1;
    }
    __syncthreads();

    __shared__ double warp_best_distances[CUDA_THREADS_PER_BLOCK / 32];
    __shared__ int warp_best_candidates[CUDA_THREADS_PER_BLOCK / 32];
    __shared__ int closest_candidate;

    int cluster_size = 1;
    while (cluster_size < point_count) {
        double local_best_distance = DBL_MAX;
        int local_best_candidate = -1;

        // Each lane examines a disjoint subset of candidates. The reduction
        // below uses candidate index as a deterministic tie breaker.
        for (int candidate = lane; candidate < point_count;
             candidate += static_cast<int>(blockDim.x)) {
            const size_t candidate_offset = row + candidate;
            if (clustered[candidate] == 0 &&
                in_cluster[candidate_offset] == 0) {
                const double candidate_distance =
                    candidate_diameters[candidate_offset];
                if (candidate_distance < threshold &&
                    isBetterCandidate(candidate_distance, candidate,
                                      local_best_distance,
                                      local_best_candidate)) {
                    local_best_distance = candidate_distance;
                    local_best_candidate = candidate;
                }
            }
        }

        // Reduce within each warp without block-wide barriers, then reduce
        // the eight warp winners. This keeps the deterministic ordering while
        // substantially reducing synchronization overhead.
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

        if ((lane & 31) == 0) {
            warp_best_distances[lane >> 5] = local_best_distance;
            warp_best_candidates[lane >> 5] = local_best_candidate;
        }
        __syncthreads();

        if (lane == 0) {
            double best_distance = warp_best_distances[0];
            int best_candidate = warp_best_candidates[0];
            for (int warp = 1; warp < CUDA_THREADS_PER_BLOCK / 32; ++warp) {
                if (isBetterCandidate(warp_best_distances[warp],
                                      warp_best_candidates[warp],
                                      best_distance, best_candidate)) {
                    best_distance = warp_best_distances[warp];
                    best_candidate = warp_best_candidates[warp];
                }
            }
            closest_candidate = best_candidate;
        }
        __syncthreads();

        const int closest = closest_candidate;
        if (closest < 0) break;

        if (lane == 0) {
            in_cluster[row + closest] = 1;
            candidate_members[row + cluster_size] = closest;
        }
        __syncthreads();

        // Add the newly selected member to every remaining candidate's
        // diameter. Existing members and globally clustered points are not
        // candidates and therefore need no update.
        for (int candidate = lane; candidate < point_count;
             candidate += static_cast<int>(blockDim.x)) {
            const size_t candidate_offset = row + candidate;
            if (clustered[candidate] == 0 &&
                in_cluster[candidate_offset] == 0) {
                const double candidate_distance =
                    distance(points[candidate], points[closest]);
                if (candidate_distance > candidate_diameters[candidate_offset]) {
                    candidate_diameters[candidate_offset] = candidate_distance;
                }
            }
        }
        __syncthreads();
        ++cluster_size;
    }

    if (lane == 0) candidate_sizes[seed] = cluster_size;
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
T* allocateDevice(const size_t count) {
    T* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                          count * sizeof(T)));
    return pointer;
}

class CudaQtClusteringBuffers {
  public:
    CudaQtClusteringBuffers(const std::vector<Point>& points,
                            const std::vector<unsigned char>& clustered)
        : point_count_(static_cast<int>(points.size())),
          matrix_elements_(points.size() * points.size()),
          device_points_(allocateDevice<Point>(points.size())),
          device_clustered_(allocateDevice<unsigned char>(points.size())),
          device_in_cluster_(allocateDevice<unsigned char>(matrix_elements_)),
          device_candidate_diameters_(allocateDevice<double>(matrix_elements_)),
          device_candidate_members_(allocateDevice<int>(matrix_elements_)),
          device_candidate_sizes_(allocateDevice<int>(points.size())) {
        CUDA_CHECK(cudaMemcpy(device_points_, points.data(),
                              points.size() * sizeof(Point),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_clustered_, clustered.data(),
                              points.size() * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    ~CudaQtClusteringBuffers() { release(); }

    CudaQtClusteringBuffers(const CudaQtClusteringBuffers&) = delete;
    CudaQtClusteringBuffers& operator=(const CudaQtClusteringBuffers&) = delete;

    void generate(const double threshold) {
        generateCandidateClustersKernel<<<point_count_, CUDA_THREADS_PER_BLOCK>>>(
            device_points_, device_clustered_, device_in_cluster_,
            device_candidate_diameters_, device_candidate_members_,
            device_candidate_sizes_, point_count_, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    void copySizesToHost(std::vector<int>& sizes) const {
        CUDA_CHECK(cudaMemcpy(sizes.data(), device_candidate_sizes_,
                              sizes.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    void copyMembersToHost(const int seed, const int member_count,
                           std::vector<int>& members) const {
        members.resize(member_count);
        if (member_count == 0) return;
        CUDA_CHECK(cudaMemcpy(members.data(),
                              device_candidate_members_ +
                                  static_cast<size_t>(seed) * point_count_,
                              members.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    void copyClusteredToDevice(const std::vector<unsigned char>& clustered) {
        CUDA_CHECK(cudaMemcpy(device_clustered_, clustered.data(),
                              clustered.size() * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    void release() {
        if (device_candidate_sizes_ != nullptr) {
            CUDA_CHECK(cudaFree(device_candidate_sizes_));
            device_candidate_sizes_ = nullptr;
        }
        if (device_candidate_members_ != nullptr) {
            CUDA_CHECK(cudaFree(device_candidate_members_));
            device_candidate_members_ = nullptr;
        }
        if (device_candidate_diameters_ != nullptr) {
            CUDA_CHECK(cudaFree(device_candidate_diameters_));
            device_candidate_diameters_ = nullptr;
        }
        if (device_in_cluster_ != nullptr) {
            CUDA_CHECK(cudaFree(device_in_cluster_));
            device_in_cluster_ = nullptr;
        }
        if (device_clustered_ != nullptr) {
            CUDA_CHECK(cudaFree(device_clustered_));
            device_clustered_ = nullptr;
        }
        if (device_points_ != nullptr) {
            CUDA_CHECK(cudaFree(device_points_));
            device_points_ = nullptr;
        }
    }

  private:
    int point_count_;
    size_t matrix_elements_;
    Point* device_points_;
    unsigned char* device_clustered_;
    unsigned char* device_in_cluster_;
    double* device_candidate_diameters_;
    int* device_candidate_members_;
    int* device_candidate_sizes_;
};

}  // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    CudaQtClusteringBuffers device_buffers(points, clustered);
    std::vector<int> candidate_sizes(N, 0);
    int clustered_count = 0;

    // The cluster-selection loop is inherently sequential: the globally
    // selected cluster changes which points are eligible in the next round.
    // Candidate construction for every seed is parallelized on the GPU.
    while (clustered_count < N) {
        device_buffers.generate(threshold);
        device_buffers.copySizesToHost(candidate_sizes);

        int max_cardinality = -1;
        int best_seed = -1;
        for (int seed = 0; seed < N; ++seed) {
            if (clustered[seed] != 0) continue;
            const int cardinality = candidate_sizes[seed];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            device_buffers.copyMembersToHost(best_seed, max_cardinality,
                                             cluster.members);
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (const int member : cluster.members) {
                if (clustered[member] == 0) {
                    clustered[member] = 1;
                    ++clustered_count;
                }
            }
            device_buffers.copyClusteredToDevice(clustered);
        } else {
            // No more clusters can be formed
            break;
        }
    }
    
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
