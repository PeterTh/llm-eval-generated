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

void checkCuda(const cudaError_t status, const char* expression,
               const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", file, line,
                     expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ double deviceDistance(const Point& p1,
                                                  const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ bool isBetterCandidate(
    const double candidate_distance, const int candidate,
    const double best_distance, const int best_candidate) {
    // The sequential implementation scans candidates in ascending order and
    // updates only on a strict improvement. Comparing the index on ties makes
    // the parallel reduction produce the same deterministic result.
    return candidate_distance < best_distance ||
           (candidate_distance == best_distance &&
            (best_candidate < 0 || candidate < best_candidate));
}

// Initialize one independent candidate-cluster trajectory per possible seed.
// A row in cluster_members is the exact ordered member sequence that the
// sequential implementation would construct for that seed.
__global__ void initializeCandidateClusters(
    const unsigned char* clustered, const int point_count,
    unsigned char* active, int* cluster_sizes, int* cluster_members) {
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= point_count) return;

    if (clustered[seed] == 0) {
        active[seed] = 1;
        cluster_sizes[seed] = 1;
        cluster_members[static_cast<size_t>(seed) * point_count] = seed;
    } else {
        active[seed] = 0;
        cluster_sizes[seed] = 0;
    }
}

// Each block owns one seed. Threads cooperatively evaluate all possible next
// points for that seed, while the reduction preserves the original
// min-distance/lowest-index tie breaking rule.
__global__ void growCandidateClusters(
    const Point* points, const int point_count, const double threshold,
    const unsigned char* clustered, const unsigned char* active,
    int* cluster_sizes, int* cluster_members, unsigned char* next_active,
    int* active_count) {
    const int seed = static_cast<int>(blockIdx.x);
    if (seed >= point_count) return;

    __shared__ double best_distances[CUDA_BLOCK_SIZE];
    __shared__ int best_candidates[CUDA_BLOCK_SIZE];

    const int thread = static_cast<int>(threadIdx.x);
    double best_distance = DBL_MAX;
    int best_candidate = -1;

    if (active[seed] != 0) {
        const int cluster_size = cluster_sizes[seed];
        const size_t row = static_cast<size_t>(seed) * point_count;

        for (int candidate = thread; candidate < point_count;
             candidate += blockDim.x) {
            if (clustered[candidate] != 0) continue;

            bool already_in_cluster = false;
            double maximum_distance = 0.0;
            for (int member_index = 0; member_index < cluster_size;
                 ++member_index) {
                const int member = cluster_members[row + member_index];
                if (member == candidate) {
                    already_in_cluster = true;
                    break;
                }

                maximum_distance = fmax(
                    maximum_distance,
                    deviceDistance(points[candidate], points[member]));
            }

            if (!already_in_cluster && maximum_distance < threshold &&
                isBetterCandidate(maximum_distance, candidate, best_distance,
                                  best_candidate)) {
                best_distance = maximum_distance;
                best_candidate = candidate;
            }
        }
    }

    best_distances[thread] = best_distance;
    best_candidates[thread] = best_candidate;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride &&
            isBetterCandidate(best_distances[thread + stride],
                              best_candidates[thread + stride],
                              best_distances[thread], best_candidates[thread])) {
            best_distances[thread] = best_distances[thread + stride];
            best_candidates[thread] = best_candidates[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        const int candidate = best_candidates[0];
        if (candidate >= 0) {
            const int old_size = cluster_sizes[seed];
            cluster_members[static_cast<size_t>(seed) * point_count + old_size] =
                candidate;
            cluster_sizes[seed] = old_size + 1;
            next_active[seed] = 1;
            atomicAdd(active_count, 1);
        } else {
            next_active[seed] = 0;
        }
    }
}

// Select the largest completed candidate cluster. Scanning seeds in order
// preserves the original first-seed tie break.
__global__ void selectBestCluster(const unsigned char* clustered,
                                  const int* cluster_sizes,
                                  const int point_count, int* result) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int best_seed = -1;
    int best_size = -1;
    for (int seed = 0; seed < point_count; ++seed) {
        if (clustered[seed] != 0) continue;
        if (cluster_sizes[seed] > best_size) {
            best_size = cluster_sizes[seed];
            best_seed = seed;
        }
    }

    result[0] = best_seed;
    result[1] = best_size;
}

} // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "CUDA clustering requires at least one GPU\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));

    const size_t point_bytes = static_cast<size_t>(N) * sizeof(Point);
    const size_t matrix_elements = static_cast<size_t>(N) * N;
    const size_t matrix_bytes = matrix_elements * sizeof(int);

    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    unsigned char* device_active = nullptr;
    unsigned char* device_next_active = nullptr;
    int* device_cluster_sizes = nullptr;
    int* device_cluster_members = nullptr;
    int* device_active_count = nullptr;
    int* device_best_cluster = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points), point_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered),
                          static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_active),
                          static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_next_active),
                          static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cluster_sizes),
                          static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cluster_members),
                          matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_active_count),
                          sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_best_cluster),
                          2 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(device_points, points.data(), point_bytes,
                          cudaMemcpyHostToDevice));

    int unclustered_count = N;
    const dim3 initialization_grid(
        (N + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    const dim3 growth_grid(N);

    // The outer QT loop remains ordered because each selected cluster changes
    // the set of eligible points for the next round. All seed trajectories in
    // one round, and all candidate distances within each trajectory, execute
    // concurrently on the GPU.
    while (unclustered_count > 0) {
        CUDA_CHECK(cudaMemcpy(device_clustered, clustered.data(),
                              static_cast<size_t>(N) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        initializeCandidateClusters<<<initialization_grid, CUDA_BLOCK_SIZE>>>(
            device_clustered, N, device_active, device_cluster_sizes,
            device_cluster_members);
        CUDA_CHECK(cudaGetLastError());

        int active_count = unclustered_count;
        while (active_count > 0) {
            CUDA_CHECK(cudaMemset(device_active_count, 0, sizeof(int)));

            growCandidateClusters<<<growth_grid, CUDA_BLOCK_SIZE>>>(
                device_points, N, threshold, device_clustered, device_active,
                device_cluster_sizes, device_cluster_members,
                device_next_active, device_active_count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&active_count, device_active_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));

            std::swap(device_active, device_next_active);
        }

        selectBestCluster<<<1, 1>>>(device_clustered, device_cluster_sizes, N,
                                    device_best_cluster);
        CUDA_CHECK(cudaGetLastError());

        int best_cluster[2] = {-1, -1};
        CUDA_CHECK(cudaMemcpy(best_cluster, device_best_cluster,
                              sizeof(best_cluster), cudaMemcpyDeviceToHost));

        const int best_seed = best_cluster[0];
        const int max_cardinality = best_cluster[1];
        if (best_seed < 0 || max_cardinality <= 0) break;

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(static_cast<size_t>(max_cardinality));
        CUDA_CHECK(cudaMemcpy(
            cluster.members.data(),
            device_cluster_members + static_cast<size_t>(best_seed) * N,
            static_cast<size_t>(max_cardinality) * sizeof(int),
            cudaMemcpyDeviceToHost));
        clusters.push_back(std::move(cluster));

        for (const int member : clusters.back().members) {
            if (clustered[member] == 0) {
                clustered[member] = 1;
                --unclustered_count;
            }
        }
    }

    CUDA_CHECK(cudaFree(device_best_cluster));
    CUDA_CHECK(cudaFree(device_active_count));
    CUDA_CHECK(cudaFree(device_cluster_members));
    CUDA_CHECK(cudaFree(device_cluster_sizes));
    CUDA_CHECK(cudaFree(device_next_active));
    CUDA_CHECK(cudaFree(device_active));
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
