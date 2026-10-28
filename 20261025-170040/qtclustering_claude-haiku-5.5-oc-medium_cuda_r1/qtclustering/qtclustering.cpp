// QT Clustering Benchmark - CUDA GPU Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// GPU mapping:
//   - Every unclustered point is tried as a seed in parallel: one CUDA thread
//     block grows the candidate cluster for one seed.
//   - Inside a block, the threads cooperatively find the next member: each
//     thread owns a strided subset of candidates, maintains the candidate's
//     maximum distance to the cluster incrementally, and a block-wide argmin
//     (ties broken by lowest index) selects the winner.
//   - The host selects the seed with the largest cluster (ties broken by the
//     lowest point index, as in the sequential scan), then re-grows that seed
//     once to record its members in insertion order.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per block used for per-seed cluster growth
static const int BLOCK_THREADS = 256;
static const int WARP_SIZE = 32;

// Upper bound on device scratch memory used for per-seed working state
static const size_t SCRATCH_BUDGET_BYTES = size_t(1) << 30;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

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

// Calculate Euclidean distance between two points (host, used for validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Device-side distance. Uses the same operation order and fused multiply-add
// as the host build, so the distances (and therefore all comparisons) match.
__device__ __forceinline__ double distanceDevice(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
}

// Ordering used for argmin: smaller value wins, ties go to the lower index
__device__ __forceinline__ bool argminBetter(double v1, int i1, double v2, int i2) {
    return v1 < v2 || (v1 == v2 && i1 < i2);
}

// Warp-level argmin; the result is valid in lane 0
__device__ __forceinline__ void warpArgmin(double& v, int& i) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, offset);
        const int oi = __shfl_down_sync(0xffffffffu, i, offset);
        if (argminBetter(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

// Block-wide argmin. On return every thread holds the same (v, i) pair.
// sv and si must provide at least WARP_SIZE entries each.
__device__ __forceinline__ void blockArgmin(double& v, int& i, double* sv, int* si) {
    warpArgmin(v, i);

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        sv[warp] = v;
        si[warp] = i;
    }
    __syncthreads();

    if (warp == 0) {
        const int numWarps = blockDim.x / WARP_SIZE;
        v = (lane < numWarps) ? sv[lane] : INFINITY;
        i = (lane < numWarps) ? si[lane] : INT_MAX;
        warpArgmin(v, i);
        if (lane == 0) {
            sv[0] = v;
            si[0] = i;
        }
    }
    __syncthreads();

    v = sv[0];
    i = si[0];
}

// Grow the candidate cluster starting from `seed` (block-cooperative).
// Must be called by all threads of the block with identical arguments except
// that `members` may be nullptr. Returns the cluster cardinality.
//   maxd[c]   : running max distance from candidate c to current members
//   inCluster : membership flags for points in the growing cluster
__device__ int growCandidateCluster(const int seed,
                                    const Point* points,
                                    const uint8_t* clustered,
                                    double* maxd,
                                    uint8_t* inCluster,
                                    const int N,
                                    const double threshold,
                                    int* members,
                                    double* sv,
                                    int* si) {
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        maxd[c] = 0.0;
        inCluster[c] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        inCluster[seed] = 1;
        if (members) members[0] = seed;
    }
    __syncthreads();

    int count = 1;
    int current = seed;

    while (count < N) {
        double bestVal = INFINITY;
        int bestIdx = INT_MAX;

        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            if (clustered[c] || inCluster[c]) continue;

            // Running max over members == max over all members of the cluster
            const double d = distanceDevice(points[c], points[current]);
            const double m = (maxd[c] < d) ? d : maxd[c];  // same as std::max
            maxd[c] = m;

            if (m < threshold && m < bestVal) {
                bestVal = m;
                bestIdx = c;
            }
        }

        blockArgmin(bestVal, bestIdx, sv, si);

        // Uniform across the block: all threads hold the same bestIdx
        if (bestIdx == INT_MAX) break;

        if (threadIdx.x == 0) {
            inCluster[bestIdx] = 1;
            if (members) members[count] = bestIdx;
        }
        current = bestIdx;
        count++;
        __syncthreads();
    }

    return count;
}

// One block per seed: computes the cardinality of the candidate cluster
__global__ void seedCardinalityKernel(const int* seeds,
                                      const Point* points,
                                      const uint8_t* clustered,
                                      const int N,
                                      const double threshold,
                                      double* maxdAll,
                                      uint8_t* inClusterAll,
                                      int* cardinalities) {
    __shared__ double sv[WARP_SIZE];
    __shared__ int si[WARP_SIZE];

    const int b = blockIdx.x;
    const size_t base = static_cast<size_t>(b) * static_cast<size_t>(N);
    const int card = growCandidateCluster(seeds[b], points, clustered,
                                          maxdAll + base, inClusterAll + base,
                                          N, threshold, nullptr, sv, si);
    if (threadIdx.x == 0) {
        cardinalities[b] = card;
    }
}

// Single block: re-grows the chosen seed and records its members in order
__global__ void recordClusterKernel(const int seed,
                                    const Point* points,
                                    const uint8_t* clustered,
                                    const int N,
                                    const double threshold,
                                    double* maxd,
                                    uint8_t* inCluster,
                                    int* members,
                                    int* count) {
    __shared__ double sv[WARP_SIZE];
    __shared__ int si[WARP_SIZE];

    const int card = growCandidateCluster(seed, points, clustered, maxd, inCluster,
                                          N, threshold, members, sv, si);
    if (threadIdx.x == 0) {
        *count = card;
    }
}

// Initialize the CUDA runtime (context creation) outside of timed sections
void initCuda() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: no CUDA-capable device found\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaFree(nullptr));
}

// Main QT clustering algorithm (GPU)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Scratch batch size: number of seeds grown concurrently
    const size_t bytesPerSeed = static_cast<size_t>(N) * (sizeof(double) + sizeof(uint8_t));
    size_t batchMax = SCRATCH_BUDGET_BYTES / bytesPerSeed;
    batchMax = std::max<size_t>(1, std::min<size_t>(batchMax, static_cast<size_t>(N)));

    Point* d_points = nullptr;
    uint8_t* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    double* d_maxd = nullptr;
    uint8_t* d_inCluster = nullptr;
    int* d_members = nullptr;
    int* d_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(uint8_t) * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_cards, sizeof(int) * batchMax));
    CUDA_CHECK(cudaMalloc(&d_maxd, sizeof(double) * batchMax * N));
    CUDA_CHECK(cudaMalloc(&d_inCluster, sizeof(uint8_t) * batchMax * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));

    std::vector<uint8_t> hClustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }
    std::vector<int> hCards(batchMax);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());
        int max_cardinality = -1;
        int best_seed = -1;

        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(), sizeof(int) * U,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_clustered, hClustered.data(), sizeof(uint8_t) * N,
                              cudaMemcpyHostToDevice));

        // Grow a candidate cluster from every unclustered seed (in batches)
        for (int start = 0; start < U; start += static_cast<int>(batchMax)) {
            const int n = std::min(static_cast<int>(batchMax), U - start);

            seedCardinalityKernel<<<n, BLOCK_THREADS>>>(d_seeds + start, d_points, d_clustered,
                                                        N, threshold, d_maxd, d_inCluster,
                                                        d_cards);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hCards.data(), d_cards, sizeof(int) * n,
                                  cudaMemcpyDeviceToHost));

            // Strictly greater: the lowest seed index wins ties
            for (int j = 0; j < n; ++j) {
                if (hCards[j] > max_cardinality) {
                    max_cardinality = hCards[j];
                    best_seed = unclustered_indices[start + j];
                }
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            recordClusterKernel<<<1, BLOCK_THREADS>>>(best_seed, d_points, d_clustered, N,
                                                      threshold, d_maxd, d_inCluster,
                                                      d_members, d_count);
            CUDA_CHECK(cudaGetLastError());

            int count = 0;
            CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> best_cluster_members(count);
            CUDA_CHECK(cudaMemcpy(best_cluster_members.data(), d_members, sizeof(int) * count,
                                  cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                hClustered[best_cluster_members[i]] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&hClustered](int idx) { return hClustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_cards));
    CUDA_CHECK(cudaFree(d_maxd));
    CUDA_CHECK(cudaFree(d_inCluster));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_count));

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
    
    // Initialize the GPU runtime before timing
    initCuda();

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
