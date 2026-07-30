// QT Clustering Benchmark - CUDA Parallelized Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// This version uses CUDA to parallelize the seed evaluation step across
// all unclustered points, and within each seed evaluation, parallelizes
// the distance computations across candidate points.

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

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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

// ---------------------------------------------------------------------------
// CUDA kernel: evaluate all unclustered seeds in parallel.
// Each block handles one seed and builds a candidate cluster by repeatedly
// adding the closest point that keeps the cluster diameter under threshold.
//
// Kernel outputs the cardinality for each seed into g_cardinalities[].
// The per-block scratch space retains the cluster member list for retrieval
// of the winning seed's members after the kernel completes.
// ---------------------------------------------------------------------------
__global__ void computeSeedCardinalitiesKernel(
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const char*   __restrict__ g_clustered,
    const int*    __restrict__ g_unclustered,
    int num_unclustered,
    double threshold,
    int num_points,
    int*          __restrict__ g_cardinalities,
    char*  g_scratch_in_cluster,   // [num_unclustered * num_points]
    int*   g_scratch_members)      // [num_unclustered * num_points]
{
    int bid = blockIdx.x;
    if (bid >= num_unclustered) return;

    int seed = g_unclustered[bid];

    // Per-block scratch space in global memory
    char* in_cluster = g_scratch_in_cluster + (size_t)bid * num_points;
    int*  members    = g_scratch_members    + (size_t)bid * num_points;

    // Initialize per-block in_cluster from global clustered array
    for (int i = threadIdx.x; i < num_points; i += blockDim.x) {
        in_cluster[i] = g_clustered[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        in_cluster[seed] = 1;
        members[0] = seed;
    }
    __syncthreads();

    // Shared memory for parallel reduction
    __shared__ double s_red_dist[1024];
    __shared__ int    s_red_cand[1024];
    __shared__ int    s_num_members;
    __shared__ bool   s_found;

    if (threadIdx.x == 0) s_num_members = 1;
    __syncthreads();

    // Iteratively grow the cluster for this seed
    while (true) {
        int n = s_num_members;

        double best_dist = 1e30;
        int    best_cand = -1;

        // Each thread scans a subset of candidate points
        for (int c = threadIdx.x; c < num_points; c += blockDim.x) {
            if (!in_cluster[c]) {
                double max_dist = 0.0;
                // Distance to all current cluster members
                for (int m = 0; m < n; ++m) {
                    int mem = members[m];
                    double dx = g_x[c] - g_x[mem];
                    double dy = g_y[c] - g_y[mem];
                    double d = sqrt(dx * dx + dy * dy);
                    if (d > max_dist) max_dist = d;
                }
                if (max_dist < threshold && max_dist < best_dist) {
                    best_dist = max_dist;
                    best_cand = c;
                }
            }
        }

        // ---- Parallel reduction to find global minimum max_dist ----
        s_red_dist[threadIdx.x] = best_dist;
        s_red_cand[threadIdx.x] = best_cand;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (s_red_dist[threadIdx.x + s] < s_red_dist[threadIdx.x]) {
                    s_red_dist[threadIdx.x] = s_red_dist[threadIdx.x + s];
                    s_red_cand[threadIdx.x] = s_red_cand[threadIdx.x + s];
                }
            }
            __syncthreads();
        }

        // Thread 0 adds the best candidate if valid
        if (threadIdx.x == 0) {
            if (s_red_cand[0] >= 0 && s_red_dist[0] < threshold) {
                members[s_num_members] = s_red_cand[0];
                in_cluster[s_red_cand[0]] = 1;
                s_num_members++;
                s_found = true;
            } else {
                s_found = false;
            }
        }
        __syncthreads();

        if (!s_found) break;
    }

    if (threadIdx.x == 0) {
        g_cardinalities[bid] = s_num_members;
    }
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm using CUDA.
// Parallelizes seed evaluation across all unclustered points on the GPU.
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());

    // ---- Allocate GPU memory ----
    double *d_x = nullptr, *d_y = nullptr;
    char   *d_clustered = nullptr;
    int    *d_unclustered = nullptr;
    int    *d_cardinalities = nullptr;
    char   *d_scratch_in_cluster = nullptr;
    int    *d_scratch_members = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_unclustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, N * sizeof(int)));

    // Upload point coordinates (constant throughout)
    std::vector<double> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) {
        h_x[i] = points[i].x;
        h_y[i] = points[i].y;
    }
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Scratch space: each block needs num_points entries for in_cluster and members.
    // Max blocks = N (all points unclustered in first iteration).
    CUDA_CHECK(cudaMalloc(&d_scratch_in_cluster, (size_t)N * N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_scratch_members,    (size_t)N * N * sizeof(int)));

    // ---- Host state ----
    std::vector<bool> clustered(N, false);
    std::vector<int>  unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    std::vector<Cluster> clusters;
    clusters.reserve(N);

    const int block_size = 256;

    while (!unclustered_indices.empty()) {
        int num_unclustered = static_cast<int>(unclustered_indices.size());

        // Upload current clustered status (use vector<char> because
        // std::vector<bool> has no data() method)
        std::vector<char> h_clustered(N);
        for (int i = 0; i < N; ++i) h_clustered[i] = clustered[i] ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(),
                              N * sizeof(char), cudaMemcpyHostToDevice));

        // Upload current unclustered indices
        CUDA_CHECK(cudaMemcpy(d_unclustered, unclustered_indices.data(),
                              num_unclustered * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Launch kernel: one block per seed
        int num_blocks = num_unclustered;
        computeSeedCardinalitiesKernel<<<num_blocks, block_size>>>(
            d_x, d_y, d_clustered, d_unclustered, num_unclustered,
            threshold, N, d_cardinalities,
            d_scratch_in_cluster, d_scratch_members);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy cardinalities back
        std::vector<int> h_cardinalities(num_unclustered);
        CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities,
                              num_unclustered * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Find seed with maximum cardinality (best cluster)
        int best_idx = -1;
        int max_cardinality = 0;
        for (int i = 0; i < num_unclustered; ++i) {
            if (h_cardinalities[i] > max_cardinality) {
                max_cardinality = h_cardinalities[i];
                best_idx = i;
            }
        }

        if (best_idx < 0 || max_cardinality <= 0) break;

        int best_seed = unclustered_indices[best_idx];

        // Retrieve the winning cluster's members from GPU scratch space
        std::vector<int> best_members(max_cardinality);
        CUDA_CHECK(cudaMemcpy(best_members.data(),
                              d_scratch_members + (size_t)best_idx * N,
                              max_cardinality * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Record the cluster
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_members);
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (int m : cluster.members) {
            clustered[m] = true;
        }

        // Remove clustered points from the unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(),
                           unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_unclustered));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_scratch_in_cluster));
    CUDA_CHECK(cudaFree(d_scratch_members));

    return clusters;
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
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

    printf("QT Clustering Benchmark (CUDA Parallel Version)\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Select and display GPU device info
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    printf("GPU Device: %s\n", props.name);

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering (CUDA accelerated)
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
