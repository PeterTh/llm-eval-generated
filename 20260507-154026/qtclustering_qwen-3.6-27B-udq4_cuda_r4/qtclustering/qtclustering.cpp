// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
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
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Kernel: compute full NxN distance matrix in parallel (2D grid)
__global__ void computeDistanceMatrixKernel(const double* __restrict__ x,
                                            const double* __restrict__ y,
                                            double* __restrict__ dist_matrix,
                                            int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    double dx = x[i] - x[j];
    double dy = y[i] - y[j];
    dist_matrix[i * N + j] = sqrt(dx * dx + dy * dy);
}

// Kernel: each thread evaluates one seed independently.
// Inside the thread we simulate generateCandidateCluster sequentially,
// iterating candidates in order 0..N-1 so tie-breaking matches the
// original sequential code exactly.
__global__ void evaluateSeedsKernel(
    const double* __restrict__ dist_matrix,
    const char*   __restrict__ clustered,
    int*          __restrict__ cardinalities,
    int*          __restrict__ members_workspace,
    int*          __restrict__ in_cluster_workspace,
    int N,
    int num_seeds,
    const int*    __restrict__ seeds,
    double threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_seeds) return;

    int seed = seeds[tid];
    int* my_members    = members_workspace    + tid * N;
    int* my_in_cluster = in_cluster_workspace + tid * N;

    // Initialise per-thread in_cluster bitset
    for (int i = 0; i < N; ++i) my_in_cluster[i] = 0;

    // Seed the cluster
    my_members[0] = seed;
    my_in_cluster[seed] = 1;
    int member_count = 1;

    // Iteratively add the closest unclustered point
    while (member_count < N) {
        int closest = -1;
        double min_diameter = 1e300;

        for (int candidate = 0; candidate < N; ++candidate) {
            if (clustered[candidate])       continue;
            if (my_in_cluster[candidate])   continue;

            // Maximum distance from candidate to any cluster member
            double max_dist = 0.0;
            for (int m = 0; m < member_count; ++m) {
                double d = dist_matrix[candidate * N + my_members[m]];
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest = candidate;
            }
        }

        if (closest < 0) break;
        my_members[member_count] = closest;
        my_in_cluster[closest]   = 1;
        ++member_count;
    }

    cardinalities[tid] = member_count;
}

// ---------------------------------------------------------------------------
// GPU-accelerated QT clustering
// ---------------------------------------------------------------------------

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());
    if (N == 0) return {};

    // Extract coordinate arrays
    std::vector<double> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) {
        h_x[i] = points[i].x;
        h_y[i] = points[i].y;
    }

    // ---- GPU allocations ----
    double *d_x, *d_y, *d_dist_matrix;
    char   *d_clustered;
    int    *d_seeds, *d_cardinalities;
    int    *d_members_workspace, *d_in_cluster_workspace;

    CUDA_CHECK(cudaMalloc(&d_x,                 N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,                 N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist_matrix,       N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered,         N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_seeds,             N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities,     N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members_workspace, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster_workspace, N * N * sizeof(int)));

    // Copy coordinates to GPU
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Compute distance matrix on GPU
    {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (N + 31) / 32);
        computeDistanceMatrixKernel<<<grid, block>>>(d_x, d_y, d_dist_matrix, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Host-side state (use char instead of bool for .data() compatibility)
    std::vector<char>  clustered(N, 0);
    std::vector<int>   unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    std::vector<int> h_cardinalities(N);
    std::vector<Cluster> clusters;

    // ---- Main clustering loop ----
    while (!unclustered_indices.empty()) {
        int num_seeds = static_cast<int>(unclustered_indices.size());

        // Transfer per-round data to GPU
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), N * sizeof(char),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                              num_seeds * sizeof(int), cudaMemcpyHostToDevice));

        // Launch parallel seed-evaluation kernel
        const int block_size = 256;
        const int grid_size  = (num_seeds + block_size - 1) / block_size;
        evaluateSeedsKernel<<<grid_size, block_size>>>(
            d_dist_matrix, d_clustered, d_cardinalities,
            d_members_workspace, d_in_cluster_workspace,
            N, num_seeds, d_seeds, threshold);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Retrieve cardinalities
        CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities,
                              num_seeds * sizeof(int), cudaMemcpyDeviceToHost));

        // Find best seed (strict >  keeps first seed with max cardinality)
        int max_cardinality = -1;
        int best_seed_idx   = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (h_cardinalities[i] > max_cardinality) {
                max_cardinality = h_cardinalities[i];
                best_seed_idx   = i;
            }
        }

        if (best_seed_idx >= 0 && max_cardinality > 0) {
            // Copy best cluster members back
            std::vector<int> best_members(max_cardinality);
            CUDA_CHECK(cudaMemcpy(best_members.data(),
                                   d_members_workspace + best_seed_idx * N,
                                   max_cardinality * sizeof(int),
                                   cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_seed_idx];
            cluster.members    = std::move(best_members);
            clusters.push_back(cluster);

            // Mark members as clustered
            for (int m : cluster.members) clustered[m] = true;

            // Shrink unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(),
                               unclustered_indices.end(),
                               [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());
        } else {
            break;
        }
    }

    // ---- Cleanup ----
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dist_matrix);
    cudaFree(d_clustered);
    cudaFree(d_seeds);
    cudaFree(d_cardinalities);
    cudaFree(d_members_workspace);
    cudaFree(d_in_cluster_workspace);

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                            points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

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

    // Perform QT clustering (GPU accelerated)
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
