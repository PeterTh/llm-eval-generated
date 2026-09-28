// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   - All pairwise distances are precomputed once on the GPU into an
//     N x N matrix, since they never change during clustering.
//   - In each outer round, every still-unclustered point is tried as a
//     candidate seed *simultaneously*: one CUDA block per seed grows a
//     candidate cluster greedily. Within a block, the threads cooperate
//     to scan all remaining candidate points in parallel and reduce to
//     find the next best point to add (parallel min-reduction).
//   - After all candidate clusters for the round have been grown, the
//     host picks the seed producing the largest cluster (matching the
//     original algorithm's tie-break of first-largest-in-index-order),
//     commits it, marks its members as clustered on the device, and the
//     next round begins.

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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                \
            exit(1);                                                          \
        }                                                                      \
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// GPU kernel: compute the full N x N pairwise distance matrix once.
__global__ void computeDistanceMatrixKernel(const Point* __restrict__ points,
                                             double* __restrict__ dist,
                                             int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        const double dx = points[i].x - points[j].x;
        const double dy = points[i].y - points[j].y;
        dist[static_cast<size_t>(i) * N + j] = sqrt(dx * dx + dy * dy);
    }
}

// GPU kernel: one block per candidate seed. Grows a greedy candidate
// cluster from that seed using the precomputed distance matrix, with the
// "find next best point" step parallelized across the block's threads.
__global__ void growCandidateClustersKernel(const double* __restrict__ dist,
                                             const bool* __restrict__ clustered,
                                             int N,
                                             double threshold,
                                             int* __restrict__ members_buf,
                                             bool* __restrict__ in_cluster_buf,
                                             int* __restrict__ cardinality_out) {
    const int seed = blockIdx.x;

    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinality_out[seed] = 0;
        return;
    }

    int* members = members_buf + static_cast<size_t>(seed) * N;
    bool* in_cluster = in_cluster_buf + static_cast<size_t>(seed) * N;

    // Reset this seed's in-cluster mask (parallelized across the block).
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        in_cluster[c] = false;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        members[0] = seed;
        in_cluster[seed] = true;
    }
    __syncthreads();

    extern __shared__ unsigned char smem[];
    double* best_val = reinterpret_cast<double*>(smem);
    int* best_idx = reinterpret_cast<int*>(best_val + blockDim.x);

    int count = 1;
    while (count < N) {
        double local_best_val = DBL_MAX;
        int local_best_idx = -1;

        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            if (clustered[c] || in_cluster[c]) continue;

            // Distance matrix is symmetric: read dist[member*N + c] instead of
            // dist[c*N + member] so that consecutive threads (consecutive c)
            // access consecutive memory addresses (coalesced global loads).
            double max_dist = 0.0;
            for (int m = 0; m < count; ++m) {
                const double d = dist[static_cast<size_t>(members[m]) * N + c];
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold && max_dist < local_best_val) {
                local_best_val = max_dist;
                local_best_idx = c;
            }
        }

        best_val[threadIdx.x] = local_best_val;
        best_idx[threadIdx.x] = local_best_idx;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (best_val[threadIdx.x + s] < best_val[threadIdx.x]) {
                    best_val[threadIdx.x] = best_val[threadIdx.x + s];
                    best_idx[threadIdx.x] = best_idx[threadIdx.x + s];
                }
            }
            __syncthreads();
        }

        const int chosen = best_idx[0];
        if (chosen < 0) break;

        if (threadIdx.x == 0) {
            members[count] = chosen;
            in_cluster[chosen] = true;
        }
        __syncthreads();
        count++;
    }

    if (threadIdx.x == 0) {
        cardinality_out[seed] = count;
    }
}

// GPU kernel: mark the winning cluster's members as clustered.
__global__ void markClusteredKernel(bool* __restrict__ clustered,
                                     const int* __restrict__ members,
                                     int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        clustered[members[i]] = true;
    }
}

// Main QT clustering algorithm - orchestrates GPU kernels per round.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    Point* d_points = nullptr;
    double* d_dist = nullptr;
    bool* d_clustered = nullptr;
    int* d_members = nullptr;
    bool* d_in_cluster = nullptr;
    int* d_cardinality = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_dist, static_cast<size_t>(N) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(N) * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(N) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, static_cast<size_t>(N) * N * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_cardinality, static_cast<size_t>(N) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, static_cast<size_t>(N) * sizeof(bool)));

    // Precompute the full pairwise distance matrix once.
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        computeDistanceMatrixKernel<<<grid, block>>>(d_points, d_dist, N);
        CUDA_CHECK(cudaGetLastError());
    }

    const int threadsPerBlock = 256;
    const size_t sharedMemBytes = threadsPerBlock * (sizeof(double) + sizeof(int));

    std::vector<int> h_cardinality(N);
    int num_unclustered = N;

    while (num_unclustered > 0) {
        growCandidateClustersKernel<<<N, threadsPerBlock, sharedMemBytes>>>(
            d_dist, d_clustered, N, threshold, d_members, d_in_cluster, d_cardinality);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_cardinality.data(), d_cardinality, static_cast<size_t>(N) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int best_seed = -1;
        int max_cardinality = -1;
        for (int i = 0; i < N; ++i) {
            if (h_cardinality[i] > max_cardinality) {
                max_cardinality = h_cardinality[i];
                best_seed = i;
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) {
            break;
        }

        std::vector<int> best_members(max_cardinality);
        CUDA_CHECK(cudaMemcpy(best_members.data(), d_members + static_cast<size_t>(best_seed) * N,
                              static_cast<size_t>(max_cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        const int blocks = (max_cardinality + threadsPerBlock - 1) / threadsPerBlock;
        markClusteredKernel<<<blocks, threadsPerBlock>>>(
            d_clustered, d_members + static_cast<size_t>(best_seed) * N, max_cardinality);
        CUDA_CHECK(cudaGetLastError());

        num_unclustered -= max_cardinality;
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_in_cluster));
    CUDA_CHECK(cudaFree(d_cardinality));

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

    // Warm up the CUDA context/driver so that its one-time initialization
    // cost is not counted as part of the algorithm's clustering time.
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
