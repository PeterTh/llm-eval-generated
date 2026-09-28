// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - Every unclustered point is tried as a seed; each candidate cluster is
//    grown by one CUDA thread block (grid-stride over seeds).
//  - Each block keeps, for every point, the maximum distance to the current
//    cluster members ("min-diameter if added"), updated incrementally after
//    each addition, and selects the next member with a block-wide argmin
//    reduction that breaks ties toward the lowest point index, exactly
//    matching the sequential candidate scan order.
//  - When it fits in GPU memory, the full pairwise distance matrix is
//    precomputed once and reused across all iterations.

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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

static const int BLOCK_SIZE = 256;
static const int MAX_GRID_BLOCKS = 512;

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

// Calculate Euclidean distance between two points (host)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Calculate Euclidean distance between two points (device)
__device__ inline double pointDist(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Precompute the full pairwise distance matrix
__global__ void distMatrixKernel(const double2* __restrict__ points,
                                 double* __restrict__ distmat,
                                 const int N) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (j < N) {
        distmat[static_cast<size_t>(i) * N + j] = pointDist(points[i], points[j]);
    }
}

// Grow a candidate cluster for each seed (one thread block per seed,
// grid-stride over seeds) and record its cardinality. For every point j,
// md[j] holds the maximum distance from j to the current cluster members,
// i.e. the cluster diameter if j were added; md[j] = -1 marks members of
// this candidate cluster and md[j] = DBL_MAX marks globally clustered
// points (both are never selected). The next member is the eligible point
// (md < threshold) with the smallest md, ties broken toward the lowest
// index to match the sequential scan.
// If members_out is non-null (single-seed rebuild launch), the members are
// recorded in selection order.
__global__ void growClustersKernel(const double2* __restrict__ points,
                                   const double* __restrict__ distmat,
                                   const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ seeds,
                                   const int num_seeds,
                                   const int N,
                                   const double threshold,
                                   double* __restrict__ workspace,
                                   int* __restrict__ cardinalities,
                                   int* __restrict__ members_out) {
    __shared__ double sval[BLOCK_SIZE];
    __shared__ int sidx[BLOCK_SIZE];
    __shared__ int chosen;

    double* md = workspace + static_cast<size_t>(blockIdx.x) * N;

    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        const int seed = seeds[s];
        const double2 seed_pt = points[seed];
        const double* seed_row =
            distmat ? distmat + static_cast<size_t>(seed) * N : nullptr;

        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            if (j == seed) {
                md[j] = -1.0;
            } else if (clustered[j]) {
                md[j] = DBL_MAX;
            } else {
                md[j] = seed_row ? seed_row[j] : pointDist(seed_pt, points[j]);
            }
        }
        if (threadIdx.x == 0 && members_out) {
            members_out[0] = seed;
        }
        __syncthreads();

        int count = 1;
        while (count < N) {
            // Block-wide argmin over eligible candidates
            double best_val = DBL_MAX;
            int best_idx = -1;
            for (int j = threadIdx.x; j < N; j += blockDim.x) {
                const double v = md[j];
                if (v >= 0.0 && v < threshold && v < best_val) {
                    best_val = v;
                    best_idx = j;
                }
            }
            sval[threadIdx.x] = best_val;
            sidx[threadIdx.x] = best_idx;
            __syncthreads();
            for (int off = blockDim.x / 2; off > 0; off >>= 1) {
                if (threadIdx.x < off) {
                    const double ov = sval[threadIdx.x + off];
                    const int oi = sidx[threadIdx.x + off];
                    if (oi >= 0 &&
                        (sidx[threadIdx.x] < 0 || ov < sval[threadIdx.x] ||
                         (ov == sval[threadIdx.x] && oi < sidx[threadIdx.x]))) {
                        sval[threadIdx.x] = ov;
                        sidx[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                chosen = sidx[0];
                if (chosen >= 0) {
                    md[chosen] = -1.0;
                    if (members_out) {
                        members_out[count] = chosen;
                    }
                }
            }
            __syncthreads();
            const int c = chosen;
            if (c < 0) break;  // No more points can be added
            count++;

            // Fold the new member's distances into md
            const double2 c_pt = points[c];
            const double* c_row =
                distmat ? distmat + static_cast<size_t>(c) * N : nullptr;
            for (int j = threadIdx.x; j < N; j += blockDim.x) {
                const double v = md[j];
                if (v >= 0.0 && v < DBL_MAX) {
                    const double d = c_row ? c_row[j] : pointDist(c_pt, points[j]);
                    if (d > v) {
                        md[j] = d;
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            cardinalities[s] = count;
        }
        __syncthreads();
    }
}

// Main QT clustering algorithm (CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Device buffers
    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;
    double* d_workspace = nullptr;
    double* d_distmat = nullptr;

    const int grid_max = std::min(N, MAX_GRID_BLOCKS);

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_workspace,
                          sizeof(double) * static_cast<size_t>(grid_max) * N));

    static_assert(sizeof(Point) == sizeof(double2), "Point/double2 layout mismatch");
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(double2) * N,
                          cudaMemcpyHostToDevice));

    // Precompute the pairwise distance matrix when it fits comfortably
    {
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const size_t distmat_bytes = sizeof(double) * static_cast<size_t>(N) * N;
        if (distmat_bytes < static_cast<size_t>(0.8 * free_mem)) {
            CUDA_CHECK(cudaMalloc(&d_distmat, distmat_bytes));
            const dim3 dm_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, N);
            distMatrixKernel<<<dm_grid, BLOCK_SIZE>>>(d_points, d_distmat, N);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    std::vector<int> cardinalities(N);
    std::vector<int> best_members_host(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                              sizeof(unsigned char) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                              sizeof(int) * num_seeds, cudaMemcpyHostToDevice));

        // Phase 1: cardinality of the candidate cluster for every seed
        const int grid = std::min(num_seeds, grid_max);
        growClustersKernel<<<grid, BLOCK_SIZE>>>(
            d_points, d_distmat, d_clustered, d_seeds, num_seeds, N, threshold,
            d_workspace, d_card, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_card,
                              sizeof(int) * num_seeds, cudaMemcpyDeviceToHost));

        // Pick the best seed: highest cardinality, first in seed order on ties
        int max_cardinality = -1;
        int best_seed = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_seed = unclustered_indices[i];
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            // Phase 2: rebuild the winning cluster, recording its members
            CUDA_CHECK(cudaMemcpy(d_seeds, &best_seed, sizeof(int),
                                  cudaMemcpyHostToDevice));
            growClustersKernel<<<1, BLOCK_SIZE>>>(
                d_points, d_distmat, d_clustered, d_seeds, 1, N, threshold,
                d_workspace, d_card, d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(best_members_host.data(), d_members,
                                  sizeof(int) * max_cardinality,
                                  cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.assign(best_members_host.begin(),
                                   best_members_host.begin() + max_cardinality);
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    cudaFree(d_points);
    cudaFree(d_clustered);
    cudaFree(d_seeds);
    cudaFree(d_card);
    cudaFree(d_members);
    cudaFree(d_workspace);
    if (d_distmat) cudaFree(d_distmat);

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

    // Initialize the CUDA context up front so one-time device setup
    // is not attributed to the clustering time
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
