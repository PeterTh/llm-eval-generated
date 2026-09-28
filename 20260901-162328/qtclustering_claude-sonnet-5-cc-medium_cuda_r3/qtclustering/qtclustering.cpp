// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy: for every round of the outer clustering loop,
// each still-unclustered point is evaluated in parallel (one CUDA thread
// block per candidate seed) to grow a full candidate cluster using the
// same greedy nearest-point-below-threshold rule as the original
// sequential algorithm. Within a block, the search for the next closest
// point below the threshold is itself parallelized across threads with a
// block-wide reduction. The host then picks the largest candidate cluster
// (identical tie-breaking rule as the original: first / smallest seed
// index wins on ties), commits it, and repeats until no points remain.

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                 \
            exit(1);                                                            \
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

// Calculate Euclidean distance between two points (host-side, used by validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Compute the full pairwise distance matrix once (reused across all rounds).
__global__ void computeDistanceMatrixKernel(const double* __restrict__ x,
                                             const double* __restrict__ y,
                                             double* __restrict__ dist,
                                             const int N) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        const double dx = x[i] - x[j];
        const double dy = y[i] - y[j];
        dist[static_cast<size_t>(i) * N + j] = sqrt(dx * dx + dy * dy);
    }
}

// One thread block per active (unclustered) seed point. Grows a candidate
// cluster greedily: repeatedly finds the not-yet-clustered, not-yet-member
// point that minimizes the resulting cluster diameter while keeping it
// strictly below the threshold, exactly mirroring the sequential
// findClosestPoint/generateCandidateCluster logic.
__global__ void growClustersKernel(const double* __restrict__ dist,
                                    const bool* __restrict__ clustered,
                                    const int* __restrict__ activeSeeds,
                                    int* __restrict__ membersOut,   // [gridDim.x][N]
                                    int* __restrict__ sizeOut,      // [gridDim.x]
                                    const int N,
                                    const double threshold) {
    extern __shared__ unsigned char sharedMem[];
    double* sh_val = reinterpret_cast<double*>(sharedMem);
    int* sh_idx = reinterpret_cast<int*>(sh_val + blockDim.x);

    const int row = blockIdx.x;
    const int seed = activeSeeds[row];
    int* members = membersOut + static_cast<size_t>(row) * N;

    __shared__ int size;
    __shared__ bool found;
    __shared__ int bestIdx;

    if (threadIdx.x == 0) {
        members[0] = seed;
        size = 1;
    }
    __syncthreads();

    while (size < N) {
        double localBestVal = DBL_MAX;
        int localBestIdx = -1;

        for (int candidate = threadIdx.x; candidate < N; candidate += blockDim.x) {
            if (clustered[candidate]) continue;

            bool isMember = false;
            double maxDist = 0.0;
            for (int m = 0; m < size; ++m) {
                const int member = members[m];
                if (member == candidate) {
                    isMember = true;
                    break;
                }
                const double d = dist[static_cast<size_t>(candidate) * N + member];
                if (d > maxDist) maxDist = d;
            }
            if (isMember) continue;

            if (maxDist < threshold) {
                if (maxDist < localBestVal ||
                    (maxDist == localBestVal && candidate < localBestIdx)) {
                    localBestVal = maxDist;
                    localBestIdx = candidate;
                }
            }
        }

        sh_val[threadIdx.x] = localBestVal;
        sh_idx[threadIdx.x] = localBestIdx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double otherVal = sh_val[threadIdx.x + stride];
                const int otherIdx = sh_idx[threadIdx.x + stride];
                const double myVal = sh_val[threadIdx.x];
                const int myIdx = sh_idx[threadIdx.x];
                if (otherIdx >= 0 &&
                    (myIdx < 0 || otherVal < myVal ||
                     (otherVal == myVal && otherIdx < myIdx))) {
                    sh_val[threadIdx.x] = otherVal;
                    sh_idx[threadIdx.x] = otherIdx;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            bestIdx = sh_idx[0];
            found = (bestIdx >= 0);
            if (found) {
                members[size] = bestIdx;
                size++;
            }
        }
        __syncthreads();

        if (!found) break;
    }

    if (threadIdx.x == 0) {
        sizeOut[row] = size;
    }
}

// ---------------------------------------------------------------------------
// Host-side GPU-driven QT clustering
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    std::vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
    }

    double *d_x = nullptr, *d_y = nullptr, *d_dist = nullptr;
    bool* d_clustered = nullptr;
    int* d_activeSeeds = nullptr;
    int* d_members = nullptr;
    int* d_size = nullptr;

    const size_t distBytes = static_cast<size_t>(N) * N * sizeof(double);
    const size_t membersBytes = static_cast<size_t>(N) * N * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, distBytes));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_activeSeeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, membersBytes));
    CUDA_CHECK(cudaMalloc(&d_size, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Precompute the full pairwise distance matrix once; reused every round.
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        computeDistanceMatrixKernel<<<grid, block>>>(d_x, d_y, d_dist, N);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<bool> clustered(N, false);
    // Device wants a plain bool array (std::vector<bool> is bit-packed).
    std::vector<unsigned char> clusteredBytes(N, 0);
    CUDA_CHECK(cudaMemcpy(d_clustered, clusteredBytes.data(), N * sizeof(bool),
                          cudaMemcpyHostToDevice));

    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    const int blockSize = 256;
    const size_t sharedBytes = blockSize * (sizeof(double) + sizeof(int));

    std::vector<int> sizes;
    std::vector<int> memberBuf(N);

    while (!unclustered_indices.empty()) {
        const int M = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_activeSeeds, unclustered_indices.data(),
                              M * sizeof(int), cudaMemcpyHostToDevice));

        growClustersKernel<<<M, blockSize, sharedBytes>>>(
            d_dist, d_clustered, d_activeSeeds, d_members, d_size, N, threshold);
        CUDA_CHECK(cudaGetLastError());

        sizes.resize(M);
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_size, M * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int bestRow = -1;
        int maxCardinality = -1;
        for (int i = 0; i < M; ++i) {
            if (sizes[i] > maxCardinality) {
                maxCardinality = sizes[i];
                bestRow = i;
            }
        }

        if (bestRow < 0 || maxCardinality <= 0) break;

        memberBuf.resize(maxCardinality);
        CUDA_CHECK(cudaMemcpy(memberBuf.data(),
                              d_members + static_cast<size_t>(bestRow) * N,
                              maxCardinality * sizeof(int), cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = unclustered_indices[bestRow];
        cluster.members.assign(memberBuf.begin(), memberBuf.end());
        clusters.push_back(cluster);

        for (int m : cluster.members) {
            clustered[m] = true;
            clusteredBytes[m] = 1;
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, clusteredBytes.data(), N * sizeof(bool),
                              cudaMemcpyHostToDevice));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dist);
    cudaFree(d_clustered);
    cudaFree(d_activeSeeds);
    cudaFree(d_members);
    cudaFree(d_size);

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
