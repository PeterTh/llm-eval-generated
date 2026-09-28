// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The GPU parallelization works as follows:
//  - The full pairwise point-distance matrix is computed once on the GPU.
//  - For every "round" of the sequential QT outer loop, one CUDA thread
//    block per still-unclustered candidate seed grows a candidate cluster
//    in parallel (all seeds are tried simultaneously). Within a block, the
//    N points are distributed across threads; a running "distance to
//    cluster" value per point is maintained incrementally and a block-wide
//    argmin reduction (with the same tie-break rule as the reference
//    implementation) selects the next point to add to that seed's cluster.
//  - The winning seed (largest cluster, first occurrence wins on ties,
//    exactly like the sequential reference) is picked on the host from the
//    per-seed cardinalities, and its member list is copied back from the
//    device.

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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                    __LINE__, cudaGetErrorString(err__));                      \
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

// Calculate Euclidean distance between two points (host-side, used for validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Compute the full N x N pairwise Euclidean distance matrix.
__global__ void computeDistanceMatrixKernel(const double* __restrict__ xs,
                                             const double* __restrict__ ys,
                                             float* __restrict__ dist,
                                             int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        double dx = xs[i] - xs[j];
        double dy = ys[i] - ys[j];
        dist[static_cast<size_t>(i) * N + j] = static_cast<float>(sqrt(dx * dx + dy * dy));
    }
}

// One block per candidate seed. Grows a candidate cluster from that seed and
// records the resulting member list and cardinality.
//
// value[i] semantics:
//   - INF  -> point i is unavailable (already globally clustered, or already
//             a member of the cluster being grown in this block)
//   - else -> current maximum distance from point i to all cluster members
//             added so far (0 before any distance has been folded in)
__global__ void growClustersKernel(const float* __restrict__ dist,
                                    const bool* __restrict__ clusteredGlobal,
                                    const int* __restrict__ seedIndices,
                                    int numSeeds,
                                    float threshold,
                                    int N,
                                    float* __restrict__ valueBuf,   // numSeeds * N
                                    int* __restrict__ membersBuf,   // numSeeds * N
                                    int* __restrict__ cardinalityOut) {
    const int b = blockIdx.x;
    if (b >= numSeeds) return;

    const int seed = seedIndices[b];
    float* value = valueBuf + static_cast<size_t>(b) * N;
    int* members = membersBuf + static_cast<size_t>(b) * N;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float kInf = HUGE_VALF;

    __shared__ int s_count;
    __shared__ int s_newMember;
    __shared__ bool s_done;

    extern __shared__ unsigned char s_mem[];
    float* s_bestVal = reinterpret_cast<float*>(s_mem);
    int* s_bestIdx = reinterpret_cast<int*>(s_bestVal + nthreads);

    // Initialize value[] based on global clustered state.
    for (int i = tid; i < N; i += nthreads) {
        value[i] = clusteredGlobal[i] ? kInf : 0.0f;
    }
    __syncthreads();

    if (tid == 0) {
        value[seed] = kInf;
        members[0] = seed;
        s_count = 1;
        s_newMember = seed;
        s_done = false;
    }
    __syncthreads();

    while (s_count < N) {
        const int newMember = s_newMember;
        const float* distRow = dist + static_cast<size_t>(newMember) * N;

        // Fold the newly added member's distances into the running max.
        for (int i = tid; i < N; i += nthreads) {
            if (value[i] != kInf) {
                value[i] = fmaxf(value[i], distRow[i]);
            }
        }
        __syncthreads();

        // Find the point with the smallest running max-distance that is
        // still below the threshold (argmin, smallest index wins ties).
        float bestVal = kInf;
        int bestIdx = -1;
        for (int i = tid; i < N; i += nthreads) {
            const float v = value[i];
            if (v < threshold && v < bestVal) {
                bestVal = v;
                bestIdx = i;
            }
        }
        s_bestVal[tid] = bestVal;
        s_bestIdx[tid] = bestIdx;
        __syncthreads();

        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float otherVal = s_bestVal[tid + stride];
                const int otherIdx = s_bestIdx[tid + stride];
                const float myVal = s_bestVal[tid];
                const int myIdx = s_bestIdx[tid];
                bool takeOther = (otherIdx >= 0) &&
                                  (myIdx < 0 || otherVal < myVal ||
                                   (otherVal == myVal && otherIdx < myIdx));
                if (takeOther) {
                    s_bestVal[tid] = otherVal;
                    s_bestIdx[tid] = otherIdx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int chosen = s_bestIdx[0];
            if (chosen < 0) {
                s_done = true;
            } else {
                value[chosen] = kInf;
                members[s_count] = chosen;
                s_count++;
                s_newMember = chosen;
            }
        }
        __syncthreads();

        if (s_done) break;
    }

    if (tid == 0) {
        cardinalityOut[b] = s_count;
    }
}

// ---------------------------------------------------------------------------
// Host-side driver
// ---------------------------------------------------------------------------

// Main QT clustering algorithm (GPU-parallel across candidate seeds)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // --- Upload point coordinates and compute the full distance matrix ---
    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        xs[i] = points[i].x;
        ys[i] = points[i].y;
    }

    double *d_xs = nullptr, *d_ys = nullptr;
    float* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xs, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float) * static_cast<size_t>(N) * N));
    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        computeDistanceMatrixKernel<<<grid, block>>>(d_xs, d_ys, d_dist, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));

    // --- Persistent device buffers reused across outer-loop rounds ---
    bool* d_clustered = nullptr;
    int* d_seedIndices = nullptr;
    float* d_valueBuf = nullptr;
    int* d_membersBuf = nullptr;
    int* d_cardinality = nullptr;

    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(bool) * N));
    CUDA_CHECK(cudaMalloc(&d_seedIndices, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_valueBuf, sizeof(float) * static_cast<size_t>(N) * N));
    CUDA_CHECK(cudaMalloc(&d_membersBuf, sizeof(int) * static_cast<size_t>(N) * N));
    CUDA_CHECK(cudaMalloc(&d_cardinality, sizeof(int) * N));

    std::vector<int> hostSeedIndices(N);
    std::vector<int> hostCardinality(N);
    std::vector<int> hostClustered(N); // used only for host<->bool conversion staging

    // Must be a power of two: the block-wide reduction below halves the
    // active thread count each step (stride >>= 1).
    int threadsPerBlock = 32;
    while (threadsPerBlock < N && threadsPerBlock < 256) {
        threadsPerBlock *= 2;
    }
    const size_t sharedMemBytes =
        static_cast<size_t>(threadsPerBlock) * (sizeof(float) + sizeof(int));

    // Stage the (initially all-false) clustered array once.
    {
        std::vector<unsigned char> initClustered(N, 0);
        CUDA_CHECK(cudaMemcpy(d_clustered, initClustered.data(), sizeof(bool) * N,
                               cudaMemcpyHostToDevice));
    }

    while (!unclustered_indices.empty()) {
        const int numSeeds = static_cast<int>(unclustered_indices.size());
        for (int i = 0; i < numSeeds; ++i) {
            hostSeedIndices[i] = unclustered_indices[i];
        }
        CUDA_CHECK(cudaMemcpy(d_seedIndices, hostSeedIndices.data(),
                               sizeof(int) * numSeeds, cudaMemcpyHostToDevice));

        growClustersKernel<<<numSeeds, threadsPerBlock, sharedMemBytes>>>(
            d_dist, d_clustered, d_seedIndices, numSeeds,
            static_cast<float>(threshold), N, d_valueBuf, d_membersBuf, d_cardinality);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hostCardinality.data(), d_cardinality,
                               sizeof(int) * numSeeds, cudaMemcpyDeviceToHost));

        // First occurrence of the maximum cardinality wins (matches the
        // sequential reference's strict "cardinality > max_cardinality").
        int max_cardinality = -1;
        int best_block = -1;
        for (int i = 0; i < numSeeds; ++i) {
            if (hostCardinality[i] > max_cardinality) {
                max_cardinality = hostCardinality[i];
                best_block = i;
            }
        }

        if (best_block >= 0 && max_cardinality > 0) {
            std::vector<int> best_cluster_members(max_cardinality);
            CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                                   d_membersBuf + static_cast<size_t>(best_block) * N,
                                   sizeof(int) * max_cardinality, cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_block];
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            for (int m : best_cluster_members) {
                clustered[m] = true;
            }

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());

            for (int i = 0; i < N; ++i) {
                hostClustered[i] = clustered[i] ? 1 : 0;
            }
            std::vector<unsigned char> packed(N);
            for (int i = 0; i < N; ++i) packed[i] = static_cast<unsigned char>(hostClustered[i]);
            CUDA_CHECK(cudaMemcpy(d_clustered, packed.data(), sizeof(bool) * N,
                                   cudaMemcpyHostToDevice));
        } else {
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seedIndices));
    CUDA_CHECK(cudaFree(d_valueBuf));
    CUDA_CHECK(cudaFree(d_membersBuf));
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
