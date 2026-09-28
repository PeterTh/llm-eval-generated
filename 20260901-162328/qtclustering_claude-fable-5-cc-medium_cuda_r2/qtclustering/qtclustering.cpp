// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: in every round of the outer loop, a candidate cluster
// must be grown greedily from each remaining unclustered seed. On the GPU
// one thread block handles one seed, and the threads of the block
// parallelize over the candidate points. Each block maintains, for every
// candidate point, the running maximum distance to the current cluster
// members, so one greedy growth step costs O(N / blockDim) instead of
// O(N * |cluster|). The host then selects the seed that produced the
// largest candidate cluster (lowest index wins ties, exactly as in the
// sequential version) and re-runs a single block to record its members.

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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(1);                                                          \
        }                                                                     \
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

__device__ inline double distDev(double x1, double y1, double x2, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

static const int BLOCK_SIZE = 256;

// Grow one candidate cluster per thread block. Each block grid-strides over
// the seeds; scratch holds one row of N running max-distances per block.
// A candidate's scratch entry is DBL_MAX when it is excluded (already
// clustered, or already a member of the growing cluster).
//
// When `members` is non-null (single-block relaunch for the winning seed),
// the member indices are recorded in the order they were added.
//
// `globalMax` (when non-null) tracks the best completed cardinality across
// all blocks. Since a candidate's running max-distance only grows, points
// already at or beyond the threshold can never join, so
// count + #viable_candidates is an upper bound on the final cardinality.
// When that bound falls strictly below the best cardinality seen so far,
// this seed cannot win (nor tie an earlier seed at globalMax) and the block
// abandons it, reporting the current count (a value < globalMax, so the
// host's winner selection is unaffected).
__global__ void candidateClusterKernel(const double* __restrict__ xs,
                                       const double* __restrict__ ys,
                                       const int N,
                                       const int* __restrict__ seeds,
                                       const int numSeeds,
                                       const unsigned char* __restrict__ clustered,
                                       const double threshold,
                                       double* __restrict__ scratch,
                                       int* __restrict__ cardinality,
                                       int* __restrict__ members,
                                       int* __restrict__ globalMax) {
    __shared__ double sval[BLOCK_SIZE];
    __shared__ int sidx[BLOCK_SIZE];
    __shared__ int scnt[BLOCK_SIZE];
    __shared__ int sgmax;

    const int tid = threadIdx.x;
    double* maxd = scratch + static_cast<size_t>(blockIdx.x) * N;

    for (int seedIdx = blockIdx.x; seedIdx < numSeeds; seedIdx += gridDim.x) {
        const int seed = seeds[seedIdx];
        const double sx = xs[seed];
        const double sy = ys[seed];

        // Initialize running max-distance of every candidate to the seed.
        for (int c = tid; c < N; c += blockDim.x) {
            maxd[c] = (clustered[c] || c == seed) ? DBL_MAX
                                                  : distDev(xs[c], ys[c], sx, sy);
        }
        if (members && tid == 0) {
            members[0] = seed;
        }
        __syncthreads();

        int count = 1;
        while (count < N) {
            // Find the candidate with the smallest running max-distance that
            // stays below the threshold; ties go to the lowest index, exactly
            // matching the sequential findClosestPoint.
            double bestv = DBL_MAX;
            int besti = -1;
            int viable = 0;
            for (int c = tid; c < N; c += blockDim.x) {
                const double v = maxd[c];
                if (v < threshold) {
                    viable++;
                    if (v < bestv) {
                        bestv = v;
                        besti = c;
                    }
                }
            }
            sval[tid] = bestv;
            sidx[tid] = besti;
            scnt[tid] = viable;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    const double ov = sval[tid + s];
                    const int oi = sidx[tid + s];
                    if (oi >= 0 && (sidx[tid] < 0 || ov < sval[tid] ||
                                    (ov == sval[tid] && oi < sidx[tid]))) {
                        sval[tid] = ov;
                        sidx[tid] = oi;
                    }
                    scnt[tid] += scnt[tid + s];
                }
                __syncthreads();
            }
            const int best = sidx[0];
            const int totalViable = scnt[0];
            // Read globalMax once (thread 0) so the break below is uniform
            // across the block. A stale value only weakens the pruning.
            if (tid == 0) {
                sgmax = globalMax ? *static_cast<volatile int*>(globalMax) : 0;
            }
            __syncthreads();
            const int gmax = sgmax;
            __syncthreads();

            if (best < 0) break;  // No more points can be added

            if (globalMax && count + totalViable < gmax) break;

            if (members && tid == 0) {
                members[count] = best;
            }
            count++;

            // Fold the new member into every candidate's running max-distance.
            const double bx = xs[best];
            const double by = ys[best];
            for (int c = tid; c < N; c += blockDim.x) {
                const double v = maxd[c];
                if (c == best) {
                    maxd[c] = DBL_MAX;
                } else if (v != DBL_MAX) {
                    const double d = distDev(xs[c], ys[c], bx, by);
                    maxd[c] = (d > v) ? d : v;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            cardinality[seedIdx] = count;
            // No-op for pruned seeds (their count is below globalMax).
            if (globalMax) atomicMax(globalMax, count);
        }
        __syncthreads();
    }
}

// Main QT clustering algorithm (GPU-accelerated)
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

    // Upload point coordinates once (structure-of-arrays for coalescing).
    std::vector<double> h_xs(N), h_ys(N);
    for (int i = 0; i < N; ++i) {
        h_xs[i] = points[i].x;
        h_ys[i] = points[i].y;
    }

    const int maxBlocks = std::min(N, 4096);

    double *d_xs, *d_ys, *d_scratch;
    int *d_seeds, *d_cardinality, *d_members, *d_globalMax;
    unsigned char* d_clustered;
    CUDA_CHECK(cudaMalloc(&d_xs, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ys, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scratch,
                          static_cast<size_t>(maxBlocks) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinality, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_globalMax, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                          N * sizeof(unsigned char), cudaMemcpyHostToDevice));

    std::vector<int> h_cardinality(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int numSeeds = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                              numSeeds * sizeof(int), cudaMemcpyHostToDevice));

        // Grow a candidate cluster from every unclustered seed in parallel.
        CUDA_CHECK(cudaMemset(d_globalMax, 0, sizeof(int)));
        const int numBlocks = std::min(numSeeds, maxBlocks);
        candidateClusterKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_xs, d_ys, N, d_seeds, numSeeds, d_clustered, threshold,
            d_scratch, d_cardinality, nullptr, d_globalMax);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_cardinality.data(), d_cardinality,
                              numSeeds * sizeof(int), cudaMemcpyDeviceToHost));

        // Pick the largest candidate cluster; first seed wins ties (the
        // seeds are stored in ascending order, matching the sequential scan).
        int max_cardinality = -1;
        int best_pos = -1;
        for (int i = 0; i < numSeeds; ++i) {
            if (h_cardinality[i] > max_cardinality) {
                max_cardinality = h_cardinality[i];
                best_pos = i;
            }
        }

        if (best_pos >= 0 && max_cardinality > 0) {
            // Re-grow the winning candidate with a single block to record
            // its member list.
            candidateClusterKernel<<<1, BLOCK_SIZE>>>(
                d_xs, d_ys, N, d_seeds + best_pos, 1, d_clustered, threshold,
                d_scratch, d_cardinality, d_members, nullptr);
            CUDA_CHECK(cudaGetLastError());

            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_pos];
            cluster.members.resize(max_cardinality);
            CUDA_CHECK(cudaMemcpy(cluster.members.data(), d_members,
                                  max_cardinality * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = 1;
            }
            CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                                  N * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice));

            clusters.push_back(std::move(cluster));

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

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_cardinality));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_globalMax));
    CUDA_CHECK(cudaFree(d_clustered));

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

    // Initialize the CUDA context before timing the clustering itself.
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
