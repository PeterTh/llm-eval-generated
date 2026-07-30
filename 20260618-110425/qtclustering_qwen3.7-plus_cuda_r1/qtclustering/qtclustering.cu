// QT Clustering Benchmark - CUDA Parallelized Version
//
// QT (Quality Threshold) clustering builds clusters by starting with a seed
// point and iteratively adding the closest point that maintains the cluster's
// diameter below a threshold.
//
// CUDA parallelization strategy:
// 1. Precompute full NxN distance matrix on GPU (embarrassingly parallel)
// 2. Main clustering loop on CPU; each iteration launches a kernel that
//    generates candidate clusters for ALL seed points in parallel
//    (one CUDA block per seed, threads parallelize the closest-point search)

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

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters (identical to original)
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

// ─── CUDA Kernels ───────────────────────────────────────────────────────────

// Compute the full NxN pairwise Euclidean distance matrix.
// dist[i*N + j] = distance(point_i, point_j)
__global__ void computeDistanceMatrixKernel(const double* __restrict__ px,
                                            const double* __restrict__ py,
                                            double* __restrict__ dist,
                                            int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        double dx = px[i] - px[j];
        double dy = py[i] - py[j];
        dist[i * N + j] = sqrt(dx * dx + dy * dy);
    }
}

// Generate candidate clusters for all seed points in parallel.
// Grid:  num_unclustered blocks (one per seed)
// Block: BLOCK_SIZE threads
//
// Each block independently grows a candidate cluster from its seed by
// iteratively finding the closest unclustered point that keeps the
// diameter below threshold.  Threads within a block cooperate to
// parallelise the closest-point search via a parallel reduction.
//
// Shared-memory layout (dynamic):
//   bool   in_cluster  [N]              – membership in this candidate cluster
//   int    members     [N]              – ordered list of members
//   int    num_members [1]              – current cluster size
//   double red_min     [BLOCK_SIZE]     – reduction buffer for min diameter
//   int    red_closest  [BLOCK_SIZE]    – reduction buffer for closest candidate
__global__ void generateCandidateClustersKernel(
        const double* __restrict__ dist_matrix,
        const char*   __restrict__ clustered,  // Changed from bool to char
        const int*    __restrict__ unclustered_indices,
        int num_unclustered,
        int N,
        double threshold,
        int* __restrict__ cluster_sizes,
        int* __restrict__ all_members) {

    const int seed_idx = blockIdx.x;
    if (seed_idx >= num_unclustered) return;

    const int tid = threadIdx.x;

    // ── dynamic shared memory ──
    extern __shared__ char smem[];

    // Compute aligned offsets (align to 8 bytes for double pointers)
    auto align8 = [](size_t x) { return (x + 7) & ~7; };
    
    size_t off_incluster = 0;
    size_t off_members   = align8(off_incluster + N * sizeof(char));
    size_t off_nummem    = align8(off_members + N * sizeof(int));
    size_t off_redmin    = align8(off_nummem + sizeof(int));
    size_t off_redclo    = align8(off_redmin + BLOCK_SIZE * sizeof(double));

    char*  s_in_cluster = (char* )(smem + off_incluster);
    int*   s_members    = (int* )(smem + off_members);
    int*   s_num_members= (int* )(smem + off_nummem);
    double* s_min_diam  = (double*)(smem + off_redmin);
    int*   s_closest    = (int* )(smem + off_redclo);

    const int seed_point = unclustered_indices[seed_idx];

    // Initialise in_cluster to false
    for (int i = tid; i < N; i += BLOCK_SIZE)
        s_in_cluster[i] = false;
    __syncthreads();

    // Seed the cluster
    if (tid == 0) {
        s_members[0]     = seed_point;
        s_in_cluster[seed_point] = true;
        s_num_members[0] = 1;
    }
    __syncthreads();

    // Iteratively grow the cluster
    while (true) {
        const int num_members = s_num_members[0];
        if (num_members >= N) break;

        // ── each thread scans a strided subset of candidates ──
        double local_min  = 1e300;
        int    local_best = -1;

        for (int cand = tid; cand < N; cand += BLOCK_SIZE) {
            if (clustered[cand] || s_in_cluster[cand]) continue;

            double max_dist = 0.0;
            for (int m = 0; m < num_members; m++) {
                // dist_matrix stored row-major: dist[member * N + candidate]
                // Threads with consecutive cand values access consecutive addresses
                // → coalesced reads
                double d = dist_matrix[s_members[m] * N + cand];
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold) {
                if (max_dist < local_min ||
                    (max_dist == local_min && cand < local_best)) {
                    local_min  = max_dist;
                    local_best = cand;
                }
            }
        }

        // ── parallel reduction across threads ──
        s_min_diam[tid] = local_min;
        s_closest[tid]  = local_best;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                bool replace = false;
                double other_min  = s_min_diam[tid + stride];
                int    other_cand = s_closest[tid + stride];
                if (other_min < s_min_diam[tid]) {
                    replace = true;
                } else if (other_min == s_min_diam[tid] &&
                           other_cand >= 0 &&
                           (s_closest[tid] < 0 || other_cand < s_closest[tid])) {
                    replace = true;
                }
                if (replace) {
                    s_min_diam[tid] = other_min;
                    s_closest[tid]  = other_cand;
                }
            }
            __syncthreads();
        }

        const int best = s_closest[0];
        if (best < 0) break;          // no more points can be added

        if (tid == 0) {
            s_members[num_members] = best;
            s_in_cluster[best]     = true;
            s_num_members[0]       = num_members + 1;
        }
        __syncthreads();
    }

    // ── write results to global memory ──
    const int final_size = s_num_members[0];
    int* out = all_members + (size_t)seed_idx * N;
    for (int i = tid; i < final_size; i += BLOCK_SIZE)
        out[i] = s_members[i];

    if (tid == 0)
        cluster_sizes[seed_idx] = final_size;
}

// ─── Host-side clustering orchestration ─────────────────────────────────────

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());
    if (N == 0) return {};

    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;

    // ── separate x/y arrays for coalesced GPU access ──
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) { px[i] = points[i].x; py[i] = points[i].y; }

    // ── allocate GPU memory ──
    double *d_px = nullptr, *d_py = nullptr, *d_dist = nullptr;
    char   *d_clustered = nullptr;  // Use char instead of bool for CUDA compatibility
    int    *d_unclustered = nullptr, *d_sizes = nullptr, *d_all_members = nullptr;

    CUDA_CHECK(cudaMalloc(&d_px,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered,    N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_unclustered,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes,        N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_members,  (size_t)N * N * sizeof(int)));

    // ── copy points and compute distance matrix on GPU ──
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    {
        dim3 blk(16, 16);
        dim3 grd((N + 15) / 16, (N + 15) / 16);
        computeDistanceMatrixKernel<<<grd, blk>>>(d_px, d_py, d_dist, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── compute shared-memory footprint with proper alignment ──
    // Align to 8 bytes for double pointers
    auto align8 = [](size_t x) { return (x + 7) & ~7; };
    
    size_t off_incluster = 0;
    size_t off_members   = align8(off_incluster + N * sizeof(char));
    size_t off_nummem    = align8(off_members + N * sizeof(int));
    size_t off_redmin    = align8(off_nummem + sizeof(int));
    size_t off_redclo    = align8(off_redmin + BLOCK_SIZE * sizeof(double));
    size_t smem_size     = off_redclo + BLOCK_SIZE * sizeof(int);

    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(generateCandidateClustersKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    // ── initialise unclustered list ──
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; i++) unclustered_indices[i] = i;

    std::vector<int> h_sizes(N);
    std::vector<int> h_members(N);
    std::vector<char> h_clustered(N);  // Use char instead of bool for CUDA compatibility

    // ── main clustering loop ──
    while (!unclustered_indices.empty()) {
        const int num_unclustered = static_cast<int>(unclustered_indices.size());

        // Pack clustered flags into a contiguous array for copy
        for (int i = 0; i < N; i++) h_clustered[i] = clustered[i] ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(),
                              N * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_unclustered, unclustered_indices.data(),
                              num_unclustered * sizeof(int), cudaMemcpyHostToDevice));

        // Launch: one block per seed
        generateCandidateClustersKernel<<<num_unclustered, BLOCK_SIZE, smem_size>>>(
            d_dist, d_clustered, d_unclustered, num_unclustered,
            N, threshold, d_sizes, d_all_members);
        CUDA_CHECK(cudaGetLastError());

        // Retrieve cluster sizes
        CUDA_CHECK(cudaMemcpy(h_sizes.data(), d_sizes,
                              num_unclustered * sizeof(int), cudaMemcpyDeviceToHost));

        // Find the seed that produced the largest candidate cluster
        int best_idx  = -1;
        int best_size = 0;
        for (int i = 0; i < num_unclustered; i++) {
            if (h_sizes[i] > best_size) {
                best_size = h_sizes[i];
                best_idx  = i;
            }
        }

        if (best_idx < 0 || best_size <= 0) break;

        // Retrieve only the winning cluster's members
        CUDA_CHECK(cudaMemcpy(h_members.data(),
                              d_all_members + (size_t)best_idx * N,
                              best_size * sizeof(int), cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = unclustered_indices[best_idx];
        cluster.members.resize(best_size);
        for (int i = 0; i < best_size; i++)
            cluster.members[i] = h_members[i];
        clusters.push_back(std::move(cluster));

        // Mark members as clustered
        for (int i = 0; i < best_size; i++)
            clustered[h_members[i]] = true;

        // Rebuild unclustered list
        std::vector<int> next;
        next.reserve(num_unclustered - best_size);
        for (int idx : unclustered_indices)
            if (!clustered[idx]) next.push_back(idx);
        unclustered_indices = std::move(next);
    }

    // ── free GPU memory ──
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_dist);
    cudaFree(d_clustered);
    cudaFree(d_unclustered);
    cudaFree(d_sizes);
    cudaFree(d_all_members);

    return clusters;
}

// ─── Validation (identical to original) ─────────────────────────────────────

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
                const double dx = points[cluster.members[i]].x - points[cluster.members[j]].x;
                const double dy = points[cluster.members[i]].y - points[cluster.members[j]].y;
                const double dist = std::sqrt(dx * dx + dy * dy);
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

    printf("QT Clustering Benchmark (CUDA)\n");
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

    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n",
           clusters_per_sec, points_per_sec);

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
