// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering builds clusters by starting with a seed
// point and iteratively adding the closest point that maintains the cluster's
// diameter below a threshold. This version uses CUDA for GPU parallelism:
// - Distance matrix precomputed on GPU
// - All seeds evaluated in parallel (one CUDA block per seed)
// - Threads within each block parallelize candidate search via block reduction
// - Incremental max_dist tracking for O(N) per iteration

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static const int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ─── CUDA Kernels ────────────────────────────────────────────────────────────

// Compute full NxN distance matrix
__global__ void computeDistanceMatrixKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    double* __restrict__ dist,
    const int N)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        double dx = px[i] - px[j];
        double dy = py[i] - py[j];
        dist[i * N + j] = sqrt(dx * dx + dy * dy);
    }
}

// Evaluate all seed points in parallel.
// Each CUDA block handles one seed; threads cooperate to find closest candidates.
// Uses incremental max_dist: when a new member is added, only distances to that
// member need to be considered, making each iteration O(N/blockDim.x).
__global__ __launch_bounds__(BLOCK_SIZE)
void evaluateAllSeedsKernel(
    const double* __restrict__ d_dist,
    const int* __restrict__ d_clustered,
    const int* __restrict__ d_unclustered,
    const double threshold,
    const int N,
    const int num_seeds,
    int* __restrict__ d_cardinalities,
    double* __restrict__ d_max_dist,    // workspace: num_seeds * N
    char* __restrict__ d_in_cluster)    // workspace: num_seeds * N
{
    const int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    const int seed = d_unclustered[seed_idx];
    const int tid = threadIdx.x;

    // This block's portion of the workspace
    double* my_max_dist = d_max_dist + (size_t)seed_idx * N;
    char* my_in_cluster = d_in_cluster + (size_t)seed_idx * N;

    // Shared memory for block-wide reduction
    __shared__ double s_min_diam[BLOCK_SIZE];
    __shared__ int s_best_idx[BLOCK_SIZE];

    // Initialize workspace
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        my_max_dist[i] = 0.0;
        my_in_cluster[i] = 0;
    }
    __syncthreads();

    // Add seed point
    if (tid == 0) my_in_cluster[seed] = 1;
    __syncthreads();

    int cardinality = 1;
    int last_added = seed;

    while (cardinality < N) {
        // Merged pass: update max_dist with distances to last_added AND
        // find the best candidate (minimum max_dist < threshold) in one loop.
        // Each thread handles its own strided subset of candidates — no
        // cross-thread data dependency within the pass.
        double local_min = threshold;
        int local_best = -1;

        for (int i = tid; i < N; i += BLOCK_SIZE) {
            if (!my_in_cluster[i] && !d_clustered[i]) {
                // Coalesced access: consecutive threads read consecutive elements
                double d = d_dist[last_added * N + i];
                if (d > my_max_dist[i]) my_max_dist[i] = d;
                double md = my_max_dist[i];
                if (md < local_min) {
                    local_min = md;
                    local_best = i;
                }
            }
        }

        // Block-wide reduction to find global best candidate
        s_min_diam[tid] = local_min;
        s_best_idx[tid] = local_best;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                double rd = s_min_diam[tid + stride];
                int ri = s_best_idx[tid + stride];
                // Tie-break: prefer lower index (matches sequential iteration order)
                if (rd < s_min_diam[tid] ||
                    (rd == s_min_diam[tid] && ri < s_best_idx[tid])) {
                    s_min_diam[tid] = rd;
                    s_best_idx[tid] = ri;
                }
            }
            __syncthreads();
        }

        const int next = s_best_idx[0];
        if (next < 0) break;

        if (tid == 0) my_in_cluster[next] = 1;
        __syncthreads();

        last_added = next;
        cardinality++;
    }

    if (tid == 0) {
        d_cardinalities[seed_idx] = cardinality;
    }
}

// Regenerate the cluster for a single seed on GPU.
// Same logic as evaluateAllSeedsKernel but also outputs member indices.
__global__ __launch_bounds__(BLOCK_SIZE)
void regenerateClusterKernel(
    const double* __restrict__ d_dist,
    const int* __restrict__ d_clustered,
    const int seed,
    const double threshold,
    const int N,
    int* __restrict__ d_members,
    int* __restrict__ d_member_count,
    double* __restrict__ d_max_dist,    // workspace: N doubles
    char* __restrict__ d_in_cluster)    // workspace: N chars
{
    const int tid = threadIdx.x;

    __shared__ double s_min_diam[BLOCK_SIZE];
    __shared__ int s_best_idx[BLOCK_SIZE];

    // Initialize workspace
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        d_max_dist[i] = 0.0;
        d_in_cluster[i] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        d_in_cluster[seed] = 1;
        d_members[0] = seed;
    }
    __syncthreads();

    int cardinality = 1;
    int last_added = seed;

    while (cardinality < N) {
        double local_min = threshold;
        int local_best = -1;

        for (int i = tid; i < N; i += BLOCK_SIZE) {
            if (!d_in_cluster[i] && !d_clustered[i]) {
                double d = d_dist[last_added * N + i];
                if (d > d_max_dist[i]) d_max_dist[i] = d;
                double md = d_max_dist[i];
                if (md < local_min) {
                    local_min = md;
                    local_best = i;
                }
            }
        }

        s_min_diam[tid] = local_min;
        s_best_idx[tid] = local_best;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                double rd = s_min_diam[tid + stride];
                int ri = s_best_idx[tid + stride];
                if (rd < s_min_diam[tid] ||
                    (rd == s_min_diam[tid] && ri < s_best_idx[tid])) {
                    s_min_diam[tid] = rd;
                    s_best_idx[tid] = ri;
                }
            }
            __syncthreads();
        }

        const int next = s_best_idx[0];
        if (next < 0) break;

        if (tid == 0) {
            d_in_cluster[next] = 1;
            d_members[cardinality] = next;
        }
        __syncthreads();

        last_added = next;
        cardinality++;
    }

    if (tid == 0) d_member_count[0] = cardinality;
}

// ─── Host functions ──────────────────────────────────────────────────────────

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

// Euclidean distance (for validation on CPU)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Main QT clustering algorithm — GPU-accelerated
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    if (N == 0) return {};

    // Separate coordinate arrays for GPU
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    // ── Allocate GPU memory ──
    double *d_px, *d_py, *d_dist;
    int *d_clustered, *d_unclustered, *d_cardinalities;
    double *d_max_dist;
    char *d_in_cluster;
    int *d_members, *d_member_count;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unclustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_member_count, sizeof(int)));

    // Workspace sized for batch processing to limit memory usage
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t dist_bytes = (size_t)N * N * sizeof(double);
    size_t other_bytes = N * sizeof(int) * 4 + N * sizeof(double) * 2 + sizeof(int);
    size_t avail = free_mem - dist_bytes - other_bytes - (100ULL << 20); // 100 MB margin
    size_t per_seed_ws = (size_t)N * (sizeof(double) + sizeof(char));
    int batch_size = (per_seed_ws > 0) ? std::min(N, (int)(avail / per_seed_ws)) : N;
    batch_size = std::max(1, batch_size);

    CUDA_CHECK(cudaMalloc(&d_max_dist, (size_t)batch_size * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, (size_t)batch_size * N * sizeof(char)));

    // ── Copy points and compute distance matrix ──
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        computeDistanceMatrixKernel<<<grid, block>>>(d_px, d_py, d_dist, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Host state ──
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered(N);
    for (int i = 0; i < N; i++) unclustered[i] = i;

    std::vector<int> h_cardinalities(batch_size);
    std::vector<int> h_clustered(N, 0);
    std::vector<int> h_members(N);

    std::vector<Cluster> clusters;

    // ── Main clustering loop ──
    while (!unclustered.empty()) {
        const int num_seeds = static_cast<int>(unclustered.size());

        // Upload current clustered flags and unclustered indices
        for (int i = 0; i < N; i++) h_clustered[i] = clustered[i] ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_unclustered, unclustered.data(),
                              num_seeds * sizeof(int), cudaMemcpyHostToDevice));

        // Evaluate seeds in batches
        int best_card = 0;
        int best_seed_global_idx = -1;

        for (int bstart = 0; bstart < num_seeds; bstart += batch_size) {
            const int bsize = std::min(batch_size, num_seeds - bstart);

            evaluateAllSeedsKernel<<<bsize, BLOCK_SIZE>>>(
                d_dist,
                d_clustered,
                d_unclustered + bstart,
                threshold, N, bsize,
                d_cardinalities,
                d_max_dist,
                d_in_cluster);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities,
                                  bsize * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i = 0; i < bsize; i++) {
                if (h_cardinalities[i] > best_card) {
                    best_card = h_cardinalities[i];
                    best_seed_global_idx = bstart + i;
                }
            }
        }

        if (best_seed_global_idx < 0 || best_card <= 0) break;

        const int best_seed = unclustered[best_seed_global_idx];

        // Regenerate the best cluster on GPU (ensures same distances used)
        regenerateClusterKernel<<<1, BLOCK_SIZE>>>(
            d_dist, d_clustered, best_seed, threshold, N,
            d_members, d_member_count, d_max_dist, d_in_cluster);
        CUDA_CHECK(cudaGetLastError());

        int mc = 0;
        CUDA_CHECK(cudaMemcpy(&mc, d_member_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_members.data(), d_members, mc * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Build cluster
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(h_members.begin(), h_members.begin() + mc);
        clusters.push_back(std::move(cluster));

        // Mark members as clustered
        for (int i = 0; i < mc; i++) {
            clustered[h_members[i]] = true;
        }

        // Rebuild unclustered list
        std::vector<int> new_unclustered;
        new_unclustered.reserve(num_seeds - mc);
        for (int u : unclustered) {
            if (!clustered[u]) new_unclustered.push_back(u);
        }
        unclustered = std::move(new_unclustered);
    }

    // ── Free GPU memory ──
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_dist);
    cudaFree(d_clustered);
    cudaFree(d_unclustered);
    cudaFree(d_cardinalities);
    cudaFree(d_max_dist);
    cudaFree(d_in_cluster);
    cudaFree(d_members);
    cudaFree(d_member_count);

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

    // Perform QT clustering (GPU-accelerated)
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
