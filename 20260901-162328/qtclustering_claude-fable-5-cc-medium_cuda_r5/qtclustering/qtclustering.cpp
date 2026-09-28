// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: in each round of the outer loop, a candidate cluster is
// grown from every unclustered seed point in parallel. A pool of persistent
// worker blocks strides over the seeds; each block grows one candidate
// cluster at a time, with its threads cooperatively scanning candidates.
// Each worker maintains an incremental per-candidate array
//   maxdist[i] = max distance from point i to the current cluster members,
// so adding a member costs O(N) instead of O(|cluster| * N). Since max is
// order-independent this is numerically identical to recomputing from
// scratch, and argmin ties are broken by the lowest point index, exactly
// matching the sequential algorithm.

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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                    __LINE__, cudaGetErrorString(err__));                     \
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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

#define BLOCK_SIZE 256

// Block-wide argmin over (value, index) pairs; ties resolved to the lowest
// index, matching the sequential scan order. Result left in s_val[0]/s_idx[0].
__device__ inline void blockArgmin(double* s_val, int* s_idx) {
    const int tid = threadIdx.x;
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double ov = s_val[tid + stride];
            const int oi = s_idx[tid + stride];
            if (ov < s_val[tid] || (ov == s_val[tid] && oi >= 0 && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }
}

// Grow a candidate cluster from seed `seed` inside a single block.
//
// The worker keeps a compacted list of still-eligible candidates in
// ping-pong scratch buffers (cand: 2N ints, mdist: 2N doubles, cpos:
// 2N double2), so each growth step only touches candidates that can still
// join. A candidate whose running max distance to the members reaches the
// threshold is dropped permanently (the max only grows). The selection is
// order-independent: ties on distance are broken by the lowest point index,
// exactly like the sequential scan, so the atomics-based compaction order
// does not affect the result.
//
// If record != nullptr, chosen members are flagged in record[] and marked
// in clustered[]. Returns the cardinality of the grown cluster.
__device__ int growCluster(const double2* __restrict__ points,
                           const unsigned char* __restrict__ clustered,
                           int* __restrict__ cand, double* __restrict__ mdist,
                           double2* __restrict__ cpos,
                           const int seed, const int N, const double threshold,
                           unsigned char* record) {
    __shared__ double s_val[BLOCK_SIZE];
    __shared__ int s_idx[BLOCK_SIZE];
    __shared__ int s_next;
    __shared__ int s_active;
    __shared__ int s_count;

    const int tid = threadIdx.x;
    const double INF = HUGE_VAL;

    // Build the initial candidate list: every point that is not globally
    // clustered and not the seed. List order is arbitrary.
    if (tid == 0) {
        s_count = 0;
        if (record) const_cast<unsigned char*>(clustered)[seed] = 1;
    }
    if (record) {
        for (int i = tid; i < N; i += BLOCK_SIZE) {
            record[i] = (i == seed) ? 1 : 0;
        }
    }
    __syncthreads();
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        if (i != seed && !clustered[i]) {
            const int k = atomicAdd(&s_count, 1);
            cand[k] = i;
            mdist[k] = 0.0;
            cpos[k] = points[i];
        }
    }
    __syncthreads();

    int active = s_count;
    __syncthreads(); // all reads done before s_count is reset in the loop
    int cur = 0;
    int members = 1;
    int newpt = seed;
    const double2 seed_pos = points[seed];
    double2 np = seed_pos;

    while (members < N && active > 0) {
        if (tid == 0) s_count = 0;
        double best_val = INF;
        int best_idx = -1;
        __syncthreads();

        const int* ci = cand + cur * N;
        const double* mi = mdist + cur * N;
        const double2* pi = cpos + cur * N;
        int* co = cand + (cur ^ 1) * N;
        double* mo = mdist + (cur ^ 1) * N;
        double2* po = cpos + (cur ^ 1) * N;

        // Fused pass: fold the newly added member into each candidate's
        // running max distance, drop ineligible candidates, compact the
        // survivors, and track the best (smallest diameter) candidate.
        for (int j = tid; j < active; j += BLOCK_SIZE) {
            const int i = ci[j];
            if (i == newpt) continue; // just became a member: drop it
            const double2 p = pi[j];
            const double dx = p.x - np.x;
            const double dy = p.y - np.y;
            const double d = sqrt(dx * dx + dy * dy);
            double m = mi[j];
            if (d > m) m = d;
            if (m < threshold) {
                const int k = atomicAdd(&s_count, 1);
                co[k] = i;
                mo[k] = m;
                po[k] = p;
                if (m < best_val || (m == best_val && i < best_idx)) {
                    best_val = m;
                    best_idx = i;
                }
            }
        }

        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();
        blockArgmin(s_val, s_idx);

        if (tid == 0) {
            s_next = s_idx[0];
            s_active = s_count;
            if (s_next >= 0 && record) {
                record[s_next] = 1;
                const_cast<unsigned char*>(clustered)[s_next] = 1;
            }
        }
        __syncthreads();
        const int next = s_next;
        active = s_active;
        __syncthreads(); // all reads done before s_count is reset next round

        if (next < 0) break; // no more points can be added
        newpt = next;
        np = points[next];
        members++;
        cur ^= 1;
    }

    return members;
}

// Phase 1: every worker block grows candidate clusters for a strided subset
// of the unclustered seeds and records each cluster's cardinality.
__global__ void candidateKernel(const double2* __restrict__ points,
                                const unsigned char* __restrict__ clustered,
                                const int* __restrict__ seeds,
                                const int num_seeds, const int N,
                                const double threshold,
                                int* __restrict__ cand_pool,
                                double* __restrict__ mdist_pool,
                                double2* __restrict__ cpos_pool,
                                int* __restrict__ cardinality) {
    const size_t off = static_cast<size_t>(blockIdx.x) * 2 * N;
    int* cand = cand_pool + off;
    double* mdist = mdist_pool + off;
    double2* cpos = cpos_pool + off;
    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        const int card = growCluster(points, clustered, cand, mdist, cpos,
                                     seeds[s], N, threshold, nullptr);
        if (threadIdx.x == 0) cardinality[s] = card;
        __syncthreads();
    }
}

// Phase 2: re-grow the winning cluster with a single block, flag its members
// in `record` and mark them globally clustered.
__global__ void growBestKernel(const double2* __restrict__ points,
                               unsigned char* clustered,
                               int* __restrict__ cand_pool,
                               double* __restrict__ mdist_pool,
                               double2* __restrict__ cpos_pool,
                               const int seed, const int N,
                               const double threshold,
                               unsigned char* record) {
    growCluster(points, clustered, cand_pool, mdist_pool, cpos_pool, seed, N,
                threshold, record);
}

// Per-GPU state for the candidate-evaluation phase
struct GpuContext {
    int device;
    int workers;
    double2* d_points;
    unsigned char* d_clustered;
    unsigned char* d_record; // GPU 0 only
    int* d_seeds;
    int* d_card;
    int* d_cand_pool;
    double* d_mdist_pool;
    double2* d_cpos_pool;
};

// Main QT clustering algorithm (GPU). The candidate-evaluation phase (grow a
// cluster from every unclustered seed) is split across all available GPUs;
// the winning cluster is re-grown on GPU 0.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus < 1) {
        fprintf(stderr, "No CUDA devices available\n");
        exit(1);
    }

    std::vector<GpuContext> gpus(num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        GpuContext& ctx = gpus[g];
        ctx.device = g;
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&ctx.d_points, sizeof(double2) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_clustered, N));
        CUDA_CHECK(cudaMalloc(&ctx.d_seeds, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_card, sizeof(int) * N));
        ctx.d_record = nullptr;
        if (g == 0) CUDA_CHECK(cudaMalloc(&ctx.d_record, N));
        CUDA_CHECK(cudaMemcpy(ctx.d_points, points.data(),
                              sizeof(double2) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(ctx.d_clustered, 0, N));

        // Size the persistent worker pool: enough blocks to saturate the
        // GPU, capped by available memory for the per-worker scratch
        // (double-buffered candidate index, running max distance and
        // coordinates: 2N ints + 2N doubles + 2N double2).
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, g));
        int workers = prop.multiProcessorCount * 16;
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const size_t per_worker = 2 * static_cast<size_t>(N) *
            (sizeof(int) + sizeof(double) + sizeof(double2));
        const size_t mem_cap = static_cast<size_t>(free_mem * 0.9) / per_worker;
        if (mem_cap < static_cast<size_t>(workers))
            workers = std::max(1, static_cast<int>(mem_cap));
        workers = std::min(workers, N);
        ctx.workers = workers;
        const size_t pool_elems = 2 * static_cast<size_t>(N) * workers;
        CUDA_CHECK(cudaMalloc(&ctx.d_cand_pool, sizeof(int) * pool_elems));
        CUDA_CHECK(cudaMalloc(&ctx.d_mdist_pool, sizeof(double) * pool_elems));
        CUDA_CHECK(cudaMalloc(&ctx.d_cpos_pool, sizeof(double2) * pool_elems));
    }

    std::vector<bool> clustered(N, false);
    std::vector<unsigned char> h_clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<int> cardinality(N);
    std::vector<unsigned char> record(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // Split the seed range across GPUs; small batches are not worth the
        // extra transfer latency, keep them on GPU 0.
        const int use_gpus = (num_seeds >= 512) ? num_gpus : 1;
        int begin = 0;
        std::vector<int> seed_begin(use_gpus), seed_count(use_gpus);
        for (int g = 0; g < use_gpus; ++g) {
            const int cnt = static_cast<int>(
                (static_cast<long long>(num_seeds) * (g + 1)) / use_gpus) - begin;
            seed_begin[g] = begin;
            seed_count[g] = cnt;
            begin += cnt;
        }

        // Launch candidate evaluation on all GPUs concurrently.
        for (int g = 0; g < use_gpus; ++g) {
            if (seed_count[g] == 0) continue;
            GpuContext& ctx = gpus[g];
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaMemcpyAsync(ctx.d_seeds,
                                       unclustered_indices.data() + seed_begin[g],
                                       sizeof(int) * seed_count[g],
                                       cudaMemcpyHostToDevice));
            const int grid = std::min(ctx.workers, seed_count[g]);
            candidateKernel<<<grid, BLOCK_SIZE>>>(
                ctx.d_points, ctx.d_clustered, ctx.d_seeds, seed_count[g], N,
                threshold, ctx.d_cand_pool, ctx.d_mdist_pool, ctx.d_cpos_pool,
                ctx.d_card);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int g = 0; g < use_gpus; ++g) {
            if (seed_count[g] == 0) continue;
            GpuContext& ctx = gpus[g];
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaMemcpy(cardinality.data() + seed_begin[g],
                                  ctx.d_card, sizeof(int) * seed_count[g],
                                  cudaMemcpyDeviceToHost));
        }

        // Pick the seed with the largest candidate cluster; the first (i.e.
        // lowest-index) seed wins ties, as in the sequential scan.
        int max_cardinality = -1;
        int best_seed = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (cardinality[i] > max_cardinality) {
                max_cardinality = cardinality[i];
                best_seed = unclustered_indices[i];
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            // Re-grow the winning cluster on GPU 0, recording members and
            // marking them clustered on the device.
            GpuContext& ctx0 = gpus[0];
            CUDA_CHECK(cudaSetDevice(ctx0.device));
            growBestKernel<<<1, BLOCK_SIZE>>>(
                ctx0.d_points, ctx0.d_clustered, ctx0.d_cand_pool,
                ctx0.d_mdist_pool, ctx0.d_cpos_pool, best_seed, N, threshold,
                ctx0.d_record);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(record.data(), ctx0.d_record, N,
                                  cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = best_seed;
            for (int i = 0; i < N; ++i) {
                if (record[i]) {
                    cluster.members.push_back(i);
                    clustered[i] = true;
                    h_clustered[i] = 1;
                }
            }
            clusters.push_back(std::move(cluster));

            // Propagate the updated clustered flags to the other GPUs.
            for (int g = 1; g < num_gpus; ++g) {
                CUDA_CHECK(cudaSetDevice(gpus[g].device));
                CUDA_CHECK(cudaMemcpy(gpus[g].d_clustered, h_clustered.data(),
                                      N, cudaMemcpyHostToDevice));
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

    for (int g = 0; g < num_gpus; ++g) {
        GpuContext& ctx = gpus[g];
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaFree(ctx.d_cpos_pool));
        CUDA_CHECK(cudaFree(ctx.d_mdist_pool));
        CUDA_CHECK(cudaFree(ctx.d_cand_pool));
        CUDA_CHECK(cudaFree(ctx.d_card));
        CUDA_CHECK(cudaFree(ctx.d_seeds));
        if (ctx.d_record) CUDA_CHECK(cudaFree(ctx.d_record));
        CUDA_CHECK(cudaFree(ctx.d_clustered));
        CUDA_CHECK(cudaFree(ctx.d_points));
    }
    CUDA_CHECK(cudaSetDevice(0));

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

    // Initialize the CUDA contexts outside the timed region
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    for (int g = num_gpus - 1; g >= 0; --g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(nullptr));
    }

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
