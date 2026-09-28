// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization scheme:
//  - MPI: the candidate seeds of each outer iteration are distributed
//    cyclically across ranks; the globally best candidate cluster is
//    selected with an Allreduce on an encoded (cardinality, seed) key.
//  - CUDA: each rank grows all of its candidate clusters on the GPU with
//    one thread block per seed. Per-candidate maximum distances to the
//    growing cluster are maintained incrementally (O(N) per added member
//    instead of O(N * |members|)), which is exactly equivalent since
//    max() over the same set of distances is order independent.
//  - OpenMP: threads drive multiple GPUs per rank concurrently, and
//    parallelize the host-side re-growth of the winning cluster as well
//    as the statistics/serialization loops.
//
// Floating point contraction is disabled on both host (-ffp-contract=off)
// and device (-fmad=false) so CPU and GPU distance computations are
// bitwise identical, preserving the sequential algorithm's semantics.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA candidate-cluster kernel
// ---------------------------------------------------------------------------
// One thread block grows one candidate cluster per seed (blocks stride over
// the seed list). Every future member must lie within `threshold` of the
// seed, so the block first compacts those neighbors into a candidate list
// and then only ever touches that list. maxd[i] holds the maximum distance
// from candidate cand[i] to all current cluster members; DBL_MAX marks
// candidates that became unavailable (added to the cluster, or exceeded the
// threshold — maxd only grows, so such a point can never be added later).
// Each step adds the eligible candidate with the smallest maxd (ties broken
// by the lowest point index), matching the sequential algorithm exactly.

static const int KERNEL_BLOCK = 256;

__global__ void candidateClusterKernel(const Point* __restrict__ points,
                                       const unsigned char* __restrict__ clustered,
                                       const int* __restrict__ seeds,
                                       const int num_seeds,
                                       const int N,
                                       const double threshold,
                                       double* __restrict__ maxd_pool,
                                       int* __restrict__ cand_pool,
                                       int* __restrict__ cards) {
    __shared__ double s_dist[KERNEL_BLOCK];
    __shared__ int s_idx[KERNEL_BLOCK];
    __shared__ int s_best;
    __shared__ int s_count;

    double* maxd = maxd_pool + static_cast<size_t>(blockIdx.x) * N;
    int* cand = cand_pool + static_cast<size_t>(blockIdx.x) * N;

    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        const int seed = seeds[s];
        const Point sp = points[seed];

        // Compact the seed's threshold neighborhood into the candidate list
        // (list order is arbitrary; selection ties break on the point index)
        if (threadIdx.x == 0) s_count = 0;
        __syncthreads();
        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            if (!clustered[c] && c != seed) {
                const double d = distance(points[c], sp);
                if (d < threshold) {
                    const int slot = atomicAdd(&s_count, 1);
                    cand[slot] = c;
                    maxd[slot] = d;
                }
            }
        }
        __syncthreads();
        const int L = s_count;

        int count = 1;
        while (count < N) {
            // Per-thread argmin over eligible candidates
            double best_d = DBL_MAX;
            int best_i = -1;
            for (int i = threadIdx.x; i < L; i += blockDim.x) {
                const double d = maxd[i];
                if (d < threshold &&
                    (d < best_d || (d == best_d && cand[i] < best_i))) {
                    best_d = d;
                    best_i = cand[i];
                }
            }
            s_dist[threadIdx.x] = best_d;
            s_idx[threadIdx.x] = best_i;
            __syncthreads();

            // Block-level (distance, index) lexicographic min reduction
            for (int off = blockDim.x / 2; off > 0; off >>= 1) {
                if (threadIdx.x < off) {
                    const double od = s_dist[threadIdx.x + off];
                    const int oi = s_idx[threadIdx.x + off];
                    if (oi >= 0 &&
                        (s_idx[threadIdx.x] < 0 || od < s_dist[threadIdx.x] ||
                         (od == s_dist[threadIdx.x] && oi < s_idx[threadIdx.x]))) {
                        s_dist[threadIdx.x] = od;
                        s_idx[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) s_best = s_idx[0];
            __syncthreads();

            const int best = s_best;
            if (best < 0) break;  // No more points can be added
            count++;

            // Incrementally fold the new member into every candidate's maxd
            const Point bp = points[best];
            for (int i = threadIdx.x; i < L; i += blockDim.x) {
                const double cur = maxd[i];
                if (cur < DBL_MAX) {
                    if (cand[i] == best) {
                        maxd[i] = DBL_MAX;
                    } else {
                        const double d = distance(points[cand[i]], bp);
                        if (d > cur) maxd[i] = (d < threshold) ? d : DBL_MAX;
                    }
                }
            }
            __syncthreads();
        }

        cards[s] = count;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Per-GPU device context
// ---------------------------------------------------------------------------
struct GpuCtx {
    int dev = -1;
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    double* d_pool = nullptr;
    int* d_cand_pool = nullptr;
    int pool_blocks = 0;
};

static void initGpuCtx(GpuCtx& g, const int dev, const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    g.dev = dev;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMalloc(&g.d_points, static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMemcpy(g.d_points, points.data(),
                          static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g.d_clustered, static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&g.d_seeds, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_cards, static_cast<size_t>(N) * sizeof(int)));

    // Size the scratch pools (one row of N doubles + N ints per resident
    // block), capped at half of the currently free device memory.
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t row = static_cast<size_t>(N) * (sizeof(double) + sizeof(int));
    size_t cap = (free_mem / 2) / row;
    if (cap < 1) cap = 1;
    g.pool_blocks = static_cast<int>(std::min<size_t>(1024, cap));
    CUDA_CHECK(cudaMalloc(&g.d_pool, static_cast<size_t>(g.pool_blocks) *
                                         static_cast<size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_cand_pool, static_cast<size_t>(g.pool_blocks) *
                                              static_cast<size_t>(N) * sizeof(int)));
}

static void freeGpuCtx(GpuCtx& g) {
    if (g.dev < 0) return;
    cudaSetDevice(g.dev);
    cudaFree(g.d_points);
    cudaFree(g.d_clustered);
    cudaFree(g.d_seeds);
    cudaFree(g.d_cards);
    cudaFree(g.d_pool);
    cudaFree(g.d_cand_pool);
    g.dev = -1;
}

// Encode (cardinality, seed) so that MPI_MAX picks the highest cardinality,
// with ties broken by the lowest seed index (matching the sequential scan
// order over the ascending unclustered list).
static inline long long encodeBest(const int card, const int seed) {
    if (card < 0) return -1;
    return (static_cast<long long>(card) << 31) |
           static_cast<long long>(INT32_MAX - seed);
}

static inline int decodeSeed(const long long v) {
    return INT32_MAX - static_cast<int>(v & 0x7FFFFFFFLL);
}

// Pick a host thread count proportional to the loop size; spawning the full
// thread pool (e.g. 256 logical CPUs) for small arrays costs far more in
// fork/join overhead than the loop itself.
static inline int hostThreads(const int work) {
    int t = work / 2048;
    if (t < 1) t = 1;
    const int m = omp_get_max_threads();
    return t > m ? m : t;
}

// ---------------------------------------------------------------------------
// Host-side (OpenMP) re-growth of the winning candidate cluster.
// Uses the same incremental max-distance scheme as the kernel and therefore
// produces the identical member list, in the identical insertion order.
// ---------------------------------------------------------------------------
static void growClusterHost(const int seed,
                            const std::vector<unsigned char>& clustered,
                            const std::vector<Point>& points,
                            const double threshold,
                            std::vector<int>& members,
                            std::vector<double>& maxd) {
    const int N = static_cast<int>(points.size());
    members.clear();
    members.push_back(seed);
    const int nt = hostThreads(N);

    const Point sp = points[seed];
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (int c = 0; c < N; ++c) {
        double d = DBL_MAX;
        if (!clustered[c] && c != seed) {
            d = distance(points[c], sp);
            if (d >= threshold) d = DBL_MAX;
        }
        maxd[c] = d;
    }

    while (static_cast<int>(members.size()) < N) {
        double best_d = DBL_MAX;
        int best_i = -1;
#pragma omp parallel num_threads(nt) if (nt > 1)
        {
            double bd = DBL_MAX;
            int bi = -1;
#pragma omp for schedule(static) nowait
            for (int c = 0; c < N; ++c) {
                const double d = maxd[c];
                if (d < threshold && d < bd) {
                    bd = d;
                    bi = c;
                }
            }
#pragma omp critical
            {
                if (bi >= 0 && (best_i < 0 || bd < best_d ||
                                (bd == best_d && bi < best_i))) {
                    best_d = bd;
                    best_i = bi;
                }
            }
        }
        if (best_i < 0) break;
        members.push_back(best_i);

        const Point bp = points[best_i];
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
        for (int c = 0; c < N; ++c) {
            const double cur = maxd[c];
            if (c == best_i) {
                maxd[c] = DBL_MAX;
            } else if (cur < DBL_MAX) {
                const double d = distance(points[c], bp);
                if (d > cur) maxd[c] = (d < threshold) ? d : DBL_MAX;
            }
        }
    }
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  std::vector<GpuCtx>& ctxs,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<Cluster> clusters;

    const int G = static_cast<int>(ctxs.size());
    std::vector<int> my_seeds;
    my_seeds.reserve((N + nprocs - 1) / nprocs);
    std::vector<int> cards(N);
    std::vector<double> maxd_host(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Cyclic distribution of the remaining seeds across ranks
        my_seeds.clear();
        for (size_t i = rank; i < unclustered_indices.size();
             i += static_cast<size_t>(nprocs)) {
            my_seeds.push_back(unclustered_indices[i]);
        }
        const int M = static_cast<int>(my_seeds.size());

        // Grow every candidate cluster on this rank's GPUs; one OpenMP
        // thread drives each device on a contiguous slice of the seeds.
        long long local_best = -1;
        if (M > 0) {
#pragma omp parallel num_threads(G) reduction(max : local_best)
            {
                const int t = omp_get_thread_num();
                const int lo = static_cast<int>(
                    static_cast<long long>(M) * t / G);
                const int hi = static_cast<int>(
                    static_cast<long long>(M) * (t + 1) / G);
                const int cnt = hi - lo;
                if (cnt > 0) {
                    GpuCtx& g = ctxs[t];
                    CUDA_CHECK(cudaSetDevice(g.dev));
                    CUDA_CHECK(cudaMemcpy(g.d_clustered, clustered.data(),
                                          static_cast<size_t>(N),
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(g.d_seeds, my_seeds.data() + lo,
                                          static_cast<size_t>(cnt) * sizeof(int),
                                          cudaMemcpyHostToDevice));
                    const int blocks = std::min(g.pool_blocks, cnt);
                    candidateClusterKernel<<<blocks, KERNEL_BLOCK>>>(
                        g.d_points, g.d_clustered, g.d_seeds, cnt, N,
                        threshold, g.d_pool, g.d_cand_pool, g.d_cards);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpy(cards.data() + lo, g.d_cards,
                                          static_cast<size_t>(cnt) * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                    for (int i = lo; i < hi; ++i) {
                        const long long e = encodeBest(cards[i], my_seeds[i]);
                        if (e > local_best) local_best = e;
                    }
                }
            }
        }

        // Global selection: highest cardinality, ties to the lowest seed
        long long global_best = -1;
        MPI_Allreduce(&local_best, &global_best, 1, MPI_LONG_LONG, MPI_MAX,
                      MPI_COMM_WORLD);
        if (global_best < 0) break;  // No more clusters can be formed

        // Every rank re-grows the winning cluster deterministically on the
        // host, so no member broadcast is needed.
        Cluster cluster;
        cluster.seed_point = decodeSeed(global_best);
        growClusterHost(cluster.seed_point, clustered, points, threshold,
                        cluster.members, maxd_host);

        // Mark all members as clustered
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            clustered[cluster.members[i]] = 1;
        }
        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

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
        const int msize = static_cast<int>(cluster.members.size());
        const int nt = hostThreads(msize * msize / 2);
#pragma omp parallel for schedule(dynamic) reduction(max : max_diameter) \
    num_threads(nt) if (nt > 1)
        for (int i = 0; i < msize; ++i) {
            for (int j = i + 1; j < msize; ++j) {
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
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Assign GPUs: ranks on the same node partition the visible devices.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev <= 0) {
        fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dev_start, dev_count;
    if (local_size >= ndev) {
        dev_start = local_rank % ndev;
        dev_count = 1;
    } else {
        const int per = ndev / local_size;
        const int extra = ndev % local_size;
        dev_start = local_rank * per + std::min(local_rank, extra);
        dev_count = per + (local_rank < extra ? 1 : 0);
    }

    std::vector<GpuCtx> ctxs(dev_count);
    for (int i = 0; i < dev_count; ++i) {
        initGpuCtx(ctxs[i], dev_start + i, points);
    }

    if (rank == 0) {
        printf("MPI ranks: %d, GPUs on rank 0: %d, OpenMP threads: %d\n",
               nprocs, dev_count, omp_get_max_threads());
    }

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, ctxs, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    for (int i = 0; i < dev_count; ++i) {
        freeGpuCtx(ctxs[i]);
    }

    int exit_code = 0;
    if (rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
