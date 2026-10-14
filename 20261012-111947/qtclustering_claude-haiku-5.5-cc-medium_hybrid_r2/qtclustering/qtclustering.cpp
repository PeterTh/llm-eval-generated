// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallel decomposition (the clustering rule and output are unchanged):
//   MPI    - every round, the candidate seeds are split across ranks, and the best
//            candidate (largest cluster, lowest seed index on ties) is picked with
//            MPI_Allreduce(MPI_MAXLOC). All ranks keep identical cluster state.
//   OpenMP - inside a rank, the team's master thread drives the GPU while the other
//            threads grow candidate clusters on the CPU, taking seeds from a shared counter.
//   CUDA   - the GPU grows one candidate cluster per thread block. Distances use the same
//            arithmetic on host and device, so both paths make identical decisions.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Only rank 0 writes to stdout; every rank holds identical results
static int g_rank = 0;

__attribute__((format(printf, 1, 2)))
static void rootPrintf(const char* fmt, ...) {
    if (g_rank != 0) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

// Abort the whole job on any CUDA error
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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

// Result of a seed search: cardinality of the grown cluster and the seed it grew from
struct Candidate {
    int card;
    int seed;
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

// Calculate Euclidean distance between two points.
// Host and device evaluate the same rounding sequence: sqrt(fma(dx, dx, dy*dy)).
// This matches the contracted form the sequential build produces, and it keeps
// CPU and GPU threshold and tie-break decisions identical.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
#ifdef __CUDA_ARCH__
    return __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
#else
    return std::sqrt(std::fma(dx, dx, dy * dy));
#endif
}

// Per-thread scratch for growing candidate clusters on the CPU
struct CpuScratch {
    std::vector<double> maxd;   // maxd[c]: largest distance from candidate c to the current members
    std::vector<int> active;    // candidates that can still join, in ascending index order

    explicit CpuScratch(size_t n) : maxd(n, 0.0) { active.reserve(n); }
};

// Grow the candidate cluster starting from `seed`: repeatedly add the unclustered point whose
// largest distance to the members is smallest, as long as that stays below the threshold.
// `unclustered` must be sorted ascending and contain `seed`.
// Returns the cardinality (size) of the cluster; members are written to *members if non-null.
int growCandidateCluster(const int seed,
                         const std::vector<int>& unclustered,
                         const std::vector<Point>& points,
                         const double threshold,
                         CpuScratch& scratch,
                         std::vector<int>* members) {
    scratch.active.clear();
    for (const int c : unclustered) {
        if (c == seed) continue;
        scratch.maxd[c] = 0.0;
        scratch.active.push_back(c);
    }

    if (members) {
        members->clear();
        members->push_back(seed);
    }
    int count = 1;
    int newest = seed;

    while (true) {
        const Point& pn = points[newest];
        double best_val = HUGE_VAL;
        int best = -1;
        size_t kept = 0;

        for (size_t k = 0; k < scratch.active.size(); ++k) {
            const int c = scratch.active[k];
            if (c == newest) continue;  // joined the cluster last round

            const double m = std::max(scratch.maxd[c], distance(points[c], pn));
            if (m >= threshold) continue;  // diameter only grows, so c can never join

            scratch.maxd[c] = m;
            scratch.active[kept++] = c;
            if (m < best_val) {  // ascending order: ties keep the lowest index
                best_val = m;
                best = c;
            }
        }
        scratch.active.resize(kept);

        if (best < 0) break;  // No more points can be added

        if (members) members->push_back(best);
        ++count;
        newest = best;
    }

    return count;
}

constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / 32;

// One thread block grows the candidate cluster of one seed, applying exactly the same greedy
// rule as growCandidateCluster(). maxd / in_cluster are this block's slices of the workspace.
__global__ void growClustersKernel(const Point* points, const unsigned char* clustered, const int N,
                                   const double threshold, const int* seeds, int* cards,
                                   double* maxd_ws, unsigned char* in_cluster_ws) {
    __shared__ double warp_val[kWarps];
    __shared__ int warp_idx[kWarps];
    __shared__ int newest;
    __shared__ int chosen;

    const int tid = threadIdx.x;
    const int seed = seeds[blockIdx.x];
    double* maxd = maxd_ws + static_cast<size_t>(blockIdx.x) * N;
    unsigned char* in_cluster = in_cluster_ws + static_cast<size_t>(blockIdx.x) * N;

    for (int c = tid; c < N; c += blockDim.x) {
        maxd[c] = 0.0;
        in_cluster[c] = 0;
    }
    __syncthreads();  // zeroing must finish before the seed is marked
    if (tid == 0) {
        in_cluster[seed] = 1;
        newest = seed;
    }
    __syncthreads();

    int count = 1;
    while (true) {
        const Point pn = points[newest];

        // Fold the newest member into every open candidate and find the best one
        double best_val = HUGE_VAL;
        int best_idx = INT_MAX;
        for (int c = tid; c < N; c += blockDim.x) {
            if (clustered[c] || in_cluster[c]) continue;
            const double cur = maxd[c];
            if (cur >= threshold) continue;  // diameter only grows, so c can never join
            const double m = fmax(cur, distance(points[c], pn));
            maxd[c] = m;
            if (m < threshold && (m < best_val || (m == best_val && c < best_idx))) {
                best_val = m;
                best_idx = c;
            }
        }

        // Block-wide argmin over (value, index); the lowest index wins ties
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, best_val, off);
            const int oi = __shfl_down_sync(0xffffffffu, best_idx, off);
            if (ov < best_val || (ov == best_val && oi < best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        const int lane = tid % 32;
        const int warp = tid / 32;
        if (lane == 0) {
            warp_val[warp] = best_val;
            warp_idx[warp] = best_idx;
        }
        __syncthreads();
        if (warp == 0) {
            const bool has_warp = lane < kWarps;
            best_val = has_warp ? warp_val[lane] : HUGE_VAL;
            best_idx = has_warp ? warp_idx[lane] : INT_MAX;
            for (int off = 16; off > 0; off >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, best_val, off);
                const int oi = __shfl_down_sync(0xffffffffu, best_idx, off);
                if (ov < best_val || (ov == best_val && oi < best_idx)) {
                    best_val = ov;
                    best_idx = oi;
                }
            }
            if (lane == 0) chosen = best_idx;
        }
        __syncthreads();

        if (chosen == INT_MAX) break;  // No more points can be added

        if (tid == 0) {
            in_cluster[chosen] = 1;
            newest = chosen;
        }
        ++count;
        __syncthreads();
    }

    if (tid == 0) cards[blockIdx.x] = count;
}

// Device-side candidate-cluster grower. Owns the device copy of the points and the
// per-block scratch used by growClustersKernel.
class GpuGrower {
  public:
    GpuGrower(const std::vector<Point>& points) : N_(static_cast<int>(points.size())) {
        const size_t bytes_per_seed = static_cast<size_t>(N_) * (sizeof(double) + sizeof(unsigned char));
        capacity_ = std::min(kMaxBatch, std::max<size_t>(1, kWorkspaceBytes / bytes_per_seed));

        CUDA_CHECK(cudaMalloc(&d_points_, N_ * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&d_clustered_, N_ * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_seeds_, capacity_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cards_, capacity_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_maxd_, capacity_ * N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_in_cluster_, capacity_ * N_ * sizeof(unsigned char)));
        h_cards_.resize(capacity_);

        CUDA_CHECK(cudaMemcpy(d_points_, points.data(), N_ * sizeof(Point), cudaMemcpyHostToDevice));
    }

    ~GpuGrower() {
        cudaFree(d_points_);
        cudaFree(d_clustered_);
        cudaFree(d_seeds_);
        cudaFree(d_cards_);
        cudaFree(d_maxd_);
        cudaFree(d_in_cluster_);
    }

    GpuGrower(const GpuGrower&) = delete;
    GpuGrower& operator=(const GpuGrower&) = delete;

    // Maximum number of seeds grown per launch
    size_t capacity() const { return capacity_; }

    void setClustered(const std::vector<unsigned char>& clustered) {
        CUDA_CHECK(cudaMemcpy(d_clustered_, clustered.data(), N_ * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }

    // Grow the candidate clusters of `count` seeds (one block each); returns their cardinalities
    const int* grow(const int* seeds, const size_t count, const double threshold) {
        CUDA_CHECK(cudaMemcpy(d_seeds_, seeds, count * sizeof(int), cudaMemcpyHostToDevice));
        growClustersKernel<<<static_cast<unsigned int>(count), kBlockThreads>>>(
            d_points_, d_clustered_, N_, threshold, d_seeds_, d_cards_, d_maxd_, d_in_cluster_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_cards_.data(), d_cards_, count * sizeof(int), cudaMemcpyDeviceToHost));
        return h_cards_.data();
    }

  private:
    static constexpr size_t kMaxBatch = 4096;
    static constexpr size_t kWorkspaceBytes = size_t{512} << 20;

    int N_;
    size_t capacity_ = 0;
    Point* d_points_ = nullptr;
    unsigned char* d_clustered_ = nullptr;
    int* d_seeds_ = nullptr;
    int* d_cards_ = nullptr;
    double* d_maxd_ = nullptr;
    unsigned char* d_in_cluster_ = nullptr;
    std::vector<int> h_cards_;
};

// Keeps the better of two candidates: the larger cluster wins, and ties go to the lower seed
// index, the order in which the sequential reference scans seeds.
inline bool isBetter(const int card, const int seed, const Candidate& best) {
    return card > best.card || (card == best.card && seed < best.seed);
}

// Seeds claimed per CPU worker counter update
constexpr size_t kCpuGrain = 8;

// Search this rank's seeds for the best candidate. The master thread drives the GPU in
// chunks, and the other OpenMP threads grow clusters on the CPU, taking seeds from a shared counter.
Candidate searchSeeds(const std::vector<int>& my_seeds,
                      const std::vector<int>& unclustered,
                      const std::vector<unsigned char>& clustered,
                      const std::vector<Point>& points,
                      const double threshold,
                      GpuGrower& gpu) {
    const size_t n = my_seeds.size();
    Candidate best{-1, INT_MAX};
    if (n == 0) return best;

    gpu.setClustered(clustered);
    std::atomic<size_t> next{0};

    #pragma omp parallel
    {
        Candidate local{-1, INT_MAX};

        if (omp_get_thread_num() == 0) {
            // GPU driver: claim a shrinking chunk of seeds and grow them as one batch
            size_t cur = next.load();
            while (cur < n) {
                const size_t take = std::min(gpu.capacity(), std::max<size_t>(1, (n - cur) / 4));
                if (!next.compare_exchange_weak(cur, cur + take)) continue;

                const int* cards = gpu.grow(my_seeds.data() + cur, take, threshold);
                for (size_t i = 0; i < take; ++i) {
                    if (isBetter(cards[i], my_seeds[cur + i], local)) {
                        local = {cards[i], my_seeds[cur + i]};
                    }
                }
                cur = next.load();
            }
        } else {
            // CPU worker: claim a small grain of seeds per counter update to limit contention
            CpuScratch scratch(points.size());
            size_t begin;
            while ((begin = next.fetch_add(kCpuGrain)) < n) {
                const size_t end = std::min(n, begin + kCpuGrain);
                for (size_t i = begin; i < end; ++i) {
                    const int seed = my_seeds[i];
                    const int card = growCandidateCluster(seed, unclustered, points, threshold, scratch, nullptr);
                    if (isBetter(card, seed, local)) {
                        local = {card, seed};
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (isBetter(local.card, local.seed, best)) best = local;
        }
    }

    return best;
}

// Main QT clustering algorithm. Every rank runs the same loop and reaches the same decisions.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nranks,
                                  GpuGrower& gpu) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    CpuScratch scratch(N);

    // Initialize unclustered indices
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // This rank's share of the seeds (round-robin over the unclustered list)
        std::vector<int> my_seeds;
        for (size_t p = rank; p < unclustered_indices.size(); p += nranks) {
            my_seeds.push_back(unclustered_indices[p]);
        }

        const Candidate local = searchSeeds(my_seeds, unclustered_indices, clustered, points, threshold, gpu);

        // Global best: largest cardinality, ties resolved to the lowest seed index
        Candidate global{};
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // If we found a cluster, add it
        if (global.seed < 0 || global.card <= 0) break;

        // Every rank regenerates the winning cluster from the same deterministic rule
        std::vector<int> members;
        const int cardinality = growCandidateCluster(global.seed, unclustered_indices, points,
                                                     threshold, scratch, &members);
        if (cardinality != global.card) {
            fprintf(stderr, "Rank %d: cluster size mismatch for seed %d (%d != %d)\n",
                    rank, global.seed, cardinality, global.card);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        Cluster cluster;
        cluster.seed_point = global.seed;
        cluster.members = members;
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (const int m : members) {
            clustered[m] = 1;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties.
// Cluster diameters are computed in parallel (clusters split across ranks, OpenMP within a rank).
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int rank,
                     const int nranks) {
    bool valid = true;

    rootPrintf("Validating clusters:\n");

    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (long long c = rank; c < static_cast<long long>(clusters.size()); c += nranks) {
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
        diameters[c] = max_diameter;
    }
    MPI_Allreduce(MPI_IN_PLACE, diameters.data(), static_cast<int>(diameters.size()),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

        if (c < 10) { // Print first 10 clusters
            rootPrintf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                       c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            rootPrintf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
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
                rootPrintf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
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

    rootPrintf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
               points.size(), clustered_count, points.size() - clustered_count);

    return valid;
}

void printUsage(const char* progName) {
    rootPrintf("Usage: %s [options]\n", progName);
    rootPrintf("Options:\n");
    rootPrintf("  -n <num>     Number of points (default: 1000)\n");
    rootPrintf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    rootPrintf("  -v           Enable validation\n");
    rootPrintf("  -r           Print results for external validation\n");
    rootPrintf("  -h           Show this help message\n");
}

// Bind this rank to a GPU and split the node's cores between the ranks that share it
void setupNodeResources(const int rank) {
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    int node_size = 1;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices visible on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % device_count));

    // Default: share a node-wide CPU budget among the ranks on this node. Measured on a 128-core,
    // 2-way SMT node, throughput peaks near 64 busy threads and drops sharply when every hardware
    // thread is busy (the GPU driver threads compete for CPU time). An explicit OMP_NUM_THREADS wins.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        constexpr int kNodeThreadBudget = 64;
        const int share = std::min(omp_get_num_procs(), kNodeThreadBudget) / node_size;
        omp_set_num_threads(std::max(1, share));
    }
}

int runBenchmark(int argc, char** argv, const int rank, const int nranks) {
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
            rootPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        rootPrintf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        return 1;
    }

    rootPrintf("QT Clustering Benchmark\n");
    rootPrintf("Number of points: %d\n", num_points);
    rootPrintf("Distance threshold: %.2f\n", threshold);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");

    setupNodeResources(rank);

    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    GpuGrower gpu(points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks, gpu);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    rootPrintf("Clustering time: %ld ms\n", cluster_time.count());
    rootPrintf("Clusters found: %zu\n", clusters.size());

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

    rootPrintf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
    rootPrintf("Average cluster size: %.2f\n", avg_cluster_size);
    rootPrintf("Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    rootPrintf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

    // Print results for external validation
    if (printResults && rank == 0) {
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
        const bool valid = validateClusters(clusters, points, threshold, rank, nranks);

        if (valid) {
            rootPrintf("Validation: PASSED\n");
            return 0;
        } else {
            rootPrintf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    g_rank = rank;

    const int rc = runBenchmark(argc, argv, rank, nranks);

    MPI_Finalize();
    return rc;
}
