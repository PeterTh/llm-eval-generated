// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization:
//   MPI    - seeds are dealt cyclically across ranks (seed s belongs to rank s % P);
//            ranks agree on each winning cluster with one allreduce and one broadcast
//   CUDA   - each rank owns one GPU; a persistent-block kernel grows one candidate
//            cluster per block, with a block-wide argmin for every growth step
//   OpenMP - host-side loops (cache validation, local best selection, result
//            copies, cluster validation) run multi-threaded
//
// Semantics match the sequential greedy algorithm exactly. A candidate cluster
// grown from a seed only depends on the points that are still unclustered
// and on its own members, so it stays exact until one of its members is
// clustered by another cluster. Cached candidates are recomputed only then.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per CUDA block used by the growth kernel
static constexpr int kGrowThreads = 256;

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

// Owning device memory buffer; grows on demand
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    void reserve(size_t n) {
        if (n <= size_) return;
        if (ptr_) CUDA_CHECK(cudaFree(ptr_));
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        size_ = n;
    }

    T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
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

// Grows the candidate cluster of every seed in a batch. One block grows one seed
// at a time; blocks pull seeds from a shared counter. Members are written to a
// global arena, so the host learns each cluster's size and offset afterwards.
//
// Each growth step adds the unclustered candidate whose maximum distance to the
// current members is smallest, subject to that distance being below the threshold.
// Ties go to the lowest index. Candidates that cannot beat the current best are
// abandoned early. The maximum only grows as members are added, so this does not
// change the chosen point.
__global__ void growClustersKernel(const Point* __restrict__ points,
                                   const uint8_t* __restrict__ clustered,
                                   const int* __restrict__ seeds,
                                   const int nseeds,
                                   const int N,
                                   const double threshold,
                                   int* __restrict__ scratch_members,
                                   uint8_t* __restrict__ scratch_flags,
                                   unsigned int* __restrict__ next_seed,
                                   int* __restrict__ arena,
                                   const unsigned long long arena_cap,
                                   unsigned long long* __restrict__ arena_used,
                                   int* __restrict__ card_out,
                                   unsigned long long* __restrict__ offset_out) {
    __shared__ double s_md[kGrowThreads];
    __shared__ int s_idx[kGrowThreads];
    __shared__ int s_si;
    __shared__ int s_next;
    __shared__ unsigned long long s_base;

    int* const members = scratch_members + static_cast<size_t>(blockIdx.x) * N;
    uint8_t* const in_cluster = scratch_flags + static_cast<size_t>(blockIdx.x) * N;
    const int tid = threadIdx.x;

    while (true) {
        if (tid == 0) s_si = static_cast<int>(atomicAdd(next_seed, 1u));
        __syncthreads();
        const int si = s_si;
        if (si >= nseeds) break;
        const int seed = seeds[si];

        if (tid == 0) {
            members[0] = seed;
            in_cluster[seed] = 1;
        }
        __syncthreads();
        int count = 1;

        while (count < N) {
            // Each thread scans a strided slice of candidates in increasing index order
            double best = threshold;
            int best_idx = INT_MAX;
            for (int c = tid; c < N; c += kGrowThreads) {
                if (clustered[c] || in_cluster[c]) continue;

                const Point pc = points[c];
                double max_d2 = 0.0;
                bool rejected = false;
                for (int i = 0; i < count; ++i) {
                    const Point pm = points[members[i]];
                    const double dx = pc.x - pm.x;
                    const double dy = pc.y - pm.y;
                    max_d2 = fmax(max_d2, dx * dx + dy * dy);
                    if (sqrt(max_d2) >= best) {
                        rejected = true;
                        break;
                    }
                }
                if (!rejected) {
                    best = sqrt(max_d2);
                    best_idx = c;
                }
            }

            // Block-wide argmin: smallest distance, then smallest index
            s_md[tid] = best;
            s_idx[tid] = best_idx;
            __syncthreads();
            for (int stride = kGrowThreads / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    const double md_other = s_md[tid + stride];
                    const int idx_other = s_idx[tid + stride];
                    if (md_other < s_md[tid] || (md_other == s_md[tid] && idx_other < s_idx[tid])) {
                        s_md[tid] = md_other;
                        s_idx[tid] = idx_other;
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                if (s_idx[0] == INT_MAX) {
                    s_next = -1;
                } else {
                    members[count] = s_idx[0];
                    in_cluster[s_idx[0]] = 1;
                    s_next = s_idx[0];
                }
            }
            __syncthreads();
            if (s_next < 0) break;
            ++count;
        }

        for (int i = tid; i < count; i += kGrowThreads) {
            in_cluster[members[i]] = 0;
        }
        if (tid == 0) s_base = atomicAdd(arena_used, static_cast<unsigned long long>(count));
        __syncthreads();

        if (s_base + count <= arena_cap) {
            for (int i = tid; i < count; i += kGrowThreads) {
                arena[s_base + i] = members[i];
            }
        }
        if (tid == 0) {
            card_out[si] = count;
            offset_out[si] = s_base;
        }
    }
}

// Per-GPU engine that grows candidate clusters for batches of seeds
class GrowthEngine {
public:
    GrowthEngine(const std::vector<Point>& points, double threshold)
        : N_(static_cast<int>(points.size())), threshold_(threshold) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int sm_count = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
        grid_ = std::min(4 * sm_count, N_);

        d_points_.reserve(N_);
        CUDA_CHECK(cudaMemcpy(d_points_.get(), points.data(), N_ * sizeof(Point),
                              cudaMemcpyHostToDevice));
        d_clustered_.reserve(N_);
        d_scratch_members_.reserve(static_cast<size_t>(grid_) * N_);
        d_scratch_flags_.reserve(static_cast<size_t>(grid_) * N_);
        CUDA_CHECK(cudaMemset(d_scratch_flags_.get(), 0, static_cast<size_t>(grid_) * N_));
        d_next_seed_.reserve(1);
        d_arena_used_.reserve(1);
    }

    // Writes the greedy candidate cluster of seeds[i] into members[i]
    void grow(const std::vector<int>& seeds, const std::vector<uint8_t>& clustered,
              std::vector<std::vector<int>>& members) {
        const int nseeds = static_cast<int>(seeds.size());
        members.assign(nseeds, {});
        if (nseeds == 0) return;

        CUDA_CHECK(cudaMemcpy(d_clustered_.get(), clustered.data(), N_ * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        d_seeds_.reserve(nseeds);
        d_card_.reserve(nseeds);
        d_offset_.reserve(nseeds);
        d_arena_.reserve(static_cast<size_t>(nseeds) * 16);
        CUDA_CHECK(cudaMemcpy(d_seeds_.get(), seeds.data(), nseeds * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Rerun with a larger arena if the batch did not fit
        unsigned long long used = 0;
        while (true) {
            CUDA_CHECK(cudaMemset(d_next_seed_.get(), 0, sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(d_arena_used_.get(), 0, sizeof(unsigned long long)));
            growClustersKernel<<<std::min(grid_, nseeds), kGrowThreads>>>(
                d_points_.get(), d_clustered_.get(), d_seeds_.get(), nseeds, N_, threshold_,
                d_scratch_members_.get(), d_scratch_flags_.get(), d_next_seed_.get(),
                d_arena_.get(), d_arena_.size(), d_arena_used_.get(), d_card_.get(),
                d_offset_.get());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&used, d_arena_used_.get(), sizeof(used),
                                  cudaMemcpyDeviceToHost));
            if (used <= d_arena_.size()) break;
            d_arena_.reserve(used);
        }

        std::vector<int> cards(nseeds);
        std::vector<unsigned long long> offsets(nseeds);
        std::vector<int> arena(used);
        CUDA_CHECK(cudaMemcpy(cards.data(), d_card_.get(), nseeds * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(offsets.data(), d_offset_.get(),
                              nseeds * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(arena.data(), d_arena_.get(), used * sizeof(int),
                              cudaMemcpyDeviceToHost));

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < nseeds; ++i) {
            const int* first = arena.data() + offsets[i];
            members[i].assign(first, first + cards[i]);
        }
    }

private:
    int N_;
    double threshold_;
    int grid_ = 1;
    DeviceBuffer<Point> d_points_;
    DeviceBuffer<uint8_t> d_clustered_;
    DeviceBuffer<int> d_scratch_members_;
    DeviceBuffer<uint8_t> d_scratch_flags_;
    DeviceBuffer<int> d_seeds_;
    DeviceBuffer<int> d_card_;
    DeviceBuffer<unsigned long long> d_offset_;
    DeviceBuffer<int> d_arena_;
    DeviceBuffer<unsigned int> d_next_seed_;
    DeviceBuffer<unsigned long long> d_arena_used_;
};

// Main QT clustering algorithm, distributed across MPI ranks.
// Seed s is owned by rank s % nranks and cached in local slot s / nranks.
// Every rank keeps the full clustered[] state and the cluster list, so all
// ranks make the same decisions.
std::vector<Cluster> qtClustering(GrowthEngine& engine, const int N, const int rank, const int nranks) {
    const int owned = (rank < N) ? (N - rank - 1) / nranks + 1 : 0;

    std::vector<uint8_t> clustered(N, 0);
    std::vector<std::vector<int>> cache(owned);
    std::vector<uint8_t> cache_valid(owned, 0);
    std::vector<Cluster> clusters;
    int remaining = N;

    while (remaining > 0) {
        // Drop cached candidates that lost a member to a cluster formed since they were grown
        #pragma omp parallel for schedule(dynamic, 64)
        for (int j = 0; j < owned; ++j) {
            const int seed = rank + j * nranks;
            if (clustered[seed]) {
                std::vector<int>().swap(cache[j]);
                cache_valid[j] = 0;
                continue;
            }
            if (cache_valid[j]) {
                for (int m : cache[j]) {
                    if (clustered[m]) {
                        cache_valid[j] = 0;
                        break;
                    }
                }
            }
        }

        // Grow the candidate clusters that are missing or stale
        std::vector<int> todo;
        for (int j = 0; j < owned; ++j) {
            const int seed = rank + j * nranks;
            if (!clustered[seed] && !cache_valid[j]) todo.push_back(seed);
        }
        if (!todo.empty()) {
            std::vector<std::vector<int>> grown;
            engine.grow(todo, clustered, grown);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < todo.size(); ++i) {
                const int j = todo[i] / nranks;
                cache[j] = std::move(grown[i]);
                cache_valid[j] = 1;
            }
        }

        // Local best: largest cardinality, ties resolved toward the smallest seed index
        uint64_t best = 0;
        #pragma omp parallel for reduction(max : best) schedule(static)
        for (int j = 0; j < owned; ++j) {
            const int seed = rank + j * nranks;
            if (clustered[seed]) continue;
            const uint64_t key = (static_cast<uint64_t>(cache[j].size()) << 32) |
                                 static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(seed));
            best = std::max(best, key);
        }
        MPI_Allreduce(MPI_IN_PLACE, &best, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        if (best == 0) break;

        // Every rank learns the winning cluster from its owner
        const int card = static_cast<int>(best >> 32);
        const int seed = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(best & 0xFFFFFFFFu));
        const int owner = seed % nranks;
        std::vector<int> members(card);
        if (rank == owner) {
            members = std::move(cache[seed / nranks]);
        }
        MPI_Bcast(members.data(), card, MPI_INT, owner, MPI_COMM_WORLD);

        for (int m : members) {
            clustered[m] = 1;
        }
        remaining -= card;

        Cluster cluster;
        cluster.members = std::move(members);
        cluster.seed_point = seed;
        clusters.push_back(std::move(cluster));
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Diameter (max distance between any two points) per cluster, computed in parallel
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
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
        diameters[c] = max_diameter;
    }

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

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

static int runBenchmark(int argc, char** argv, int rank, int nranks, int local_rank) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }

    // Each rank drives one GPU
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    GrowthEngine engine(points, threshold);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(engine, num_points, rank, nranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long max_cluster_time_ms = cluster_time.count();
    MPI_Allreduce(MPI_IN_PLACE, &max_cluster_time_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(max_cluster_time_ms);

    // Results are identical on every rank; only the root reports them
    if (rank != 0) return 0;

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Ranks on the same node share its cores and are numbered locally for GPU selection
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / local_size));
    }

    const int ret = runBenchmark(argc, argv, rank, nranks, local_rank);
    MPI_Finalize();
    return ret;
}
