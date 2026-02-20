// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// MPI: distribute seed evaluation across ranks per clustering iteration.
// OpenMP: parallelize local seed evaluation within each rank.
// CUDA: accelerate the inner "find closest point" candidate search.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// Generate synthetic 2D point data in clusters (rank 0, then broadcast).
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (group_cnt < 1) group_cnt = 1; // ensure progress for small N

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

inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------- CUDA accelerated candidate search ----------------

static constexpr int CUDA_BLOCK = 256;

__device__ __forceinline__ double dist_dev(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void find_best_candidate_kernel(const Point* __restrict__ points,
                                          const unsigned char* __restrict__ clustered,
                                          const unsigned char* __restrict__ in_cluster,
                                          const int* __restrict__ members,
                                          int members_count,
                                          double threshold,
                                          int N,
                                          double* __restrict__ block_best_val,
                                          int* __restrict__ block_best_idx) {
    __shared__ double s_val[CUDA_BLOCK];
    __shared__ int s_idx[CUDA_BLOCK];

    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    double best_val = INFINITY;
    int best_idx = -1;

    for (int cand = blockIdx.x * blockDim.x + tid; cand < N; cand += stride) {
        if (clustered[cand] || in_cluster[cand]) continue;

        double maxd = 0.0;
        const Point pc = points[cand];

        // Compute max distance to current cluster members; early exit once threshold exceeded.
        for (int i = 0; i < members_count; ++i) {
            const int m = members[i];
            const double d = dist_dev(pc, points[m]);
            if (d > maxd) {
                maxd = d;
                if (maxd >= threshold) break;
            }
        }

        if (maxd < threshold) {
            if (maxd < best_val || (maxd == best_val && cand < best_idx)) {
                best_val = maxd;
                best_idx = cand;
            }
        }
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const double v2 = s_val[tid + offset];
            const int i2 = s_idx[tid + offset];
            const double v1 = s_val[tid];
            const int i1 = s_idx[tid];

            const bool take2 = (v2 < v1) || (v2 == v1 && i2 >= 0 && (i1 < 0 || i2 < i1));
            if (take2) {
                s_val[tid] = v2;
                s_idx[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_best_val[blockIdx.x] = s_val[0];
        block_best_idx[blockIdx.x] = s_idx[0];
    }
}

struct CudaGlobal {
    int N = 0;
    int device = 0;
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;

    CudaGlobal() = default;

    void init(const std::vector<Point>& points, int device_id) {
        device = device_id;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");

        N = static_cast<int>(points.size());
        cudaCheck(cudaMalloc((void**)&d_points, sizeof(Point) * N), "cudaMalloc d_points");
        cudaCheck(cudaMalloc((void**)&d_clustered, sizeof(unsigned char) * N), "cudaMalloc d_clustered");

        cudaCheck(cudaMemcpy(d_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice),
                  "cudaMemcpy points H2D");
        cudaCheck(cudaMemset(d_clustered, 0, sizeof(unsigned char) * N), "cudaMemset d_clustered");

        // Prefer L1 cache for this memory-bound kernel.
        cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), "cudaDeviceSetCacheConfig");
    }

    void updateClustered(const std::vector<unsigned char>& clustered) {
        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), sizeof(unsigned char) * N, cudaMemcpyHostToDevice),
                  "cudaMemcpy clustered H2D");
    }

    ~CudaGlobal() {
        if (d_points) cudaFree(d_points);
        if (d_clustered) cudaFree(d_clustered);
    }
};

struct CudaWorkspace {
    int N = 0;
    int grid = 0;
    cudaStream_t stream{};

    unsigned char* d_in_cluster = nullptr;
    int* d_members = nullptr;

    double* d_block_best_val = nullptr;
    int* d_block_best_idx = nullptr;

    double* h_block_best_val = nullptr; // pinned
    int* h_block_best_idx = nullptr;    // pinned

    explicit CudaWorkspace(const CudaGlobal& g) {
        N = g.N;
        cudaCheck(cudaSetDevice(g.device), "cudaSetDevice(ws)");
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

        cudaCheck(cudaMalloc((void**)&d_in_cluster, sizeof(unsigned char) * N), "cudaMalloc d_in_cluster");
        cudaCheck(cudaMalloc((void**)&d_members, sizeof(int) * N), "cudaMalloc d_members");

        grid = std::min(1024, std::max(1, (N + CUDA_BLOCK - 1) / CUDA_BLOCK));
        cudaCheck(cudaMalloc((void**)&d_block_best_val, sizeof(double) * grid), "cudaMalloc d_block_best_val");
        cudaCheck(cudaMalloc((void**)&d_block_best_idx, sizeof(int) * grid), "cudaMalloc d_block_best_idx");

        cudaCheck(cudaHostAlloc((void**)&h_block_best_val, sizeof(double) * grid, cudaHostAllocPortable),
                  "cudaHostAlloc h_block_best_val");
        cudaCheck(cudaHostAlloc((void**)&h_block_best_idx, sizeof(int) * grid, cudaHostAllocPortable),
                  "cudaHostAlloc h_block_best_idx");
    }

    void resetInCluster() {
        cudaCheck(cudaMemsetAsync(d_in_cluster, 0, sizeof(unsigned char) * N, stream), "cudaMemsetAsync in_cluster");
    }

    void setInCluster(int idx) {
        const unsigned char one = 1;
        cudaCheck(cudaMemcpyAsync(d_in_cluster + idx, &one, sizeof(unsigned char), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync set in_cluster");
    }

    void setMember(int pos, int idx) {
        cudaCheck(cudaMemcpyAsync(d_members + pos, &idx, sizeof(int), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync set member");
    }

    int findClosestPoint(const CudaGlobal& g,
                         int members_count,
                         double threshold) {
        find_best_candidate_kernel<<<grid, CUDA_BLOCK, 0, stream>>>(
            g.d_points, g.d_clustered, d_in_cluster, d_members, members_count, threshold, N,
            d_block_best_val, d_block_best_idx);
        cudaCheck(cudaGetLastError(), "kernel launch");

        cudaCheck(cudaMemcpyAsync(h_block_best_val, d_block_best_val, sizeof(double) * grid,
                                 cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync block_best_val D2H");
        cudaCheck(cudaMemcpyAsync(h_block_best_idx, d_block_best_idx, sizeof(int) * grid,
                                 cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync block_best_idx D2H");
        cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        double best_val = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (int b = 0; b < grid; ++b) {
            const double v = h_block_best_val[b];
            const int i = h_block_best_idx[b];
            if (i < 0) continue;
            if (v < best_val || (v == best_val && i < best_idx)) {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx;
    }

    ~CudaWorkspace() {
        if (stream) cudaStreamDestroy(stream);
        if (d_in_cluster) cudaFree(d_in_cluster);
        if (d_members) cudaFree(d_members);
        if (d_block_best_val) cudaFree(d_block_best_val);
        if (d_block_best_idx) cudaFree(d_block_best_idx);
        if (h_block_best_val) cudaFreeHost(h_block_best_val);
        if (h_block_best_idx) cudaFreeHost(h_block_best_idx);
    }
};

int generateCandidateClusterCuda(const int seed_point,
                                const std::vector<unsigned char>& clustered,
                                const CudaGlobal& g,
                                CudaWorkspace& ws,
                                const double threshold,
                                std::vector<int>* cluster_members) {
    (void)clustered; // clustered is on device (g.d_clustered), host copy passed for semantics parity.

    std::vector<int> members;
    members.reserve(g.N);

    ws.resetInCluster();
    ws.setInCluster(seed_point);
    ws.setMember(0, seed_point);

    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < g.N) {
        const int closest = ws.findClosestPoint(g, static_cast<int>(members.size()), threshold);
        if (closest < 0) break;

        ws.setInCluster(closest);
        ws.setMember(static_cast<int>(members.size()), closest);
        members.push_back(closest);
    }

    if (cluster_members) {
        *cluster_members = std::move(members);
    }
    return static_cast<int>(cluster_members ? cluster_members->size() : members.size());
}

// ---------------- MPI + OpenMP distributed QT clustering ----------------

struct LocalBest {
    int cardinality = -1;
    int seed = -1;
    std::vector<int> members;
};

static inline bool betterSeed(int card_a, int seed_a, int card_b, int seed_b) {
    // Prefer higher cardinality; for ties, prefer smaller seed index (matches sequential scan order).
    if (card_a != card_b) return card_a > card_b;
    if (seed_b < 0) return seed_a >= 0;
    if (seed_a < 0) return false;
    return seed_a < seed_b;
}

std::vector<Cluster> qtClusteringHybridMPI(const std::vector<Point>& points,
                                          const double threshold,
                                          MPI_Comm comm) {
    int rank = 0, world = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world);

    const int N = static_cast<int>(points.size());

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    // Select GPU per-node using shared-memory local rank.
    int local_rank = 0;
    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    const int device_id = (device_count > 0) ? (local_rank % device_count) : 0;

    CudaGlobal g;
    g.init(points, device_id);

    // Create a bounded number of per-thread CUDA workspaces serially to avoid
    // multi-threaded CUDA allocations (which can stall or deadlock on some runtimes).
    int max_ws_threads = omp_get_max_threads();
    max_ws_threads = std::min(max_ws_threads, 8);
    if (max_ws_threads < 1) max_ws_threads = 1;

    std::vector<std::unique_ptr<CudaWorkspace>> workspaces;
    workspaces.reserve(max_ws_threads);
    for (int t = 0; t < max_ws_threads; ++t) {
        workspaces.emplace_back(new CudaWorkspace(g));
    }

    std::vector<Cluster> clusters;

    while (!unclustered_indices.empty()) {
        // Update clustered flags on device once per outer iteration.
        g.updateClustered(clustered);

        LocalBest local_best;

        int effective_threads = std::min(static_cast<int>(workspaces.size()), static_cast<int>(unclustered_indices.size()));
        if (effective_threads < 1) effective_threads = 1;

        #pragma omp parallel num_threads(effective_threads)
        {
            // CUDA uses a per-host-thread current device; ensure each OpenMP thread selects
            // the correct device before using streams/allocations created on it.
            cudaCheck(cudaSetDevice(g.device), "cudaSetDevice(omp)");
            CudaWorkspace& ws = *workspaces[omp_get_thread_num()];

            LocalBest thread_best;
            thread_best.cardinality = -1;
            thread_best.seed = -1;

            #pragma omp for schedule(dynamic, 1) nowait
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                if (static_cast<int>(i % static_cast<size_t>(world)) != rank) continue;

                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                std::vector<int> candidate_members;
                const int card = generateCandidateClusterCuda(seed, clustered, g, ws, threshold, &candidate_members);

                if (betterSeed(card, seed, thread_best.cardinality, thread_best.seed)) {
                    thread_best.cardinality = card;
                    thread_best.seed = seed;
                    thread_best.members = std::move(candidate_members);
                }
            }

            #pragma omp critical
            {
                if (betterSeed(thread_best.cardinality, thread_best.seed, local_best.cardinality, local_best.seed)) {
                    local_best = std::move(thread_best);
                }
            }
        }

        int local_card = local_best.cardinality;
        int local_seed = local_best.seed;

        std::vector<int> all_cards;
        std::vector<int> all_seeds;
        if (rank == 0) {
            all_cards.resize(world);
            all_seeds.resize(world);
        }

        MPI_Gather(&local_card, 1, MPI_INT, rank == 0 ? all_cards.data() : nullptr, 1, MPI_INT, 0, comm);
        MPI_Gather(&local_seed, 1, MPI_INT, rank == 0 ? all_seeds.data() : nullptr, 1, MPI_INT, 0, comm);

        int winner_rank = 0;
        int best_card = -1;
        int best_seed = -1;

        if (rank == 0) {
            winner_rank = -1;
            for (int r = 0; r < world; ++r) {
                const int c = all_cards[r];
                const int s = all_seeds[r];
                if (betterSeed(c, s, best_card, best_seed)) {
                    best_card = c;
                    best_seed = s;
                    winner_rank = r;
                }
            }
        }

        MPI_Bcast(&winner_rank, 1, MPI_INT, 0, comm);
        MPI_Bcast(&best_card, 1, MPI_INT, 0, comm);
        MPI_Bcast(&best_seed, 1, MPI_INT, 0, comm);

        if (winner_rank < 0 || best_seed < 0 || best_card <= 0) {
            break;
        }

        // Root obtains members from the winning rank, then broadcasts to all ranks.
        std::vector<int> best_members;

        if (rank == winner_rank) {
            best_members = local_best.members;
        }

        if (rank == 0 && winner_rank != 0) {
            int mcount = 0;
            MPI_Recv(&mcount, 1, MPI_INT, winner_rank, 100, comm, MPI_STATUS_IGNORE);
            best_members.resize(mcount);
            MPI_Recv(best_members.data(), mcount, MPI_INT, winner_rank, 101, comm, MPI_STATUS_IGNORE);
        } else if (rank == winner_rank && winner_rank != 0) {
            const int mcount = static_cast<int>(best_members.size());
            MPI_Send(&mcount, 1, MPI_INT, 0, 100, comm);
            MPI_Send(best_members.data(), mcount, MPI_INT, 0, 101, comm);
        }

        int mcount_bcast = 0;
        if (rank == 0) mcount_bcast = static_cast<int>(best_members.size());
        MPI_Bcast(&mcount_bcast, 1, MPI_INT, 0, comm);
        if (rank != 0) best_members.resize(mcount_bcast);
        MPI_Bcast(best_members.data(), mcount_bcast, MPI_INT, 0, comm);

        // Update clustered/unclustered state consistently on all ranks.
        for (int idx : best_members) {
            clustered[idx] = 1;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = std::move(best_members);
            clusters.push_back(std::move(cluster));
        }

        // Synchronize between iterations for determinism and consistent timing.
        MPI_Barrier(comm);
    }

    return clusters;
}

// Validation: check clusters satisfy QT properties (rank 0 only).
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
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
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
    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

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
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
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

    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }

    // Broadcast points so every rank can evaluate seeds independently.
    MPI_Bcast(points.data(), static_cast<int>(sizeof(Point) * points.size()), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const std::vector<Cluster> clusters = qtClusteringHybridMPI(points, threshold, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long cluster_ms = static_cast<long>(max_time * 1000.0);
        printf("Clustering time: %ld ms\n", cluster_ms);
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
               total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = max_time;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

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
            const bool ok = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
