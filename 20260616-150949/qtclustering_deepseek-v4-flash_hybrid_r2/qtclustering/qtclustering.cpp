// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH  = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point { double x, y; };

struct Cluster {
    std::vector<int> members;
    int              seed_point;
};

// ------------------------------------------------------------------
// CUDA error-checking helper
// ------------------------------------------------------------------
#define CUDA_ABORT(ans) do {                                          \
    cudaError_t _e_ = (ans);                                          \
    if (_e_ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %d: %s at %s:%d\n",               \
                _e_, cudaGetErrorString(_e_), __FILE__, __LINE__);    \
        MPI_Abort(MPI_COMM_WORLD, _e_);                               \
    }                                                                 \
} while(0)

// ====================================================================
// CUDA kernel: build an entire candidate cluster for one seed in a
// single kernel launch (one block).  All threads in the block
// cooperate to parallelise distance computations; shared memory is
// used for the per-iteration reduction.
//
// The kernel writes the cluster members and size to global output
// buffers; the host reads them back after the kernel finishes.
// ====================================================================
__global__ void buildOneClusterKernel(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const char*   __restrict__ clustered_global,
    int    seed,
    int    N,
    double threshold,
    char*  __restrict__ in_cluster,    // N bytes scratch (zeroed on entry)
    int*   __restrict__ members_out,   // output: member indices
    int*   __restrict__ size_out)      // output: cluster cardinality
{
    int tid = threadIdx.x;
    int total = blockDim.x;

    // --- initialise per-block in_cluster ---
    for (int c = tid; c < N; c += total) in_cluster[c] = 0;
    __syncthreads();
    if (tid == 0) { in_cluster[seed] = 1; members_out[0] = seed; }
    __syncthreads();

    extern __shared__ double s_diam[];
    int* s_idx = reinterpret_cast<int*>(s_diam + blockDim.x);

    int size = 1;

    while (size < N) {
        // ---- each thread computes the best candidate among its points ----
        // Iterate only over known cluster members (stored in members_out[0..size-1])
        // instead of scanning all N points — this is the key optimisation.
        int    local_best      = -1;
        double local_best_diam = threshold;

        for (int c = tid; c < N; c += total) {
            if (clustered_global[c] || in_cluster[c]) continue;

            double max_dist = 0.0;
            for (int mi = 0; mi < size; ++mi) {
                int m = members_out[mi];
                double dx = points_x[c] - points_x[m];
                double dy = points_y[c] - points_y[m];
                double d  = sqrt(dx * dx + dy * dy);
                if (d > max_dist) max_dist = d;
            }

            // Tie-break by lower point index to match sequential semantics
            if (max_dist < local_best_diam ||
                (max_dist == local_best_diam && c < local_best)) {
                local_best_diam = max_dist;
                local_best      = c;
            }
        }

        // ---- block-wide reduction ----
        s_diam[tid] = local_best_diam;
        s_idx[tid]  = local_best;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                // Tie-break by lower index for deterministic output
                bool better = (s_diam[tid + s] < s_diam[tid]) ||
                    (s_diam[tid + s] == s_diam[tid] &&
                     s_idx[tid + s] < s_idx[tid]);
                if (better) {
                    s_diam[tid] = s_diam[tid + s];
                    s_idx[tid]  = s_idx[tid + s];
                }
            }
            __syncthreads();
        }

        int best = s_idx[0];

        if (best < 0) {
            if (tid == 0) *size_out = size;
            return;
        }

        if (tid == 0) {
            in_cluster[best]  = 1;
            members_out[size] = best;
        }
        __syncthreads();   // make in_cluster visible to all threads next iter
        ++size;
    }

    if (tid == 0) *size_out = N;   // all points fit in cluster (rare)
}

// ====================================================================
// GPUContext – per-MPI-rank RAII wrapper for persistent GPU allocations
// and operations.
// ====================================================================
class GPUContext {
    double *d_x_           = nullptr;
    double *d_y_           = nullptr;
    char   *d_clustered_   = nullptr;
    char   *d_in_cluster_  = nullptr;   // per-kernel scratch (N bytes)
    int    *d_members_out_ = nullptr;   // output buffer (N ints)
    int    *d_size_out_    = nullptr;   // output scalar
    int     N_             = 0;
    int     dev_id_        = 0;
    bool    ok_            = false;

public:
    bool isOk() const { return ok_; }

    bool init(const std::vector<Point>& points, int mpi_rank, int /*mpi_size*/) {
        N_ = static_cast<int>(points.size());
        int dev_cnt = 0;
        cudaGetDeviceCount(&dev_cnt);
        if (dev_cnt == 0) {
            fprintf(stderr, "Rank %d: no CUDA devices found\n", mpi_rank);
            return false;
        }
        dev_id_ = mpi_rank % dev_cnt;
        cudaSetDevice(dev_id_);

        CUDA_ABORT(cudaMalloc(&d_x_,           N_ * sizeof(double)));
        CUDA_ABORT(cudaMalloc(&d_y_,           N_ * sizeof(double)));
        CUDA_ABORT(cudaMalloc(&d_clustered_,   N_ * sizeof(char)));
        CUDA_ABORT(cudaMalloc(&d_in_cluster_,  N_ * sizeof(char)));
        CUDA_ABORT(cudaMalloc(&d_members_out_, N_ * sizeof(int)));
        CUDA_ABORT(cudaMalloc(&d_size_out_,    sizeof(int)));

        // Copy point coordinates to device once (flat arrays)
        std::vector<double> hx(N_), hy(N_);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_; ++i) {
            hx[i] = points[i].x;
            hy[i] = points[i].y;
        }
        CUDA_ABORT(cudaMemcpy(d_x_, hx.data(), N_ * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_ABORT(cudaMemcpy(d_y_, hy.data(), N_ * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_ABORT(cudaMemset(d_clustered_,  0, N_ * sizeof(char)));
        CUDA_ABORT(cudaMemset(d_in_cluster_, 0, N_ * sizeof(char)));
        ok_ = true;
        return true;
    }

    void syncClustered(const char* h_clustered) {
        CUDA_ABORT(cudaMemcpy(d_clustered_, h_clustered, N_ * sizeof(char),
                              cudaMemcpyHostToDevice));
    }

    // Build a cluster on the GPU starting from 'seed'.
    // Returns cluster cardinality; 'members' is filled with member indices.
    int buildCluster(int seed, double threshold,
                     std::vector<int>& members) {
        const int threads = 256;
        size_t smem = threads * (sizeof(double) + sizeof(int));

        buildOneClusterKernel<<<1, threads, smem>>>(
            d_x_, d_y_, d_clustered_,
            seed, N_, threshold,
            d_in_cluster_, d_members_out_, d_size_out_);
        CUDA_ABORT(cudaGetLastError());

        // Read back the cluster size
        int size = 0;
        CUDA_ABORT(cudaMemcpy(&size, d_size_out_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_ABORT(cudaStreamSynchronize(0));

        if (size > 0) {
            members.resize(size);
            CUDA_ABORT(cudaMemcpy(members.data(), d_members_out_,
                                  size * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }
        return size;
    }

    ~GPUContext() { cleanup(); }

    void cleanup() {
        auto free_dev = [](auto*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        free_dev(d_x_);
        free_dev(d_y_);
        free_dev(d_clustered_);
        free_dev(d_in_cluster_);
        free_dev(d_members_out_);
        free_dev(d_size_out_);
        ok_ = false;
    }
};

// One GPU context per MPI rank (each rank drives its own device).
static GPUContext g_gpu;

// ====================================================================
// Synthetic data generation (identical semantics to original)
// ====================================================================
void generateSyntheticData(std::vector<Point>& points, const int N,
                           unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R      = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (group_cnt > (N - count)) group_cnt = N - count;
        while (group_cnt > 0) {
            const double sign  = (frand() < 0.5) ? -1.0 : 1.0;
            const double r     = frand() * R;
            const double dx    = (2.0 * frand() - 1.0) * r;
            const double dy    = std::sqrt(r * r - dx * dx) * sign;
            const double x     = cntr_x + dx;
            const double y     = cntr_y + dy;
            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT) continue;
            points[count] = {x, y};
            ++count;
            --group_cnt;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ====================================================================
// OpenMP-parallelised findClosestPoint (included as part of the hybrid
// approach for CPU-resident distance computation).
// ====================================================================
int findClosestPointOMP(const std::vector<int>& members,
                         const std::vector<char>& clustered,
                         const std::vector<char>& in_cluster,
                         const std::vector<Point>& points,
                         double threshold, int N) {
    int    best      = -1;
    double best_diam = threshold;

    #pragma omp parallel
    {
        int    local_best      = -1;
        double local_best_diam = threshold;

        #pragma omp for nowait
        for (int c = 0; c < N; ++c) {
            if (clustered[c] || in_cluster[c]) continue;
            double max_dist = 0.0;
            for (size_t i = 0; i < members.size(); ++i) {
                double d = distance(points[c], points[members[i]]);
                if (d > max_dist) max_dist = d;
            }
            if (max_dist < local_best_diam) {
                local_best_diam = max_dist;
                local_best      = c;
            }
        }

        #pragma omp critical
        {
            if (local_best_diam < best_diam) {
                best_diam = local_best_diam;
                best      = local_best;
            }
        }
    }
    return best;
}

// ====================================================================
// Candidate cluster generation — delegates the entire greedy cluster
// construction to a single GPU kernel invocation per seed.
// ====================================================================
int generateCandidateClusterHybrid(int seed_point,
                                    std::vector<char>& /*clustered*/,
                                    const std::vector<Point>& /*points*/,
                                    double threshold, int /*N*/,
                                    std::vector<int>* out_members) {
    std::vector<int> members;
    int size = g_gpu.buildCluster(seed_point, threshold, members);
    if (out_members) *out_members = std::move(members);
    return size;
}

// ====================================================================
// Main QT clustering — MPI-parallelised outer loop.
//
// Every iteration:
//   1. Partition the unclustered seeds among ranks.
//   2. Each rank evaluates its seeds (GPU-accelerated).
//   3. MPI_Allreduce / MPI_MAXLOC finds the globally best cluster.
//   4. The winning rank broadcasts the cluster contents.
//   5. All ranks update their local clustered[] and unclustered lists.
// ====================================================================
std::vector<Cluster> qtClusteringHybrid(const std::vector<Point>& points,
                                         double threshold,
                                         int mpi_rank, int mpi_size,
                                         MPI_Comm comm) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);

    std::vector<int> unclustered;
    unclustered.reserve(N);
    for (int i = 0; i < N; ++i) unclustered.push_back(i);

    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        const int U = static_cast<int>(unclustered.size());

        // ---- 1. Partition seeds among ranks ----
        int lo = (U * mpi_rank)      / mpi_size;
        int hi = (U * (mpi_rank + 1)) / mpi_size;

        // ---- 2. Each rank evaluates its local seeds ----
        int    best_card   = -1;
        int    best_seed   = -1;
        std::vector<int> best_members;

        for (int i = lo; i < hi; ++i) {
            int s = unclustered[i];
            std::vector<int> cand;
            int card = generateCandidateClusterHybrid(
                           s, clustered, points, threshold, N, &cand);
            if (card > best_card) {
                best_card   = card;
                best_seed   = s;
                best_members.swap(cand);
            }
        }

        // ---- 3. Global reduction — which rank has the best cluster? ----
        struct { int card; int rank; } local{best_card, mpi_rank}, global;
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, comm);
        const int winner_rank = global.rank;

        // ---- 4. Broadcast winning cluster contents to all ranks ----
        int    global_seed = -1;
        int    global_size = 0;
        std::vector<int> global_members;

        if (mpi_rank == winner_rank) {
            global_seed = best_seed;
            global_size = static_cast<int>(best_members.size());
            MPI_Bcast(&global_seed, 1, MPI_INT, winner_rank, comm);
            MPI_Bcast(&global_size, 1, MPI_INT, winner_rank, comm);
            MPI_Bcast(best_members.data(), global_size, MPI_INT,
                      winner_rank, comm);
            global_members.swap(best_members);
        } else {
            MPI_Bcast(&global_seed, 1, MPI_INT, winner_rank, comm);
            MPI_Bcast(&global_size, 1, MPI_INT, winner_rank, comm);
            global_members.resize(global_size);
            MPI_Bcast(global_members.data(), global_size, MPI_INT,
                      winner_rank, comm);
        }

        if (global_size <= 0) break;

        // ---- 5. Record cluster and update local state ----
        clusters.push_back({global_members, global_seed});

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < global_size; ++i) {
            clustered[global_members[i]] = 1;
        }

        // Sync the updated clustered array to the GPU.
        g_gpu.syncClustered(clustered.data());

        // Remove clustered points from the unclustered list.
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                [&](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
    }

    return clusters;
}

// ====================================================================
// Validation (identical semantics to original)
// ====================================================================
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      double threshold) {
    bool valid = true;
    printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cl = clusters[c];
        double max_d = 0.0;
        for (size_t i = 0; i < cl.members.size(); ++i)
            for (size_t j = i + 1; j < cl.members.size(); ++j) {
                double d = distance(points[cl.members[i]],
                                    points[cl.members[j]]);
                if (d > max_d) max_d = d;
            }
        if (c < 10)
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cl.members.size(), cl.seed_point, max_d);
        if (max_d > threshold * 1.001) {
            printf("ERROR: Cluster %zu diameter %.4f > threshold %.4f\n",
                   c, max_d, threshold);
            valid = false;
        }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            int m = clusters[c].members[i];
            if (membership[m] >= 0) {
                printf("ERROR: Point %d in multiple clusters (%d and %zu)\n",
                       m, membership[m], c);
                valid = false;
            }
            membership[m] = static_cast<int>(c);
        }
    int cnt = 0;
    for (auto m : membership) if (m >= 0) ++cnt;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), cnt, points.size() - cnt);
    return valid;
}

void printUsage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ====================================================================
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int    num_points   = 1000;
    double threshold    = 2.0;
    bool   validate     = false;
    bool   printResults = false;

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            num_points = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            threshold   = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (strcmp(argv[i], "-r") == 0)
            printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0)
            printf("Error: num_points=%d, threshold=%.2f\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark  (MPI+OpenMP+CUDA hybrid)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    // ---- Generate data (all ranks produce identical copies) ----
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // ---- Initialise GPU (per rank) ----
    {
        int dev_cnt;
        cudaGetDeviceCount(&dev_cnt);
        if (mpi_rank == 0)
            printf("CUDA devices available: %d\n", dev_cnt);
    }
    bool gpu_ok = g_gpu.init(points, mpi_rank, mpi_size);
    if (!gpu_ok) {
        fprintf(stderr, "Rank %d: GPU init failed -- aborting.\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Clustering ----
    auto t0 = std::chrono::high_resolution_clock::now();

    auto clusters = qtClusteringHybrid(points, threshold,
                                        mpi_rank, mpi_size,
                                        MPI_COMM_WORLD);

    CUDA_ABORT(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        t1 - t0).count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // ---- Rank 0 reports timing and statistics ----
    if (mpi_rank == 0) {
        auto ms = std::chrono::milliseconds(max_ms);
        printf("Clustering time: %ld ms\n", ms.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster     = 0;
        for (const auto& cl : clusters) {
            int sz = static_cast<int>(cl.members.size());
            total_clustered += sz;
            if (sz > max_cluster) max_cluster = sz;
        }
        double avg = clusters.empty()
                         ? 0.0
                         : static_cast<double>(total_clustered) /
                               clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg);
        printf("Maximum cluster size: %d\n", max_cluster);

        double sec = ms.count() / 1000.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters.size() / sec, num_points / sec);
    }

    // ---- Optional external-validation output (rank 0 only) ----
    if (printResults && mpi_rank == 0) {
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c)
            for (size_t i = 0; i < clusters[c].members.size(); ++i)
                membership[clusters[c].members[i]] =
                    static_cast<int>(c);
        for (int m : membership)
            membershipData.push_back(static_cast<double>(m));
        print_results(membershipData, "ClusterMembership");
    }

    // ---- Validation (rank 0 only) ----
    if (validate && mpi_rank == 0) {
        bool ok = validateClusters(clusters, points, threshold);
        printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
    }

    g_gpu.cleanup();
    MPI_Finalize();
    return 0;
}
