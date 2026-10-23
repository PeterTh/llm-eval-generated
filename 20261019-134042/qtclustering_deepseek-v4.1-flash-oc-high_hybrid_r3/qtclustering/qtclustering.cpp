#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallel strategy:
//   * MPI distributes the candidate seed points of every outer clustering
//     iteration over the ranks (one GPU per rank is expected).
//   * CUDA evaluates all candidate clusters of a rank concurrently, using
//     one thread block per seed point and an incremental max-distance
//     formulation that is mathematically identical to the original
//     findClosestPoint() implementation.
//   * OpenMP builds the pairwise distance matrix, evaluates a small slice of
//     the seed candidates in parallel with the asynchronous GPU kernels, and
//     helps with validation.
//
// All ranks keep an identical copy of the (small) clustering state, so the
// sequential tie-breaking rules of the reference implementation are kept
// exactly: maximum cardinality wins, ties are resolved by the smallest seed
// index, and candidates are considered in increasing index order.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Upper bound on N for which a full double precision distance matrix is used.
static const long long MATRIX_MAX_N = 32768;
// Device workspace budget for the per-seed max-distance rows.
static const size_t WORKSPACE_BUDGET_BYTES = 256ull * 1024ull * 1024ull;
static const int MAX_BLOCK_THREADS = 256;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

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

// Device copy of the distance function. --fmad=false keeps the double
// arithmetic bit-identical to the host implementation.
__device__ __forceinline__ double deviceDistance(double x1, double y1,
                                                 double x2, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

// One thread block per seed point. The block greedily grows the candidate
// cluster while keeping, for every point, the maximum distance to the
// points already in the cluster. This is exactly the quantity recomputed
// by the reference findClosestPoint().
template <bool USE_MATRIX>
__global__ void __launch_bounds__(MAX_BLOCK_THREADS)
candidateClusterKernel(const int* __restrict__ seeds,
                       const double* __restrict__ dist_matrix,
                       const double* __restrict__ px,
                       const double* __restrict__ py,
                       const unsigned char* __restrict__ clustered,
                       const double threshold,
                       const int N,
                       double* __restrict__ workspace,
                       unsigned int* __restrict__ out_cardinality) {
    const int block = blockIdx.x;
    const int seed = seeds[block];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const double inf = INFINITY;

    double* max_dist = workspace + static_cast<size_t>(block) * static_cast<size_t>(N);

    double seed_x = 0.0;
    double seed_y = 0.0;
    if (!USE_MATRIX) {
        seed_x = px[seed];
        seed_y = py[seed];
    }

    for (int c = tid; c < N; c += nthreads) {
        double d;
        if (clustered[c]) {
            d = inf;
        } else if (USE_MATRIX) {
            d = dist_matrix[static_cast<size_t>(seed) * static_cast<size_t>(N) + c];
        } else {
            d = deviceDistance(seed_x, seed_y, px[c], py[c]);
        }
        if (c == seed) {
            d = inf;
        }
        max_dist[c] = d;
    }
    __syncthreads();

    unsigned int cardinality = 1;
    __shared__ double s_val[MAX_BLOCK_THREADS];
    __shared__ int s_idx[MAX_BLOCK_THREADS];

    while (true) {
        double local_min = inf;
        int local_idx = INT_MAX;

        for (int c = tid; c < N; c += nthreads) {
            const double v = max_dist[c];
            if (v < threshold && (v < local_min || (v == local_min && c < local_idx))) {
                local_min = v;
                local_idx = c;
            }
        }

        s_val[tid] = local_min;
        s_idx[tid] = local_idx;
        __syncthreads();

        for (int s = nthreads >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                const double ov = s_val[tid + s];
                const int oi = s_idx[tid + s];
                const double cv = s_val[tid];
                const int ci = s_idx[tid];
                if (ov < cv || (ov == cv && oi < ci)) {
                    s_val[tid] = ov;
                    s_idx[tid] = oi;
                }
            }
            __syncthreads();
        }

        const int best = s_idx[0];
        if (best == INT_MAX) {
            break;
        }

        if (USE_MATRIX) {
            const double* row = dist_matrix + static_cast<size_t>(best) * static_cast<size_t>(N);
            for (int c = tid; c < N; c += nthreads) {
                const double d = row[c];
                if (d > max_dist[c]) {
                    max_dist[c] = d;
                }
            }
        } else {
            const double bx = px[best];
            const double by = py[best];
            for (int c = tid; c < N; c += nthreads) {
                const double d = deviceDistance(bx, by, px[c], py[c]);
                if (d > max_dist[c]) {
                    max_dist[c] = d;
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            max_dist[best] = inf;
        }
        __syncthreads();

        ++cardinality;
    }

    if (tid == 0) {
        out_cardinality[block] = cardinality;
    }
}

// Host version of the candidate cluster generation. Used to materialise the
// members of the winning cluster. It follows the same incremental
// max-distance formulation as the CUDA kernel.
int generateCandidateCluster(const int seed_point,
                             const unsigned char* clustered,
                             const std::vector<Point>& points,
                             const double* dist_matrix,
                             const double threshold,
                             const int point_count,
                             std::vector<int>& members) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> max_dist(point_count);
    const bool have_matrix = (dist_matrix != nullptr);

    members.clear();
    members.reserve(32);
    members.push_back(seed_point);

    if (have_matrix) {
        const double* row = dist_matrix + static_cast<size_t>(seed_point) *
                                           static_cast<size_t>(point_count);
        for (int c = 0; c < point_count; ++c) {
            double d = clustered[c] ? inf : row[c];
            if (c == seed_point) d = inf;
            max_dist[c] = d;
        }
    } else {
        const Point seed = points[seed_point];
        for (int c = 0; c < point_count; ++c) {
            double d = clustered[c] ? inf : distance(seed, points[c]);
            if (c == seed_point) d = inf;
            max_dist[c] = d;
        }
    }

    while (true) {
        double global_min = inf;
        int global_idx = INT_MAX;

        for (int c = 0; c < point_count; ++c) {
            const double v = max_dist[c];
            if (v < threshold && (v < global_min || (v == global_min && c < global_idx))) {
                global_min = v;
                global_idx = c;
            }
        }
        if (global_idx == INT_MAX) break;

        const int best = global_idx;
        if (have_matrix) {
            const double* row = dist_matrix + static_cast<size_t>(best) *
                                               static_cast<size_t>(point_count);
            for (int c = 0; c < point_count; ++c) {
                const double d = row[c];
                if (d > max_dist[c]) max_dist[c] = d;
            }
        } else {
            const Point best_point = points[best];
            for (int c = 0; c < point_count; ++c) {
                const double d = distance(best_point, points[c]);
                if (d > max_dist[c]) max_dist[c] = d;
            }
        }

        max_dist[best] = inf;
        members.push_back(best);
    }

    return static_cast<int>(members.size());
}

struct GpuBuffers {
    double* dist_matrix = nullptr;
    double* px = nullptr;
    double* py = nullptr;
    unsigned char* clustered = nullptr;
    int* seeds = nullptr;
    unsigned int* cardinality = nullptr;
    double* workspace = nullptr;
    int workspace_rows = 0;

    ~GpuBuffers() {
        if (dist_matrix) cudaFree(dist_matrix);
        if (px) cudaFree(px);
        if (py) cudaFree(py);
        if (clustered) cudaFree(clustered);
        if (seeds) cudaFree(seeds);
        if (cardinality) cudaFree(cardinality);
        if (workspace) cudaFree(workspace);
    }
};

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int rank_count,
                                  const int node_rank_count) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<Cluster> clusters;

    GpuBuffers gpu;

    // Decide whether a full distance matrix fits comfortably on this rank's
    // share of the GPU. If it does not, distances are evaluated on the fly.
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const long long n64 = static_cast<long long>(N);
    const long long matrix_bytes = n64 * n64 * static_cast<long long>(sizeof(double));
    const long long local_budget = static_cast<long long>(free_bytes) /
                                   static_cast<long long>(std::max(1, node_rank_count));
    // Use the most constrained rank's budget so that every rank follows the
    // same distance-evaluation path.
    long long global_budget = local_budget;
    MPI_Allreduce(&local_budget, &global_budget, 1, MPI_LONG_LONG, MPI_MIN,
                  MPI_COMM_WORLD);
    bool use_matrix = (N <= MATRIX_MAX_N) &&
                      (matrix_bytes <= (global_budget * 3) / 4);

    std::vector<double> host_matrix;
    if (use_matrix) {
        host_matrix.resize(static_cast<size_t>(N) * static_cast<size_t>(N));
        // OpenMP: build the pairwise distance matrix once.
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const Point pi = points[i];
            double* row = host_matrix.data() + static_cast<size_t>(i) * static_cast<size_t>(N);
            for (int j = 0; j < N; ++j) {
                row[j] = distance(pi, points[j]);
            }
        }

        if (cudaMalloc(reinterpret_cast<void**>(&gpu.dist_matrix),
                       static_cast<size_t>(matrix_bytes)) != cudaSuccess) {
            cudaGetLastError();
            gpu.dist_matrix = nullptr;
            use_matrix = false;
            std::vector<double>().swap(host_matrix);
        } else if (cudaMemcpy(gpu.dist_matrix, host_matrix.data(),
                              static_cast<size_t>(matrix_bytes),
                              cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(gpu.dist_matrix);
            gpu.dist_matrix = nullptr;
            use_matrix = false;
            std::vector<double>().swap(host_matrix);
        }
    }

    // If any rank could not allocate the matrix, all ranks fall back to
    // on-the-fly distances to keep the numerical path identical everywhere.
    const int local_matrix_ok = use_matrix ? 1 : 0;
    int global_matrix_ok = 1;
    MPI_Allreduce(&local_matrix_ok, &global_matrix_ok, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (use_matrix && global_matrix_ok == 0) {
        cudaFree(gpu.dist_matrix);
        gpu.dist_matrix = nullptr;
        std::vector<double>().swap(host_matrix);
        use_matrix = false;
    }

    // Point coordinates are always kept on the device for the on-the-fly
    // distance path.
    std::vector<double> host_px(N), host_py(N);
    for (int i = 0; i < N; ++i) {
        host_px[i] = points[i].x;
        host_py[i] = points[i].y;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.px), sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.py), sizeof(double) * N));
    CUDA_CHECK(cudaMemcpy(gpu.px, host_px.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.py, host_py.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.clustered), N));
    CUDA_CHECK(cudaMemcpy(gpu.clustered, clustered.data(), N, cudaMemcpyHostToDevice));

    // Workspace capacity is based on the largest seed count of this rank,
    // which occurs in the very first iteration.
    const int max_local_seeds = (N > rank) ? ((N - rank + rank_count - 1) / rank_count) : 0;
    const size_t per_seed_values = static_cast<size_t>(N);
    int batch_capacity = 1;
    if (per_seed_values > 0) {
        const size_t rows_by_budget = WORKSPACE_BUDGET_BYTES /
                                      (per_seed_values * sizeof(double));
        batch_capacity = static_cast<int>(
            std::max<size_t>(1, std::min<size_t>(rows_by_budget,
                                                 static_cast<size_t>(INT_MAX))));
    }
    if (max_local_seeds > 0) {
        const int rows = std::min(max_local_seeds, batch_capacity);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.workspace),
                              static_cast<size_t>(rows) * per_seed_values * sizeof(double)));
        gpu.workspace_rows = rows;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.seeds), sizeof(int) * rows));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.cardinality), sizeof(unsigned int) * rows));
    }

    std::vector<int> local_seeds;
    std::vector<unsigned int> host_cardinality;
    int block_threads = MAX_BLOCK_THREADS;
    if (N < 128) {
        block_threads = 64;
    } else if (N < 256) {
        block_threads = 128;
    }

    while (!unclustered_indices.empty()) {
        const int unclustered_count = static_cast<int>(unclustered_indices.size());

        // Round-robin distribution of the current seed candidates over ranks.
        const int local_count = (unclustered_count > rank)
                                    ? ((unclustered_count - rank + rank_count - 1) / rank_count)
                                    : 0;

        // Collective phase: choose the globally best candidate and materialise
        // its members. Returns false when no further cluster can be formed.
        auto finish_iteration = [&](uint64_t local_key) -> bool {
            uint64_t global_key = 0;
            MPI_Allreduce(&local_key, &global_key, 1, MPI_UINT64_T, MPI_MAX,
                          MPI_COMM_WORLD);

            const int best_cardinality = static_cast<int>(global_key >> 32);
            const int best_seed = static_cast<int>(
                0xFFFFFFFFu - static_cast<unsigned int>(global_key & 0xFFFFFFFFu));

            // The owner of the winning seed reconstructs its members exactly.
            const size_t position = static_cast<size_t>(
                std::lower_bound(unclustered_indices.begin(), unclustered_indices.end(),
                                 best_seed) - unclustered_indices.begin());
            const int owner = static_cast<int>(position % rank_count);

            int member_count = best_cardinality;
            std::vector<int> members;
            if (rank == owner) {
                generateCandidateCluster(best_seed, clustered.data(), points,
                                         use_matrix ? host_matrix.data() : nullptr,
                                         threshold, N, members);
                member_count = static_cast<int>(members.size());
            }

            MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
            members.resize(member_count);
            if (member_count > 0) {
                MPI_Bcast(members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);
            }

            if (member_count <= 0) {
                return false;
            }

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = std::move(members);
            clusters.push_back(std::move(cluster));

            for (size_t i = 0; i < clusters.back().members.size(); ++i) {
                clustered[clusters.back().members[i]] = 1;
            }

            size_t write = 0;
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                const int idx = unclustered_indices[i];
                if (!clustered[idx]) {
                    unclustered_indices[write++] = idx;
                }
            }
            unclustered_indices.resize(write);
            return true;
        };

        if (local_count > 0) {
            local_seeds.resize(local_count);
            for (int i = 0; i < local_count; ++i) {
                local_seeds[i] = unclustered_indices[rank + static_cast<size_t>(i) * rank_count];
            }

            CUDA_CHECK(cudaMemcpy(gpu.clustered, clustered.data(), N, cudaMemcpyHostToDevice));

            // A small slice of the seeds is evaluated by the CPU with OpenMP
            // while the GPU is busy with the rest. The chosen fraction keeps
            // the CPU work mostly hidden behind the asynchronous GPU kernel.
            const int cpu_count = (N >= 4096 && local_count >= 16) ? (local_count / 16) : 0;
            const int gpu_count = local_count - cpu_count;

            unsigned int local_best_cardinality = 0;
            int local_best_seed = INT_MAX;

            auto update_best = [&](unsigned int card, int seed) {
                if (card > local_best_cardinality ||
                    (card == local_best_cardinality && seed < local_best_seed)) {
                    local_best_cardinality = card;
                    local_best_seed = seed;
                }
            };

            auto run_cpu_seeds = [&]() {
                if (cpu_count <= 0) return;
                unsigned int cpu_best_cardinality = 0;
                int cpu_best_seed = INT_MAX;
#pragma omp parallel if (cpu_count > 1)
                {
                    unsigned int thread_best_cardinality = 0;
                    int thread_best_seed = INT_MAX;
                    std::vector<int> thread_members;
#pragma omp for schedule(static)
                    for (int i = 0; i < cpu_count; ++i) {
                        const int seed = local_seeds[gpu_count + i];
                        thread_members.clear();
                        const int card = generateCandidateCluster(
                            seed, clustered.data(), points,
                            use_matrix ? host_matrix.data() : nullptr,
                            threshold, N, thread_members);
                        if (static_cast<unsigned int>(card) > thread_best_cardinality ||
                            (static_cast<unsigned int>(card) == thread_best_cardinality &&
                             seed < thread_best_seed)) {
                            thread_best_cardinality = static_cast<unsigned int>(card);
                            thread_best_seed = seed;
                        }
                    }
#pragma omp critical
                    {
                        if (thread_best_cardinality > cpu_best_cardinality ||
                            (thread_best_cardinality == cpu_best_cardinality &&
                             thread_best_seed < cpu_best_seed)) {
                            cpu_best_cardinality = thread_best_cardinality;
                            cpu_best_seed = thread_best_seed;
                        }
                    }
                }
                update_best(cpu_best_cardinality, cpu_best_seed);
            };

            const int batch_size = std::min(std::max(gpu_count, 1), gpu.workspace_rows);
            host_cardinality.resize(batch_size);

            auto launch_gpu_batch = [&](int base, int count) {
                CUDA_CHECK(cudaMemcpy(gpu.seeds, local_seeds.data() + base,
                                      sizeof(int) * count, cudaMemcpyHostToDevice));
                if (use_matrix) {
                    candidateClusterKernel<true><<<count, block_threads>>>(
                        gpu.seeds, gpu.dist_matrix, gpu.px, gpu.py, gpu.clustered,
                        threshold, N, gpu.workspace, gpu.cardinality);
                } else {
                    candidateClusterKernel<false><<<count, block_threads>>>(
                        gpu.seeds, gpu.dist_matrix, gpu.px, gpu.py, gpu.clustered,
                        threshold, N, gpu.workspace, gpu.cardinality);
                }
                CUDA_CHECK(cudaGetLastError());
            };

            auto collect_gpu_batch = [&](int base, int count) {
                CUDA_CHECK(cudaMemcpy(host_cardinality.data(), gpu.cardinality,
                                      sizeof(unsigned int) * count,
                                      cudaMemcpyDeviceToHost));
                for (int i = 0; i < count; ++i) {
                    update_best(host_cardinality[i], local_seeds[base + i]);
                }
            };

            if (gpu_count > 0) {
                const int first_count = std::min(gpu_count, batch_size);
                // Launch asynchronously, then overlap the CPU seeds with it.
                launch_gpu_batch(0, first_count);
                run_cpu_seeds();
                collect_gpu_batch(0, first_count);

                for (int base = first_count; base < gpu_count; base += batch_size) {
                    const int count = std::min(batch_size, gpu_count - base);
                    launch_gpu_batch(base, count);
                    collect_gpu_batch(base, count);
                }
            } else {
                run_cpu_seeds();
            }

            // Pack (cardinality, inverted seed) so that MPI_MAX implements
            // "largest cluster, smallest seed index on a tie".
            const uint64_t local_key =
                (static_cast<uint64_t>(local_best_cardinality) << 32) |
                static_cast<uint64_t>(0xFFFFFFFFu - static_cast<unsigned int>(local_best_seed));

            if (!finish_iteration(local_key)) {
                break;
            }
        } else {
            // This rank has no seeds, but it must still participate in the
            // collective operations so that the clustering state stays
            // identical on every rank.
            if (!finish_iteration(0)) {
                break;
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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

// The GNU OpenMP runtime defaults to actively spinning idle worker threads.
// While the CUDA kernels are running that would waste other CPUs of the node,
// so the process re-executes itself once with a passive wait policy. The
// environment is read by libgomp at load time, hence the re-exec.
// OpenMPI binds each rank to one physical core by default. For this hybrid
// code that would force all OpenMP workers of a rank onto a single core, so
// the affinity mask is widened to the CPUs available to the node before the
// OpenMP thread pool is created.
static void widenCpuAffinity() {
    cpu_set_t self;
    CPU_ZERO(&self);
    if (sched_getaffinity(0, sizeof(self), &self) != 0) {
        return;
    }
    int best_count = CPU_COUNT(&self);
    cpu_set_t best = self;

    const pid_t candidates[2] = {getppid(), 1};
    for (int i = 0; i < 2; ++i) {
        cpu_set_t other;
        CPU_ZERO(&other);
        if (sched_getaffinity(candidates[i], sizeof(other), &other) == 0) {
            const int count = CPU_COUNT(&other);
            if (count > best_count) {
                best_count = count;
                best = other;
            }
        }
    }

    if (best_count > CPU_COUNT(&self)) {
        sched_setaffinity(0, sizeof(best), &best);
    }
}

static void ensurePassiveOpenMP(char** argv) {
    if (getenv("QTCLUSTERING_OMP_ENV_READY") != nullptr) {
        return;
    }
    setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    setenv("GOMP_SPINCOUNT", "0", 1);
    setenv("QTCLUSTERING_OMP_ENV_READY", "1", 1);
    execv("/proc/self/exe", argv);
    // If execv is unavailable the program still runs, only with spinning
    // idle OpenMP threads.
}

int main(int argc, char** argv) {
    ensurePassiveOpenMP(argv);

    MPI_Init(&argc, &argv);

    widenCpuAffinity();

    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    // Determine how many ranks share this node and bind one GPU per local rank.
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    int node_rank_count = 1;
    if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_size(node_comm, &node_rank_count);
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA error: no usable GPU found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % device_count));

    // Give each rank on a node a fair share of the available CPU threads.
    const int cpu_threads = omp_get_num_procs();
    const int threads_per_rank = std::max(1, cpu_threads / std::max(1, node_rank_count));
    omp_set_num_threads(threads_per_rank);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, rank_count, node_rank_count);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time.count()));
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
        const double clusters_per_sec = time_sec > 0.0 ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = time_sec > 0.0 ? num_points / time_sec : 0.0;
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
    }
    
    // Validation
    bool valid = true;
    if (validate) {
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        int local_valid = valid ? 1 : 0;
        int global_valid = 1;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = (global_valid != 0);
    }

    if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
