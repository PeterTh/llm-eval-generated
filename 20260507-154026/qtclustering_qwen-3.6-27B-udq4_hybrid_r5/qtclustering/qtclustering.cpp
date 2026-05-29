// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// Parallelization strategy:
//   - MPI:  Distribute seed candidate evaluation across MPI ranks.
//           Each rank holds a full copy of the data; seeds are round-robin
//           distributed. After evaluation, an Allgather finds the global
//           best cluster, and a Bcast distributes the winning members.
//   - OpenMP: Parallelize seed evaluation within each rank. Each thread
//             gets its own CUDA buffers so GPU calls are independent.
//   - CUDA:   Accelerate the hot-path distance computation. A single kernel
//             computes the max distance from every candidate to all cluster
//             members in parallel.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH  = 20.0;
static const double MAX_HEIGHT = 20.0;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Per-rank local best result for MPI reduction
struct LocalResult {
    int cardinality;
    int seed;
};

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernel: for every candidate compute max distance to all members
// ---------------------------------------------------------------------------
__global__ void maxDistKernel(
    const double* __restrict__ cand_x,
    const double* __restrict__ cand_y,
    const double* __restrict__ mem_x,
    const double* __restrict__ mem_y,
    int nc, int nm,
    double* __restrict__ out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;

    double cx = cand_x[c];
    double cy = cand_y[c];
    double md = 0.0;

    for (int m = 0; m < nm; ++m) {
        double dx = cx - mem_x[m];
        double dy = cy - mem_y[m];
        double d  = sqrt(dx * dx + dy * dy);
        if (d > md) md = d;
    }

    out[c] = md;
}

// ---------------------------------------------------------------------------
// Data generation (identical to original for deterministic results)
// ---------------------------------------------------------------------------
void generateSyntheticData(std::vector<Point>& points, const int N,
                           unsigned int seed = 42)
{
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x    = frand() * MAX_WIDTH;
        const double cntr_y    = frand() * MAX_HEIGHT;
        const double R         = frand() * min_dim / 2.0;
        int group_cnt          = static_cast<int>(frand() * (N / 30.0));

        if (group_cnt > (N - count))
            group_cnt = N - count;

        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r    = frand() * R;
            const double dx   = (2.0 * frand() - 1.0) * r;
            const double dy   = std::sqrt(r * r - dx * dx) * sign;
            const double x    = cntr_x + dx;
            const double y    = cntr_y + dy;

            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT)
                continue;

            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA-accelerated findClosestPoint
// ---------------------------------------------------------------------------
static int findClosestPointCUDA(
    const std::vector<int>& cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    const std::vector<Point>& points,
    const double threshold,
    const int point_count,
    double* d_cx, double* d_cy,
    double* d_mx, double* d_my,
    double* d_out,
    double* h_cx, double* h_cy,
    double* h_mx, double* h_my,
    double* h_out)
{
    // Collect candidates
    std::vector<int> candidates;
    candidates.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        if (!clustered[i] && !in_cluster[i])
            candidates.push_back(i);
    }
    if (candidates.empty()) return -1;

    const int nc = static_cast<int>(candidates.size());
    const int nm = static_cast<int>(cluster_members.size());

    // Stage coordinates into pinned host memory
    for (int i = 0; i < nc; ++i) {
        h_cx[i] = points[candidates[i]].x;
        h_cy[i] = points[candidates[i]].y;
    }
    for (int i = 0; i < nm; ++i) {
        h_mx[i] = points[cluster_members[i]].x;
        h_my[i] = points[cluster_members[i]].y;
    }

    // Copy to device (pinned host -> device is fast)
    CUDA_CHECK(cudaMemcpy(d_cx, h_cx, nc * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, h_cy, nc * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx, h_mx, nm * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my, h_my, nm * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Launch kernel
    const int bs = std::min(256, nc);
    const int nb = (nc + bs - 1) / bs;
    maxDistKernel<<<nb, bs>>>(d_cx, d_cy, d_mx, d_my, nc, nm, d_out);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nc * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Scan for best candidate (smallest max_diameter that is < threshold)
    int closest_point = -1;
    double min_diam   = std::numeric_limits<double>::max();
    for (int i = 0; i < nc; ++i) {
        const double md = h_out[i];
        if (md < threshold && md < min_diam) {
            min_diam      = md;
            closest_point = candidates[i];
        }
    }
    return closest_point;
}

// ---------------------------------------------------------------------------
// CUDA-accelerated generateCandidateCluster
// ---------------------------------------------------------------------------
static int generateCandidateClusterCUDA(
    const int seed_point,
    const std::vector<bool>& clustered,
    const std::vector<Point>& points,
    const double threshold,
    const int point_count,
    double* d_cx, double* d_cy,
    double* d_mx, double* d_my,
    double* d_out,
    double* h_cx, double* h_cy,
    double* h_mx, double* h_my,
    double* h_out,
    std::vector<int>* cluster_members = nullptr)
{
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int>  members;
    members.reserve(point_count);

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPointCUDA(
            members, clustered, in_cluster, points, threshold, point_count,
            d_cx, d_cy, d_mx, d_my, d_out,
            h_cx, h_cy, h_mx, h_my, h_out);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// ---------------------------------------------------------------------------
// Main QT clustering – MPI + OpenMP + CUDA
// ---------------------------------------------------------------------------
static std::vector<Cluster> qtClustering(
    const std::vector<Point>& points,
    const double threshold,
    int rank, int num_ranks)
{
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int>  unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    std::vector<Cluster> clusters;

    while (!unclustered_indices.empty()) {
        // --- distribute seeds round-robin across ranks ---
        std::vector<int> local_seeds;
        local_seeds.reserve(unclustered_indices.size() / num_ranks + 1);
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            if (static_cast<int>(i) % num_ranks == rank)
                local_seeds.push_back(unclustered_indices[i]);
        }

        LocalResult local_best{-1, -1};
        std::vector<int> local_best_members;

        // --- OpenMP: each thread evaluates a subset of seeds ---
        #pragma omp parallel
        {
            // Per-thread CUDA device buffers
            double *td_cx, *td_cy, *td_mx, *td_my, *td_out;
            CUDA_CHECK(cudaMalloc(&td_cx, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&td_cy, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&td_mx, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&td_my, N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&td_out, N * sizeof(double)));

            // Per-thread pinned host buffers
            double *th_cx, *th_cy, *th_mx, *th_my, *th_out;
            CUDA_CHECK(cudaMallocHost(&th_cx, N * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&th_cy, N * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&th_mx, N * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&th_my, N * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&th_out, N * sizeof(double)));

            // Thread-local best
            LocalResult thread_best{-1, -1};
            std::vector<int> thread_best_members;

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < static_cast<int>(local_seeds.size()); ++i) {
                const int seed = local_seeds[i];
                if (clustered[seed]) continue;

                std::vector<int> members;
                const int card = generateCandidateClusterCUDA(
                    seed, clustered, points, threshold, N,
                    td_cx, td_cy, td_mx, td_my, td_out,
                    th_cx, th_cy, th_mx, th_my, th_out,
                    &members);

                // Update thread-local best (no synchronisation needed)
                if (card > thread_best.cardinality ||
                    (card == thread_best.cardinality &&
                     (thread_best.seed < 0 || seed < thread_best.seed))) {
                    thread_best.cardinality    = card;
                    thread_best.seed           = seed;
                    thread_best_members        = std::move(members);
                }
            }

            // Merge into rank-local best (one critical entry per thread)
            #pragma omp critical
            {
                if (thread_best.cardinality > local_best.cardinality ||
                    (thread_best.cardinality == local_best.cardinality &&
                     (local_best.seed < 0 || thread_best.seed < local_best.seed))) {
                    local_best.cardinality     = thread_best.cardinality;
                    local_best.seed            = thread_best.seed;
                    local_best_members         = std::move(thread_best_members);
                }
            }

            CUDA_CHECK(cudaFree(td_cx));
            CUDA_CHECK(cudaFree(td_cy));
            CUDA_CHECK(cudaFree(td_mx));
            CUDA_CHECK(cudaFree(td_my));
            CUDA_CHECK(cudaFree(td_out));
            CUDA_CHECK(cudaFreeHost(th_cx));
            CUDA_CHECK(cudaFreeHost(th_cy));
            CUDA_CHECK(cudaFreeHost(th_mx));
            CUDA_CHECK(cudaFreeHost(th_my));
            CUDA_CHECK(cudaFreeHost(th_out));
        } // end omp parallel

        // --- MPI: find global best across ranks ---
        std::vector<LocalResult> all_results(num_ranks);
        MPI_Allgather(&local_best, 2, MPI_INT,
                      all_results.data(), 2, MPI_INT, MPI_COMM_WORLD);

        int global_max_card = -1;
        int global_best_seed = -1;
        int root_rank = -1;
        for (int r = 0; r < num_ranks; ++r) {
            if (all_results[r].cardinality > global_max_card ||
                (all_results[r].cardinality == global_max_card &&
                 (global_best_seed < 0 ||
                  all_results[r].seed < global_best_seed))) {
                global_max_card = all_results[r].cardinality;
                global_best_seed = all_results[r].seed;
                root_rank = r;
            }
        }

        if (global_best_seed < 0 || global_max_card <= 0) break;

        // Broadcast winning cluster members from root rank
        int member_count = 0;
        if (rank == root_rank)
            member_count = static_cast<int>(local_best_members.size());
        MPI_Bcast(&member_count, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

        std::vector<int> best_members(member_count);
        if (rank == root_rank)
            best_members = local_best_members;
        MPI_Bcast(best_members.data(), member_count, MPI_INT,
                  root_rank, MPI_COMM_WORLD);

        // Record cluster
        Cluster c;
        c.seed_point = global_best_seed;
        c.members    = std::move(best_members);
        clusters.push_back(std::move(c));

        // Update state on every rank (use clusters.back().members)
        for (int m : clusters.back().members)
            clustered[m] = true;
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(),
                           unclustered_indices.end(),
                           [&clustered](int idx){ return clustered[idx]; }),
            unclustered_indices.end());
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation (OpenMP-parallelised diameter check)
// ---------------------------------------------------------------------------
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold)
{
    bool valid = true;

    printf("Validating clusters:\n");

    int error_count = 0;
    #pragma omp parallel for schedule(static) reduction(+:error_count)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cl = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cl.members.size(); ++i) {
            for (size_t j = i + 1; j < cl.members.size(); ++j) {
                const double d = distance(points[cl.members[i]],
                                          points[cl.members[j]]);
                if (d > max_diameter) max_diameter = d;
            }
        }

        if (c < 10) {
            #pragma omp critical
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cl.members.size(), cl.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001)
            error_count++;
    }

    if (error_count > 0) {
        printf("ERROR: %d clusters have diameter > threshold\n", error_count);
        valid = false;
    }

    // Check for duplicate memberships (sequential – inherently ordered)
    std::vector<int> membership(points.size(), -1);
    int dup_count = 0;
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int m = clusters[c].members[i];
            if (membership[m] >= 0) {
                dup_count++;
                continue;
            }
            membership[m] = static_cast<int>(c);
        }
    }
    if (dup_count > 0) {
        printf("ERROR: %d points appear in multiple clusters\n", dup_count);
        valid = false;
    }

    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i)
        if (membership[i] >= 0) clustered_count++;

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count,
           points.size() - static_cast<size_t>(clustered_count));

    return valid;
}

// ---------------------------------------------------------------------------
// Usage and main
// ---------------------------------------------------------------------------
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Initialise CUDA on every rank
    CUDA_CHECK(cudaSetDevice(0));

    int   num_points   = 1000;
    double threshold   = 2.0;
    int   validate     = 0;
    int   printResults = 0;

    // Parse arguments on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                num_points = atoi(argv[++i]);
            else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
                threshold = atof(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)
                validate = 1;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = 1;
            else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters
    MPI_Bcast(&num_points,   1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold,    1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT,    0, MPI_COMM_WORLD);

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("MPI ranks: %d\n", num_ranks);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate data on rank 0, broadcast to all ranks
    std::vector<Point> points(num_points);
    if (rank == 0)
        generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- clustering ----
    auto t0 = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, num_ranks);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    // ---- results (rank 0 only) ----
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            if (sz > max_cluster_size) max_cluster_size = sz;
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
        const double points_per_sec   = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (size_t i = 0; i < clusters[c].members.size(); ++i)
                    membership[clusters[c].members[i]] = static_cast<int>(c);
            for (int m : membership)
                membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool ok = validateClusters(clusters, points, threshold);
            if (ok) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
