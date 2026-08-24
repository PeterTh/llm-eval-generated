// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Version
//
// Parallelization strategy:
//   MPI:    distribute seed evaluations across ranks
//   OpenMP: parallelize seed evaluations within each rank
//   CUDA:   accelerate findClosestPoint distance computations on GPU

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

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (group_cnt > (N - count)) group_cnt = N - count;
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
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

// CUDA kernel: for each candidate point, compute max distance to cluster members
__global__ void computeMaxDistsKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ members,
    int num_members,
    const int* __restrict__ skip,
    double* __restrict__ max_dists,
    int N,
    double threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (skip[tid]) { max_dists[tid] = 1e308; return; }

    double cx = px[tid], cy = py[tid];
    double md = 0.0;
    for (int i = 0; i < num_members; i++) {
        int m = members[i];
        double dx = cx - px[m];
        double dy = cy - py[m];
        double d = sqrt(dx * dx + dy * dy);
        if (d > md) md = d;
        if (md >= threshold) break;
    }
    max_dists[tid] = md;
}

// Per-OpenMP-thread GPU resources
struct ThreadGPUResources {
    int* d_members;
    int* d_skip;
    double* d_max_dists;
    double* h_max_dists; // pinned host memory for async transfer
    cudaStream_t stream;
};

// GPU-accelerated findClosestPoint
int findClosestPointGPU(
    const double* d_px, const double* d_py,
    ThreadGPUResources& res,
    const std::vector<int>& members,
    const std::vector<int>& skip,
    int N, double threshold)
{
    int nm = static_cast<int>(members.size());
    cudaMemcpyAsync(res.d_members, members.data(), nm * sizeof(int),
                    cudaMemcpyHostToDevice, res.stream);
    cudaMemcpyAsync(res.d_skip, skip.data(), N * sizeof(int),
                    cudaMemcpyHostToDevice, res.stream);

    int bs = 256;
    int gs = (N + bs - 1) / bs;
    computeMaxDistsKernel<<<gs, bs, 0, res.stream>>>(
        d_px, d_py, res.d_members, nm, res.d_skip, res.d_max_dists, N, threshold);

    cudaMemcpyAsync(res.h_max_dists, res.d_max_dists, N * sizeof(double),
                    cudaMemcpyDeviceToHost, res.stream);
    cudaStreamSynchronize(res.stream);

    int closest = -1;
    double min_d = 1e308;
    for (int i = 0; i < N; i++) {
        double d = res.h_max_dists[i];
        if (d < threshold && d < min_d) { min_d = d; closest = i; }
    }
    return closest;
}

// Build candidate cluster for a seed using GPU-accelerated closest point search
int generateCandidateClusterGPU(
    int seed, const std::vector<int>& clustered_int,
    const double* d_px, const double* d_py,
    ThreadGPUResources& res,
    double threshold, int N,
    std::vector<int>* out_members)
{
    std::vector<int> skip(clustered_int);
    std::vector<int> members;
    skip[seed] = 1;
    members.push_back(seed);

    while (static_cast<int>(members.size()) < N) {
        int c = findClosestPointGPU(d_px, d_py, res, members, skip, N, threshold);
        if (c < 0) break;
        skip[c] = 1;
        members.push_back(c);
    }

    if (out_members) *out_members = members;
    return static_cast<int>(members.size());
}

// Main QT clustering with hybrid MPI+OpenMP+CUDA
std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                   int mpi_rank, int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<int> clustered_int(N, 0);
    std::vector<int> unclustered;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; i++) unclustered.push_back(i);

    // Upload points to GPU in SoA layout
    std::vector<double> h_px(N), h_py(N);
    for (int i = 0; i < N; i++) { h_px[i] = points[i].x; h_py[i] = points[i].y; }

    double *d_px, *d_py;
    cudaMalloc(&d_px, N * sizeof(double));
    cudaMalloc(&d_py, N * sizeof(double));
    cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate per-OpenMP-thread GPU resources
    int num_threads = omp_get_max_threads();
    std::vector<ThreadGPUResources> tres(num_threads);
    for (int t = 0; t < num_threads; t++) {
        cudaMalloc(&tres[t].d_members, N * sizeof(int));
        cudaMalloc(&tres[t].d_skip, N * sizeof(int));
        cudaMalloc(&tres[t].d_max_dists, N * sizeof(double));
        cudaMallocHost(&tres[t].h_max_dists, N * sizeof(double));
        cudaStreamCreate(&tres[t].stream);
    }

    while (!unclustered.empty()) {
        int num_seeds = static_cast<int>(unclustered.size());
        int per_rank = (num_seeds + mpi_size - 1) / mpi_size;
        int my_start = std::min(mpi_rank * per_rank, num_seeds);
        int my_end = std::min(my_start + per_rank, num_seeds);

        // Thread-local best candidates
        std::vector<int> t_card(num_threads, -1);
        std::vector<int> t_seed(num_threads, N);
        std::vector<std::vector<int>> t_members(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic)
            for (int i = my_start; i < my_end; i++) {
                int seed = unclustered[i];
                if (clustered_int[seed]) continue;

                std::vector<int> cand;
                int card = generateCandidateClusterGPU(
                    seed, clustered_int, d_px, d_py,
                    tres[tid], threshold, N, &cand);

                if (card > t_card[tid] ||
                    (card == t_card[tid] && seed < t_seed[tid])) {
                    t_card[tid] = card;
                    t_seed[tid] = seed;
                    t_members[tid] = std::move(cand);
                }
            }
        }

        // Reduce across threads to find rank-local best
        int local_card = -1, local_seed = N;
        std::vector<int> local_members;
        for (int t = 0; t < num_threads; t++) {
            if (t_card[t] > local_card ||
                (t_card[t] == local_card && t_seed[t] < local_seed)) {
                local_card = t_card[t];
                local_seed = t_seed[t];
                local_members = std::move(t_members[t]);
            }
        }

        // MPI global reduction: encode (cardinality, seed) into a single score
        // Higher score = higher cardinality, with ties broken by lower seed index
        long long local_score = (local_card > 0)
            ? (long long)local_card * N + (N - 1 - local_seed) : -1;
        long long global_score;
        MPI_Allreduce(&local_score, &global_score, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

        if (global_score < 0) break;

        // Determine which rank holds the winning cluster
        int winner = (local_score == global_score) ? mpi_rank : mpi_size;
        int global_winner;
        MPI_Allreduce(&winner, &global_winner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        int best_seed = N - 1 - static_cast<int>(global_score % N);

        // Broadcast winning cluster members from winner rank
        int nm;
        if (mpi_rank == global_winner) nm = static_cast<int>(local_members.size());
        MPI_Bcast(&nm, 1, MPI_INT, global_winner, MPI_COMM_WORLD);

        std::vector<int> best_members(nm);
        if (mpi_rank == global_winner) best_members = local_members;
        MPI_Bcast(best_members.data(), nm, MPI_INT, global_winner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int m : best_members) clustered_int[m] = 1;
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                          [&clustered_int](int idx) { return clustered_int[idx]; }),
            unclustered.end());
    }

    // Cleanup GPU resources
    for (int t = 0; t < num_threads; t++) {
        cudaFree(tres[t].d_members);
        cudaFree(tres[t].d_skip);
        cudaFree(tres[t].d_max_dists);
        cudaFreeHost(tres[t].h_max_dists);
        cudaStreamDestroy(tres[t].stream);
    }
    cudaFree(d_px);
    cudaFree(d_py);

    return clusters;
}

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
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", mpi_size, omp_get_max_threads());
    }

    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long long local_cluster_time_ms = static_cast<long long>(cluster_time.count());
    long long global_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &global_cluster_time_ms, 1,
               MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Clustering time: %lld ms\n", global_cluster_time_ms);
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

        const double time_sec = global_cluster_time_ms / 1000.0;
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
