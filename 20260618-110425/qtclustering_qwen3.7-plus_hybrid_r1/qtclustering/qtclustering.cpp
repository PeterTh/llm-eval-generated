// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// Parallelization strategy:
//   CUDA  – precompute the full NxN distance matrix on the GPU
//   MPI   – distribute candidate seed evaluation across processes
//   OpenMP – parallelize the candidate scan inside findClosestPoint

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

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ---------------------------------------------------------------------------
// CUDA kernel – compute symmetric distance matrix
// ---------------------------------------------------------------------------
__global__ void computeDistanceMatrixKernel(const double* __restrict__ px,
                                            const double* __restrict__ py,
                                            double* __restrict__ dist,
                                            int N) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N && i < j) {
        double dx = px[i] - px[j];
        double dy = py[i] - py[j];
        double d = sqrt(dx * dx + dy * dy);
        dist[i * N + j] = d;
        dist[j * N + i] = d;
    } else if (i < N && j < N && i == j) {
        dist[i * N + j] = 0.0;
    }
}

// Compute full NxN distance matrix on the GPU, return on host
static void computeDistanceMatrixGPU(const std::vector<Point>& points,
                                     std::vector<double>& dist_matrix) {
    const int N = static_cast<int>(points.size());
    dist_matrix.resize(static_cast<size_t>(N) * N);

    std::vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) { hx[i] = points[i].x; hy[i] = points[i].y; }

    double *dx, *dy, *dd;
    cudaMalloc(&dx, N * sizeof(double));
    cudaMalloc(&dy, N * sizeof(double));
    cudaMalloc(&dd, static_cast<size_t>(N) * N * sizeof(double));

    cudaMemcpy(dx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    computeDistanceMatrixKernel<<<grid, block>>>(dx, dy, dd, N);
    cudaDeviceSynchronize();

    cudaMemcpy(dist_matrix.data(), dd, static_cast<size_t>(N) * N * sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dd);
}

// ---------------------------------------------------------------------------
// Data generation (identical to original – sequential, deterministic)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// OpenMP-parallel findClosestPoint using precomputed distance matrix
// ---------------------------------------------------------------------------
static int findClosestPointOMP(const std::vector<int>& cluster_members,
                               const std::vector<bool>& clustered,
                               const std::vector<bool>& in_cluster,
                               const double* __restrict__ dist_matrix,
                               const double threshold,
                               const int point_count) {
    const int num_members = static_cast<int>(cluster_members.size());

    // Cache member indices for better access pattern
    std::vector<int> members_cache(cluster_members.begin(), cluster_members.end());

    // Per-thread results to avoid critical section contention
    const int nthreads = omp_get_max_threads();
    std::vector<double> thread_min_diam(nthreads, std::numeric_limits<double>::max());
    std::vector<int> thread_closest(nthreads, -1);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double local_min = std::numeric_limits<double>::max();
        int local_closest = -1;

        #pragma omp for nowait schedule(static)
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            const double* __restrict__ row = dist_matrix + static_cast<size_t>(candidate) * point_count;
            double max_dist = 0.0;
            for (int k = 0; k < num_members; ++k) {
                const double d = row[members_cache[k]];
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold && max_dist < local_min) {
                local_min = max_dist;
                local_closest = candidate;
            }
        }

        thread_min_diam[tid] = local_min;
        thread_closest[tid] = local_closest;
    }

    // Serial reduction over thread results
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (int t = 0; t < nthreads; ++t) {
        if (thread_min_diam[t] < min_diameter ||
            (thread_min_diam[t] == min_diameter && thread_closest[t] < closest_point)) {
            min_diameter = thread_min_diam[t];
            closest_point = thread_closest[t];
        }
    }

    return closest_point;
}

// ---------------------------------------------------------------------------
// Generate candidate cluster (uses OpenMP-parallel findClosestPoint)
// ---------------------------------------------------------------------------
static int generateCandidateClusterOMP(int seed_point,
                                       const std::vector<bool>& clustered,
                                       const double* dist_matrix,
                                       double threshold,
                                       int point_count,
                                       std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    members.reserve(64);

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPointOMP(members, clustered, in_cluster,
                                                dist_matrix, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// ---------------------------------------------------------------------------
// Main hybrid MPI + OpenMP + CUDA QT clustering
// ---------------------------------------------------------------------------
static std::vector<Cluster> qtClusteringHybrid(const std::vector<Point>& points,
                                                const double* dist_matrix,
                                                double threshold) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // --- Each MPI rank evaluates a strided subset of seeds ---------------
        int local_max_card = -1;
        int local_best_pos = -1;   // position in unclustered_indices (for tie-break)
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        for (int i = mpi_rank; i < num_seeds; i += mpi_size) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            const int card = generateCandidateClusterOMP(seed, clustered, dist_matrix,
                                                         threshold, N, &candidate_members);

            if (card > local_max_card ||
                (card == local_max_card && i < local_best_pos)) {
                local_max_card = card;
                local_best_pos = i;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // --- MPI reduction: find global best seed ----------------------------
        // Pack (cardinality, position) so ties are broken by earliest position
        // in unclustered_indices, matching the sequential algorithm.
        struct CardPos { int card; int pos; int rank; };
        CardPos local_cp = { local_max_card, local_best_pos, mpi_rank };

        // Gather all results to find the global winner
        // Use 3 separate arrays for card, pos, rank to avoid struct padding issues
        std::vector<int> all_cards(mpi_size), all_pos(mpi_size), all_ranks(mpi_size);
        MPI_Allgather(&local_cp.card, 1, MPI_INT, all_cards.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&local_cp.pos, 1, MPI_INT, all_pos.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&local_cp.rank, 1, MPI_INT, all_ranks.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Find the winner: max cardinality, then min position for ties
        CardPos winner = { -1, std::numeric_limits<int>::max(), -1 };
        for (int r = 0; r < mpi_size; ++r) {
            if (all_cards[r] > winner.card ||
                (all_cards[r] == winner.card && all_pos[r] < winner.pos)) {
                winner = { all_cards[r], all_pos[r], all_ranks[r] };
            }
        }

        int winner_rank = winner.rank;

        if (winner.card <= 0) break;

        // Broadcast best seed and cardinality from the winning rank
        int best_seed = (mpi_rank == winner_rank) ? local_best_seed : -1;
        int best_card = winner.card;
        MPI_Bcast(&best_seed, 1, MPI_INT, winner_rank, MPI_COMM_WORLD);
        MPI_Bcast(&best_card, 1, MPI_INT, winner_rank, MPI_COMM_WORLD);

        if (best_seed < 0 || best_card <= 0) break;

        // Broadcast cluster members from the winning rank
        std::vector<int> best_members(best_card);
        if (mpi_rank == winner_rank) {
            best_members = std::move(local_best_members);
        }
        MPI_Bcast(best_members.data(), best_card, MPI_INT, winner_rank, MPI_COMM_WORLD);

        // All ranks update identically
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_members);
        clusters.push_back(cluster);

        for (int m : cluster.members) clustered[m] = true;

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation (identical semantics to original)
// ---------------------------------------------------------------------------
static inline double point_distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x, dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points,
                             double threshold) {
    bool valid = true;
    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i)
            for (size_t j = i + 1; j < cluster.members.size(); ++j)
                max_diameter = std::max(max_diameter,
                    point_distance(points[cluster.members[i]], points[cluster.members[j]]));

        if (c < 10)
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }

    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i)
        if (membership[i] >= 0) clustered_count++;

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    return valid;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Set OpenMP threads to avoid CPU oversubscription
    // Use (total_cores / mpi_size) threads per rank
    int total_cores = omp_get_num_procs();
    int threads_per_rank = std::max(1, total_cores / mpi_size);
    omp_set_num_threads(threads_per_rank);

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
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
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
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI ranks: %d, OpenMP threads: %d\n", mpi_size, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (identical on all ranks)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // ---- CUDA: precompute distance matrix on GPU ---------------------------
    if (mpi_rank == 0) printf("Computing distance matrix on GPU...\n");
    auto cluster_start = std::chrono::high_resolution_clock::now();
    auto gpu_start = std::chrono::high_resolution_clock::now();

    // Select GPU based on MPI rank (one rank per GPU)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        cudaSetDevice(mpi_rank % num_gpus);
    }

    std::vector<double> dist_matrix;
    computeDistanceMatrixGPU(points, dist_matrix);

    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
    if (mpi_rank == 0) printf("GPU distance matrix time: %ld ms\n", gpu_time.count());

    // ---- Hybrid MPI+OpenMP clustering --------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    const std::vector<Cluster> clusters = qtClusteringHybrid(points, dist_matrix.data(), threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    const double local_cluster_time_ms =
        std::chrono::duration<double, std::milli>(cluster_end - cluster_start).count();
    double cluster_time_ms = 0.0;
    MPI_Reduce(&local_cluster_time_ms, &cluster_time_ms, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Clustering time: %.3f ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0, max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }

        const double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = cluster_time_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && mpi_rank == 0) {
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

    // Validation (rank 0 only)
    if (validate) {
        bool valid = false;
        if (mpi_rank == 0) {
            valid = validateClusters(clusters, points, threshold);
        }
        // Broadcast validation result
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
