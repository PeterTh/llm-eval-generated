// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Version
//
// Parallelization strategy:
// - MPI: Distribute independent seed evaluations across ranks
// - CUDA: Accelerate distance computation (findClosestPoint) on GPU
// - OpenMP: Parallelize validation and statistics computation

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

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err));         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

// ============================================================
// Data Generation (sequential, deterministic - unchanged)
// ============================================================

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ============================================================
// CUDA Kernel: compute max distance from each candidate to cluster
// ============================================================

__global__ void computeMaxDistances(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ members,
    int cluster_size,
    const int* __restrict__ candidates,
    int num_candidates,
    double* __restrict__ max_dists)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    int candidate = candidates[idx];
    double max_dist = 0.0;

    for (int i = 0; i < cluster_size; ++i) {
        int member = members[i];
        double dx = px[candidate] - px[member];
        double dy = py[candidate] - py[member];
        double dist = sqrt(dx * dx + dy * dy);
        if (dist > max_dist) max_dist = dist;
    }

    max_dists[idx] = max_dist;
}

// ============================================================
// GPU Context - pre-allocated device memory
// ============================================================

static double* d_px = nullptr;
static double* d_py = nullptr;
static int*    d_members = nullptr;
static int*    d_candidates = nullptr;
static double* d_max_dists = nullptr;

static void initGPU(int N) {
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidates, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_dists, N * sizeof(double)));
}

static void cleanupGPU() {
    if (d_px)         { CUDA_CHECK(cudaFree(d_px));         d_px = nullptr; }
    if (d_py)         { CUDA_CHECK(cudaFree(d_py));         d_py = nullptr; }
    if (d_members)    { CUDA_CHECK(cudaFree(d_members));    d_members = nullptr; }
    if (d_candidates) { CUDA_CHECK(cudaFree(d_candidates)); d_candidates = nullptr; }
    if (d_max_dists)  { CUDA_CHECK(cudaFree(d_max_dists));  d_max_dists = nullptr; }
}

static void uploadPoints(const std::vector<Point>& points, int N) {
    // Convert AoS -> SoA for coalesced GPU memory access
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
}

// ============================================================
// GPU-accelerated findClosestPoint
// ============================================================

static int findClosestPointGPU(
    const std::vector<int>& cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    const double threshold,
    int N)
{
    // Build candidate list (preserving index order for deterministic tie-breaking)
    std::vector<int> candidates;
    candidates.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (!clustered[i] && !in_cluster[i]) {
            candidates.push_back(i);
        }
    }

    if (candidates.empty()) return -1;

    int num_candidates = static_cast<int>(candidates.size());
    int cluster_size   = static_cast<int>(cluster_members.size());

    // Upload variable data to GPU
    CUDA_CHECK(cudaMemcpy(d_members,    cluster_members.data(),
                          cluster_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(),
                          num_candidates * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (num_candidates + blockSize - 1) / blockSize;
    computeMaxDistances<<<gridSize, blockSize>>>(
        d_px, d_py,
        d_members, cluster_size,
        d_candidates, num_candidates,
        d_max_dists);

    // Download results (cudaMemcpy is synchronous)
    std::vector<double> max_dists(num_candidates);
    CUDA_CHECK(cudaMemcpy(max_dists.data(), d_max_dists,
                          num_candidates * sizeof(double), cudaMemcpyDeviceToHost));

    // Find best candidate (same logic as original: strictly less than min_diameter)
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int i = 0; i < num_candidates; ++i) {
        if (max_dists[i] < threshold && max_dists[i] < min_diameter) {
            min_diameter = max_dists[i];
            closest_point = candidates[i];
        }
    }

    return closest_point;
}

// ============================================================
// GPU-accelerated generateCandidateCluster
// ============================================================

static int generateCandidateClusterGPU(
    int seed_point,
    const std::vector<bool>& clustered,
    const double threshold,
    int N,
    std::vector<int>* cluster_members = nullptr)
{
    std::vector<bool> in_cluster(N, false);
    std::vector<int>  members;
    members.reserve(N);

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < N) {
        int closest = findClosestPointGPU(members, clustered, in_cluster, threshold, N);
        if (closest < 0) break;

        in_cluster[closest] = true;
        members.push_back(closest);
    }

    int cardinality = static_cast<int>(members.size());

    if (cluster_members) {
        *cluster_members = std::move(members);
    }

    return cardinality;
}

// ============================================================
// MPI-distributed QT Clustering
// ============================================================

static std::vector<Cluster> qtClusteringHybrid(
    const std::vector<Point>& points,
    const double threshold,
    int rank, int size)
{
    int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int>  unclustered_indices(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    std::vector<Cluster> clusters;

    // Upload points to GPU once (each rank independently)
    uploadPoints(points, N);

    while (true) {
        int num_unclustered = static_cast<int>(unclustered_indices.size());

        // Synchronize unclustered count across ranks
        int global_count;
        if (rank == 0) global_count = num_unclustered;
        MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (global_count == 0) break;

        // Distribute seeds: rank r handles indices [start, next_start)
        int my_start   = (rank * global_count) / size;
        int next_start = ((rank + 1) * global_count) / size;

        // Evaluate my subset of seeds
        int local_best_card  = -1;
        int local_best_seed  = -1;
        std::vector<int> local_best_members;

        for (int i = my_start; i < next_start; ++i) {
            int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            int cardinality = generateCandidateClusterGPU(
                seed, clustered, threshold, N, &candidate_members);

            // Tie-breaking: prefer higher cardinality, then lower seed index
            if (cardinality > local_best_card ||
                (cardinality == local_best_card && seed < local_best_seed)) {
                local_best_card    = cardinality;
                local_best_seed    = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // Gather (cardinality, seed, member_count) from all ranks to rank 0
        int send_buf[3] = {local_best_card, local_best_seed,
                           static_cast<int>(local_best_members.size())};
        std::vector<int> recv_buf(size * 3);
        MPI_Gather(send_buf, 3, MPI_INT, recv_buf.data(), 3, MPI_INT,
                   0, MPI_COMM_WORLD);

        // Rank 0 picks global best
        int global_best_card  = -1;
        int global_best_seed  = -1;
        int best_rank         = -1;
        int best_member_count = 0;

        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                int card = recv_buf[r * 3];
                int seed = recv_buf[r * 3 + 1];
                int mc   = recv_buf[r * 3 + 2];
                if (card > global_best_card ||
                    (card == global_best_card && seed < global_best_seed)) {
                    global_best_card  = card;
                    global_best_seed  = seed;
                    best_rank         = r;
                    best_member_count = mc;
                }
            }
        }

        // Broadcast best seed, best rank, best card, and member count
        MPI_Bcast(&global_best_card,  1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_best_seed,  1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&best_rank,         1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&best_member_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (global_best_seed < 0 || global_best_card <= 0) break;

        // Broadcast cluster members from the best rank to all ranks
        std::vector<int> cluster_members(best_member_count);
        if (rank == best_rank) {
            cluster_members = local_best_members;
        }
        MPI_Bcast(cluster_members.data(), best_member_count, MPI_INT,
                  best_rank, MPI_COMM_WORLD);

        // All ranks update clustered[] and unclustered_indices identically
        for (int m : cluster_members) {
            clustered[m] = true;
        }
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());

        // Only rank 0 records the cluster for output
        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = global_best_seed;
            cluster.members = cluster_members;
            clusters.push_back(cluster);
        }
    }

    return clusters;
}

// ============================================================
// OpenMP-accelerated Validation
// ============================================================

static bool validateClusters(
    const std::vector<Cluster>& clusters,
    const std::vector<Point>& points,
    const double threshold)
{
    bool valid = true;
    printf("Validating clusters:\n");

    size_t num_clusters = clusters.size();

    // Compute diameters in parallel
    std::vector<double> diameters(num_clusters, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < num_clusters; ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(
                    points[cluster.members[i]],
                    points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        diameters[c] = max_diameter;
    }

    // Print first 10 clusters (sequential)
    for (size_t c = 0; c < std::min(num_clusters, (size_t)10); ++c) {
        printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
               c, clusters[c].members.size(), clusters[c].seed_point,
               diameters[c]);
    }

    // Check diameter validity in parallel
    #pragma omp parallel for reduction(||:valid)
    for (size_t c = 0; c < num_clusters; ++c) {
        if (diameters[c] > threshold * 1.001) {
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < num_clusters; ++c) {
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

    // Count clustered points in parallel
    int clustered_count = 0;
    #pragma omp parallel for reduction(+:clustered_count)
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count,
           points.size() - static_cast<size_t>(clustered_count));

    return valid;
}

// ============================================================
// Main
// ============================================================

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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assign GPU device per rank (round-robin for multi-GPU)
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count > 1) {
        CUDA_CHECK(cudaSetDevice(rank % device_count));
    }

    int num_points = 1000;
    double threshold = 2.0;
    int validate     = 0;
    int printResults = 0;

    // Parse arguments on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                num_points = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                threshold = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
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

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&num_points,   1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold,    1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT,    0, MPI_COMM_WORLD);

    // Generate data on rank 0, then broadcast to all ranks
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(reinterpret_cast<double*>(points.data()),
              num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print header (rank 0 only)
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    // Initialize GPU on each rank
    initGPU(num_points);

    // Perform QT clustering (MPI + CUDA hybrid)
    auto cluster_start = std::chrono::high_resolution_clock::now();
    std::vector<Cluster> clusters = qtClusteringHybrid(points, threshold, rank, size);
    auto cluster_end   = std::chrono::high_resolution_clock::now();
    auto cluster_time  = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    // Statistics (rank 0 only, with OpenMP)
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;

        #pragma omp parallel for reduction(+:total_clustered) reduction(max:max_cluster_size)
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
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
        const double points_per_sec   = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
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

    // Validation (rank 0 only, with OpenMP)
    if (validate && rank == 0) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
            cleanupGPU();
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            cleanupGPU();
            MPI_Finalize();
            return 1;
        }
    }

    cleanupGPU();
    MPI_Finalize();

    return 0;
}
