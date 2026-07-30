// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   MPI     - Distribute seed evaluation across processes
//   OpenMP  - Parallelize seed evaluation within each MPI rank (master
//             thread uses GPU, worker threads use CPU)
//   CUDA    - Accelerate distance computations in findClosestPoint on GPU

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#ifndef __CUDACC__
#error "This file must be compiled with a CUDA compiler (nvcc)"
#endif

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err_ = call;                                                  \
    if (err_ != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %d: %s at %s:%d\n",                      \
                (int)err_, cudaGetErrorString(err_), __FILE__, __LINE__);     \
        MPI_Abort(MPI_COMM_WORLD, 1);                                         \
    }                                                                         \
} while(0)

static const double MAX_WIDTH  = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point { double x, y; };

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// =============================================================================
// CUDA kernel: compute max distance from each candidate to all cluster members
// =============================================================================
__global__ void computeMaxDistsKernel(
    const double* points_x,
    const double* points_y,
    int            point_count,
    const int*     cluster_members,
    int            cluster_size,
    const char*    clustered,
    const char*    in_cluster,
    double*        max_dists_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;

    // Skip points already globally clustered or already in this cluster
    if (clustered[idx] || in_cluster[idx]) {
        max_dists_out[idx] = -1.0;   // sentinel value
        return;
    }

    double max_d = 0.0;
    for (int i = 0; i < cluster_size; ++i) {
        int m = cluster_members[i];
        double dx = points_x[idx] - points_x[m];
        double dy = points_y[idx] - points_y[m];
        double d = sqrt(dx * dx + dy * dy);
        if (d > max_d) max_d = d;
    }
    max_dists_out[idx] = max_d;
}

// =============================================================================
// Persistent GPU device pointers
// =============================================================================
static double* d_points_x       = nullptr;
static double* d_points_y       = nullptr;
static char*   d_clustered      = nullptr;
static char*   d_in_cluster     = nullptr;
static double* d_max_dists      = nullptr;
static int*    d_cluster_members = nullptr;
static int     gpu_allocated_n  = 0;

static void allocateGPUMemory(int n) {
    if (gpu_allocated_n >= n) return;
    if (gpu_allocated_n > 0) {
        CUDA_CHECK(cudaFree(d_points_x));
        CUDA_CHECK(cudaFree(d_points_y));
        CUDA_CHECK(cudaFree(d_clustered));
        CUDA_CHECK(cudaFree(d_in_cluster));
        CUDA_CHECK(cudaFree(d_max_dists));
        CUDA_CHECK(cudaFree(d_cluster_members));
    }
    CUDA_CHECK(cudaMalloc(&d_points_x,        n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_points_y,        n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered,       n * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster,      n * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_max_dists,       n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cluster_members, n * sizeof(int)));
    gpu_allocated_n = n;
}

static void uploadPointsToGPU(const std::vector<Point>& points) {
    int n = static_cast<int>(points.size());
    std::vector<double> h_x(n), h_y(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = points[i].x;
        h_y[i] = points[i].y;
    }
    CUDA_CHECK(cudaMemcpy(d_points_x, h_x.data(),
                           n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_y, h_y.data(),
                           n * sizeof(double), cudaMemcpyHostToDevice));
}

static void freeGPUMemory() {
    if (gpu_allocated_n > 0) {
        cudaFree(d_points_x);       d_points_x       = nullptr;
        cudaFree(d_points_y);       d_points_y       = nullptr;
        cudaFree(d_clustered);      d_clustered      = nullptr;
        cudaFree(d_in_cluster);     d_in_cluster     = nullptr;
        cudaFree(d_max_dists);      d_max_dists      = nullptr;
        cudaFree(d_cluster_members); d_cluster_members = nullptr;
        gpu_allocated_n = 0;
    }
}

// =============================================================================
// Host distance utility
// =============================================================================
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// =============================================================================
// GPU-accelerated findClosestPoint
// =============================================================================
static int findClosestPointGPU(
    const std::vector<int>& cluster_members,
    const char*             h_in_cluster,
    double                  threshold,
    int                     point_count)
{
    int cluster_size = static_cast<int>(cluster_members.size());

    // Copy cluster members to device
    CUDA_CHECK(cudaMemcpy(d_cluster_members, cluster_members.data(),
                           cluster_size * sizeof(int), cudaMemcpyHostToDevice));

    // Copy in_cluster array to device
    CUDA_CHECK(cudaMemcpy(d_in_cluster, h_in_cluster,
                           point_count * sizeof(char), cudaMemcpyHostToDevice));

    // Launch kernel
    const int blockSize = 256;
    int gridSize = (point_count + blockSize - 1) / blockSize;
    computeMaxDistsKernel<<<gridSize, blockSize>>>(
        d_points_x, d_points_y, point_count,
        d_cluster_members, cluster_size,
        d_clustered, d_in_cluster, d_max_dists);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<double> h_max_dists(point_count);
    CUDA_CHECK(cudaMemcpy(h_max_dists.data(), d_max_dists,
                           point_count * sizeof(double), cudaMemcpyDeviceToHost));

    // Find the best candidate (minimizes max distance, below threshold)
    int    best     = -1;
    double bestDist = threshold;
    for (int i = 0; i < point_count; ++i) {
        double d = h_max_dists[i];
        if (d >= 0.0 && d < bestDist) {
            bestDist = d;
            best     = i;
        }
    }
    return best;
}

// =============================================================================
// GPU-accelerated generateCandidateCluster
// =============================================================================
static int generateCandidateClusterGPU(
    const int                seed_point,
    const std::vector<Point>& /* points */,
    double                   threshold,
    int                      point_count,
    std::vector<int>*        cluster_members_ptr)
{
    std::vector<char> in_cluster(point_count, 0);
    std::vector<int>  members;
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        int closest = findClosestPointGPU(members,
                                           in_cluster.data(),
                                           threshold, point_count);
        if (closest < 0) break;

        in_cluster[closest] = 1;
        members.push_back(closest);
    }

    int card = static_cast<int>(members.size());
    if (cluster_members_ptr)
        *cluster_members_ptr = std::move(members);

    return card;
}

// =============================================================================
// CPU-only findClosestPoint  (called from worker OpenMP threads)
// =============================================================================
static int findClosestPointCPU(
    const std::vector<int>&    cluster_members,
    const std::vector<char>&   clustered,
    const std::vector<char>&   in_cluster,
    const std::vector<Point>&  points,
    double                     threshold,
    int                        point_count)
{
    int    best     = -1;
    double bestDist = threshold;

    for (int c = 0; c < point_count; ++c) {
        if (clustered[c] || in_cluster[c]) continue;

        double max_d = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            int m = cluster_members[i];
            max_d = std::max(max_d, distance(points[c], points[m]));
        }
        if (max_d < bestDist) {
            bestDist = max_d;
            best     = c;
        }
    }
    return best;
}

// =============================================================================
// CPU-only generateCandidateCluster  (called from worker OpenMP threads)
// =============================================================================
static int generateCandidateClusterCPU(
    const int                seed_point,
    const std::vector<char>& clustered,
    const std::vector<Point>& points,
    double                   threshold,
    int                      point_count,
    std::vector<int>*        cluster_members_ptr)
{
    std::vector<char> in_cluster(point_count, 0);
    std::vector<int>  members;
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        int closest = findClosestPointCPU(members, clustered, in_cluster,
                                           points, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = 1;
        members.push_back(closest);
    }

    int card = static_cast<int>(members.size());
    if (cluster_members_ptr)
        *cluster_members_ptr = std::move(members);

    return card;
}

// =============================================================================
// Generate synthetic 2D point data in clusters
// =============================================================================
void generateSyntheticData(std::vector<Point>& points, const int N,
                           unsigned int seed = 42)
{
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
        if (group_cnt > (N - count)) group_cnt = N - count;

        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r    = frand() * R;
            const double dx   = (2.0 * frand() - 1.0) * r;
            const double dy   = std::sqrt(r * r - dx * dx) * sign;
            const double x    = cntr_x + dx;
            const double y    = cntr_y + dy;

            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;

            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// =============================================================================
// Main QT clustering algorithm  (MPI + OpenMP + CUDA)
// =============================================================================
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold)
{
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int>  unclustered_indices;
    std::vector<Cluster> clusters;
    clusters.reserve(N);

    for (int i = 0; i < N; ++i)
        unclustered_indices.push_back(i);

    // Initialize device clustered array to all zeros
    CUDA_CHECK(cudaMemset(d_clustered, 0, N * sizeof(char)));

    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed       = -1;
        std::vector<int> local_best_members;

        // ----- Parallel seed evaluation (OpenMP + CUDA) -----
        // Thread 0 uses GPU acceleration, worker threads use CPU.
        #pragma omp parallel
        {
            int tid       = omp_get_thread_num();
            int nthreads  = omp_get_num_threads();

            int priv_max_card = -1;
            int priv_best_seed = -1;
            std::vector<int> priv_best_members;

            // Manual work partitioning to avoid workshare deadlock
            // (thread 0 does GPU work, others do CPU work)
            if (tid == 0) {
                // GPU-accelerated seed evaluation (serial on master thread)
                for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                    int seed = unclustered_indices[i];
                    if (seed % num_ranks != rank) continue;
                    if (clustered[seed]) continue;

                    std::vector<int> candidate_members;
                    int card = generateCandidateClusterGPU(
                        seed, points, threshold, N,
                        &candidate_members);

                    if (card > priv_max_card) {
                        priv_max_card    = card;
                        priv_best_seed   = seed;
                        priv_best_members = std::move(candidate_members);
                    }
                }
            } else {
                // CPU-based seed evaluation (parallel on worker threads)
                for (size_t i = static_cast<size_t>(tid);
                     i < unclustered_indices.size();
                     i += static_cast<size_t>(nthreads))
                {
                    int seed = unclustered_indices[i];
                    if (seed % num_ranks != rank) continue;
                    if (clustered[seed]) continue;

                    std::vector<int> candidate_members;
                    int card = generateCandidateClusterCPU(
                        seed, clustered, points,
                        threshold, N, &candidate_members);

                    if (card > priv_max_card) {
                        priv_max_card    = card;
                        priv_best_seed   = seed;
                        priv_best_members = std::move(candidate_members);
                    }
                }
            }

            #pragma omp critical
            {
                if (priv_max_card > local_max_cardinality) {
                    local_max_cardinality = priv_max_card;
                    local_best_seed       = priv_best_seed;
                    local_best_members    = std::move(priv_best_members);
                }
            }
        }

        // ----- MPI: find global best cluster -----
        struct { int cardinality; int rank_id; } local_info, global_info;
        local_info.cardinality = local_max_cardinality;
        local_info.rank_id     = rank;
        MPI_Allreduce(&local_info, &global_info, 1, MPI_2INT,
                      MPI_MAXLOC, MPI_COMM_WORLD);

        int global_best_seed = -1;
        int members_size     = 0;
        std::vector<int> global_best_members;

        if (rank == global_info.rank_id) {
            global_best_seed  = local_best_seed;
            members_size      = static_cast<int>(local_best_members.size());
            global_best_members = std::move(local_best_members);
        }

        MPI_Bcast(&global_best_seed, 1, MPI_INT,
                  global_info.rank_id, MPI_COMM_WORLD);
        MPI_Bcast(&members_size, 1, MPI_INT,
                  global_info.rank_id, MPI_COMM_WORLD);

        if (rank != global_info.rank_id) {
            global_best_members.resize(members_size);
        }
        if (members_size > 0) {
            MPI_Bcast(global_best_members.data(), members_size, MPI_INT,
                      global_info.rank_id, MPI_COMM_WORLD);
        }

        if (global_best_seed >= 0 && global_info.cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = global_best_seed;
            cluster.members    = global_best_members;
            clusters.push_back(std::move(cluster));

            // Mark members as clustered on host
            for (int m : global_best_members)
                clustered[m] = 1;

            // Sync device copy of clustered array
            CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                                   N * sizeof(char), cudaMemcpyHostToDevice));

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(),
                               unclustered_indices.end(),
                               [&](int idx) { return clustered[idx]; }),
                unclustered_indices.end());
        } else {
            break;
        }
    }

    return clusters;
}

// =============================================================================
// Validation: check that clusters satisfy QT clustering properties
// OpenMP parallelized for the pairwise distance loops.
// =============================================================================
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>&   points,
                      double                      threshold)
{
    bool valid = true;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) printf("Validating clusters:\n");

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const int n_members = static_cast<int>(cluster.members.size());
        double max_diameter = 0.0;

        // Parallelize the O(n^2) pairwise distance loop
        #pragma omp parallel for reduction(max:max_diameter)
        for (int i = 0; i < n_members; ++i) {
            for (int j = i + 1; j < n_members; ++j) {
                double dist = distance(points[cluster.members[i]],
                                       points[cluster.members[j]]);
                if (dist > max_diameter) max_diameter = dist;
            }
        }

        if (c < 10 && rank == 0) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            #pragma omp critical
            {
                printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                       c, max_diameter, threshold);
                valid = false;
            }
        }
    }

    // Check for duplicate memberships (serial, very fast)
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            int member = clusters[c].members[i];
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
    for (size_t i = 0; i < membership.size(); ++i)
        if (membership[i] >= 0) clustered_count++;

    if (rank == 0) {
        printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
               points.size(), clustered_count,
               points.size() - clustered_count);
    }

    return valid;
}

// =============================================================================
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Discover and set CUDA device (round-robin among ranks on multi-GPU nodes)
    int n_devices = 0;
    cudaError_t ce = cudaGetDeviceCount(&n_devices);
    if (ce != cudaSuccess) {
        fprintf(stderr, "No CUDA-capable device found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (n_devices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % n_devices));
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate    = false;
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
        printf("QT Clustering Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI processes: %d\n", num_ranks);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA devices available: %d\n", n_devices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (same seed on all ranks)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Allocate GPU memory and upload point data
    allocateGPUMemory(num_points);
    uploadPointsToGPU(points);

    // ---- Perform QT clustering ----
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered  = 0;
        int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }

        double avg_cluster_size = clusters.empty() ? 0.0
            : static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        double time_sec = cluster_time.count() / 1000.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters.size() / time_sec, num_points / time_sec);
    }

    // Print results for external validation (-r flag, rank 0 only)
    if (printResults && rank == 0) {
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

    // Validation (-v flag)
    if (validate) {
        bool valid = validateClusters(clusters, points, threshold);
        if (rank == 0)
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        freeGPUMemory();
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    freeGPUMemory();
    MPI_Finalize();
    return 0;
}
