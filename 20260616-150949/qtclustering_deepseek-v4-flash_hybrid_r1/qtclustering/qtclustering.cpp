// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   MPI:     Distribute seed evaluation across compute nodes/ranks
//   OpenMP:  Parallelize seed evaluation loop within each rank (CPU cores)
//   CUDA:    Accelerate findClosestPoint distance computations on GPU

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

#include "../common/results_output.hpp"

// Wrapper for omp_get_thread_num (host-only function)
static inline int get_omp_thread_num() {
    return omp_get_thread_num();
}

// =========================================================================
// CUDA Kernel and Device Code
// =========================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

// Kernel: evaluate max distance from each candidate point to all cluster members
// Each thread handles one candidate point with grid-stride loop for arbitrary sizes
// Uses unsigned char (1 byte) instead of bool due to std::vector<bool> specialization
__global__ void findClosestKernel(
    const double* points_x,
    const double* points_y,
    const unsigned char* clustered,
    const unsigned char* in_cluster,
    const int* cluster_members,
    int cluster_size,
    int point_count,
    double threshold,
    double* results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int candidate = idx; candidate < point_count; candidate += stride) {
        // Skip already clustered points
        if (clustered[candidate] || in_cluster[candidate]) {
            results[candidate] = 1e308;
            continue;
        }

        // Compute max distance from this candidate to all current cluster members
        double max_dist = 0.0;
        for (int j = 0; j < cluster_size; ++j) {
            int member = cluster_members[j];
            double dx = points_x[candidate] - points_x[member];
            double dy = points_y[candidate] - points_y[member];
            double dist = sqrt(dx * dx + dy * dy);
            if (dist > max_dist) max_dist = dist;
        }

        // Store result: distance if within threshold, infinity otherwise
        results[candidate] = (max_dist < threshold) ? max_dist : 1e308;
    }
}

// CUDA device data (unsigned char used instead of bool because std::vector<bool>
// is a bitset specialization without data() access)
static double*       d_points_x = nullptr;
static double*       d_points_y = nullptr;
static unsigned char* d_clustered = nullptr;
static unsigned char* d_in_cluster = nullptr;
static int*          d_cluster_members = nullptr;
static double*       d_results = nullptr;
static bool          cuda_initialized = false;

// Initialize CUDA device memory with point data
static void initCUDA(const std::vector<double>& px,
                     const std::vector<double>& py,
                     int n) {
    if (cuda_initialized) return;

    CUDA_CHECK(cudaMalloc(&d_points_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_points_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_cluster_members, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_points_x, px.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_y, py.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Initialize working arrays to 0 (false)
    CUDA_CHECK(cudaMemset(d_clustered, 0, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_in_cluster, 0, n * sizeof(unsigned char)));

    cuda_initialized = true;
}

// Clean up CUDA device memory
static void freeCUDA() {
    if (!cuda_initialized) return;
    cudaFree(d_points_x);
    cudaFree(d_points_y);
    cudaFree(d_clustered);
    cudaFree(d_in_cluster);
    cudaFree(d_cluster_members);
    cudaFree(d_results);
    d_points_x = nullptr;
    d_points_y = nullptr;
    d_clustered = nullptr;
    d_in_cluster = nullptr;
    d_cluster_members = nullptr;
    d_results = nullptr;
    cuda_initialized = false;
}

// Helper: convert std::vector<bool> to std::vector<unsigned char> for CUDA transfer
static std::vector<unsigned char> boolVecToUChar(const std::vector<bool>& src) {
    std::vector<unsigned char> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i] ? 1 : 0;
    }
    return dst;
}

// CUDA-accelerated findClosestPoint implementation
// Uses GPU to compute distances for all candidates in parallel
static int findClosestPointCUDA(
    const std::vector<int>& cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    int point_count,
    double threshold)
{
    int cluster_size = static_cast<int>(cluster_members.size());

    // Convert std::vector<bool> (bitset) to unsigned char arrays for CUDA
    std::vector<unsigned char> h_clustered = boolVecToUChar(clustered);
    std::vector<unsigned char> h_in_cluster = boolVecToUChar(in_cluster);

    // Copy working data to device
    CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(),
                          point_count * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_cluster, h_in_cluster.data(),
                          point_count * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_members, cluster_members.data(),
                          cluster_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Launch kernel: enough blocks to saturate GPU
    int threads = 256;
    int blocks = std::min((point_count + threads - 1) / threads, 1024);

    findClosestKernel<<<blocks, threads>>>(
        d_points_x, d_points_y,
        d_clustered, d_in_cluster,
        d_cluster_members, cluster_size,
        point_count, threshold,
        d_results
    );

    // Read results back to host (implicit synchronization)
    std::vector<double> h_results(point_count);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
                          point_count * sizeof(double),
                          cudaMemcpyDeviceToHost));



    // Host-side reduction: find candidate with minimum diameter
    double min_dist = 1e308;
    int min_idx = -1;
    for (int i = 0; i < point_count; ++i) {
        if (h_results[i] < min_dist) {
            min_dist = h_results[i];
            min_idx = i;
        }
    }

    return min_dist < threshold ? min_idx : -1;
}

// Synchronize CUDA device
static void syncCUDA() {
    if (cuda_initialized) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Update the clustered array on the GPU after an outer iteration
static void updateClusteredCUDA(const std::vector<bool>& clustered, int n) {
    if (cuda_initialized) {
        std::vector<unsigned char> h_clustered = boolVecToUChar(clustered);
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(),
                              n * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }
}

#endif // __CUDACC__

// =========================================================================
// Host Data Structures and Constants
// =========================================================================

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// =========================================================================
// Generate synthetic 2D point data in clusters
// =========================================================================
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

// =========================================================================
// Calculate Euclidean distance between two points
// =========================================================================
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// =========================================================================
// Find closest point using CPU (sequential, for non-GPU threads)
// =========================================================================
static int findClosestPointCPU(
    const std::vector<int>& cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    const std::vector<Point>& points,
    double threshold,
    int point_count)
{
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            if (dist > max_dist) max_dist = dist;
        }

        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// =========================================================================
// Find the closest unclustered point to the current cluster
// Thread 0 (inside OMP parallel) uses CUDA acceleration, others use CPU
// =========================================================================
static int findClosestPoint(
    const std::vector<int>& cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    const std::vector<Point>& points,
    double threshold,
    int point_count)
{
#ifdef __CUDACC__
    // Thread 0 uses GPU acceleration; critical section ensures exclusive access
    if (get_omp_thread_num() == 0) {
        int best;
        #pragma omp critical (cuda_findclosest)
        {
            best = findClosestPointCUDA(cluster_members, clustered,
                                        in_cluster, point_count, threshold);
        }
        return best;
    }
#endif
    // CPU fallback for OpenMP threads > 0 (or if CUDA is not compiled)
    return findClosestPointCPU(cluster_members, clustered, in_cluster,
                               points, threshold, point_count);
}

// =========================================================================
// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
// =========================================================================
static int generateCandidateCluster(
    int seed_point,
    const std::vector<bool>& clustered,
    const std::vector<Point>& points,
    double threshold,
    int point_count,
    std::vector<int>* cluster_members = nullptr)
{
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster,
                                             points, threshold, point_count);
        if (closest < 0) break;

        in_cluster[closest] = true;
        members.push_back(closest);
    }

    int cluster_size = static_cast<int>(members.size());

    if (cluster_members) {
        *cluster_members = std::move(members);
    }

    return cluster_size;
}

// =========================================================================
// Main QT clustering algorithm with MPI + OpenMP + CUDA parallelization
// =========================================================================
static std::vector<Cluster> qtClustering(
    const std::vector<Point>& points,
    double threshold,
    int mpi_rank,
    int mpi_size)
{
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    while (!unclustered_indices.empty()) {
        int total = static_cast<int>(unclustered_indices.size());

        // Static partition of seeds across MPI ranks
        int chunk = total / mpi_size;
        int rem = total % mpi_size;
        int start_idx = mpi_rank * chunk + std::min(mpi_rank, rem);
        int end_idx = start_idx + chunk + (mpi_rank < rem ? 1 : 0);

        if (start_idx >= end_idx) {
            end_idx = start_idx;
        }

        // Each rank finds the best seed in its partition
        // Seeded candidates are evaluated iteratively (CUDA or CPU path)
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        for (int si = start_idx; si < end_idx; ++si) {
            int seed = unclustered_indices[si];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold, N, &candidate_members);

            if (cardinality > local_max_cardinality) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

#ifdef __CUDACC__
        // Ensure all GPU work is complete before MPI communication
        syncCUDA();
#endif

        // Exchange local bests across all MPI ranks using Allgather
        struct LocalInfo {
            int cardinality;
            int seed;
            int rank;
        };
        LocalInfo local_info;
        local_info.cardinality = local_max_cardinality;
        local_info.seed = local_best_seed;
        local_info.rank = mpi_rank;

        std::vector<LocalInfo> all_info(mpi_size);
        MPI_Allgather(&local_info, sizeof(LocalInfo), MPI_BYTE,
                      all_info.data(), sizeof(LocalInfo), MPI_BYTE,
                      MPI_COMM_WORLD);

        // Determine global best: max cardinality, min seed index for ties
        int global_best_cardinality = -1;
        int global_best_seed = -1;
        int winning_rank = -1;

        for (int r = 0; r < mpi_size; ++r) {
            if (all_info[r].cardinality > global_best_cardinality ||
                (all_info[r].cardinality == global_best_cardinality &&
                 all_info[r].seed >= 0 &&
                 (global_best_seed < 0 || all_info[r].seed < global_best_seed))) {
                global_best_cardinality = all_info[r].cardinality;
                global_best_seed = all_info[r].seed;
                winning_rank = all_info[r].rank;
            }
        }

        if (global_best_cardinality <= 0 || global_best_seed < 0) {
            break;
        }

        // Broadcast cluster members from the winning rank
        int member_count = 0;
        std::vector<int> global_best_members;

        if (mpi_rank == winning_rank) {
            member_count = static_cast<int>(local_best_members.size());
        }
        MPI_Bcast(&member_count, 1, MPI_INT, winning_rank, MPI_COMM_WORLD);

        if (mpi_rank == winning_rank) {
            global_best_members = local_best_members;
        } else {
            global_best_members.resize(member_count);
        }
        MPI_Bcast(global_best_members.data(), member_count, MPI_INT,
                  winning_rank, MPI_COMM_WORLD);

        // All ranks add this cluster and update their state
        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = global_best_members;
        clusters.push_back(std::move(cluster));

        for (int m : global_best_members) {
            clustered[m] = true;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );

#ifdef __CUDACC__
        // Update clustered array on GPU for the next iteration
        updateClusteredCUDA(clustered, N);
#endif
    }

    return clusters;
}

// =========================================================================
// Validation: check that clusters satisfy the QT clustering properties
// =========================================================================
static bool validateClusters(const std::vector<Cluster>& clusters,
                            const std::vector<Point>& points,
                            double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Check each cluster (parallelized with OpenMP across clusters)
    #pragma omp parallel for reduction(&&: valid)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                if (dist > max_diameter) max_diameter = dist;
            }
        }

        if (c < 10) {
            #pragma omp critical (val_print)
            {
                printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                       c, cluster.members.size(), cluster.seed_point,
                       max_diameter);
            }
        }

        if (max_diameter > threshold * 1.001) {
            #pragma omp critical (val_print)
            {
                printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                       c, max_diameter, threshold);
            }
            valid = false;
        }
    }

    // Check for duplicate memberships (serial, fast)
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

// =========================================================================
// Print usage information
// =========================================================================
static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// =========================================================================
// Main entry point: MPI init, CUDA init, run clustering, print results
// =========================================================================
int main(int argc, char** argv) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Parse command line arguments (all ranks parse identically)
    int num_points = 1000;
    double threshold = 2.0;
    bool do_validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            do_validate = true;
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
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    // Rank 0 prints benchmark info
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI ranks: %d\n", mpi_size);
        printf("Validation: %s\n", do_validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (all ranks produce identical data with same seed)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

#ifdef __CUDACC__
    // Initialize CUDA on each MPI rank (round-robin GPU assignment)
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int dev = mpi_rank % device_count;
            CUDA_CHECK(cudaSetDevice(dev));
        }

        // Prepare separate x/y arrays for GPU
        std::vector<double> px(num_points), py(num_points);
        for (int i = 0; i < num_points; ++i) {
            px[i] = points[i].x;
            py[i] = points[i].y;
        }
        initCUDA(px, py, num_points);
    }
#endif

    // Run the parallel QT clustering algorithm
    auto cluster_start = std::chrono::high_resolution_clock::now();

    std::vector<Cluster> clusters = qtClustering(points, threshold,
                                                  mpi_rank, mpi_size);

#ifdef __CUDACC__
    syncCUDA();
#endif
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);

    // Rank 0 prints results and statistics
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;

        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            if (size > max_cluster_size) max_cluster_size = size;
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
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }

    // Print results for external verification (rank 0 only)
    if (printResults && mpi_rank == 0) {
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (int m : clusters[c].members) {
                membership[m] = static_cast<int>(c);
            }
        }
        for (int m : membership) {
            membershipData.push_back(static_cast<double>(m));
        }
        print_results(membershipData, "ClusterMembership");
    }

    // Validation (rank 0 runs it, result broadcast for consistent exit code)
    if (do_validate) {
        bool valid = false;
        if (mpi_rank == 0) {
            valid = validateClusters(clusters, points, threshold);
        }
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

#ifdef __CUDACC__
        freeCUDA();
#endif
        MPI_Finalize();
        return valid ? 0 : 1;
    }

#ifdef __CUDACC__
    freeCUDA();
#endif

    MPI_Finalize();
    return 0;
}
