// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Version
//
// Parallelization strategy:
// - MPI: Distribute seed point evaluations across cluster nodes
// - OpenMP: Parallelize seed evaluation within each node
// - CUDA: GPU-accelerate the max-distance computation in findClosestPoint

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

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ============================================================
// CUDA kernel for GPU-accelerated max-distance computation
// ============================================================

// Each block handles one candidate point. Threads within the block
// cooperate to compute the max distance from that candidate to all
// current cluster members via shared-memory reduction.
__global__ void computeMaxDistancesKernel(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const int* __restrict__ members,
    const int num_members,
    const int N,
    double* __restrict__ max_distances)
{
    const int candidate = blockIdx.x;
    if (candidate >= N) return;

    extern __shared__ double sdata[];

    const double cx = points_x[candidate];
    const double cy = points_y[candidate];

    double local_max = 0.0;
    for (int m = threadIdx.x; m < num_members; m += blockDim.x) {
        const int midx = members[m];
        const double dx = cx - points_x[midx];
        const double dy = cy - points_y[midx];
        const double dist = sqrt(dx * dx + dy * dy);
        if (dist > local_max) local_max = dist;
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_distances[candidate] = sdata[0];
    }
}

// ============================================================
// Persistent CUDA context to avoid per-call allocation overhead
// ============================================================

struct CudaContext {
    double* d_px = nullptr;
    double* d_py = nullptr;
    int* d_members = nullptr;
    double* d_max_distances = nullptr;
    int N = 0;
    bool initialized = false;
    bool points_uploaded = false;
};

static CudaContext g_cuda_ctx;

static void initCudaContext(int N) {
    if (g_cuda_ctx.initialized && g_cuda_ctx.N == N) return;
    if (g_cuda_ctx.initialized) {
        cudaFree(g_cuda_ctx.d_px);
        cudaFree(g_cuda_ctx.d_py);
        cudaFree(g_cuda_ctx.d_members);
        cudaFree(g_cuda_ctx.d_max_distances);
    }
    cudaMalloc(&g_cuda_ctx.d_px, N * sizeof(double));
    cudaMalloc(&g_cuda_ctx.d_py, N * sizeof(double));
    cudaMalloc(&g_cuda_ctx.d_members, N * sizeof(int));
    cudaMalloc(&g_cuda_ctx.d_max_distances, N * sizeof(double));
    g_cuda_ctx.N = N;
    g_cuda_ctx.initialized = true;
    g_cuda_ctx.points_uploaded = false;
}

static void cleanupCudaContext() {
    if (!g_cuda_ctx.initialized) return;
    cudaFree(g_cuda_ctx.d_px);
    cudaFree(g_cuda_ctx.d_py);
    cudaFree(g_cuda_ctx.d_members);
    cudaFree(g_cuda_ctx.d_max_distances);
    g_cuda_ctx.initialized = false;
}

// GPU-accelerated findClosestPoint using persistent device memory
static int findClosestPointCUDA(
    const std::vector<Point>& points,
    const std::vector<int>& cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    const double threshold,
    const int point_count)
{
    const int N = point_count;
    const int num_members = static_cast<int>(cluster_members.size());

    initCudaContext(N);

    // Upload points once (they never change)
    if (!g_cuda_ctx.points_uploaded) {
        std::vector<double> h_px(N), h_py(N);
        for (int i = 0; i < N; i++) {
            h_px[i] = points[i].x;
            h_py[i] = points[i].y;
        }
        cudaMemcpy(g_cuda_ctx.d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(g_cuda_ctx.d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        g_cuda_ctx.points_uploaded = true;
    }

    // Upload members array (changes each call)
    cudaMemcpy(g_cuda_ctx.d_members, cluster_members.data(),
               num_members * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    const int threadsPerBlock = 128;
    const size_t sharedMemSize = threadsPerBlock * sizeof(double);
    computeMaxDistancesKernel<<<N, threadsPerBlock, sharedMemSize>>>(
        g_cuda_ctx.d_px, g_cuda_ctx.d_py,
        g_cuda_ctx.d_members, num_members, N,
        g_cuda_ctx.d_max_distances);

    // Download results
    std::vector<double> h_max_distances(N);
    cudaMemcpy(h_max_distances.data(), g_cuda_ctx.d_max_distances,
               N * sizeof(double), cudaMemcpyDeviceToHost);

    // Find best candidate on CPU (preserves tie-breaking by lowest index)
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < N; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;
        const double max_dist = h_max_distances[candidate];
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// ============================================================
// Data generation (identical to original, deterministic)
// ============================================================

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };

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

// ============================================================
// Core clustering functions
// ============================================================

// findClosestPoint wrapper - uses CUDA with OpenMP critical section
// to serialize GPU access across threads within a rank
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int result;
    #pragma omp critical(cuda_access)
    {
        result = findClosestPointCUDA(points, cluster_members, clustered,
                                      in_cluster, threshold, point_count);
    }
    return result;
}

// Generate a candidate cluster starting from a seed point
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, points,
                                             threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// ============================================================
// Main QT clustering with MPI + OpenMP + CUDA
// ============================================================

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;

        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // Distribute seeds across MPI ranks (block distribution)
        const int seeds_per_rank = (num_seeds + mpi_size - 1) / mpi_size;
        const int start_idx = mpi_rank * seeds_per_rank;
        const int end_idx = std::min(start_idx + seeds_per_rank, num_seeds);

        // Each rank evaluates its assigned seeds in parallel with OpenMP
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_cluster_members;

        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic)
            for (int i = start_idx; i < end_idx; ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                std::vector<int> candidate_members;
                const int cardinality = generateCandidateCluster(
                    seed, clustered, points, threshold, N, &candidate_members);

                #pragma omp critical(local_best)
                {
                    if (cardinality > local_max_cardinality) {
                        local_max_cardinality = cardinality;
                        local_best_seed = seed;
                        local_best_cluster_members = candidate_members;
                    }
                }
            }
        }

        // Gather cardinalities and seeds from all ranks
        std::vector<int> all_cardinalities(mpi_size);
        std::vector<int> all_seeds(mpi_size);
        std::vector<int> all_members_sizes(mpi_size);

        MPI_Gather(&local_max_cardinality, 1, MPI_INT,
                   all_cardinalities.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_best_seed, 1, MPI_INT,
                   all_seeds.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        int local_size = static_cast<int>(local_best_cluster_members.size());
        MPI_Gather(&local_size, 1, MPI_INT,
                   all_members_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Gather member arrays with Gatherv
        std::vector<int> recv_counts;
        std::vector<int> displacements;
        std::vector<int> all_members_flat;

        if (mpi_rank == 0) {
            recv_counts.resize(mpi_size);
            displacements.resize(mpi_size, 0);
            for (int r = 0; r < mpi_size; r++) {
                recv_counts[r] = all_members_sizes[r];
                if (r > 0) displacements[r] = displacements[r-1] + recv_counts[r-1];
            }
            int total = (mpi_size > 0) ? displacements[mpi_size-1] + recv_counts[mpi_size-1] : 0;
            all_members_flat.resize(total);
        }

        MPI_Gatherv(local_best_cluster_members.data(), local_size, MPI_INT,
                    all_members_flat.data(), recv_counts.data(),
                    displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

        // Rank 0 determines global best
        if (mpi_rank == 0) {
            int global_max = -1;
            int global_best_seed = -1;
            std::vector<int> global_best_members;

            for (int r = 0; r < mpi_size; r++) {
                if (all_cardinalities[r] > global_max) {
                    global_max = all_cardinalities[r];
                    global_best_seed = all_seeds[r];
                    int offset = displacements[r];
                    int sz = recv_counts[r];
                    global_best_members.assign(all_members_flat.begin() + offset,
                                              all_members_flat.begin() + offset + sz);
                }
            }

            max_cardinality = global_max;
            best_seed = global_best_seed;
            best_cluster_members = global_best_members;
        }

        // Broadcast best cluster to all ranks
        MPI_Bcast(&max_cardinality, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&best_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int best_members_size = static_cast<int>(best_cluster_members.size());
        MPI_Bcast(&best_members_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi_rank != 0) {
            best_cluster_members.resize(best_members_size);
        }
        MPI_Bcast(best_cluster_members.data(), best_members_size, MPI_INT, 0, MPI_COMM_WORLD);

        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());
        } else {
            break;
        }
    }

    return clusters;
}

// ============================================================
// Validation
// ============================================================

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
                const double dx = points[cluster.members[i]].x - points[cluster.members[j]].x;
                const double dy = points[cluster.members[i]].y - points[cluster.members[j]].y;
                const double dist = std::sqrt(dx * dx + dy * dy);
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

void print_usage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    if (mpi_rank == 0) {
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
                print_usage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                print_usage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int val_int = validate ? 1 : 0;
    int res_int = printResults ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&res_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = val_int != 0;
    printResults = res_int != 0;

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", mpi_size);
    }

    // All ranks generate identical data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
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

        const double time_sec = cluster_time.count() / 1000.0;
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
            }
        }
    }

    cleanupCudaContext();
    MPI_Finalize();
    return 0;
}
