// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// Parallelization strategy:
// - CUDA: Precompute NxN pairwise distance matrix on GPU
// - MPI: Distribute seed point evaluations across MPI ranks
// - OpenMP: Parallelize seed evaluations within each MPI rank

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: compute pairwise Euclidean distance matrix
__global__ void computeDistanceMatrixKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    double* __restrict__ dist,
    const int N)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N && i < j) {
        const double dx = px[i] - px[j];
        const double dy = py[i] - py[j];
        const double d = sqrt(dx * dx + dy * dy);
        dist[i * N + j] = d;
        dist[j * N + i] = d;
    }
}

// Compute full NxN distance matrix on GPU, return on host
void computeDistanceMatrixGPU(const std::vector<Point>& points,
                               std::vector<double>& dist_matrix) {
    const int N = static_cast<int>(points.size());

    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    double *d_px, *d_py, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_px, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)N * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dist, 0, (size_t)N * N * sizeof(double)));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    computeDistanceMatrixKernel<<<grid, block>>>(d_px, d_py, d_dist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dist_matrix.resize((size_t)N * N);
    CUDA_CHECK(cudaMemcpy(dist_matrix.data(), d_dist,
                           (size_t)N * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_dist));
}

// Generate synthetic 2D point data in clusters (identical to original)
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

// Find closest unclustered point that maintains diameter < threshold
// Uses precomputed distance matrix with early termination
inline int findClosestPoint(const int* __restrict__ cluster_members,
                            const int n_members,
                            const char* __restrict__ clustered,
                            const char* __restrict__ in_cluster,
                            const double* __restrict__ dist_matrix,
                            const double threshold,
                            const int N) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < N; ++candidate) {
        if (clustered[candidate] | in_cluster[candidate]) continue;

        const double* __restrict__ row = dist_matrix + (size_t)candidate * N;
        double max_dist = 0.0;

        for (int i = 0; i < n_members; ++i) {
            const double dist = row[cluster_members[i]];
            if (dist > max_dist) max_dist = dist;
            if (max_dist >= threshold) break; // early termination
        }

        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// Generate candidate cluster from seed using precomputed distances
// Uses pre-allocated buffers to avoid repeated allocations
int generateCandidateCluster(const int seed_point,
                            const char* __restrict__ clustered,
                            const double* __restrict__ dist_matrix,
                            const double threshold,
                            const int N,
                            char* __restrict__ in_cluster_buf,
                            int* __restrict__ members_buf,
                            std::vector<int>* cluster_members = nullptr) {
    std::fill(in_cluster_buf, in_cluster_buf + N, 0);
    int n_members = 0;

    in_cluster_buf[seed_point] = 1;
    members_buf[n_members++] = seed_point;

    while (n_members < N) {
        const int closest = findClosestPoint(members_buf, n_members, clustered,
                                             in_cluster_buf, dist_matrix,
                                             threshold, N);
        if (closest < 0) break;
        in_cluster_buf[closest] = 1;
        members_buf[n_members++] = closest;
    }

    if (cluster_members) {
        cluster_members->resize(n_members);
        std::copy(members_buf, members_buf + n_members, cluster_members->begin());
    }

    return n_members;
}

// Main QT clustering algorithm with MPI + OpenMP parallelization
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double* dist_matrix,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    std::vector<Cluster> clusters;
    clusters.reserve(N / 10);

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Determine max number of OpenMP threads
    const int num_threads = omp_get_max_threads();

    // Pre-allocate thread-local buffers
    // Each thread needs its own in_cluster and members buffers
    std::vector<char> in_cluster_bufs((size_t)num_threads * N, 0);
    std::vector<int> members_bufs((size_t)num_threads * N, 0);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_unclustered = static_cast<int>(unclustered_indices.size());

        // Block distribution of seeds across MPI ranks
        const int rank_start = (int)((long long)num_unclustered * mpi_rank / mpi_size);
        const int rank_end = (int)((long long)num_unclustered * (mpi_rank + 1) / mpi_size);

        // Local best for this rank
        int local_max_card = -1;
        int local_best_seed = N;  // Initialize to N for proper tie-breaking
        std::vector<int> local_best_members;

        // OpenMP parallel evaluation of seeds assigned to this rank
        // Store per-thread results for deterministic reduction
        std::vector<int> thread_max_card(num_threads, -1);
        std::vector<int> thread_best_seed(num_threads, N);
        std::vector<std::vector<int>> thread_best_members(num_threads);

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            char* my_in_cluster = in_cluster_bufs.data() + (size_t)tid * N;
            int* my_members = members_bufs.data() + (size_t)tid * N;

            #pragma omp for schedule(dynamic, 1)
            for (int idx = rank_start; idx < rank_end; ++idx) {
                const int seed = unclustered_indices[idx];
                if (clustered[seed]) continue;

                std::vector<int> candidate_members;
                const int cardinality = generateCandidateCluster(
                    seed, clustered.data(), dist_matrix, threshold, N,
                    my_in_cluster, my_members, &candidate_members);

                if (cardinality > thread_max_card[tid] ||
                    (cardinality == thread_max_card[tid] && seed < thread_best_seed[tid])) {
                    thread_max_card[tid] = cardinality;
                    thread_best_seed[tid] = seed;
                    thread_best_members[tid] = std::move(candidate_members);
                }
            }
        }

        // Deterministic reduction across threads
        for (int tid = 0; tid < num_threads; ++tid) {
            if (thread_max_card[tid] > local_max_card ||
                (thread_max_card[tid] == local_max_card && thread_best_seed[tid] < local_best_seed)) {
                local_max_card = thread_max_card[tid];
                local_best_seed = thread_best_seed[tid];
                local_best_members = std::move(thread_best_members[tid]);
            }
        }

        // MPI Allreduce to find global best seed
        // Use two-step reduction for deterministic tie-breaking:
        // 1. Find maximum cardinality across all ranks
        // 2. Among ranks with max cardinality, find minimum seed
        int global_max_card;
        MPI_Allreduce(&local_max_card, &global_max_card, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Only consider seeds with maximum cardinality for tie-breaking
        // local_best_seed is N if no valid seed found on this rank
        int candidate_seed = (local_max_card == global_max_card && local_best_seed < N) ? local_best_seed : N;
        int global_best_seed;
        MPI_Allreduce(&candidate_seed, &global_best_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (global_best_seed >= N) {
            break; // No more clusters
        }

        // All ranks regenerate the best cluster independently (deterministic)
        std::vector<int> best_cluster_members;
        {
            char* tmp_in_cluster = in_cluster_bufs.data();
            int* tmp_members = members_bufs.data();
            generateCandidateCluster(global_best_seed, clustered.data(), dist_matrix,
                                     threshold, N, tmp_in_cluster, tmp_members,
                                     &best_cluster_members);
        }

        // All ranks update clustered array (deterministic)
        for (int m : best_cluster_members) {
            clustered[m] = 1;
        }

        // Add cluster
        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = std::move(best_cluster_members);
        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    return clusters;
}

// Validation function
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

    // GPU assignment (round-robin)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        cudaSetDevice(mpi_rank % num_gpus);
    }

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

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (MPI + OpenMP + CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        if (num_gpus > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            printf("CUDA device: %s\n", prop.name);
        }
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (all ranks, same seed = same data)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Compute distance matrix on GPU
    std::vector<double> dist_matrix;
    computeDistanceMatrixGPU(points, dist_matrix);

    // Perform QT clustering (MPI + OpenMP)
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    const std::vector<Cluster> clusters = qtClustering(points, dist_matrix.data(), threshold);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    long cluster_time_ms = (long)((end_time - start_time) * 1000.0);

    // Only rank 0 outputs results
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time_ms);
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

        const double time_sec = cluster_time_ms / 1000.0;
        const double clusters_per_sec = time_sec > 0 ? clusters.size() / time_sec : 0;
        const double points_per_sec = time_sec > 0 ? num_points / time_sec : 0;
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
