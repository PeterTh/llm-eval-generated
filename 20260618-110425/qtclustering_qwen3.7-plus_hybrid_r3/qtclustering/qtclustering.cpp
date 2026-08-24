// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// Parallelization strategy:
// - CUDA: Precompute N×N distance matrix on GPU; GPU-accelerated findClosestPoint
// - OpenMP: Parallelize seed evaluation loop across CPU cores
// - MPI: Distribute seeds across ranks, Allreduce for global best cluster

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ==================== CUDA Kernels ====================

__global__ void compute_dist_matrix_kernel(const Point* __restrict__ d_points,
                                           double* __restrict__ d_dist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx % N;

    if (i == j) {
        d_dist[idx] = 0.0;
    } else if (i < j) {
        double dx = d_points[i].x - d_points[j].x;
        double dy = d_points[i].y - d_points[j].y;
        double dist = sqrt(dx * dx + dy * dy);
        d_dist[(size_t)i * N + j] = dist;
        d_dist[(size_t)j * N + i] = dist;
    }
}

__global__ void find_closest_kernel(
    const double* __restrict__ d_dist,
    const int* __restrict__ d_members,
    const int* __restrict__ d_clustered,
    const int* __restrict__ d_in_cluster,
    int num_members,
    int N,
    double threshold,
    double* __restrict__ d_max_dists,
    int* __restrict__ d_valid
) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= N) return;

    if (d_clustered[candidate] || d_in_cluster[candidate]) {
        d_valid[candidate] = 0;
        d_max_dists[candidate] = 1e300;
        return;
    }

    double max_dist = 0.0;
    size_t row_offset = (size_t)candidate * N;
    for (int i = 0; i < num_members; i++) {
        double dist = d_dist[row_offset + d_members[i]];
        if (dist > max_dist) max_dist = dist;
    }

    if (max_dist < threshold) {
        d_valid[candidate] = 1;
        d_max_dists[candidate] = max_dist;
    } else {
        d_valid[candidate] = 0;
        d_max_dists[candidate] = 1e300;
    }
}

// ==================== GPU State ====================

struct GPUState {
    Point* d_points = nullptr;
    double* d_dist_matrix = nullptr;
    int* d_clustered = nullptr;
    int* d_in_cluster = nullptr;
    int* d_members = nullptr;
    double* d_max_dists = nullptr;
    int* d_valid = nullptr;
    double* h_dist_matrix = nullptr;
    int N = 0;
    bool initialized = false;
};

void initGPU(GPUState& state, const std::vector<Point>& points) {
    state.N = (int)points.size();
    int N = state.N;

    CUDA_CHECK(cudaMalloc(&state.d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&state.d_dist_matrix, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_in_cluster, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_max_dists, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.d_valid, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(state.d_points, points.data(), N * sizeof(Point),
                          cudaMemcpyHostToDevice));

    int total = N * N;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    compute_dist_matrix_kernel<<<numBlocks, blockSize>>>(
        state.d_points, state.d_dist_matrix, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    state.h_dist_matrix = new double[(size_t)N * N];
    CUDA_CHECK(cudaMemcpy(state.h_dist_matrix, state.d_dist_matrix,
                          (size_t)N * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(state.d_clustered, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.d_in_cluster, 0, N * sizeof(int)));

    state.initialized = true;
}

void freeGPU(GPUState& state) {
    if (!state.initialized) return;
    cudaFree(state.d_points);
    cudaFree(state.d_dist_matrix);
    cudaFree(state.d_clustered);
    cudaFree(state.d_in_cluster);
    cudaFree(state.d_members);
    cudaFree(state.d_max_dists);
    cudaFree(state.d_valid);
    delete[] state.h_dist_matrix;
    state.initialized = false;
}

// ==================== Data Generation ====================

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

// ==================== CPU findClosestPoint ====================

static int findClosestPointCPU(const std::vector<int>& cluster_members,
                               const std::vector<bool>& clustered,
                               const std::vector<bool>& in_cluster,
                               const double* dist_matrix,
                               const double threshold,
                               const int N) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    const int num_members = (int)cluster_members.size();

    for (int candidate = 0; candidate < N; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist = 0.0;
        const double* row = &dist_matrix[(size_t)candidate * N];
        for (int i = 0; i < num_members; ++i) {
            double dist = row[cluster_members[i]];
            if (dist > max_dist) max_dist = dist;
        }

        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// ==================== GPU findClosestPoint ====================

static int findClosestPointGPU(GPUState& state,
                               const std::vector<int>& cluster_members,
                               const std::vector<bool>& clustered,
                               const std::vector<bool>& in_cluster,
                               const double threshold) {
    int N = state.N;
    int num_members = (int)cluster_members.size();

    std::vector<int> h_clustered(N), h_in_cluster(N);
    for (int i = 0; i < N; i++) {
        h_clustered[i] = clustered[i] ? 1 : 0;
        h_in_cluster[i] = in_cluster[i] ? 1 : 0;
    }
    CUDA_CHECK(cudaMemcpy(state.d_clustered, h_clustered.data(), N * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_in_cluster, h_in_cluster.data(), N * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_members, cluster_members.data(),
                          num_members * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    find_closest_kernel<<<numBlocks, blockSize>>>(
        state.d_dist_matrix, state.d_members, state.d_clustered, state.d_in_cluster,
        num_members, N, threshold, state.d_max_dists, state.d_valid);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_max_dists(N);
    std::vector<int> h_valid(N);
    CUDA_CHECK(cudaMemcpy(h_max_dists.data(), state.d_max_dists, N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid.data(), state.d_valid, N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int best = -1;
    double min_d = std::numeric_limits<double>::max();
    for (int candidate = 0; candidate < N; candidate++) {
        if (h_valid[candidate] && h_max_dists[candidate] < min_d) {
            min_d = h_max_dists[candidate];
            best = candidate;
        }
    }

    return best;
}

// ==================== Generate Candidate Cluster ====================

static int generateCandidateClusterCPU(int seed,
                                      const std::vector<bool>& clustered,
                                      const double* dist_matrix,
                                      double threshold, int N,
                                      std::vector<int>* members_out) {
    std::vector<bool> in_cluster(N, false);
    std::vector<int> members;

    in_cluster[seed] = true;
    members.push_back(seed);

    while ((int)members.size() < N) {
        int closest = findClosestPointCPU(members, clustered, in_cluster,
                                          dist_matrix, threshold, N);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (members_out) *members_out = members;
    return (int)members.size();
}

// GPU-accelerated candidate cluster generation (used for sequential regeneration)
static int generateCandidateClusterGPU(int seed,
                                       GPUState& state,
                                       const std::vector<bool>& clustered,
                                       double threshold,
                                       std::vector<int>& members_out) {
    int N = state.N;

    members_out.clear();
    members_out.push_back(seed);

    // Copy clustered to GPU
    std::vector<int> h_clustered(N);
    for (int i = 0; i < N; i++) h_clustered[i] = clustered[i] ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(state.d_clustered, h_clustered.data(), N * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_in_cluster, 0, N * sizeof(int)));

    int one = 1;
    CUDA_CHECK(cudaMemcpy(state.d_in_cluster + seed, &one, sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_members, &seed, sizeof(int), cudaMemcpyHostToDevice));

    std::vector<double> h_max_dists(N);
    std::vector<int> h_valid(N);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    while ((int)members_out.size() < N) {
        int num_members = (int)members_out.size();

        find_closest_kernel<<<numBlocks, blockSize>>>(
            state.d_dist_matrix, state.d_members, state.d_clustered, state.d_in_cluster,
            num_members, N, threshold, state.d_max_dists, state.d_valid);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_max_dists.data(), state.d_max_dists, N * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_valid.data(), state.d_valid, N * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int best = -1;
        double min_d = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < N; candidate++) {
            if (h_valid[candidate] && h_max_dists[candidate] < min_d) {
                min_d = h_max_dists[candidate];
                best = candidate;
            }
        }

        if (best < 0) break;

        CUDA_CHECK(cudaMemcpy(state.d_in_cluster + best, &one, sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_members + num_members, &best, sizeof(int),
                              cudaMemcpyHostToDevice));

        members_out.push_back(best);
    }

    return (int)members_out.size();
}

// ==================== Main QT Clustering (MPI + OpenMP + CUDA) ====================

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         double threshold,
                                         int rank, int world_size,
                                         GPUState& gpu_state) {
    const int N = (int)points.size();
    const double* dist_matrix = gpu_state.h_dist_matrix;

    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    unclustered_indices.reserve(N);
    for (int i = 0; i < N; i++) unclustered_indices.push_back(i);

    while (!unclustered_indices.empty()) {
        int num_unclustered = (int)unclustered_indices.size();

        // Phase 1: Parallel seed evaluation (OpenMP within each MPI rank)
        int local_best_card = -1;
        int local_best_seed = INT_MAX;

        #pragma omp parallel
        {
            int thread_best_card = -1;
            int thread_best_seed = INT_MAX;

            #pragma omp for schedule(dynamic, 4)
            for (int i = rank; i < num_unclustered; i += world_size) {
                int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                int cardinality = generateCandidateClusterCPU(
                    seed, clustered, dist_matrix, threshold, N, nullptr);

                if (cardinality > thread_best_card ||
                    (cardinality == thread_best_card && seed < thread_best_seed)) {
                    thread_best_card = cardinality;
                    thread_best_seed = seed;
                }
            }

            #pragma omp critical
            {
                if (thread_best_card > local_best_card ||
                    (thread_best_card == local_best_card &&
                     thread_best_seed < local_best_seed)) {
                    local_best_card = thread_best_card;
                    local_best_seed = thread_best_seed;
                }
            }
        }

        // Phase 2: MPI Allreduce to find global best cardinality
        int global_best_card = 0;
        MPI_Allreduce(&local_best_card, &global_best_card, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD);

        if (global_best_card <= 0) break;

        // Phase 3: MPI Allreduce to find global best seed (tiebreak: lowest seed)
        int my_seed_for_min = (local_best_card == global_best_card) ?
                              local_best_seed : INT_MAX;
        int global_best_seed = INT_MAX;
        MPI_Allreduce(&my_seed_for_min, &global_best_seed, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);

        if (global_best_seed == INT_MAX) break;

        // Determine owner rank (strided distribution)
        int best_pos = -1;
        for (int i = 0; i < num_unclustered; i++) {
            if (unclustered_indices[i] == global_best_seed) {
                best_pos = i;
                break;
            }
        }
        int owner_rank = best_pos % world_size;

        // Phase 4: Owner regenerates best cluster using GPU-accelerated findClosestPoint
        std::vector<int> cluster_members(global_best_card, 0);
        if (rank == owner_rank) {
            std::vector<int> gpu_members;
            generateCandidateClusterGPU(global_best_seed, gpu_state, clustered,
                                        threshold, gpu_members);
            cluster_members = gpu_members;
        }

        // Phase 5: Broadcast cluster members to all ranks
        MPI_Bcast(cluster_members.data(), global_best_card, MPI_INT,
                  owner_rank, MPI_COMM_WORLD);

        // Create cluster and update state
        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = std::move(cluster_members);
        clusters.push_back(std::move(cluster));

        for (int m : clusters.back().members) {
            clustered[m] = true;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    return clusters;
}

// ==================== Validation ====================

static inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

static bool validateClusters(const std::vector<Cluster>& clusters,
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

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ==================== Main ====================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_points = 1000;
    double threshold = 2.0;
    int validate_i = 0;
    int printResults_i = 0;

    // Parse command line arguments on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                num_points = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                threshold = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
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

    // Broadcast parameters to all ranks
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool validate = (validate_i != 0);
    bool printResults = (printResults_i != 0);

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n",
               world_size, omp_get_max_threads());
    }

    // Generate synthetic data on rank 0, broadcast to all ranks
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize GPU: compute distance matrix on CUDA device
    GPUState gpu_state;
    initGPU(gpu_state, points);

    // Perform QT clustering with MPI + OpenMP + CUDA
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, world_size, gpu_state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long local_cluster_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &cluster_time_ms, 1, MPI_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // Output results (rank 0 only)
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;

        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = (int)clusters[i].members.size();
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
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        // Print results for external validation
        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                    membership[clusters[c].members[i]] = (int)c;
                }
            }
            for (int m : membership) {
                membershipData.push_back(static_cast<double>(m));
            }
            print_results(membershipData, "ClusterMembership");
        }

        // Validation
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    freeGPU(gpu_state);
    MPI_Finalize();
    return 0;
}
