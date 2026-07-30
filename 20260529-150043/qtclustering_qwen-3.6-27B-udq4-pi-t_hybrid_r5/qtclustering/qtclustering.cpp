// QT Clustering Benchmark - Hybrid Parallel Version (MPI + OpenMP + CUDA)
//
// QT (Quality Threshold) clustering builds clusters by starting with a seed
// point and iteratively adding the closest point that maintains the cluster's
// diameter below a threshold.
//
// Parallelization strategy:
//   MPI:    Distribute seed candidates across ranks; each rank independently
//           generates candidate clusters, then finds global best.
//   OpenMP: Parallelize data generation and distance reduction on host.
//   CUDA:   Parallelize distance computations (findClosestPoint) on GPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <omp.h>
#include <vector>

#include <mpi.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Kernel: compute max distance from each candidate to all cluster members.
//   out[candidate] = max_{member in cluster} distance(candidate, member)
//   out[candidate] = -1.0 if candidate is already clustered or in_cluster
// Each thread handles one candidate and checks ALL cluster members.
__global__ void computeMaxDistancesKernel(
    const double* d_px, const double* d_py,
    const int* d_in_cluster, const int* d_clustered,
    const int* d_cluster_members,
    int cluster_size, int point_count,
    double* out)
{
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;

    if (candidate >= point_count) return;

    double max_dist = 0.0;
    const double cx = d_px[candidate];
    const double cy = d_py[candidate];

    // Each thread checks ALL cluster members for its candidate
    for (int i = 0; i < cluster_size; i++) {
        int m = d_cluster_members[i];
        double dx = cx - d_px[m];
        double dy = cy - d_py[m];
        double d = sqrt(dx * dx + dy * dy);
        if (d > max_dist) max_dist = d;
    }

    // Check if candidate is eligible
    if (d_in_cluster[candidate] || d_clustered[candidate]) {
        out[candidate] = -1.0;
    } else {
        out[candidate] = max_dist;
    }
}

// ---------------------------------------------------------------------------
// Host-side cluster generation (CUDA-accelerated)
// ---------------------------------------------------------------------------

// Device memory handles (allocated once per MPI rank, reused across calls)
static double*  d_px = nullptr;
static double*  d_py = nullptr;
static int*     d_in_cluster = nullptr;
static int*     d_clustered = nullptr;
static int*     d_cluster_members = nullptr;
static double*  d_distances = nullptr;
static int      d_max_points = 0;

void initDeviceMemory(int N) {
    if (d_max_points >= N) return;
    d_max_points = N;

    cudaMalloc(&d_px, N * sizeof(double));
    cudaMalloc(&d_py, N * sizeof(double));
    cudaMalloc(&d_in_cluster, N * sizeof(int));
    cudaMalloc(&d_clustered, N * sizeof(int));
    cudaMalloc(&d_cluster_members, N * sizeof(int));
    cudaMalloc(&d_distances, N * sizeof(double));
}

void freeDeviceMemory() {
    cudaFree(d_px); d_px = nullptr;
    cudaFree(d_py); d_py = nullptr;
    cudaFree(d_in_cluster); d_in_cluster = nullptr;
    cudaFree(d_clustered); d_clustered = nullptr;
    cudaFree(d_cluster_members); d_cluster_members = nullptr;
    cudaFree(d_distances); d_distances = nullptr;
    d_max_points = 0;
}

// Generate a candidate cluster starting from a seed point.
// Uses CUDA for distance computations. Returns cluster cardinality.
int generateCandidateCluster(
    int seed_point,
    const std::vector<int>& clustered,
    const double* px, const double* py,
    double threshold, int point_count,
    std::vector<int>* cluster_members_out)
{
    std::vector<int> members;
    members.reserve(point_count);
    std::vector<int> in_cluster(point_count, 0);

    // Upload points to device
    cudaMemcpy(d_px, px, point_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, point_count * sizeof(double), cudaMemcpyHostToDevice);

    // Upload clustered array
    cudaMemcpy(d_clustered, clustered.data(), point_count * sizeof(int),
               cudaMemcpyHostToDevice);

    // Add seed point
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        int cluster_size = static_cast<int>(members.size());

        // Upload in_cluster and cluster_members to device
        cudaMemcpy(d_in_cluster, in_cluster.data(), point_count * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_members, members.data(),
                   cluster_size * sizeof(int), cudaMemcpyHostToDevice);

        // Launch distance computation kernel
        const int block_size = 256;
        const int num_blocks = (point_count + block_size - 1) / block_size;

        computeMaxDistancesKernel<<<num_blocks, block_size>>>(
            d_px, d_py, d_in_cluster, d_clustered,
            d_cluster_members, cluster_size, point_count, d_distances);

        cudaDeviceSynchronize();

        // Copy distances back to host
        std::vector<double> h_distances(point_count);
        cudaMemcpy(h_distances.data(), d_distances,
                   point_count * sizeof(double), cudaMemcpyDeviceToHost);

        // Find best candidate: minimum valid distance (OpenMP parallel)
        int best = -1;
        double best_d = 1e300;
        #pragma omp parallel for reduction(min:best_d) schedule(static)
        for (int i = 0; i < point_count; i++) {
            if (h_distances[i] >= 0.0 && h_distances[i] < best_d) {
                best_d = h_distances[i];
            }
        }
        // Second pass to find the index
        for (int i = 0; i < point_count; i++) {
            if (h_distances[i] >= 0.0 && h_distances[i] == best_d) {
                best = i;
                break;
            }
        }

        if (best < 0 || best_d >= threshold) break;

        in_cluster[best] = 1;
        members.push_back(best);
    }

    // Compute result BEFORE copying members
    int result = static_cast<int>(members.size());
    if (cluster_members_out) {
        *cluster_members_out = members;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Data generation (OpenMP-parallelized)
// ---------------------------------------------------------------------------

void generateSyntheticData(std::vector<double>& px, std::vector<double>& py,
                           const int N, unsigned int seed = 42) {
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    unsigned int local_seed = seed;

    auto frand = [&local_seed]() mutable {
        return rand_r(&local_seed) / static_cast<double>(RAND_MAX);
    };

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

        while (group_cnt > 0 && count < N) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;

            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                group_cnt--;
                continue;
            }

            px[count] = x;
            py[count] = y;
            count++;
            group_cnt--;
        }
    }
}

// ---------------------------------------------------------------------------
// Distance helper
// ---------------------------------------------------------------------------

inline double distance_pt(const double x1, const double y1,
                          const double x2, const double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (MPI + CUDA + OpenMP)
// ---------------------------------------------------------------------------

struct ClusterResult {
    int seed_point;
    int cardinality;
    std::vector<int> members;
};

std::vector<ClusterResult> qtClustering(
    const double* px, const double* py,
    double threshold, int N,
    int mpi_rank, int mpi_size)
{
    std::vector<int> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<ClusterResult> clusters;

    for (int i = 0; i < N; i++) {
        unclustered_indices.push_back(i);
    }

    while (!unclustered_indices.empty()) {
        int total = static_cast<int>(unclustered_indices.size());

        // Calculate local seed range for this rank
        int base = total / mpi_size;
        int remainder = total % mpi_size;
        int local_start = (mpi_rank < remainder) ? mpi_rank * (base + 1)
                                                  : remainder * base +
                                                    (mpi_rank - remainder) * base;
        int local_end = local_start +
            ((mpi_rank < remainder) ? (base + 1) : base);

        ClusterResult local_best;
        local_best.seed_point = -1;
        local_best.cardinality = 0;

        // Sequential seed evaluation (GPU requires serialization)
        for (int si = local_start; si < local_end; si++) {
            int seed = unclustered_indices[si];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            int cardinality = generateCandidateCluster(
                seed, clustered, px, py, threshold, N,
                &candidate_members);

            if (cardinality > local_best.cardinality) {
                local_best.cardinality = cardinality;
                local_best.seed_point = seed;
                local_best.members = std::move(candidate_members);
            }
        }

        // --- MPI: find global best cluster ---
        ClusterResult global_best;
        global_best.seed_point = -1;
        global_best.cardinality = 0;

        // Allreduce on cardinality
        int global_max_card = 0;
        MPI_Allreduce(&local_best.cardinality, &global_max_card, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_card <= 0) break;

        // Gather cardinalities from all ranks
        int local_card = local_best.cardinality;
        int* all_cards = new int[mpi_size];
        MPI_Gather(&local_card, 1, MPI_INT, all_cards, 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        // Find best rank
        int best_rank = 0;
        if (mpi_rank == 0) {
            for (int r = 1; r < mpi_size; r++) {
                if (all_cards[r] > all_cards[best_rank]) {
                    best_rank = r;
                }
            }
        }
        MPI_Bcast(&best_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast best cluster from best_rank to all ranks
        if (mpi_rank == best_rank) {
            global_best = local_best;  // Copy local best to global best
            MPI_Bcast(&local_card, 1, MPI_INT, mpi_rank, MPI_COMM_WORLD);
            if (local_card > 0) {
                MPI_Bcast(local_best.members.data(), local_card,
                          MPI_INT, mpi_rank, MPI_COMM_WORLD);
            }
            MPI_Bcast(&local_best.seed_point, 1, MPI_INT,
                      mpi_rank, MPI_COMM_WORLD);
        } else {
            int recv_card = 0;
            MPI_Bcast(&recv_card, 1, MPI_INT, best_rank, MPI_COMM_WORLD);
            if (recv_card > 0) {
                global_best.members.resize(recv_card);
                MPI_Bcast(global_best.members.data(), recv_card,
                          MPI_INT, best_rank, MPI_COMM_WORLD);
            }
            MPI_Bcast(&global_best.seed_point, 1, MPI_INT,
                      best_rank, MPI_COMM_WORLD);
            global_best.cardinality = recv_card;
        }

        delete[] all_cards;

        // Mark members as clustered and update unclustered list
        if (global_best.cardinality > 0) {
            for (int m : global_best.members) {
                clustered[m] = 1;
            }

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(),
                               unclustered_indices.end(),
                               [&clustered](int idx) {
                                   return clustered[idx];
                               }),
                unclustered_indices.end());
        } else {
            break;
        }

        clusters.push_back(global_best);
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool validateClusters(const std::vector<ClusterResult>& clusters,
                      const double* px, const double* py,
                      double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); c++) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); i++) {
            for (size_t j = i + 1; j < cluster.members.size(); j++) {
                double dist = distance_pt(
                    px[cluster.members[i]], py[cluster.members[i]],
                    px[cluster.members[j]], py[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<std::pair<int, int>> point_cluster;
    for (size_t c = 0; c < clusters.size(); c++) {
        for (int m : clusters[c].members) {
            point_cluster.push_back({m, static_cast<int>(c)});
        }
    }
    std::sort(point_cluster.begin(), point_cluster.end());

    for (size_t i = 1; i < point_cluster.size(); i++) {
        if (point_cluster[i].first == point_cluster[i-1].first) {
            printf("ERROR: Point %d in multiple clusters (%d and %d)\n",
                   point_cluster[i].first, point_cluster[i-1].second,
                   point_cluster[i].second);
            valid = false;
        }
    }

    int clustered_count = static_cast<int>(point_cluster.size());
    printf("Total points: clustered=%d\n", clustered_count);

    return valid;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; i++) {
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
               num_points, threshold);
        return 1;
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Initialize CUDA - select GPU for this rank
    int num_gpus = 1;
    cudaGetDeviceCount(&num_gpus);
    int gpu_id = mpi_rank % num_gpus;
    cudaSetDevice(gpu_id);

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (MPI=%d, OpenMP, CUDA)\n", mpi_size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<double> px(num_points);
    std::vector<double> py(num_points);

    if (mpi_rank == 0) {
        generateSyntheticData(px, py, num_points);
    }

    // Broadcast data to all ranks
    MPI_Bcast(px.data(), num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(py.data(), num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize CUDA device memory
    initDeviceMemory(num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    std::vector<ClusterResult> clusters = qtClustering(
        px.data(), py.data(), threshold, num_points, mpi_rank, mpi_size);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate statistics
    int total_clustered = 0;
    int max_cluster_size = 0;
    for (const auto& cl : clusters) {
        int size = static_cast<int>(cl.members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }

    double avg_cluster_size = clusters.empty() ? 0.0 :
        static_cast<double>(total_clustered) / clusters.size();

    // Print results from rank 0
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        double time_sec = cluster_time.count() / 1000.0;
        if (time_sec > 0) {
            printf("Performance: %.1f clusters/s, %.1f points/s\n",
                   clusters.size() / time_sec, num_points / time_sec);
        }
    }

    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); c++) {
            for (int m : clusters[c].members) {
                membership[m] = static_cast<int>(c);
            }
        }
        for (int m : membership) {
            membershipData.push_back(static_cast<double>(m));
        }
        print_results(membershipData, "ClusterMembership");
    }

    // Validation
    if (validate && mpi_rank == 0) {
        bool valid = validateClusters(clusters, px.data(), py.data(), threshold);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    freeDeviceMemory();
    MPI_Finalize();

    return 0;
}
