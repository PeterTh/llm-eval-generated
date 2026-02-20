// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization:
//   CUDA  - precomputes the NxN pairwise distance matrix on GPU
//   MPI   - distributes seed candidates across ranks each iteration
//   OpenMP - parallelizes seed processing within each MPI rank

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

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters
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

// Calculate Euclidean distance between two points (used by validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// CUDA kernel: compute full NxN pairwise distance matrix
__global__ void computeDistanceMatrixKernel(const double* __restrict__ px,
                                             const double* __restrict__ py,
                                             double* __restrict__ dist,
                                             int N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(N) * N;
    if (idx >= total) return;
    int i = static_cast<int>(idx / N);
    int j = static_cast<int>(idx % N);
    double dx = px[i] - px[j];
    double dy = py[i] - py[j];
    dist[idx] = sqrt(dx * dx + dy * dy);
}

// Compute distance matrix on GPU
void computeDistanceMatrixGPU(const std::vector<Point>& points,
                               std::vector<double>& dist_matrix, int N) {
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    size_t N2 = static_cast<size_t>(N) * N;
    double *d_px, *d_py, *d_dist;

    cudaMalloc(&d_px, N * sizeof(double));
    cudaMalloc(&d_py, N * sizeof(double));
    cudaMalloc(&d_dist, N2 * sizeof(double));

    cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = static_cast<int>((N2 + threads_per_block - 1) / threads_per_block);
    computeDistanceMatrixKernel<<<num_blocks, threads_per_block>>>(d_px, d_py, d_dist, N);
    cudaDeviceSynchronize();

    dist_matrix.resize(N2);
    cudaMemcpy(dist_matrix.data(), d_dist, N2 * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_dist);
}

// Find the closest unclustered point using precomputed distance matrix
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const double* dist_matrix,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    const int num_members = static_cast<int>(cluster_members.size());

    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist = 0.0;
        for (int mi = 0; mi < num_members; ++mi) {
            const double dist = dist_matrix[static_cast<size_t>(candidate) * point_count
                                            + cluster_members[mi]];
            if (dist > max_dist) max_dist = dist;
        }

        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// Generate a candidate cluster starting from a seed point
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const double* dist_matrix,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster,
                                              dist_matrix, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm - MPI distributes seeds, OpenMP parallelises within rank
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold,
                                   const double* dist_matrix) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // Each MPI rank handles a round-robin subset of seeds
        std::vector<int> my_seed_positions;
        for (int i = rank; i < num_seeds; i += nprocs) {
            my_seed_positions.push_back(i);
        }
        const int my_count = static_cast<int>(my_seed_positions.size());

        int local_best_card = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        // OpenMP: each thread finds its thread-local best, then merge
        #pragma omp parallel
        {
            int t_best_card = -1;
            int t_best_seed = -1;
            std::vector<int> t_best_members;

            #pragma omp for schedule(dynamic) nowait
            for (int k = 0; k < my_count; ++k) {
                const int seed = unclustered_indices[my_seed_positions[k]];

                std::vector<int> candidate_members;
                const int cardinality = generateCandidateCluster(
                    seed, clustered, dist_matrix, threshold, N, &candidate_members);

                if (cardinality > t_best_card ||
                    (cardinality == t_best_card && seed < t_best_seed)) {
                    t_best_card = cardinality;
                    t_best_seed = seed;
                    t_best_members = std::move(candidate_members);
                }
            }

            #pragma omp critical
            {
                if (t_best_card > local_best_card ||
                    (t_best_card == local_best_card &&
                     t_best_seed >= 0 &&
                     (local_best_seed < 0 || t_best_seed < local_best_seed))) {
                    local_best_card = t_best_card;
                    local_best_seed = t_best_seed;
                    local_best_members = std::move(t_best_members);
                }
            }
        }

        // MPI: gather (cardinality, seed) from every rank and pick global best
        int local_info[2] = {local_best_card, local_best_seed};
        std::vector<int> all_info(2 * nprocs);
        MPI_Allgather(local_info, 2, MPI_INT,
                      all_info.data(), 2, MPI_INT, MPI_COMM_WORLD);

        int global_best_card = -1;
        int global_best_seed = -1;
        int winning_rank = -1;
        for (int r = 0; r < nprocs; ++r) {
            int r_card = all_info[2 * r];
            int r_seed = all_info[2 * r + 1];
            if (r_card > global_best_card ||
                (r_card == global_best_card && r_seed >= 0 &&
                 (global_best_seed < 0 || r_seed < global_best_seed))) {
                global_best_card = r_card;
                global_best_seed = r_seed;
                winning_rank = r;
            }
        }

        if (winning_rank < 0 || global_best_card <= 0) break;

        // Broadcast winning cluster members from the rank that found the best
        int member_count = 0;
        if (rank == winning_rank) member_count = static_cast<int>(local_best_members.size());
        MPI_Bcast(&member_count, 1, MPI_INT, winning_rank, MPI_COMM_WORLD);

        std::vector<int> best_members(member_count);
        if (rank == winning_rank) best_members = local_best_members;
        MPI_Bcast(best_members.data(), member_count, MPI_INT, winning_rank, MPI_COMM_WORLD);

        // All ranks update state identically
        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (size_t i = 0; i < best_members.size(); ++i) {
            clustered[best_members[i]] = true;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPUs round-robin across MPI ranks
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(rank % num_devices);
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
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               nprocs, omp_get_max_threads(), num_devices);
    }

    // All ranks generate identical data (deterministic seed)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // CUDA: precompute NxN distance matrix on GPU
    std::vector<double> dist_matrix;
    computeDistanceMatrixGPU(points, dist_matrix, num_points);

    // Perform QT clustering with MPI + OpenMP
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold,
                                                        dist_matrix.data());

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (rank == 0) {
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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
