// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Version
// Uses MPI to distribute seed evaluation across ranks, OpenMP to parallelize within a rank,
// and CUDA to accelerate distance computations for cluster membership testing.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <iostream>

#include <mpi.h>
#include <omp.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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

#ifdef USE_CUDA
// CUDA kernel: compute distances from candidate to each cluster member
__global__ void compute_distances_kernel(const double* members_x, const double* members_y,
                                         double cand_x, double cand_y, double* out_dists, int mcount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mcount) {
        double dx = cand_x - members_x[idx];
        double dy = cand_y - members_y[idx];
        out_dists[idx] = sqrt(dx * dx + dy * dy);
    }
}

// GPU helper: compute maximum distance from candidate to cluster members using CUDA
static double gpu_max_distance(int candidate,
                               const std::vector<int>& cluster_members,
                               const std::vector<Point>& points) {
    const int mcount = static_cast<int>(cluster_members.size());
    if (mcount == 0) return 0.0;

    // Host arrays
    std::vector<double> h_members_x(mcount), h_members_y(mcount), h_dists(mcount);
    for (int i = 0; i < mcount; ++i) {
        const Point& p = points[cluster_members[i]];
        h_members_x[i] = p.x;
        h_members_y[i] = p.y;
    }
    const double cand_x = points[candidate].x;
    const double cand_y = points[candidate].y;

    // Device arrays
    double *d_members_x = nullptr, *d_members_y = nullptr, *d_dists = nullptr;
    const size_t bytes = mcount * sizeof(double);
    cudaError_t cerr;
    cerr = cudaMalloc(&d_members_x, bytes); if (cerr != cudaSuccess) { return 1e300; }
    cerr = cudaMalloc(&d_members_y, bytes); if (cerr != cudaSuccess) { cudaFree(d_members_x); return 1e300; }
    cerr = cudaMalloc(&d_dists, bytes);    if (cerr != cudaSuccess) { cudaFree(d_members_x); cudaFree(d_members_y); return 1e300; }

    cudaMemcpy(d_members_x, h_members_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_members_y, h_members_y.data(), bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (mcount + threads - 1) / threads;
    compute_distances_kernel<<<blocks, threads>>>(d_members_x, d_members_y, cand_x, cand_y, d_dists, mcount);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dists.data(), d_dists, bytes, cudaMemcpyDeviceToHost);

    double maxd = 0.0;
    for (int i = 0; i < mcount; ++i) maxd = std::max(maxd, h_dists[i]);

    cudaFree(d_members_x); cudaFree(d_members_y); cudaFree(d_dists);
    return maxd;
}
#else
// CPU fallback for GPU distance computation (used when CUDA not available)
static double gpu_max_distance(int candidate,
                               const std::vector<int>& cluster_members,
                               const std::vector<Point>& points) {
    double maxd = 0.0;
    for (size_t i = 0; i < cluster_members.size(); ++i) {
        const Point& p = points[cluster_members[i]];
        double dx = points[candidate].x - p.x;
        double dy = points[candidate].y - p.y;
        double d = std::sqrt(dx*dx + dy*dy);
        if (d > maxd) maxd = d;
    }
    return maxd;
}
#endif

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
        if (group_cnt > (N - count)) group_cnt = N - count;
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(std::max(0.0, r * r - dx * dx)) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count] = {x, y};
            count++; group_cnt--;
        }
    }
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Uses GPU to compute max distance from candidate to cluster members
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    // Parallelize candidate testing with OpenMP
    #pragma omp parallel
    {
        int local_closest = -1;
        double local_min = std::numeric_limits<double>::max();

        #pragma omp for schedule(dynamic, 64)
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            // Use GPU to compute max distance from candidate to cluster members
            double max_dist = gpu_max_distance(candidate, cluster_members, points);

            if (max_dist < threshold && max_dist < local_min) {
                local_min = max_dist;
                local_closest = candidate;
            }
        }

        #pragma omp critical
        {
            if (local_closest >= 0 && local_min < min_diameter) {
                min_diameter = local_min;
                closest_point = local_closest;
            }
        }
    }
    return closest_point;
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
        const int closest = findClosestPoint(members, clustered, in_cluster, points, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm using MPI to distribute seed evaluation and OpenMP to parallelize within rank
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;
    std::vector<Cluster> clusters;

    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    while (!unclustered_indices.empty()) {
        // Each rank evaluates only a subset of seeds (striping by point id)
        int local_max_card = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        // Parallelize seed evaluations with OpenMP within each rank
        #pragma omp parallel
        {
            int thread_max = -1;
            int thread_seed = -1;
            std::vector<int> thread_members;

            #pragma omp for schedule(dynamic)
            for (size_t idx = 0; idx < unclustered_indices.size(); ++idx) {
                int seed = unclustered_indices[idx];
                if ((seed % mpi_size) != mpi_rank) continue; // distribute seeds across ranks
                if (clustered[seed]) continue;

                std::vector<int> cand_members;
                int card = generateCandidateCluster(seed, clustered, points, threshold, N, &cand_members);

                if (card > thread_max) {
                    thread_max = card;
                    thread_seed = seed;
                    thread_members = std::move(cand_members);
                }
            }

            #pragma omp critical
            {
                if (thread_max > local_max_card) {
                    local_max_card = thread_max;
                    local_best_seed = thread_seed;
                    local_best_members = std::move(thread_members);
                }
            }
        }

        // Gather local bests to root (rank 0)
        struct { int card; int seed; } local_pair{local_max_card, local_best_seed}, global_pairs[256];
        // Note: assume mpi_size <= 256 for simplicity (typical clusters)
        MPI_Gather(&local_pair, 2, MPI_INT, global_pairs, 2, MPI_INT, 0, MPI_COMM_WORLD);

        int global_max = -1;
        int owner_rank = -1;
        int chosen_seed = -1;
        std::vector<int> chosen_members;

        if (mpi_rank == 0) {
            for (int r = 0; r < mpi_size; ++r) {
                int c = global_pairs[r].card;
                int s = global_pairs[r].seed;
                if (c > global_max) { global_max = c; owner_rank = r; chosen_seed = s; }
            }
            // Root collects members from owner_rank if not itself
            if (global_max > 0) {
                if (owner_rank == 0) {
                    chosen_members = local_best_members;
                } else {
                    // receive size then members
                    int msize = 0;
                    MPI_Recv(&msize, 1, MPI_INT, owner_rank, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    chosen_members.resize(msize);
                    if (msize > 0) MPI_Recv(chosen_members.data(), msize, MPI_INT, owner_rank, 124, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            // Non-root: send members to root if this rank is owner
            if (local_max_card > 0 && local_best_seed == local_pair.seed) {
                // if this rank might be owner, send to root when asked; but simpler: send to root if owner later
            }
        }

        // If this rank was owner and not root, send its members to root when root requests via tag
        // Determine if we are owner by having root broadcast owner_rank later.
        MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&owner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&chosen_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi_rank == owner_rank && owner_rank != 0) {
            int msize = static_cast<int>(local_best_members.size());
            MPI_Send(&msize, 1, MPI_INT, 0, 123, MPI_COMM_WORLD);
            if (msize > 0) MPI_Send(local_best_members.data(), msize, MPI_INT, 0, 124, MPI_COMM_WORLD);
        }

        // Root will broadcast the chosen members to all ranks
        int chosen_size = 0;
        if (mpi_rank == 0) chosen_size = static_cast<int>(chosen_members.size());
        MPI_Bcast(&chosen_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpi_rank != 0) chosen_members.resize(chosen_size);
        if (chosen_size > 0) MPI_Bcast(chosen_members.data(), chosen_size, MPI_INT, 0, MPI_COMM_WORLD);

        // If no cluster found globally, break
        if (global_max <= 0) break;

        // Add cluster on root and update clustered array on all ranks
        if (mpi_rank == 0) {
            Cluster cluster; cluster.seed_point = chosen_seed; cluster.members = chosen_members;
            clusters.push_back(cluster);
        }

        // Mark all members as clustered locally and rebuild unclustered_indices
        for (int m : chosen_members) clustered[m] = true;

        std::vector<int> new_unclustered;
        new_unclustered.reserve(unclustered_indices.size());
        for (int idx : unclustered_indices) if (!clustered[idx]) new_unclustered.push_back(idx);
        unclustered_indices.swap(new_unclustered);
    }

    // Ensure root's clusters is returned to all ranks by broadcasting serialized data if needed; here return root's clusters on all ranks for simplicity
    int root_cluster_count = 0;
    if (mpi_rank == 0) root_cluster_count = static_cast<int>(clusters.size());
    MPI_Bcast(&root_cluster_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) clusters.resize(root_cluster_count);
    // For simplicity, do not broadcast full cluster details to non-root in this implementation because validation and printing typically done on root.

    return clusters;
}

// Validation remains the same (CPU)
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
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, max_diameter);
        if (max_diameter > threshold * 1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter, threshold); valid = false; }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) for (size_t i = 0; i < clusters[c].members.size(); ++i) {
        int member = clusters[c].members[i]; if (membership[member] >= 0) { printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c); valid = false; } membership[member] = static_cast<int>(c);
    }
    int clustered_count = 0; for (size_t i = 0; i < membership.size(); ++i) if (membership[i] >= 0) clustered_count++;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count, points.size() - clustered_count);
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
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0, mpi_size = 1; MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Parse command line arguments (after MPI init to allow MPI args)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) { num_points = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) { threshold = atof(argv[++i]); }
        else if (strcmp(argv[i], "-v") == 0) { validate = true; }
        else if (strcmp(argv[i], "-r") == 0) { printResults = true; }
        else if (strcmp(argv[i], "-h") == 0) { if (mpi_rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        MPI_Finalize(); return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Set OpenMP threads from environment if provided
    const char* omp_env = getenv("OMP_NUM_THREADS"); if (omp_env) omp_set_num_threads(atoi(omp_env));

    // Generate synthetic data on all ranks (deterministic seed)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);

    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0; int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) { int s = static_cast<int>(clusters[i].members.size()); total_clustered += s; max_cluster_size = std::max(max_cluster_size, s); }
        double avg_cluster_size = clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        double time_sec = cluster_time.count() / 1000.0; printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters.size() / time_sec, num_points / time_sec);

        if (printResults) {
            std::vector<double> membershipData; membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) for (size_t i = 0; i < clusters[c].members.size(); ++i) membership[clusters[c].members[i]] = static_cast<int>(c);
            for (int m : membership) membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize(); return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
