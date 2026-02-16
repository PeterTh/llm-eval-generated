// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Version
// Adds hybrid parallelism: MPI distributes seed evaluation across ranks,
// OpenMP parallelizes inner loops on host, CUDA computes full pairwise
// distance matrix on the device for fast distance queries.

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

// Simple CUDA error check
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { 
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); 
    MPI_Abort(MPI_COMM_WORLD, -1);
}} while(0)

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// CUDA kernel to compute full NxN distance matrix (row-major: dist[i*N + j])
extern "C" __global__ void computeDistanceMatrix(const Point* pts, double* dist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx >= total) return;
    int i = idx / N;
    int j = idx % N;
    double dx = pts[i].x - pts[j].x;
    double dy = pts[i].y - pts[j].y;
    dist[idx] = sqrt(dx*dx + dy*dy);
}

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
            const double dy = std::sqrt(fmax(0.0, r * r - dx * dx)) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count] = {x, y};
            count++; group_cnt--;
        }
    }
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Uses precomputed distance matrix 'dist' (row-major, size N*N)
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const double* dist, // pointer to N x N matrix
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Parallelize candidate loop with OpenMP; each thread keeps a local best
    #pragma omp parallel
    {
        int local_best = -1;
        double local_min = std::numeric_limits<double>::max();
        #pragma omp for nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double max_dist = 0.0;
            // compute max distance from candidate to all members
            for (size_t i = 0; i < cluster_members.size(); ++i) {
                const int member = cluster_members[i];
                double d = dist[member * point_count + candidate];
                max_dist = (d > max_dist) ? d : max_dist;
                if (max_dist >= local_min) break; // prune
            }
            if (max_dist < threshold && max_dist < local_min) {
                local_min = max_dist;
                local_best = candidate;
            }
        }
        #pragma omp critical
        {
            if (local_best >= 0 && local_min < min_diameter) {
                min_diameter = local_min;
                closest_point = local_best;
            }
        }
    }
    return closest_point;
}

// Generate a candidate cluster starting from a seed point using distance matrix
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const double* dist,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, dist, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm with MPI coordination
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const double* dist) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices; unclustered_indices.reserve(N);
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    int world_size=1, rank=0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (!unclustered_indices.empty()) {
        int local_max = -1;
        int local_seed = -1;
        std::vector<int> local_members;
        // Partition seeds among ranks by index
        for (size_t idx = rank; idx < unclustered_indices.size(); idx += world_size) {
            const int seed = unclustered_indices[idx];
            if (clustered[seed]) continue;
            std::vector<int> candidate_members;
            int card = generateCandidateCluster(seed, clustered, dist, threshold, N, &candidate_members);
            if (card > local_max) {
                local_max = card;
                local_seed = seed;
                local_members.swap(candidate_members);
            }
        }
        // Gather local_max and seeds to all ranks to pick global best
        std::vector<int> all_max(world_size);
        std::vector<int> all_seed(world_size);
        MPI_Allgather(&local_max, 1, MPI_INT, all_max.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&local_seed, 1, MPI_INT, all_seed.data(), 1, MPI_INT, MPI_COMM_WORLD);

        int global_max = -1;
        int winner = -1;
        for (int r = 0; r < world_size; ++r) {
            if (all_max[r] > global_max) { global_max = all_max[r]; winner = r; }
        }
        if (global_max <= 0) break; // no more clusters

        // Broadcast members from winner to all ranks
        int members_size = 0;
        if (rank == winner) members_size = static_cast<int>(local_members.size());
        MPI_Bcast(&members_size, 1, MPI_INT, winner, MPI_COMM_WORLD);
        std::vector<int> best_members(members_size);
        if (rank == winner) {
            if (members_size) std::memcpy(best_members.data(), local_members.data(), members_size * sizeof(int));
        }
        if (members_size) MPI_Bcast(best_members.data(), members_size, MPI_INT, winner, MPI_COMM_WORLD);

        // All ranks update clustered state and the cluster list
        Cluster cluster;
        cluster.seed_point = (rank == winner) ? local_seed : all_seed[winner];
        cluster.members = best_members;
        clusters.push_back(cluster);
        for (int m : best_members) clustered[m] = true;

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dx = points[cluster.members[i]].x - points[cluster.members[j]].x;
                const double dy = points[cluster.members[i]].y - points[cluster.members[j]].y;
                const double dist = sqrt(dx*dx + dy*dy);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        if (rank == 0 && c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        if (max_diameter > threshold * 1.001) { if (rank == 0) printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter, threshold); valid = false; }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) { if (rank == 0) printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c); valid = false; }
            membership[member] = static_cast<int>(c);
        }
    }
    int clustered_count = 0; for (size_t i = 0; i < membership.size(); ++i) if (membership[i] >= 0) clustered_count++;
    if (rank == 0) printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count, points.size() - clustered_count);
    return valid;
}

void printUsage(const char* progName) {
    int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Usage: %s [options]\n", progName);
        printf("Options:\n");
        printf("  -n <num>     Number of points (default: 1000)\n");
        printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
        printf("  -v           Enable validation\n");
        printf("  -r           Print results for external validation\n");
        printf("  -h           Show this help message\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 prints usage/errors)
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (num_points <= 0 || threshold <= 0.0) { if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold); MPI_Finalize(); return 1; }

    if (rank == 0) {
        printf("QT Clustering Benchmark (hybrid MPI/OpenMP/CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", world_size, omp_get_max_threads());
    }

    // Generate synthetic data on rank 0 and broadcast to all ranks
    std::vector<Point> points;
    points.resize(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Allocate and compute distance matrix on GPU (rank 0 computes and broadcasts to others)
    std::vector<double> dist_host;
    dist_host.resize(static_cast<size_t>(num_points) * num_points);

    if (rank == 0) {
        Point* d_points = nullptr; double* d_dist = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_points, num_points * sizeof(Point)));
        CUDA_CHECK(cudaMalloc((void**)&d_dist, num_points * (size_t)num_points * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_points, points.data(), num_points * sizeof(Point), cudaMemcpyHostToDevice));
        const size_t total = (size_t)num_points * num_points;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);
        computeDistanceMatrix<<<blocks, threads>>>(d_points, d_dist, num_points);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dist_host.data(), d_dist, total * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_points)); CUDA_CHECK(cudaFree(d_dist));
    }
    // Broadcast distance matrix to all ranks
    MPI_Bcast(dist_host.data(), num_points * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform QT clustering (all ranks compute the same clusters via coordinated MPI algorithm)
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, dist_host.data());
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0; int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size; max_cluster_size = std::max(max_cluster_size, size);
        }
        const double avg_cluster_size = clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        const double time_sec = cluster_time.count() / 1000.0; const double clusters_per_sec = clusters.size() / time_sec; const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData; membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) for (size_t i = 0; i < clusters[c].members.size(); ++i) membership[clusters[c].members[i]] = static_cast<int>(c);
            for (int m : membership) membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) { printf("Validation: PASSED\n"); MPI_Finalize(); return 0; } else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
