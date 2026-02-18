// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// Parallelization strategy:
//  - CUDA: precompute the full pairwise squared-distance matrix (O(N^2)) on GPU once.
//  - MPI: distribute seed evaluations across ranks in each outer iteration.
//  - OpenMP: parallelize seed evaluations within each rank.
//
// Semantics match the original sequential implementation, including deterministic
// tie-breaking on seed order (first best seed in the unclustered list).

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

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void compute_dist2_kernel(const Point* __restrict__ points,
                                    double* __restrict__ dist2,
                                    int N) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= N || j >= N) return;
    const double dx = points[i].x - points[j].x;
    const double dy = points[i].y - points[j].y;
    dist2[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = dx * dx + dy * dy;
}

static void computeDistanceMatrixCUDA(const std::vector<Point>& points,
                                      std::vector<double>& dist2,
                                      int mpi_rank) {
    const int N = static_cast<int>(points.size());
    const size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const int device = mpi_rank % deviceCount;
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");

    Point* d_points = nullptr;
    double* d_dist2 = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_points), sizeof(Point) * static_cast<size_t>(N)), "cudaMalloc(d_points)");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_dist2), sizeof(double) * NN), "cudaMalloc(d_dist2)");

    cudaCheck(cudaMemcpy(d_points, points.data(), sizeof(Point) * static_cast<size_t>(N), cudaMemcpyHostToDevice),
              "cudaMemcpy(points)");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    compute_dist2_kernel<<<grid, block>>>(d_points, d_dist2, N);
    cudaCheck(cudaGetLastError(), "compute_dist2_kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaCheck(cudaMemcpy(dist2.data(), d_dist2, sizeof(double) * NN, cudaMemcpyDeviceToHost),
              "cudaMemcpy(dist2)");

    cudaCheck(cudaFree(d_dist2), "cudaFree(d_dist2)");
    cudaCheck(cudaFree(d_points), "cudaFree(d_points)");
}

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
        // Make sure we don't make more points than we need
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

// Calculate Euclidean distance between two points (used for validation/statistics only)
inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

static inline double dist2_at(const std::vector<double>& dist2, int N, int a, int b) {
    return dist2[static_cast<size_t>(a) * static_cast<size_t>(N) + static_cast<size_t>(b)];
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold.
// Uses squared distances to avoid sqrt while preserving ordering/threshold semantics.
static int findClosestPointDist2(const std::vector<int>& cluster_members,
                                 const std::vector<unsigned char>& clustered,
                                 const std::vector<unsigned char>& in_cluster,
                                 const std::vector<double>& dist2,
                                 const double threshold2,
                                 const int point_count) {
    int closest_point = -1;
    double min_diameter2 = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist2 = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double d2 = dist2_at(dist2, point_count, candidate, member);
            max_dist2 = std::max(max_dist2, d2);
        }

        if (max_dist2 < threshold2 && max_dist2 < min_diameter2) {
            min_diameter2 = max_dist2;
            closest_point = candidate;
        }
    }

    return closest_point;
}

static int generateCandidateClusterDist2(const int seed_point,
                                         const std::vector<unsigned char>& clustered,
                                         const std::vector<double>& dist2,
                                         const double threshold2,
                                         const int point_count,
                                         std::vector<int>* cluster_members = nullptr) {
    std::vector<unsigned char> in_cluster(static_cast<size_t>(point_count), 0);
    std::vector<int> members;
    members.reserve(static_cast<size_t>(point_count));

    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPointDist2(members, clustered, in_cluster, dist2, threshold2, point_count);
        if (closest < 0) break;
        in_cluster[closest] = 1;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

static std::vector<Cluster> qtClusteringHybridMPI(const std::vector<Point>& points,
                                                  const std::vector<double>& dist2,
                                                  const double threshold,
                                                  int mpi_rank,
                                                  int mpi_size) {
    const int N = static_cast<int>(points.size());
    const double threshold2 = threshold * threshold;

    std::vector<unsigned char> clustered(static_cast<size_t>(N), 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(static_cast<size_t>(N));

    std::vector<Cluster> clusters;
    if (mpi_rank == 0) clusters.reserve(static_cast<size_t>(N));

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    while (!unclustered_indices.empty()) {
        // Build local list of positions owned by this rank (block-cyclic by current unclustered order).
        std::vector<int> local_positions;
        local_positions.reserve((unclustered_indices.size() + static_cast<size_t>(mpi_size) - 1) / static_cast<size_t>(mpi_size));
        for (int pos = mpi_rank; pos < static_cast<int>(unclustered_indices.size()); pos += mpi_size) {
            local_positions.push_back(pos);
        }

        struct { int val; int loc; } local_best{ -1, std::numeric_limits<int>::max() };

        #pragma omp parallel
        {
            int t_val = -1;
            int t_loc = std::numeric_limits<int>::max();

            #pragma omp for schedule(dynamic, 1) nowait
            for (int k = 0; k < static_cast<int>(local_positions.size()); ++k) {
                const int pos = local_positions[k];
                const int seed = unclustered_indices[static_cast<size_t>(pos)];
                if (clustered[seed]) continue;

                const int cardinality = generateCandidateClusterDist2(seed, clustered, dist2, threshold2, N, nullptr);
                if (cardinality > t_val || (cardinality == t_val && pos < t_loc)) {
                    t_val = cardinality;
                    t_loc = pos;
                }
            }

            #pragma omp critical
            {
                if (t_val > local_best.val || (t_val == local_best.val && t_loc < local_best.loc)) {
                    local_best.val = t_val;
                    local_best.loc = t_loc;
                }
            }
        }

        struct { int val; int loc; } global_best{ -1, std::numeric_limits<int>::max() };
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global_best.val < 0 || global_best.loc < 0 || global_best.loc >= static_cast<int>(unclustered_indices.size())) {
            break;
        }

        const int best_pos = global_best.loc;
        const int best_seed = unclustered_indices[static_cast<size_t>(best_pos)];
        const int owner = best_pos % mpi_size;

        std::vector<int> best_cluster_members;
        int member_count = 0;

        if (mpi_rank == owner) {
            const int card = generateCandidateClusterDist2(best_seed, clustered, dist2, threshold2, N, &best_cluster_members);
            member_count = card;
        }

        MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        best_cluster_members.resize(static_cast<size_t>(member_count));
        MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(std::move(cluster));
        }

        for (int idx : best_cluster_members) clustered[idx] = 1;

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[static_cast<size_t>(idx)] != 0; }),
            unclustered_indices.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        // Check diameter (max distance between any two points)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
    // Check for duplicate memberships
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
    
    // Count clustered points
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

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int printResults = 0;

    if (mpi_rank == 0) {
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

        if (num_points <= 0 || threshold <= 0.0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            MPI_Finalize();
            return 1;
        }

        printf("QT Clustering Benchmark (Hybrid MPI + OpenMP + CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Point> points(static_cast<size_t>(num_points));
    generateSyntheticData(points, num_points);

    // Allocate and compute pairwise squared distances (CUDA, per-rank).
    const size_t NN = static_cast<size_t>(num_points) * static_cast<size_t>(num_points);
    std::vector<double> dist2;
    try {
        dist2.resize(NN);
    } catch (...) {
        if (mpi_rank == 0) fprintf(stderr, "Failed to allocate distance matrix (%zu doubles)\n", NN);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    computeDistanceMatrixCUDA(points, dist2, mpi_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringHybridMPI(points, dist2, threshold, mpi_rank, mpi_size);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_end = std::chrono::high_resolution_clock::now();

    const long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start).count();
    long long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Clustering time: %lld ms\n", max_ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }

        const double avg_cluster_size = clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = max_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / std::max(1e-12, time_sec);
        const double points_per_sec = num_points / std::max(1e-12, time_sec);
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(static_cast<size_t>(num_points));
            std::vector<int> membership(static_cast<size_t>(num_points), -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                    membership[static_cast<size_t>(clusters[c].members[i])] = static_cast<int>(c);
                }
            }
            for (int m : membership) membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool ok = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
