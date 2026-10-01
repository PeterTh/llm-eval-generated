// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

struct CudaContext {
    Point* d_points = nullptr;
    int* d_members = nullptr;
    double* d_max_dist = nullptr;
    int point_count = 0;
};

inline void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void computeMaxDistances(const Point* points,
                                    const int* members,
                                    const int member_count,
                                    const int point_count,
                                    double* max_dist) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;

    const Point p = points[idx];
    double max_d = 0.0;
    for (int i = 0; i < member_count; ++i) {
        const Point m = points[members[i]];
        const double dx = p.x - m.x;
        const double dy = p.y - m.y;
        const double dist = dx * dx + dy * dy;
        if (dist > max_d) {
            max_d = dist;
        }
    }
    max_dist[idx] = max_d;
}

void initCudaContext(const std::vector<Point>& points, CudaContext& ctx) {
    ctx.point_count = static_cast<int>(points.size());
    checkCuda(cudaMalloc(&ctx.d_points, ctx.point_count * sizeof(Point)), "cudaMalloc points");
    checkCuda(cudaMalloc(&ctx.d_members, ctx.point_count * sizeof(int)), "cudaMalloc members");
    checkCuda(cudaMalloc(&ctx.d_max_dist, ctx.point_count * sizeof(double)), "cudaMalloc max_dist");
    checkCuda(cudaMemcpy(ctx.d_points, points.data(),
                         ctx.point_count * sizeof(Point),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy points");
}

void destroyCudaContext(CudaContext& ctx) {
    cudaFree(ctx.d_points);
    cudaFree(ctx.d_members);
    cudaFree(ctx.d_max_dist);
    ctx.d_points = nullptr;
    ctx.d_members = nullptr;
    ctx.d_max_dist = nullptr;
    ctx.point_count = 0;
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     CudaContext& cuda_ctx,
                     std::vector<double>& host_max_dist,
                     const double threshold_sq,
                     const int point_count) {
    const int member_count = static_cast<int>(cluster_members.size());
    checkCuda(cudaMemcpy(cuda_ctx.d_members, cluster_members.data(),
                         member_count * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy members");

    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;
    computeMaxDistances<<<blocks, threads>>>(cuda_ctx.d_points, cuda_ctx.d_members,
                                             member_count, point_count, cuda_ctx.d_max_dist);
    checkCuda(cudaGetLastError(), "computeMaxDistances launch");
    checkCuda(cudaDeviceSynchronize(), "computeMaxDistances sync");
    checkCuda(cudaMemcpy(host_max_dist.data(), cuda_ctx.d_max_dist,
                         point_count * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy max_dist");

    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

#pragma omp parallel
    {
        int local_closest = -1;
        double local_min = std::numeric_limits<double>::max();

#pragma omp for nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            const double max_dist = host_max_dist[candidate];
            if (max_dist < threshold_sq &&
                (max_dist < local_min ||
                 (max_dist == local_min &&
                  (local_closest < 0 || candidate < local_closest)))) {
                local_min = max_dist;
                local_closest = candidate;
            }
        }

#pragma omp critical
        {
            if (local_closest >= 0 &&
                (local_min < min_diameter ||
                 (local_min == min_diameter &&
                  (closest_point < 0 || local_closest < closest_point)))) {
                min_diameter = local_min;
                closest_point = local_closest;
            }
        }
    }

    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                             const std::vector<bool>& clustered,
                             const double threshold,
                             const int point_count,
                             CudaContext& cuda_ctx,
                             std::vector<double>& host_max_dist,
                             std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    const double threshold_sq = threshold * threshold;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, cuda_ctx,
                                             host_max_dist, threshold_sq, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  CudaContext& cuda_ctx,
                                  const int mpi_rank,
                                  const int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    std::vector<double> host_max_dist(N);

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_best_size = -1;
        int local_best_seed = INT_MAX;

        // Try each unclustered point as a seed (distributed by seed id)
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed] || (seed % mpi_size) != mpi_rank) continue;

            const int cardinality = generateCandidateCluster(seed, clustered,
                                                             threshold, N,
                                                             cuda_ctx,
                                                             host_max_dist);

            if (cardinality > local_best_size ||
                (cardinality == local_best_size && seed < local_best_seed)) {
                local_best_size = cardinality;
                local_best_seed = seed;
            }
        }

        int local_pair[2];
        local_pair[0] = local_best_size;
        local_pair[1] = (local_best_size >= 0) ? local_best_seed : INT_MAX;

        int global_pair[2];
        MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        const int best_size = global_pair[0];
        if (best_size < 0) {
            break;
        }

        const int best_seed = global_pair[1];
        const int owner = best_seed % mpi_size;

        std::vector<int> best_cluster_members;
        if (mpi_rank == owner) {
            generateCandidateCluster(best_seed, clustered, threshold, N,
                                     cuda_ctx, host_max_dist,
                                     &best_cluster_members);
        }

        int member_count = (mpi_rank == owner)
                               ? static_cast<int>(best_cluster_members.size())
                               : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (mpi_rank != owner) {
            best_cluster_members.resize(member_count);
        }
        MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);

        if (best_seed >= 0 && member_count > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }

            // Remove clustered points from unclustered list
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
    bool validate = false;
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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int device_count = 0;
    const cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {
        if (mpi_rank == 0) {
            printf("Error: No CUDA-capable device detected.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(mpi_rank % device_count), "cudaSetDevice");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    CudaContext cuda_ctx;
    initCudaContext(points, cuda_ctx);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold,
                                                       cuda_ctx, mpi_rank, mpi_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long cluster_time_ms = cluster_time.count();
    MPI_Allreduce(MPI_IN_PLACE, &cluster_time_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(cluster_time_ms);
    
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    if (mpi_rank == 0) {
#pragma omp parallel for reduction(+:total_clustered) reduction(max:max_cluster_size)
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
    }
    
    const double avg_cluster_size = clusters.empty() ? 0.0 :
        static_cast<double>(total_clustered) / clusters.size();
    
    if (mpi_rank == 0) {
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
    }
    
    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    if (mpi_rank == 0) {
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        // Serialize cluster membership for hashing
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
    
    // Validation
    int exit_code = 0;
    if (validate && mpi_rank == 0) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
            exit_code = 0;
        } else {
            printf("Validation: FAILED\n");
            exit_code = 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    destroyCudaContext(cuda_ctx);
    MPI_Finalize();
    return exit_code;
}
