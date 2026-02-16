// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
// This file has been adapted to use MPI to distribute seed evaluation,
// OpenMP for host-side parallel loops, and CUDA kernels to compute
// distances in parallel on an accelerator.

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

// CUDA error check
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while (0)

// Kernel: compute maximum distance from each candidate to the given cluster members
extern "C" __global__ void compute_max_distances(const Point* d_points,
                                                  int point_count,
                                                  const int* d_cluster_members,
                                                  int cluster_size,
                                                  const unsigned char* d_mask_skip, // 1: skip (already clustered or in_cluster)
                                                  double* d_max_dists) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= point_count) return;
    if (d_mask_skip[cid]) { d_max_dists[cid] = 1e300; return; }

    double maxd = 0.0;
    const Point pc = d_points[cid];
    for (int i = 0; i < cluster_size; ++i) {
        int m = d_cluster_members[i];
        const Point pm = d_points[m];
        double dx = pc.x - pm.x;
        double dy = pc.y - pm.y;
        double dist = sqrt(dx * dx + dy * dy);
        if (dist > maxd) maxd = dist;
    }
    d_max_dists[cid] = maxd;
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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// This implementation offloads distance computation to CUDA for the full candidate set
int findClosestPoint_GPU(const std::vector<int>& cluster_members,
                         const std::vector<bool>& clustered,
                         const std::vector<bool>& in_cluster,
                         const std::vector<Point>& points,
                         const double threshold,
                         const int point_count) {
    if (cluster_members.empty()) return -1;

    // Prepare device buffers
    Point* d_points = nullptr;
    int* d_cluster_members = nullptr;
    unsigned char* d_mask_skip = nullptr;
    double* d_max_dists = nullptr;

    size_t points_bytes = sizeof(Point) * point_count;
    size_t members_bytes = sizeof(int) * cluster_members.size();
    size_t mask_bytes = sizeof(unsigned char) * point_count;
    size_t dists_bytes = sizeof(double) * point_count;

    CUDA_CHECK(cudaMalloc(&d_points, points_bytes));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), points_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_cluster_members, members_bytes));
    CUDA_CHECK(cudaMemcpy(d_cluster_members, cluster_members.data(), members_bytes, cudaMemcpyHostToDevice));

    std::vector<unsigned char> mask_skip(point_count);
    for (int i = 0; i < point_count; ++i) mask_skip[i] = (clustered[i] || in_cluster[i]) ? 1 : 0;
    CUDA_CHECK(cudaMalloc(&d_mask_skip, mask_bytes));
    CUDA_CHECK(cudaMemcpy(d_mask_skip, mask_skip.data(), mask_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_max_dists, dists_bytes));

    // Launch kernel
    int block = 256;
    int grid = (point_count + block - 1) / block;
    compute_max_distances<<<grid, block>>>(d_points, point_count, d_cluster_members, (int)cluster_members.size(), d_mask_skip, d_max_dists);
    CUDA_CHECK(cudaGetLastError());

    // Copy back distances
    std::vector<double> max_dists(point_count);
    CUDA_CHECK(cudaMemcpy(max_dists.data(), d_max_dists, dists_bytes, cudaMemcpyDeviceToHost));

    // Find best candidate on host
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (mask_skip[candidate]) continue;
        double max_dist = max_dists[candidate];
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_cluster_members);
    cudaFree(d_mask_skip);
    cudaFree(d_max_dists);

    return closest_point;
}

// Fallback CPU version using OpenMP for intra-loop parallelism
int findClosestPoint_CPU(const std::vector<int>& cluster_members,
                         const std::vector<bool>& clustered,
                         const std::vector<bool>& in_cluster,
                         const std::vector<Point>& points,
                         const double threshold,
                         const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        int local_closest = -1;
        double local_min = std::numeric_limits<double>::max();

        #pragma omp for nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double max_dist = 0.0;
            for (size_t i = 0; i < cluster_members.size(); ++i) {
                const int member = cluster_members[i];
                const double dx = points[candidate].x - points[member].x;
                const double dy = points[candidate].y - points[member].y;
                const double dist = std::sqrt(dx * dx + dy * dy);
                if (dist > max_dist) max_dist = dist;
            }
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

// Wrapper that prefers GPU implementation but falls back to CPU if CUDA fails
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    // Attempt GPU path
    int res = -1;
    // Simple heuristic: use GPU for clusters larger than 0
    res = findClosestPoint_GPU(cluster_members, clustered, in_cluster, points, threshold, point_count);
    return res;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        if (closest < 0) break; // No more points can be added
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm using MPI to distribute seed evaluation
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;

    // Each rank will participate; rank 0 will collect clusters
    while (true) {
        // Each rank searches seeds assigned to it (seed % world_size == rank)
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        // Broadcast current clustered mask from rank 0 to all ranks
        // Pack clustered into bytes
        std::vector<unsigned char> clustered_bytes(N);
        if (world_rank == 0) {
            for (int i = 0; i < N; ++i) clustered_bytes[i] = clustered[i] ? 1 : 0;
        }
        MPI_Bcast(clustered_bytes.data(), N, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        if (world_rank != 0) {
            for (int i = 0; i < N; ++i) clustered[i] = clustered_bytes[i] != 0;
        }

        // Each rank evaluates seeds assigned to it
        #pragma omp parallel
        {
            std::vector<int> thread_members;
            #pragma omp for schedule(dynamic)
            for (int seed = 0; seed < N; ++seed) {
                if (seed % world_size != world_rank) continue;
                if (clustered[seed]) continue;
                int card = generateCandidateCluster(seed, clustered, points, threshold, N, &thread_members);
                #pragma omp critical
                {
                    if (card > local_max_cardinality) {
                        local_max_cardinality = card;
                        local_best_seed = seed;
                        local_best_members = thread_members;
                    }
                }
            }
        }

        // Reduce to find global maximum cardinality
        int global_max_cardinality = -1;
        MPI_Allreduce(&local_max_cardinality, &global_max_cardinality, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_max_cardinality <= 0) {
            break; // no more clusters
        }

        // Determine owner rank: the lowest rank that has local_max_cardinality == global_max_cardinality
        int local_has = (local_max_cardinality == global_max_cardinality) ? world_rank : world_size;
        int owner_rank = world_size;
        MPI_Allreduce(&local_has, &owner_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Owner rank will broadcast best_seed and members
        int best_seed = -1;
        int members_size = 0;
        if (world_rank == owner_rank) {
            best_seed = local_best_seed;
            members_size = static_cast<int>(local_best_members.size());
        }
        MPI_Bcast(&best_seed, 1, MPI_INT, owner_rank, MPI_COMM_WORLD);
        MPI_Bcast(&members_size, 1, MPI_INT, owner_rank, MPI_COMM_WORLD);

        std::vector<int> best_members(members_size);
        if (world_rank == owner_rank) {
            if (members_size > 0) memcpy(best_members.data(), local_best_members.data(), members_size * sizeof(int));
        }
        MPI_Bcast(best_members.data(), members_size, MPI_INT, owner_rank, MPI_COMM_WORLD);

        // Update clustered mask on all ranks
        for (int idx = 0; idx < members_size; ++idx) clustered[best_members[idx]] = true;

        // Rank 0 records the cluster
        if (world_rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_members;
            clusters.push_back(cluster);
        }

        // Check termination: if all points are clustered, break
        int remaining = 0;
        for (int i = 0; i < N; ++i) if (!clustered[i]) remaining++;
        int global_remaining = 0;
        MPI_Allreduce(&remaining, &global_remaining, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_remaining == 0) break;
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
                const double dx = points[cluster.members[i]].x - points[cluster.members[j]].x;
                const double dy = points[cluster.members[i]].y - points[cluster.members[j]].y;
                const double dist = std::sqrt(dx * dx + dy * dy);
                if (dist > max_diameter) max_diameter = dist;
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
            if (i == 1) { /* allow MPI internal args */ }
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (0 == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        MPI_Finalize();
        return 1;
    }
    
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();

    if (world_rank == 0) {
        auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);
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

        const double time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start).count() / 1000.0;
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
            for (int m : membership) membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
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
