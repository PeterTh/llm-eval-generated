// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

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

struct DistanceMatrix {
    Point* device_points = nullptr;
    double* matrix = nullptr;
};

inline size_t distIndex(const int i, const int j, const int point_count) {
    return static_cast<size_t>(i) * static_cast<size_t>(point_count) + static_cast<size_t>(j);
}

inline double distAt(const double* dist_matrix, const int point_count, const int i, const int j) {
    return dist_matrix[distIndex(i, j, point_count)];
}

inline void checkCuda(const cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", message, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void computeDistanceMatrixKernel(const Point* points, double* dist_matrix, int point_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = point_count * point_count;
    if (idx >= total) return;
    const int i = idx / point_count;
    const int j = idx - i * point_count;
    const double dx = points[i].x - points[j].x;
    const double dy = points[i].y - points[j].y;
    dist_matrix[idx] = sqrt(dx * dx + dy * dy);
}

DistanceMatrix computeDistanceMatrixCUDA(const std::vector<Point>& points, const int device_id) {
    DistanceMatrix dm;
    const int point_count = static_cast<int>(points.size());
    if (point_count == 0) {
        return dm;
    }

    const size_t points_bytes = sizeof(Point) * static_cast<size_t>(point_count);
    const size_t matrix_bytes =
        sizeof(double) * static_cast<size_t>(point_count) * static_cast<size_t>(point_count);

    checkCuda(cudaMallocManaged(&dm.device_points, points_bytes), "cudaMallocManaged points");
    checkCuda(cudaMallocManaged(&dm.matrix, matrix_bytes), "cudaMallocManaged distance matrix");
    std::memcpy(dm.device_points, points.data(), points_bytes);

    checkCuda(cudaMemPrefetchAsync(dm.device_points, points_bytes, device_id),
              "cudaMemPrefetchAsync points");
    checkCuda(cudaMemPrefetchAsync(dm.matrix, matrix_bytes, device_id),
              "cudaMemPrefetchAsync distance matrix");

    const int threads = 256;
    const int total = point_count * point_count;
    const int blocks = (total + threads - 1) / threads;
    computeDistanceMatrixKernel<<<blocks, threads>>>(dm.device_points, dm.matrix, point_count);
    checkCuda(cudaGetLastError(), "computeDistanceMatrixKernel launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize distance kernel");

    checkCuda(cudaMemPrefetchAsync(dm.matrix, matrix_bytes, cudaCpuDeviceId),
              "cudaMemPrefetchAsync distance matrix to CPU");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize distance prefetch");

    return dm;
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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const double* dist_matrix,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    #pragma omp parallel
    {
        int local_point = -1;
        double local_min = std::numeric_limits<double>::max();

        #pragma omp for schedule(static) nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            // Skip if already clustered or already in this cluster
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            // Calculate the maximum distance from candidate to all cluster members
            double max_dist = 0.0;
            for (size_t i = 0; i < cluster_members.size(); ++i) {
                const int member = cluster_members[i];
                const double dist = distAt(dist_matrix, point_count, candidate, member);
                if (dist > max_dist) max_dist = dist;
            }
            
            // If adding this point keeps diameter below threshold and is better than current best
            if (max_dist < threshold) {
                if (max_dist < local_min ||
                    (max_dist == local_min && (local_point < 0 || candidate < local_point))) {
                    local_min = max_dist;
                    local_point = candidate;
                }
            }
        }

        #pragma omp critical
        {
            if (local_point >= 0) {
                if (local_min < min_diameter ||
                    (local_min == min_diameter &&
                     (closest_point < 0 || local_point < closest_point))) {
                    min_diameter = local_min;
                    closest_point = local_point;
                }
            }
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                             const std::vector<bool>& clustered,
                             const double* dist_matrix,
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
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, dist_matrix,
                                             threshold, point_count);
        
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

// Main QT clustering algorithm (MPI across seeds, OpenMP within candidate search)
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double* dist_matrix,
                                     const double threshold,
                                     const int world_rank,
                                     const int world_size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    unclustered_indices.reserve(N);
    clusters.reserve(N);
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int total_unclustered = static_cast<int>(unclustered_indices.size());
        const int local_begin = (world_rank * total_unclustered) / world_size;
        const int local_end = ((world_rank + 1) * total_unclustered) / world_size;

        int local_best_cardinality = -1;
        int local_best_order = std::numeric_limits<int>::max();
        int local_best_seed = -1;
        std::vector<int> local_best_members;
        
        // Try assigned unclustered points as seeds
        for (int idx = local_begin; idx < local_end; ++idx) {
            const int seed = unclustered_indices[idx];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, dist_matrix, 
                                                       threshold, N, 
                                                       &candidate_members);
            
            if (cardinality > local_best_cardinality ||
                (cardinality == local_best_cardinality && idx < local_best_order)) {
                local_best_cardinality = cardinality;
                local_best_order = idx;
                local_best_seed = seed;
                local_best_members.swap(candidate_members);
            }
        }
        
        int local_info[4] = {
            local_best_cardinality,
            local_best_order,
            local_best_seed,
            world_rank
        };
        std::vector<int> all_info(4 * world_size);
        MPI_Allgather(local_info, 4, MPI_INT, all_info.data(), 4, MPI_INT, MPI_COMM_WORLD);

        int best_cardinality = -1;
        int best_order = std::numeric_limits<int>::max();
        int best_seed = -1;
        int best_rank = 0;
        if (world_rank == 0) {
            for (int r = 0; r < world_size; ++r) {
                const int card = all_info[4 * r];
                const int order = all_info[4 * r + 1];
                const int seed = all_info[4 * r + 2];
                const int rank = all_info[4 * r + 3];
                if (card > best_cardinality ||
                    (card == best_cardinality && order < best_order)) {
                    best_cardinality = card;
                    best_order = order;
                    best_seed = seed;
                    best_rank = rank;
                }
            }
        }

        int best_info[3] = {best_cardinality, best_seed, best_rank};
        MPI_Bcast(best_info, 3, MPI_INT, 0, MPI_COMM_WORLD);
        best_cardinality = best_info[0];
        best_seed = best_info[1];
        best_rank = best_info[2];
        
        if (best_seed < 0 || best_cardinality <= 0) {
            break;
        }

        std::vector<int> best_cluster_members;
        if (world_rank == best_rank) {
            best_cluster_members = local_best_members;
        }

        int member_count =
            (world_rank == best_rank) ? static_cast<int>(best_cluster_members.size()) : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, best_rank, MPI_COMM_WORLD);
        if (world_rank != best_rank) {
            best_cluster_members.resize(member_count);
        }
        MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT, best_rank, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_cluster_members);
        clusters.push_back(cluster);
        
        // Mark all members as clustered
        for (size_t i = 0; i < clusters.back().members.size(); ++i) {
            clustered[clusters.back().members[i]] = true;
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

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                      const double* dist_matrix,
                      const int point_count,
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
                const double dist = distAt(dist_matrix, point_count,
                                           cluster.members[i], cluster.members[j]);
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
    std::vector<int> membership(point_count, -1);
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
    
    printf("Total points: %d, Clustered: %d, Unclustered: %d\n",
           point_count, clustered_count, point_count - clustered_count);
    
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            printf("ERROR: MPI does not provide required thread support.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int parse_status = 1;
    
    // Parse command line arguments
    if (world_rank == 0) {
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
                parse_status = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_status = 0;
                break;
            }
        }
        
        if (parse_status == 1 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
            parse_status = 0;
        }
    }

    MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_status != 1) {
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int validate_flag = validate ? 1 : 0;
    int results_flag = printResults ? 1 : 0;
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&results_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_flag != 0);
    printResults = (results_flag != 0);
    
    if (world_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n",
               world_size, omp_get_max_threads());
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        if (world_rank == 0) {
            printf("ERROR: No CUDA devices detected.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int device_id = world_rank % device_count;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");

    DistanceMatrix dist_matrix = computeDistanceMatrixCUDA(points, device_id);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters =
        qtClusteringMPI(points, dist_matrix.matrix, threshold, world_rank, world_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_end = MPI_Wtime();
    const double cluster_time_local = cluster_end - cluster_start;
    double cluster_time_max = 0.0;
    MPI_Reduce(&cluster_time_local, &cluster_time_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        printf("Clustering time: %.3f ms\n", cluster_time_max * 1000.0);
        printf("Clusters found: %zu\n", clusters.size());
    }
    
    // Calculate statistics and performance metrics
    if (world_rank == 0) {
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
        
        // Performance metrics
        const double time_sec = cluster_time_max;
        const double clusters_per_sec = time_sec > 0.0 ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = time_sec > 0.0 ? num_points / time_sec : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (printResults && world_rank == 0) {
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
    if (validate) {
        bool valid = true;
        if (world_rank == 0) {
            valid = validateClusters(clusters, dist_matrix.matrix, num_points, threshold);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (dist_matrix.device_points) {
        checkCuda(cudaFree(dist_matrix.device_points), "cudaFree points");
    }
    if (dist_matrix.matrix) {
        checkCuda(cudaFree(dist_matrix.matrix), "cudaFree distance matrix");
    }

    MPI_Finalize();
    return exit_code;
}
