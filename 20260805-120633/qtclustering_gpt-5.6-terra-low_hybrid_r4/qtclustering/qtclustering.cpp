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

// One CUDA pass amortizes the otherwise dominant O(N^2) geometry work over
// every QT phase.  Squared distances preserve all threshold/order comparisons.
__global__ void buildDistanceMatrix(const Point* points, double* distances, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const double dx = points[row].x - points[col].x;
        const double dy = points[row].y - points[col].y;
        distances[static_cast<size_t>(row) * n + col] = dx * dx + dy * dy;
    }
}

static void cudaCheck(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static std::vector<double> makeDistanceMatrix(const std::vector<Point>& points, int rank) {
    const size_t entries = points.size() * points.size();
    std::vector<double> matrix(entries);
    // A single rank owns the accelerator work; the matrix is then shared by
    // MPI, avoiding redundant GPU work on every rank.
    if (rank == 0) {
        Point* d_points = nullptr;
        double* d_matrix = nullptr;
        cudaCheck(cudaMalloc(&d_points, points.size() * sizeof(Point)), "cudaMalloc(points)");
        cudaCheck(cudaMalloc(&d_matrix, entries * sizeof(double)), "cudaMalloc(matrix)");
        cudaCheck(cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice), "cudaMemcpy(points)");
        const dim3 block(16, 16);
        const dim3 grid((points.size() + block.x - 1) / block.x, (points.size() + block.y - 1) / block.y);
        buildDistanceMatrix<<<grid, block>>>(d_points, d_matrix, static_cast<int>(points.size()));
        cudaCheck(cudaGetLastError(), "buildDistanceMatrix launch");
        cudaCheck(cudaDeviceSynchronize(), "buildDistanceMatrix synchronization");
        cudaCheck(cudaMemcpy(matrix.data(), d_matrix, entries * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(matrix)");
        cudaFree(d_matrix);
        cudaFree(d_points);
    }
    if (entries > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Distance matrix exceeds MPI count limit\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    MPI_Bcast(matrix.data(), static_cast<int>(entries), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return matrix;
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<unsigned char>& clustered,
                     const std::vector<unsigned char>& in_cluster,
                     const std::vector<double>& distances,
                     const double threshold_squared, const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distances[static_cast<size_t>(candidate) * point_count + member];
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold_squared && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<double>& distances,
                              const double threshold_squared,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<unsigned char> in_cluster(point_count, 0);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, distances,
                                             threshold_squared, point_count);
        
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
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const std::vector<double> distances = makeDistanceMatrix(points, rank);
    const double threshold_squared = threshold * threshold;
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_cardinality = -1;
        int local_seed = std::numeric_limits<int>::max();
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        #pragma omp parallel
        {
            int thread_cardinality = -1;
            int thread_seed = std::numeric_limits<int>::max();
            // Cyclic ownership keeps the MPI work balanced as candidate costs
            // diverge, while OpenMP dynamically balances local seed work.
            #pragma omp for schedule(dynamic, 1) nowait
            for (long long i = rank; i < static_cast<long long>(unclustered_indices.size()); i += world_size) {
            const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;
                const int cardinality = generateCandidateCluster(seed, clustered, distances,
                    threshold_squared, N, nullptr);
                if (cardinality > thread_cardinality ||
                    (cardinality == thread_cardinality && seed < thread_seed)) {
                    thread_cardinality = cardinality;
                    thread_seed = seed;
                }
            }
            #pragma omp critical
            if (thread_cardinality > local_cardinality ||
                (thread_cardinality == local_cardinality && thread_seed < local_seed)) {
                local_cardinality = thread_cardinality;
                local_seed = thread_seed;
            }
        }

        int local_result[2] = {local_cardinality, local_seed};
        std::vector<int> results(rank == 0 ? 2 * world_size : 0);
        MPI_Gather(local_result, 2, MPI_INT, results.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
        int max_cardinality = -1, best_seed = -1;
        if (rank == 0) {
            for (int r = 0; r < world_size; ++r) {
                const int cardinality = results[2 * r];
                const int seed = results[2 * r + 1];
                if (cardinality > max_cardinality ||
                    (cardinality == max_cardinality && seed < best_seed)) {
                    max_cardinality = cardinality;
                    best_seed = seed;
                }
            }
            if (best_seed >= 0) generateCandidateCluster(best_seed, clustered, distances,
                threshold_squared, N, &best_cluster_members);
        }
        
        // If we found a cluster, add it
        if (rank == 0 && best_seed >= 0 && max_cardinality > 0) {
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
                unclustered_indices.end()
            );
        } else if (rank == 0) {
            // No more clusters can be formed
            break;
        }
        int continue_clustering = (rank == 0 && best_seed >= 0 && max_cardinality > 0) ? 1 : 0;
        MPI_Bcast(&continue_clustering, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!continue_clustering) break;
        MPI_Bcast(clustered.data(), N, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        // Non-root ranks keep only distributed state; root alone retains output clusters.
        if (rank != 0) {
            unclustered_indices.erase(std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                [&clustered](int idx) { return clustered[idx] != 0; }), unclustered_indices.end());
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
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
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
            printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        MPI_Finalize(); return 1;
    }
    
    if (mpi_rank == 0) printf("QT Clustering Benchmark\n");
    if (mpi_rank == 0) printf("Number of points: %d\n", num_points);
    if (mpi_rank == 0) printf("Distance threshold: %.2f\n", threshold);
    if (mpi_rank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long long local_cluster_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long long cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &cluster_time_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) printf("Clustering time: %lld ms\n", cluster_time_ms);
    if (mpi_rank != 0) { MPI_Finalize(); return 0; }
    printf("Clusters found: %zu\n", clusters.size());
    
    // Calculate statistics and performance metrics
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
    const double time_sec = cluster_time_ms / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", 
           clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults) {
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
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
