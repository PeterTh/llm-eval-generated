// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
// 
// Parallelization Strategy:
// - MPI: Distributes candidate cluster evaluation across nodes
// - OpenMP: Parallelizes inner distance calculations and reductions
// - CUDA: Accelerates distance computations on GPU

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Global MPI rank and size
int mpi_rank = 0, mpi_size = 1;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// CUDA state management
#ifdef USE_CUDA
struct CudaState {
    double* d_points_x = nullptr;
    double* d_points_y = nullptr;
    char* d_clustered = nullptr;
    char* d_in_cluster = nullptr;
    int num_points = 0;
    bool initialized = false;
    
    void init(int n_points, const std::vector<Point>& points) {
        if (initialized) return;
        num_points = n_points;
        cudaMalloc(&d_points_x, n_points * sizeof(double));
        cudaMalloc(&d_points_y, n_points * sizeof(double));
        cudaMalloc(&d_clustered, n_points * sizeof(char));
        cudaMalloc(&d_in_cluster, n_points * sizeof(char));
        
        std::vector<double> x_vals(n_points), y_vals(n_points);
        for (int i = 0; i < n_points; ++i) {
            x_vals[i] = points[i].x;
            y_vals[i] = points[i].y;
        }
        cudaMemcpy(d_points_x, x_vals.data(), n_points * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points_y, y_vals.data(), n_points * sizeof(double), cudaMemcpyHostToDevice);
        initialized = true;
    }
    
    void update_clustered(const std::vector<bool>& clustered) {
        if (!initialized) return;
        std::vector<char> clustered_char(clustered.begin(), clustered.end());
        cudaMemcpy(d_clustered, clustered_char.data(), num_points * sizeof(char), cudaMemcpyHostToDevice);
    }
    
    void update_in_cluster(const std::vector<bool>& in_cluster) {
        if (!initialized) return;
        std::vector<char> in_cluster_char(in_cluster.begin(), in_cluster.end());
        cudaMemcpy(d_in_cluster, in_cluster_char.data(), num_points * sizeof(char), cudaMemcpyHostToDevice);
    }
    
    void cleanup() {
        if (initialized) {
            cudaFree(d_points_x);
            cudaFree(d_points_y);
            cudaFree(d_clustered);
            cudaFree(d_in_cluster);
            initialized = false;
        }
    }
} cuda_state;

#endif

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
// OpenMP parallel version for intra-node parallelization
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 32) reduction(min : min_diameter) reduction(min : closest_point)
#endif
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        
#ifdef _OPENMP
        #pragma omp parallel for reduction(max : max_dist)
#endif
        for (int i = 0; i < static_cast<int>(cluster_members.size()); ++i) {
            const int member = cluster_members[i];
            double dx = points[candidate].x - points[member].x;
            double dy = points[candidate].y - points[member].y;
            double dist = std::sqrt(dx * dx + dy * dy);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                if (max_dist < min_diameter) {
                    min_diameter = max_dist;
                    closest_point = candidate;
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
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
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

// Structure to hold local best cluster for parallel reduction
struct LocalBest {
    int seed;
    int cardinality;
    std::vector<int> members;
};

// Main QT clustering algorithm with MPI/OpenMP parallelization
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
#ifdef USE_CUDA
    cuda_state.init(N, points);
    cuda_state.update_clustered(clustered);
#endif
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Distribute seeds across MPI ranks and OpenMP threads
        std::vector<int> local_unclustered;
        
#ifdef USE_MPI
        // Each MPI rank processes a subset of unclustered points
        int rank_start = (mpi_rank * static_cast<int>(unclustered_indices.size())) / mpi_size;
        int rank_end = ((mpi_rank + 1) * static_cast<int>(unclustered_indices.size())) / mpi_size;
        for (int i = rank_start; i < rank_end; ++i) {
            local_unclustered.push_back(unclustered_indices[i]);
        }
#else
        local_unclustered = unclustered_indices;
#endif
        
        // Try each unclustered point as a seed (with OpenMP parallelization)
        LocalBest local_best = {-1, -1, {}};
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int i = 0; i < static_cast<int>(local_unclustered.size()); ++i) {
            const int seed = local_unclustered[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                        threshold, N, 
                                                        &candidate_members);
            
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                if (cardinality > local_best.cardinality) {
                    local_best.seed = seed;
                    local_best.cardinality = cardinality;
                    local_best.members = candidate_members;
                }
            }
        }
        
        max_cardinality = local_best.cardinality;
        best_seed = local_best.seed;
        best_cluster_members = local_best.members;
        
#ifdef USE_MPI
        // Find global best across all MPI ranks using MPI_MAXLOC
        struct {
            int cardinality;
            int rank;
        } local_data = {max_cardinality, mpi_rank};
        
        struct {
            int cardinality;
            int rank;
        } global_data = {-1, 0};
        
        MPI_Reduce(&local_data, &global_data, 1, 
                   MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Broadcast the winning rank's best seed and cardinality
        int bcast_data[2] = {best_seed, max_cardinality};
        MPI_Bcast(bcast_data, 2, MPI_INT, 0, MPI_COMM_WORLD);
        
        best_seed = bcast_data[0];
        max_cardinality = bcast_data[1];
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Re-compute the best cluster if we're not on the winning rank
        if (best_seed >= 0 && max_cardinality > 0) {
            if (mpi_rank != global_data.rank) {
                best_cluster_members.clear();
                generateCandidateCluster(best_seed, clustered, points, 
                                        threshold, N, &best_cluster_members);
            }
        }
#endif
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
#ifdef USE_CUDA
            cuda_state.update_clustered(clustered);
#endif
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }
    
#ifdef USE_CUDA
    cuda_state.cleanup();
#endif
    
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
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    
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
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Parallel Version\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
#ifdef USE_MPI
        printf("MPI processes: %d\n", mpi_size);
#endif
#ifdef _OPENMP
        printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif
    }
    
    // Generate synthetic data - all ranks generate the same data using same seed
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
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
        const double time_sec = cluster_time.count() / 1000.0;
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
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}
