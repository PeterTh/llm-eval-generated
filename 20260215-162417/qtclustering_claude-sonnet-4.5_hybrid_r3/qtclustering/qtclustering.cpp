// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Version
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

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

// Forward declaration of CUDA function
extern "C" void computeMaxDistances_cuda(const Point* d_points, const int* d_members,
                                         int member_count, const int* d_candidates,
                                         int num_candidates, double* d_results,
                                         double threshold);

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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Find the closest unclustered point using CUDA for distance computations
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const Point* d_points,
                     const double threshold,
                     const int point_count,
                     int* d_members,
                     int* d_candidates,
                     double* d_results) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    const int member_count = static_cast<int>(cluster_members.size());
    
    // Collect candidate points
    std::vector<int> candidates;
    for (int i = 0; i < point_count; ++i) {
        if (!clustered[i] && !in_cluster[i]) {
            candidates.push_back(i);
        }
    }
    
    if (candidates.empty()) return -1;
    
    // Copy members to device
    CUDA_CHECK(cudaMemcpy(d_members, cluster_members.data(), 
                          member_count * sizeof(int), cudaMemcpyHostToDevice));
    
    // Process candidates in batches
    const int batch_size = 1024;
    for (size_t start = 0; start < candidates.size(); start += batch_size) {
        int count = std::min(batch_size, static_cast<int>(candidates.size() - start));
        
        // Copy batch to device
        CUDA_CHECK(cudaMemcpy(d_candidates, &candidates[start], 
                              count * sizeof(int), cudaMemcpyHostToDevice));
        
        // Compute max distances on GPU
        computeMaxDistances_cuda(d_points, d_members, member_count,
                                d_candidates, count, d_results, threshold);
        
        // Copy results back
        std::vector<double> results(count);
        CUDA_CHECK(cudaMemcpy(results.data(), d_results, 
                              count * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Find best candidate in batch
        for (int i = 0; i < count; ++i) {
            double max_dist = results[i];
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest_point = candidates[start + i];
            }
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const Point* d_points,
                              const double threshold,
                              const int point_count,
                              int* d_members,
                              int* d_candidates,
                              double* d_results,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, d_points, 
                                             threshold, point_count, d_members, 
                                             d_candidates, d_results);
        
        if (closest < 0) break;
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm - Hybrid MPI+OpenMP+CUDA version
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Allocate GPU memory
    Point* d_points;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    
    // Copy points to GPU
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    
    // Allocate per-thread GPU buffers
    const int max_threads = omp_get_max_threads();
    std::vector<int*> d_members_array(max_threads);
    std::vector<int*> d_candidates_array(max_threads);
    std::vector<double*> d_results_array(max_threads);
    
    for (int t = 0; t < max_threads; ++t) {
        CUDA_CHECK(cudaMalloc(&d_members_array[t], N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_candidates_array[t], 1024 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_results_array[t], 1024 * sizeof(double)));
    }
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_cluster_members;
        
        const int total_seeds = static_cast<int>(unclustered_indices.size());
        
        // Use OpenMP to parallelize seed evaluation
        #pragma omp parallel
        {
            int thread_max_cardinality = -1;
            int thread_best_seed = -1;
            std::vector<int> thread_best_members;
            
            #pragma omp for schedule(dynamic, 4)
            for (int i = rank; i < total_seeds; i += size) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;
                
                int thread_id = omp_get_thread_num();
                std::vector<int> candidate_members;
                const int cardinality = generateCandidateCluster(seed, clustered, d_points, 
                                                           threshold, N, 
                                                           d_members_array[thread_id], 
                                                           d_candidates_array[thread_id],
                                                           d_results_array[thread_id], 
                                                           &candidate_members);
                
                if (cardinality > thread_max_cardinality ||
                    (cardinality == thread_max_cardinality && seed < thread_best_seed)) {
                    thread_max_cardinality = cardinality;
                    thread_best_seed = seed;
                    thread_best_members = candidate_members;
                }
            }
            
            #pragma omp critical
            {
                if (thread_max_cardinality > local_max_cardinality ||
                    (thread_max_cardinality == local_max_cardinality && 
                     thread_best_seed < local_best_seed)) {
                    local_max_cardinality = thread_max_cardinality;
                    local_best_seed = thread_best_seed;
                    local_best_cluster_members = thread_best_members;
                }
            }
        }
        
        // MPI reduction to find global best cluster
        struct { int cardinality; int seed; } local_best = {local_max_cardinality, local_best_seed};
        struct { int cardinality; int seed; } global_best;
        
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        // Resolve ties by minimum seed
        int min_seed = (local_max_cardinality == global_best.cardinality) ? local_best_seed : INT_MAX;
        MPI_Allreduce(&min_seed, &global_best.seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        // Broadcast best cluster
        int best_cluster_size = 0;
        if (local_best_seed == global_best.seed && 
            local_max_cardinality == global_best.cardinality) {
            best_cluster_size = static_cast<int>(local_best_cluster_members.size());
        }
        
        struct { int size; int rank_id; } size_rank = {best_cluster_size, rank};
        struct { int size; int rank_id; } max_size_rank;
        
        MPI_Allreduce(&size_rank, &max_size_rank, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        best_cluster_size = max_size_rank.size;
        MPI_Bcast(&best_cluster_size, 1, MPI_INT, max_size_rank.rank_id, MPI_COMM_WORLD);
        
        std::vector<int> best_cluster_members(best_cluster_size);
        if (rank == max_size_rank.rank_id) {
            best_cluster_members = local_best_cluster_members;
        }
        
        MPI_Bcast(best_cluster_members.data(), best_cluster_size, MPI_INT, 
                  max_size_rank.rank_id, MPI_COMM_WORLD);
        
        if (global_best.seed >= 0 && global_best.cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = global_best.seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }
    }
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_points));
    for (int t = 0; t < max_threads; ++t) {
        CUDA_CHECK(cudaFree(d_members_array[t]));
        CUDA_CHECK(cudaFree(d_candidates_array[t]));
        CUDA_CHECK(cudaFree(d_results_array[t]));
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    if (rank == 0) {
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
    }
    
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int deviceId = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    // Initialize CUDA context and synchronize
    cudaFree(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // When running with multiple MPI ranks, limit OpenMP threads to avoid context issues
    if (size > 1) {
        omp_set_num_threads(1);
    }
    
    // Barrier to ensure all ranks have initialized CUDA
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", deviceCount);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);
    
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
        
        int total_clustered = 0;
        int max_cluster_size = 0;
        
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int clust_size = static_cast<int>(clusters[i].members.size());
            total_clustered += clust_size;
            max_cluster_size = std::max(max_cluster_size, clust_size);
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
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
