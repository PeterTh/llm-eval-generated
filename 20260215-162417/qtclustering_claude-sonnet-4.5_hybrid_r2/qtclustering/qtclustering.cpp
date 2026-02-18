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

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) do {} while(0)
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

#ifdef USE_CUDA
// CUDA kernel to compute distances from a candidate point to cluster members
__global__ void computeMaxDistanceKernel(const Point* points, const int* members, 
                                          int num_members, int candidate,
                                          double* max_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double shared_max[256];
    shared_max[threadIdx.x] = 0.0;
    
    if (idx < num_members) {
        int member = members[idx];
        double dx = points[candidate].x - points[member].x;
        double dy = points[candidate].y - points[member].y;
        double dist = sqrt(dx * dx + dy * dy);
        shared_max[threadIdx.x] = dist;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(max_dist), 
                  __double_as_longlong(shared_max[0]));
    }
}

// CUDA atomic max for double
__device__ static inline void atomicMax(unsigned long long* address, unsigned long long val) {
    unsigned long long old = *address, assumed;
    if (__longlong_as_double(old) >= __longlong_as_double(val)) return;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, val);
    } while (assumed != old && __longlong_as_double(old) < __longlong_as_double(val));
}
#endif

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
#ifdef USE_CUDA
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count,
                     Point* d_points,
                     int* d_members,
                     double* d_max_dist) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Copy members to device
    CUDA_CHECK(cudaMemcpy(d_members, cluster_members.data(), 
                          cluster_members.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Try each unclustered point as a candidate (parallelized with OpenMP)
    #pragma omp parallel
    {
        int local_closest = -1;
        double local_min_diameter = std::numeric_limits<double>::max();
        
        #pragma omp for schedule(dynamic, 32)
        for (int candidate = 0; candidate < point_count; ++candidate) {
            // Skip if already clustered or already in this cluster
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            // Calculate the maximum distance from candidate to all cluster members on GPU
            double h_max_dist = 0.0;
            CUDA_CHECK(cudaMemcpy(d_max_dist, &h_max_dist, sizeof(double), cudaMemcpyHostToDevice));
            
            int num_members = cluster_members.size();
            int blockSize = 256;
            int numBlocks = (num_members + blockSize - 1) / blockSize;
            
            computeMaxDistanceKernel<<<numBlocks, blockSize>>>(
                d_points, d_members, num_members, candidate, d_max_dist);
            
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_max_dist, d_max_dist, sizeof(double), cudaMemcpyDeviceToHost));
            
            // If adding this point keeps diameter below threshold and is better than current best
            if (h_max_dist < threshold && h_max_dist < local_min_diameter) {
                local_min_diameter = h_max_dist;
                local_closest = candidate;
            }
        }
        
        // Thread-safe update of global best
        #pragma omp critical
        {
            if (local_min_diameter < min_diameter) {
                min_diameter = local_min_diameter;
                closest_point = local_closest;
            }
        }
    }
    
    return closest_point;
}
#else
// CPU-only version using OpenMP
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count,
                     void* d_points = nullptr,
                     void* d_members = nullptr,
                     void* d_max_dist = nullptr) {
    (void)d_points; (void)d_members; (void)d_max_dist; // Unused in CPU version
    
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate (parallelized with OpenMP)
    #pragma omp parallel
    {
        int local_closest = -1;
        double local_min_diameter = std::numeric_limits<double>::max();
        
        #pragma omp for schedule(dynamic, 32)
        for (int candidate = 0; candidate < point_count; ++candidate) {
            // Skip if already clustered or already in this cluster
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            // Calculate the maximum distance from candidate to all cluster members
            double max_dist = 0.0;
            for (size_t i = 0; i < cluster_members.size(); ++i) {
                const int member = cluster_members[i];
                const double dist = distance(points[candidate], points[member]);
                max_dist = std::max(max_dist, dist);
            }
            
            // If adding this point keeps diameter below threshold and is better than current best
            if (max_dist < threshold && max_dist < local_min_diameter) {
                local_min_diameter = max_dist;
                local_closest = candidate;
            }
        }
        
        // Thread-safe update of global best
        #pragma omp critical
        {
            if (local_min_diameter < min_diameter) {
                min_diameter = local_min_diameter;
                closest_point = local_closest;
            }
        }
    }
    
    return closest_point;
}
#endif

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
#ifdef USE_CUDA
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              Point* d_points,
                              int* d_members,
                              double* d_max_dist,
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
                                             threshold, point_count, d_points, d_members, d_max_dist);
        
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
#else
// CPU-only version
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              void* d_points = nullptr,
                              void* d_members = nullptr,
                              void* d_max_dist = nullptr,
                              std::vector<int>* cluster_members = nullptr) {
    (void)d_points; (void)d_members; (void)d_max_dist; // Unused in CPU version
    
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
#endif

// Main QT clustering algorithm with MPI+OpenMP+CUDA hybrid parallelization
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  int mpi_rank, int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
#ifdef USE_CUDA
    // Allocate GPU memory
    Point* d_points;
    int* d_members;
    double* d_max_dist;
    
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_dist, sizeof(double)));
    
    // Copy points to GPU
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
#else
    // CPU-only - use nullptr for GPU pointers
    (void)N; // Suppress unused warning if needed
#endif
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Determine which seeds this rank will process
        int total_seeds = 0;
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            if (!clustered[unclustered_indices[i]]) total_seeds++;
        }
        
        if (total_seeds == 0) break;
        
        // Divide seeds among MPI ranks
        int seeds_per_rank = (total_seeds + mpi_size - 1) / mpi_size;
        int start_seed = mpi_rank * seeds_per_rank;
        int end_seed = std::min(start_seed + seeds_per_rank, total_seeds);
        
        struct LocalBest {
            int cardinality;
            int seed;
            std::vector<int> members;
        };
        
        LocalBest local_best;
        local_best.cardinality = -1;
        local_best.seed = -1;
        
        // OpenMP parallelization within MPI rank
        #pragma omp parallel
        {
            LocalBest thread_best;
            thread_best.cardinality = -1;
            thread_best.seed = -1;
            
            #pragma omp for schedule(dynamic, 1) nowait
            for (int seed_idx = start_seed; seed_idx < end_seed; ++seed_idx) {
                // Map to actual unclustered index
                int count = 0;
                int seed = -1;
                for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                    if (!clustered[unclustered_indices[i]]) {
                        if (count == seed_idx) {
                            seed = unclustered_indices[i];
                            break;
                        }
                        count++;
                    }
                }
                
                if (seed < 0 || clustered[seed]) continue;
                
                std::vector<int> candidate_members;
#ifdef USE_CUDA
                const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                           threshold, N, 
                                                           d_points, d_members, d_max_dist,
                                                           &candidate_members);
#else
                const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                           threshold, N, 
                                                           nullptr, nullptr, nullptr,
                                                           &candidate_members);
#endif
                
                if (cardinality > thread_best.cardinality) {
                    thread_best.cardinality = cardinality;
                    thread_best.seed = seed;
                    thread_best.members = candidate_members;
                }
            }
            
            // Thread-safe reduction
            #pragma omp critical
            {
                if (thread_best.cardinality > local_best.cardinality) {
                    local_best = thread_best;
                }
            }
        }
        
        // MPI reduction to find global best cluster
        struct {
            int cardinality;
            int rank;
        } local_data, global_data;
        
        local_data.cardinality = local_best.cardinality;
        local_data.rank = mpi_rank;
        
        MPI_Allreduce(&local_data, &global_data, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        // Broadcast best cluster from winning rank
        int best_seed = local_best.seed;
        int best_size = local_best.cardinality;
        
        MPI_Bcast(&best_seed, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);
        MPI_Bcast(&best_size, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);
        
        if (best_size <= 0 || best_seed < 0) break;
        
        // Allocate space for members
        std::vector<int> best_cluster_members(best_size);
        
        if (mpi_rank == global_data.rank) {
            best_cluster_members = local_best.members;
        }
        
        MPI_Bcast(best_cluster_members.data(), best_size, MPI_INT, global_data.rank, MPI_COMM_WORLD);
        
        // Add cluster (only rank 0 stores it)
        if (mpi_rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
        }
        
        // Mark all members as clustered (all ranks)
        for (int i = 0; i < best_size; ++i) {
            clustered[best_cluster_members[i]] = true;
        }
        
        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }
    
#ifdef USE_CUDA
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_max_dist));
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
    if (mpi_rank == 0) {
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
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
#ifdef USE_CUDA
        printf("CUDA: enabled\n");
#else
        printf("CUDA: disabled (CPU-only)\n");
#endif
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (all ranks generate same data)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Synchronize before clustering
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    // Only rank 0 prints results and validates
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
