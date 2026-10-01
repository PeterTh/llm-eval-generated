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
    } while (0)

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

// CUDA kernel to compute distance matrix
__global__ void computeDistanceMatrixKernel(const double* points_x, const double* points_y,
                                           int N, double* dist_matrix) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N && j < N && j > i) {
        double dx = points_x[i] - points_x[j];
        double dy = points_y[i] - points_y[j];
        double dist = sqrt(dx * dx + dy * dy);
        dist_matrix[i * N + j] = dist;
        dist_matrix[j * N + i] = dist;
    } else if (i == j && i < N) {
        dist_matrix[i * N + j] = 0.0;
    }
}

// GPU helper class to manage distance matrix
class GPUDistanceMatrix {
private:
    double *d_points_x, *d_points_y, *d_dist_matrix;
    double *h_dist_matrix;
    int N;
    bool initialized;

public:
    GPUDistanceMatrix() : d_points_x(nullptr), d_points_y(nullptr), 
                          d_dist_matrix(nullptr), h_dist_matrix(nullptr),
                          N(0), initialized(false) {}
    
    ~GPUDistanceMatrix() {
        cleanup();
    }
    
    void initialize(const std::vector<Point>& points) {
        N = static_cast<int>(points.size());
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_points_x, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_points_y, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dist_matrix, N * N * sizeof(double)));
        
        // Allocate host memory
        h_dist_matrix = new double[N * N];
        
        // Copy points to device
        std::vector<double> points_x(N), points_y(N);
        for (int i = 0; i < N; ++i) {
            points_x[i] = points[i].x;
            points_y[i] = points[i].y;
        }
        
        CUDA_CHECK(cudaMemcpy(d_points_x, points_x.data(), N * sizeof(double), 
                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_points_y, points_y.data(), N * sizeof(double), 
                   cudaMemcpyHostToDevice));
        
        // Compute distance matrix on GPU
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                     (N + blockDim.y - 1) / blockDim.y);
        
        computeDistanceMatrixKernel<<<gridDim, blockDim>>>(
            d_points_x, d_points_y, N, d_dist_matrix);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy back to host
        CUDA_CHECK(cudaMemcpy(h_dist_matrix, d_dist_matrix, N * N * sizeof(double),
                   cudaMemcpyDeviceToHost));
        
        initialized = true;
    }
    
    inline double getDistance(int i, int j) const {
        return h_dist_matrix[i * N + j];
    }
    
    void cleanup() {
        if (initialized) {
            CUDA_CHECK(cudaFree(d_points_x));
            CUDA_CHECK(cudaFree(d_points_y));
            CUDA_CHECK(cudaFree(d_dist_matrix));
            delete[] h_dist_matrix;
            initialized = false;
        }
    }
};

// Find the closest unclustered point using pre-computed distances
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const GPUDistanceMatrix& dist_matrix,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    #pragma omp parallel for schedule(dynamic, 16) if(point_count > 100)
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = dist_matrix.getDistance(candidate, member);
            max_dist = std::max(max_dist, dist);
        }
        
        if (max_dist < threshold && max_dist < min_diameter) {
            #pragma omp critical
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
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const GPUDistanceMatrix& dist_matrix,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, 
                                             dist_matrix, threshold, point_count);
        
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
                                  const double threshold,
                                  int rank, int size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Pre-compute distance matrix on GPU
    GPUDistanceMatrix dist_matrix;
    dist_matrix.initialize(points);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_unclustered = static_cast<int>(unclustered_indices.size());
        
        // Divide work among MPI ranks
        const int seeds_per_rank = (num_unclustered + size - 1) / size;
        const int start_idx = rank * seeds_per_rank;
        const int end_idx = std::min(start_idx + seeds_per_rank, num_unclustered);
        
        // Local best for this rank
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_cluster_members;
        
        // Try each unclustered point in this rank's range as a seed (with OpenMP)
        #pragma omp parallel
        {
            int thread_max_cardinality = -1;
            int thread_best_seed = -1;
            std::vector<int> thread_best_members;
            
            #pragma omp for schedule(dynamic, 1) nowait
            for (int i = start_idx; i < end_idx; ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;
                
                std::vector<int> candidate_members;
                const int cardinality = generateCandidateCluster(
                    seed, clustered, dist_matrix, threshold, N, &candidate_members);
                
                if (cardinality > thread_max_cardinality) {
                    thread_max_cardinality = cardinality;
                    thread_best_seed = seed;
                    thread_best_members = candidate_members;
                }
            }
            
            #pragma omp critical
            {
                if (thread_max_cardinality > local_max_cardinality) {
                    local_max_cardinality = thread_max_cardinality;
                    local_best_seed = thread_best_seed;
                    local_best_cluster_members = thread_best_members;
                }
            }
        }
        
        // MPI reduction to find global best
        struct {
            int cardinality;
            int rank;
        } local_data, global_data;
        
        local_data.cardinality = local_max_cardinality;
        local_data.rank = rank;
        
        MPI_Allreduce(&local_data, &global_data, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        int best_seed = local_best_seed;
        int max_cardinality = global_data.cardinality;
        std::vector<int> best_cluster_members = local_best_cluster_members;
        
        // Broadcast the winning cluster from the rank that found it
        MPI_Bcast(&best_seed, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);
        
        int member_count = static_cast<int>(best_cluster_members.size());
        MPI_Bcast(&member_count, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);
        
        if (rank != global_data.rank) {
            best_cluster_members.resize(member_count);
        }
        MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT, 
                  global_data.rank, MPI_COMM_WORLD);
        
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
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
    
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    auto distance = [](const Point& p1, const Point& p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (same on all ranks for consistency)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, size);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time_ms = cluster_time.count();
    long max_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &max_cluster_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", max_cluster_time_ms);
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
        
        const double time_sec = max_cluster_time_ms / 1000.0;
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
