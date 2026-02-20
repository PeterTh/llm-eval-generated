// QT Clustering Benchmark - CUDA-Accelerated Version
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
#include <cfloat>
#include <cstdint>
#include <limits>
#include <vector>

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

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                            \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

constexpr int kCandidateBlockSize = 256;
constexpr int kReduceBlockSize = 256;
constexpr int kFlagBlockSize = 256;

__global__ void computeDistancesKernel(const Point* points,
                                       double* distances,
                                       int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const double dx = points[row].x - points[col].x;
        const double dy = points[row].y - points[col].y;
        distances[row * n + col] = dx * dx + dy * dy;
    }
}

__global__ void computeCandidateMaxKernel(const double* distances,
                                          const int* members,
                                          int member_count,
                                          const uint8_t* clustered,
                                          const uint8_t* in_cluster,
                                          double threshold_sq,
                                          int n,
                                          double* out_values,
                                          int* out_indices) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= n) return;

    out_indices[candidate] = candidate;
    if (clustered[candidate] || in_cluster[candidate]) {
        out_values[candidate] = DBL_MAX;
        return;
    }

    double max_dist = 0.0;
    for (int i = 0; i < member_count; ++i) {
        const int member = members[i];
        const double dist = distances[candidate * n + member];
        if (dist > max_dist) {
            max_dist = dist;
            if (max_dist >= threshold_sq) {
                break;
            }
        }
    }

    out_values[candidate] = (max_dist < threshold_sq) ? max_dist : DBL_MAX;
}

template <int BLOCK_SIZE>
__global__ void reduceMinKernel(const double* values,
                                const int* indices,
                                double* out_values,
                                int* out_indices,
                                int n) {
    __shared__ double shared_vals[BLOCK_SIZE];
    __shared__ int shared_idx[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int start = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    double best_val = DBL_MAX;
    int best_idx = -1;

    if (start < n) {
        best_val = values[start];
        best_idx = indices[start];
    }

    const int second = start + BLOCK_SIZE;
    if (second < n) {
        const double other_val = values[second];
        const int other_idx = indices[second];
        if (other_val < best_val || (other_val == best_val && other_idx < best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double other_val = shared_vals[tid + stride];
            const int other_idx = shared_idx[tid + stride];
            if (other_val < shared_vals[tid] ||
                (other_val == shared_vals[tid] && other_idx < shared_idx[tid])) {
                shared_vals[tid] = other_val;
                shared_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_values[blockIdx.x] = shared_vals[0];
        out_indices[blockIdx.x] = shared_idx[0];
    }
}

__global__ void setFlagsKernel(uint8_t* flags,
                               const int* indices,
                               int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        flags[indices[idx]] = 1;
    }
}

struct CudaContext {
    explicit CudaContext(const std::vector<Point>& points)
        : n(static_cast<int>(points.size())) {
        CUDA_CHECK(cudaSetDevice(0));
        const size_t point_bytes = static_cast<size_t>(n) * sizeof(Point);
        const size_t dist_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(double);
        const size_t flag_bytes = static_cast<size_t>(n) * sizeof(uint8_t);
        const size_t index_bytes = static_cast<size_t>(n) * sizeof(int);
        const size_t value_bytes = static_cast<size_t>(n) * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_points, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_distances, dist_bytes));
        CUDA_CHECK(cudaMalloc(&d_clustered, flag_bytes));
        CUDA_CHECK(cudaMalloc(&d_in_cluster, flag_bytes));
        CUDA_CHECK(cudaMalloc(&d_members, index_bytes));
        CUDA_CHECK(cudaMalloc(&d_candidate_values, value_bytes));
        CUDA_CHECK(cudaMalloc(&d_candidate_indices, index_bytes));

        int reduce_blocks = (n + (kReduceBlockSize * 2 - 1)) / (kReduceBlockSize * 2);
        if (reduce_blocks < 1) {
            reduce_blocks = 1;
        }
        const size_t reduce_value_bytes = static_cast<size_t>(reduce_blocks) * sizeof(double);
        const size_t reduce_index_bytes = static_cast<size_t>(reduce_blocks) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_reduce_values, reduce_value_bytes));
        CUDA_CHECK(cudaMalloc(&d_reduce_indices, reduce_index_bytes));

        CUDA_CHECK(cudaMemcpy(d_points, points.data(), point_bytes, cudaMemcpyHostToDevice));

        const dim3 block(16, 16);
        const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        computeDistancesKernel<<<grid, block>>>(d_points, d_distances, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_clustered, 0, flag_bytes));
        CUDA_CHECK(cudaMemset(d_in_cluster, 0, flag_bytes));
    }

    ~CudaContext() {
        cudaFree(d_points);
        cudaFree(d_distances);
        cudaFree(d_clustered);
        cudaFree(d_in_cluster);
        cudaFree(d_members);
        cudaFree(d_candidate_values);
        cudaFree(d_candidate_indices);
        cudaFree(d_reduce_values);
        cudaFree(d_reduce_indices);
    }

    void resetInCluster() {
        CUDA_CHECK(cudaMemset(d_in_cluster, 0, static_cast<size_t>(n) * sizeof(uint8_t)));
    }

    void setInCluster(int idx) {
        const uint8_t one = 1;
        CUDA_CHECK(cudaMemcpy(d_in_cluster + idx, &one, sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    void setMember(int member_idx, int point_idx) {
        CUDA_CHECK(cudaMemcpy(d_members + member_idx, &point_idx, sizeof(int), cudaMemcpyHostToDevice));
    }

    void markClustered(const std::vector<int>& members) {
        if (members.empty()) return;
        CUDA_CHECK(cudaMemcpy(d_members, members.data(),
                              members.size() * sizeof(int), cudaMemcpyHostToDevice));
        const int blocks = static_cast<int>((members.size() + kFlagBlockSize - 1) / kFlagBlockSize);
        setFlagsKernel<<<blocks, kFlagBlockSize>>>(d_clustered, d_members,
                                                   static_cast<int>(members.size()));
        CUDA_CHECK(cudaGetLastError());
    }

    int findClosestPoint(int member_count, double threshold_sq) {
        const int blocks = (n + kCandidateBlockSize - 1) / kCandidateBlockSize;
        computeCandidateMaxKernel<<<blocks, kCandidateBlockSize>>>(
            d_distances, d_members, member_count, d_clustered, d_in_cluster,
            threshold_sq, n, d_candidate_values, d_candidate_indices);
        CUDA_CHECK(cudaGetLastError());

        double min_val = DBL_MAX;
        int min_idx = -1;
        reduceMin(d_candidate_values, d_candidate_indices, n, min_val, min_idx);
        if (min_val == DBL_MAX) {
            return -1;
        }
        return min_idx;
    }

private:
    void reduceMin(double* values, int* indices, int count, double& out_val, int& out_idx) {
        int num = count;
        double* current_vals = values;
        int* current_idx = indices;
        double* temp_vals = d_reduce_values;
        int* temp_idx = d_reduce_indices;

        while (true) {
            const int blocks = (num + (kReduceBlockSize * 2 - 1)) / (kReduceBlockSize * 2);
            reduceMinKernel<kReduceBlockSize><<<blocks, kReduceBlockSize>>>(
                current_vals, current_idx, temp_vals, temp_idx, num);
            CUDA_CHECK(cudaGetLastError());

            if (blocks == 1) {
                CUDA_CHECK(cudaMemcpy(&out_val, temp_vals, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&out_idx, temp_idx, sizeof(int), cudaMemcpyDeviceToHost));
                return;
            }

            num = blocks;
            std::swap(current_vals, temp_vals);
            std::swap(current_idx, temp_idx);
        }
    }

    int n = 0;
    Point* d_points = nullptr;
    double* d_distances = nullptr;
    uint8_t* d_clustered = nullptr;
    uint8_t* d_in_cluster = nullptr;
    int* d_members = nullptr;
    double* d_candidate_values = nullptr;
    int* d_candidate_indices = nullptr;
    double* d_reduce_values = nullptr;
    int* d_reduce_indices = nullptr;
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

// Generate a candidate cluster starting from a seed point using CUDA
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                             CudaContext& cuda_ctx,
                             const double threshold_sq,
                             const int point_count,
                             std::vector<int>* cluster_members = nullptr) {
    std::vector<int> members;
    members.reserve(point_count);

    cuda_ctx.resetInCluster();
    cuda_ctx.setInCluster(seed_point);
    cuda_ctx.setMember(0, seed_point);
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = cuda_ctx.findClosestPoint(static_cast<int>(members.size()),
                                                      threshold_sq);
        if (closest < 0) break;
        cuda_ctx.setInCluster(closest);
        cuda_ctx.setMember(static_cast<int>(members.size()), closest);
        members.push_back(closest);
    }

    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    CudaContext cuda_ctx(points);
    const double threshold_sq = threshold * threshold;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, cuda_ctx,
                                                             threshold_sq, N,
                                                             &candidate_members);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = 1;
            }
            cuda_ctx.markClustered(best_cluster_members);
            
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
