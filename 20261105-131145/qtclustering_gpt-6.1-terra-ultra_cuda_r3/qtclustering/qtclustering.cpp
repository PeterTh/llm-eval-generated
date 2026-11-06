// QT Clustering Benchmark - CUDA Version
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
#include <cfloat>
#include <cstdint>
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

// Check all CUDA operations, including asynchronous kernel failures.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Each row contains distances from one possible member to all candidates.
__global__ void calculateDistances(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int member = static_cast<int>(i / n);
        const int candidate = static_cast<int>(i % n);
        const double dx = points[candidate].x - points[member].x;
        const double dy = points[candidate].y - points[member].y;
        distances[i] = sqrt(dx * dx + dy * dy);
    }
}

constexpr int CUDA_THREADS = 256;

// A block grows one seed's cluster. The maximum distance for each candidate
// only needs updating against the most recently added member. Negative values
// denote selected (-1) or ineligible (-2) candidates.
__global__ void growCandidates(const double* distances, const unsigned char* clustered,
                               double* states, int* sizes, int n, double threshold,
                               bool use_shared) {
    const int seed = blockIdx.x;
    const int tid = threadIdx.x;
    if (clustered[seed]) {
        if (tid == 0) sizes[seed] = 0;
        return;
    }
    extern __shared__ double shared_state[];
    __shared__ double reduction_dist[CUDA_THREADS / 32];
    __shared__ int reduction_idx[CUDA_THREADS / 32];
    __shared__ int selected;
    double* const row = states + static_cast<size_t>(seed) * n;
    double* const state = use_shared ? shared_state : row;
    const double* const initial = distances + static_cast<size_t>(seed) * n;
    for (int i = tid; i < n; i += blockDim.x) {
        if (i == seed) {
            state[i] = -1.0;
        } else if (clustered[i]) {
            state[i] = -2.0;
        } else {
            const double d = initial[i];
            // std::max(0.0, d) in the original ignores a NaN distance.
            const double maximum = d > 0.0 ? d : 0.0;
            state[i] = maximum < threshold && maximum < DBL_MAX ? maximum : -2.0;
        }
    }
    __syncthreads();
    int count = 1;
    while (count < n) {
        double local_dist = DBL_MAX;
        int local_idx = n;
        for (int i = tid; i < n; i += blockDim.x) {
            const double d = state[i];
            if (d >= 0.0 && (d < local_dist || (d == local_dist && i < local_idx))) {
                local_dist = d;
                local_idx = i;
            }
        }
        // Reduce within warps first, then have one warp combine the eight winners.
        // Index is the secondary key, matching the sequential scan's tie break.
        for (int offset = 16; offset > 0; offset /= 2) {
            const double d = __shfl_down_sync(0xffffffff, local_dist, offset);
            const int i = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (d < local_dist || (d == local_dist && i < local_idx)) {
                local_dist = d;
                local_idx = i;
            }
        }
        const int lane = tid % 32;
        const int warp = tid / 32;
        if (lane == 0) {
            reduction_dist[warp] = local_dist;
            reduction_idx[warp] = local_idx;
        }
        __syncthreads();
        if (warp == 0) {
            local_dist = lane < CUDA_THREADS / 32 ? reduction_dist[lane] : DBL_MAX;
            local_idx = lane < CUDA_THREADS / 32 ? reduction_idx[lane] : n;
            for (int offset = 16; offset > 0; offset /= 2) {
                const double d = __shfl_down_sync(0xffffffff, local_dist, offset);
                const int i = __shfl_down_sync(0xffffffff, local_idx, offset);
                if (d < local_dist || (d == local_dist && i < local_idx)) {
                    local_dist = d;
                    local_idx = i;
                }
            }
            if (lane == 0) selected = local_idx;
        }
        __syncthreads();
        const int member = selected;
        if (member == n) break;
        const double* const new_distances = distances + static_cast<size_t>(member) * n;
        for (int i = tid; i < n; i += blockDim.x) {
            double current = state[i];
            if (i == member) {
                state[i] = -1.0;
            } else if (current >= 0.0) {
                const double d = new_distances[i];
                if (d > current) current = d;
                state[i] = current < threshold && current < DBL_MAX ? current : -2.0;
            }
        }
        ++count;
        __syncthreads();
    }
    if (use_shared) {
        for (int i = tid; i < n; i += blockDim.x) row[i] = state[i];
    }
    if (tid == 0) sizes[seed] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t count = static_cast<size_t>(n);
    if (count > SIZE_MAX / count / sizeof(double)) {
        fprintf(stderr, "Point count is too large for the CUDA distance matrix\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t matrix_bytes = count * count * sizeof(double);
    Point* device_points = nullptr;
    double *distances = nullptr, *states = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_sizes = nullptr;
    cudaCheck(cudaMalloc(&device_points, count * sizeof(Point)), "allocate points");
    cudaCheck(cudaMalloc(&distances, matrix_bytes), "allocate distance matrix");
    cudaCheck(cudaMalloc(&states, matrix_bytes), "allocate candidate states");
    cudaCheck(cudaMalloc(&device_clustered, count), "allocate clustered flags");
    cudaCheck(cudaMalloc(&device_sizes, count * sizeof(int)), "allocate candidate sizes");
    cudaCheck(cudaMemcpy(device_points, points.data(), count * sizeof(Point), cudaMemcpyHostToDevice),
              "copy points");
    const size_t blocks = std::min<size_t>((count * count + CUDA_THREADS - 1) / CUDA_THREADS, 65535);
    calculateDistances<<<static_cast<unsigned int>(blocks), CUDA_THREADS>>>(device_points, distances, n);
    cudaCheck(cudaGetLastError(), "launch distance calculation");

    // Limit per-block shared memory so multiple seed blocks can run concurrently.
    const bool use_shared = count * sizeof(double) <= 32 * 1024;
    const size_t shared_bytes = use_shared ? count * sizeof(double) : 0;
    std::vector<unsigned char> clustered(count, 0);
    std::vector<int> sizes(count);
    std::vector<double> winning_state(count);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        cudaCheck(cudaMemcpy(device_clustered, clustered.data(), count, cudaMemcpyHostToDevice),
                  "copy clustered flags");
        growCandidates<<<n, CUDA_THREADS, shared_bytes>>>(distances, device_clustered, states,
                                                          device_sizes, n, threshold, use_shared);
        cudaCheck(cudaGetLastError(), "launch candidate clustering");
        cudaCheck(cudaMemcpy(sizes.data(), device_sizes, count * sizeof(int), cudaMemcpyDeviceToHost),
                  "copy candidate sizes");
        int best_seed = -1;
        int best_size = 0;
        for (int i = 0; i < n; ++i) {
            if (sizes[i] > best_size) {
                best_size = sizes[i];
                best_seed = i;
            }
        }
        if (best_seed < 0) break;
        cudaCheck(cudaMemcpy(winning_state.data(), states + static_cast<size_t>(best_seed) * n,
                             count * sizeof(double), cudaMemcpyDeviceToHost), "copy winning cluster");
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.reserve(best_size);
        // Membership order does not affect subsequent rounds or the output.
        for (int i = 0; i < n; ++i) {
            if (winning_state[i] == -1.0) {
                cluster.members.push_back(i);
                clustered[i] = 1;
            }
        }
        remaining -= best_size;
        clusters.push_back(std::move(cluster));
    }
    cudaCheck(cudaFree(device_sizes), "free candidate sizes");
    cudaCheck(cudaFree(device_clustered), "free clustered flags");
    cudaCheck(cudaFree(states), "free candidate states");
    cudaCheck(cudaFree(distances), "free distances");
    cudaCheck(cudaFree(device_points), "free points");
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
