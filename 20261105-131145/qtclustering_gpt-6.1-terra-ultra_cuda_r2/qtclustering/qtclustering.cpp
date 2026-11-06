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

// Each block builds one seed's candidate cluster.  A candidate's diameter
// only changes when a new member is added, so retain its running maximum.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__device__ double pointDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Match the native host compiler's contraction of dx*dx + dy*dy.
    return sqrt(__fma_rn(dx, dx, dy * dy));
}

__global__ void computeDistances(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int a = static_cast<int>(i / n);
        const int b = static_cast<int>(i % n);
        distances[i] = pointDistance(points[a], points[b]);
    }
}

constexpr int BLOCK_SIZE = 256;
__global__ void buildCandidates(const Point* points, const double* distances,
                                const unsigned char* clustered, double* scores,
                                int* sizes, int n, double threshold, int first_seed) {
    const int row = blockIdx.x;
    const int seed = first_seed + row;
    if (clustered[seed]) {
        if (threadIdx.x == 0) sizes[row] = 0;
        return;
    }
    double* row_scores = scores + static_cast<size_t>(row) * n;
    __shared__ double best_distance[BLOCK_SIZE / 32];
    __shared__ int best_index[BLOCK_SIZE / 32];
    __shared__ int newest;
    __shared__ int count;
    if (threadIdx.x == 0) {
        newest = seed;
        count = 1;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        row_scores[i] = clustered[i] ? -2.0 : 0.0;
    __syncthreads();

    while (true) {
        const int member = newest;
        double local_best = DBL_MAX;
        int local_index = -1;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double value = row_scores[i];
            if (i == member) {
                row_scores[i] = -1.0;
                continue;
            }
            if (value < 0.0) continue;
            const double d = distances ? distances[static_cast<size_t>(member) * n + i]
                                       : pointDistance(points[member], points[i]);
            // Equivalent to std::max(value, d), including for a NaN d.
            if (d > value) value = d;
            row_scores[i] = value;
            if (value < threshold &&
                (value < local_best || (value == local_best && i < local_index))) {
                local_best = value;
                local_index = i;
            }
        }
        // Reduce within warps first, then reduce the eight warp winners.
        for (int offset = 16; offset > 0; offset /= 2) {
            const double d = __shfl_down_sync(0xffffffff, local_best, offset);
            const int other = __shfl_down_sync(0xffffffff, local_index, offset);
            if (other >= 0 && (local_index < 0 || d < local_best ||
                               (d == local_best && other < local_index))) {
                local_best = d;
                local_index = other;
            }
        }
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane == 0) {
            best_distance[warp] = local_best;
            best_index[warp] = local_index;
        }
        __syncthreads();
        if (warp == 0) {
            local_best = lane < BLOCK_SIZE / 32 ? best_distance[lane] : DBL_MAX;
            local_index = lane < BLOCK_SIZE / 32 ? best_index[lane] : -1;
            for (int offset = 16; offset > 0; offset /= 2) {
                const double d = __shfl_down_sync(0xffffffff, local_best, offset);
                const int other = __shfl_down_sync(0xffffffff, local_index, offset);
                if (other >= 0 && (local_index < 0 || d < local_best ||
                                   (d == local_best && other < local_index))) {
                    local_best = d;
                    local_index = other;
                }
            }
            if (lane == 0) {
                newest = local_index;
                if (newest >= 0) ++count;
            }
        }
        __syncthreads();
        if (newest < 0) break;
    }
    if (threadIdx.x == 0) sizes[row] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(double);
    const size_t matrix_bytes = static_cast<size_t>(n) * row_bytes;
    Point* device_points = nullptr;
    double* device_distances = nullptr;
    double* device_scores = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&device_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&device_clustered, n));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // Cache distances when there is ample space; otherwise calculate them in
    // the candidate kernel.  Cap the candidate rows independently of N.
    if (matrix_bytes <= free_bytes / 2) {
        CUDA_CHECK(cudaMalloc(&device_distances, matrix_bytes));
        const size_t blocks = std::min<size_t>((static_cast<size_t>(n) * n + 255) / 256, 65535);
        computeDistances<<<static_cast<unsigned int>(blocks), 256>>>(device_points, device_distances, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = std::min<size_t>(free_bytes / 2, size_t{512} * 1024 * 1024);
    const int batch_size = static_cast<int>(std::min<size_t>(n, std::max<size_t>(1, budget / row_bytes)));
    CUDA_CHECK(cudaMalloc(&device_scores, static_cast<size_t>(batch_size) * row_bytes));
    CUDA_CHECK(cudaMalloc(&device_sizes, static_cast<size_t>(batch_size) * sizeof(int)));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<double> best_scores(n);
    std::vector<int> sizes(batch_size);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        CUDA_CHECK(cudaMemcpy(device_clustered, clustered.data(), n, cudaMemcpyHostToDevice));
        int best_seed = -1, best_size = 0;
        for (int first = 0; first < n; first += batch_size) {
            const int count = std::min(batch_size, n - first);
            buildCandidates<<<count, BLOCK_SIZE>>>(device_points, device_distances, device_clustered,
                                                   device_scores, device_sizes, n, threshold, first);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), device_sizes, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
            int best_row = -1;
            for (int row = 0; row < count; ++row) {
                if (sizes[row] > best_size) {
                    best_size = sizes[row];
                    best_seed = first + row;
                    best_row = row;
                }
            }
            if (best_row >= 0)
                CUDA_CHECK(cudaMemcpy(best_scores.data(), device_scores + static_cast<size_t>(best_row) * n,
                                      row_bytes, cudaMemcpyDeviceToHost));
        }
        // If the largest candidate is a singleton, every remaining seed is
        // also a singleton and the original tie rule picks them in index order.
        if (best_size == 1) {
            for (int i = 0; i < n; ++i) {
                if (!clustered[i]) clusters.push_back(Cluster{{i}, i});
            }
            break;
        }
        Cluster cluster;
        cluster.seed_point = best_seed;
        for (int i = 0; i < n; ++i) {
            if (best_scores[i] == -1.0) {
                cluster.members.push_back(i);
                clustered[i] = 1;
            }
        }
        remaining -= best_size;
        clusters.push_back(std::move(cluster));
    }
    CUDA_CHECK(cudaFree(device_sizes));
    CUDA_CHECK(cudaFree(device_scores));
    if (device_distances) CUDA_CHECK(cudaFree(device_distances));
    CUDA_CHECK(cudaFree(device_clustered));
    CUDA_CHECK(cudaFree(device_points));
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
