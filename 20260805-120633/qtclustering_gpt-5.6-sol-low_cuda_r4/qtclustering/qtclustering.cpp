// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <stdexcept>
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

static void cudaCheck(cudaError_t e, const char* operation) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}

// One block constructs one seed's candidate.  max_distance is updated
// incrementally, avoiding a rescan of all existing members at every step.
__global__ void distanceMatrixKernel(const Point* points, int n, double* distances) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * n;
    if (index < total) distances[index] = distance(points[index / n], points[index % n]);
}

__global__ void candidateClustersKernel(const double* distances, const unsigned char* clustered,
                                         int n, double threshold, int* cardinalities,
                                         int* member_order, double* max_distance) {
    const int seed = blockIdx.x;
    if (seed >= n || clustered[seed]) return;
    const int tid = threadIdx.x;
    double* row = max_distance + static_cast<size_t>(seed) * n;
    int* output = member_order + static_cast<size_t>(seed) * n;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];
    __shared__ int chosen;

    for (int i = tid; i < n; i += blockDim.x)
        row[i] = (clustered[i] || i == seed) ? DBL_MAX : distances[static_cast<size_t>(seed) * n + i];
    if (tid == 0) { output[0] = seed; cardinalities[seed] = 1; }
    __syncthreads();

    for (int count = 1; count < n; ++count) {
        double value = DBL_MAX;
        int index = -1;
        for (int i = tid; i < n; i += blockDim.x) {
            const double d = row[i];
            if (d < threshold && (d < value || (d == value && (index < 0 || i < index)))) {
                value = d; index = i;
            }
        }
        best_distance[tid] = value; best_index[tid] = index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride) {
                const double other = best_distance[tid + stride];
                const int oi = best_index[tid + stride];
                if (oi >= 0 && (best_index[tid] < 0 || other < best_distance[tid] ||
                    (other == best_distance[tid] && oi < best_index[tid]))) {
                    best_distance[tid] = other; best_index[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            chosen = best_index[0];
            if (chosen >= 0) { output[count] = chosen; cardinalities[seed] = count + 1; row[chosen] = DBL_MAX; }
        }
        __syncthreads();
        if (chosen < 0) break;
        for (int i = tid; i < n; i += blockDim.x) {
            if (row[i] != DBL_MAX) row[i] = fmax(row[i], distances[static_cast<size_t>(chosen) * n + i]);
        }
        __syncthreads();
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    Point* d_points = nullptr; unsigned char* d_clustered = nullptr;
    int* d_cardinalities = nullptr; int* d_members = nullptr;
    double* d_max_distance = nullptr; double* d_distances = nullptr;
    const size_t matrix_count = static_cast<size_t>(N) * N;
    cudaCheck(cudaMalloc(&d_points, N * sizeof(Point)), "cudaMalloc points");
    cudaCheck(cudaMalloc(&d_clustered, N), "cudaMalloc clustered");
    cudaCheck(cudaMalloc(&d_cardinalities, N * sizeof(int)), "cudaMalloc cardinalities");
    cudaCheck(cudaMalloc(&d_members, matrix_count * sizeof(int)), "cudaMalloc members");
    cudaCheck(cudaMalloc(&d_max_distance, matrix_count * sizeof(double)), "cudaMalloc workspace");
    cudaCheck(cudaMalloc(&d_distances, matrix_count * sizeof(double)), "cudaMalloc distances");
    cudaCheck(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice), "copy points");
    distanceMatrixKernel<<<static_cast<unsigned>((matrix_count + 255) / 256), 256>>>(d_points, N, d_distances);
    cudaCheck(cudaGetLastError(), "launch distance kernel");
    std::vector<int> cardinalities(N);

    // Main clustering loop. Every iteration evaluates all remaining seeds concurrently.
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), N, cudaMemcpyHostToDevice), "copy flags");
        cudaCheck(cudaMemset(d_cardinalities, 0, N * sizeof(int)), "clear cardinalities");
        candidateClustersKernel<<<N, 256>>>(d_distances, d_clustered, N, threshold,
                                             d_cardinalities, d_members, d_max_distance);
        cudaCheck(cudaGetLastError(), "launch candidate kernel");
        cudaCheck(cudaMemcpy(cardinalities.data(), d_cardinalities, N * sizeof(int), cudaMemcpyDeviceToHost), "copy cardinalities");
        for (int seed : unclustered_indices) {
            const int cardinality = cardinalities[seed];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }
        if (best_seed >= 0)
            { best_cluster_members.resize(max_cardinality); cudaCheck(cudaMemcpy(best_cluster_members.data(), d_members + static_cast<size_t>(best_seed) * N, max_cardinality * sizeof(int), cudaMemcpyDeviceToHost), "copy best cluster"); }
        
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
    
    cudaFree(d_distances); cudaFree(d_max_distance); cudaFree(d_members); cudaFree(d_cardinalities);
    cudaFree(d_clustered); cudaFree(d_points);
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
