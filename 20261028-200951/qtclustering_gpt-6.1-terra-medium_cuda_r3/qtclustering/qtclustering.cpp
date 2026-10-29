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

// Each block builds one candidate cluster. The maximum distance for each
// candidate is updated only for the newly added member at each iteration.
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void buildCandidates(const Point* points, const unsigned char* clustered,
                                int n, double threshold, double* distances,
                                int* members, int* counts) {
    const int seed = blockIdx.x;
    if (clustered[seed]) {
        if (threadIdx.x == 0) counts[seed] = 0;
        return;
    }
    constexpr int threads = 256;
    __shared__ double best_distance[threads];
    __shared__ int best_index[threads];
    __shared__ int selected;
    __shared__ int size;
    const size_t base = static_cast<size_t>(seed) * n;
    if (threadIdx.x == 0) {
        members[base] = seed;
        selected = seed;
        size = 1;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        distances[base + i] = (clustered[i] || i == seed) ? INFINITY : 0.0;
    __syncthreads();

    while (true) {
        const int last = selected;
        double local_best = INFINITY;
        int local_index = -1;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double d = distances[base + i];
            if (d != INFINITY) {
                double dx = points[i].x - points[last].x;
                double dy = points[i].y - points[last].y;
                double current = sqrt(dx * dx + dy * dy);
                if (current > d) d = current;
                distances[base + i] = d;
                if (d < threshold && (d < local_best ||
                    (d == local_best && i < local_index))) {
                    local_best = d;
                    local_index = i;
                }
            }
        }
        best_distance[threadIdx.x] = local_best;
        best_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = threads / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double other = best_distance[threadIdx.x + stride];
                int index = best_index[threadIdx.x + stride];
                if (index >= 0 && (best_index[threadIdx.x] < 0 ||
                    other < best_distance[threadIdx.x] ||
                    (other == best_distance[threadIdx.x] && index < best_index[threadIdx.x]))) {
                    best_distance[threadIdx.x] = other;
                    best_index[threadIdx.x] = index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected = best_index[0];
            if (selected >= 0) {
                members[base + size] = selected;
                ++size;
                distances[base + selected] = INFINITY;
            }
        }
        __syncthreads();
        if (selected < 0 || size == n) break;
    }
    if (threadIdx.x == 0) counts[seed] = size;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t matrix = static_cast<size_t>(n) * n;
    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    double* device_distances = nullptr;
    int *device_members = nullptr, *device_counts = nullptr;
    cudaCheck(cudaMalloc(&device_points, n * sizeof(Point)), "allocate points");
    cudaCheck(cudaMalloc(&device_clustered, n * sizeof(unsigned char)), "allocate flags");
    cudaCheck(cudaMalloc(&device_distances, matrix * sizeof(double)), "allocate distances");
    cudaCheck(cudaMalloc(&device_members, matrix * sizeof(int)), "allocate members");
    cudaCheck(cudaMalloc(&device_counts, n * sizeof(int)), "allocate counts");
    cudaCheck(cudaMemcpy(device_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice), "copy points");
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> counts(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        cudaCheck(cudaMemcpy(device_clustered, clustered.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy flags");
        buildCandidates<<<n, 256>>>(device_points, device_clustered, n, threshold,
                                    device_distances, device_members, device_counts);
        cudaCheck(cudaGetLastError(), "launch candidate clusters");
        cudaCheck(cudaMemcpy(counts.data(), device_counts, n * sizeof(int), cudaMemcpyDeviceToHost), "copy counts");
        int seed = -1, count = 0;
        for (int i = 0; i < n; ++i) {
            if (counts[i] > count) { seed = i; count = counts[i]; }
        }
        if (seed < 0) break;
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(count);
        cudaCheck(cudaMemcpy(cluster.members.data(), device_members + static_cast<size_t>(seed) * n,
                             count * sizeof(int), cudaMemcpyDeviceToHost), "copy winning cluster");
        for (int member : cluster.members) clustered[member] = 1;
        remaining -= count;
        clusters.push_back(std::move(cluster));
    }
    cudaCheck(cudaFree(device_counts), "free counts");
    cudaCheck(cudaFree(device_members), "free members");
    cudaCheck(cudaFree(device_distances), "free distances");
    cudaCheck(cudaFree(device_clustered), "free flags");
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
