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

// Each block builds one seed's candidate cluster.  Threads maintain the
// maximum distance to the members seen so far and reduce to the next point.
// The index is the secondary key, matching the sequential scan's tie rule.
#include <cuda_runtime.h>

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

__global__ void calculateDistances(const Point* points, double* distances, int n) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * n;
    for (; k < total; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int a = static_cast<int>(k / n), b = static_cast<int>(k % n);
        double dx = points[a].x - points[b].x;
        double dy = points[a].y - points[b].y;
        distances[k] = sqrt(dx * dx + dy * dy);
    }
}

constexpr int BLOCK_SIZE = 256;
__global__ void buildCandidates(const double* distances, const unsigned char* clustered,
                                double* maxima, int* members, int* sizes,
                                int n, double threshold) {
    const int seed = blockIdx.x;
    if (clustered[seed]) { if (threadIdx.x == 0) sizes[seed] = 0; return; }
    const size_t base = static_cast<size_t>(seed) * n;
    __shared__ double best_dist[BLOCK_SIZE];
    __shared__ int best_index[BLOCK_SIZE];
    __shared__ int last, count;
    if (threadIdx.x == 0) { last = seed; count = 1; members[base] = seed; }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        maxima[base + i] = (clustered[i] || i == seed) ? DBL_MAX : 0.0;
    __syncthreads();
    while (true) {
        const int current = last;
        double local_dist = DBL_MAX;
        int local_index = -1;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double value = maxima[base + i];
            if (value != DBL_MAX) {
                value = fmax(value, distances[static_cast<size_t>(current) * n + i]);
                maxima[base + i] = value;
                if (value < threshold && (value < local_dist ||
                    (value == local_dist && (local_index < 0 || i < local_index)))) {
                    local_dist = value;
                    local_index = i;
                }
            }
        }
        best_dist[threadIdx.x] = local_dist;
        best_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                const int other = best_index[threadIdx.x + stride];
                const double d = best_dist[threadIdx.x + stride];
                if (other >= 0 && (best_index[threadIdx.x] < 0 ||
                    d < best_dist[threadIdx.x] ||
                    (d == best_dist[threadIdx.x] && other < best_index[threadIdx.x]))) {
                    best_dist[threadIdx.x] = d;
                    best_index[threadIdx.x] = other;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            last = best_index[0];
            if (last >= 0) {
                members[base + count++] = last;
                maxima[base + last] = DBL_MAX;
            }
        }
        __syncthreads();
        if (last < 0) break;
    }
    if (threadIdx.x == 0) sizes[seed] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t cells = static_cast<size_t>(n) * n;
    Point* d_points;
    double *d_distances, *d_maxima;
    int *d_members, *d_sizes;
    unsigned char* d_clustered;
    CUDA_CHECK(cudaMalloc(&d_points, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_distances, cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxima, cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_members, cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clustered, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, n * sizeof(unsigned char)));
    calculateDistances<<<static_cast<int>(std::min<size_t>((cells + 255) / 256, 65535)), 256>>>(d_points, d_distances, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> sizes(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        buildCandidates<<<n, BLOCK_SIZE>>>(d_distances, d_clustered, d_maxima,
                                           d_members, d_sizes, n, threshold);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes, n * sizeof(int), cudaMemcpyDeviceToHost));
        int best = -1;
        for (int i = 0; i < n; ++i)
            if (sizes[i] > 0 && (best < 0 || sizes[i] > sizes[best])) best = i;
        if (best < 0) break;
        Cluster cluster;
        cluster.seed_point = best;
        cluster.members.resize(sizes[best]);
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), d_members + static_cast<size_t>(best) * n,
                              cluster.members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int member : cluster.members) clustered[member] = 1;
        remaining -= sizes[best];
        clusters.push_back(std::move(cluster));
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_maxima));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_clustered));
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
