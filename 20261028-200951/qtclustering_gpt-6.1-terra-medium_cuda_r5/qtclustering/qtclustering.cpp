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

// Each block builds one seed's candidate cluster. The maximum distance to
// the current members is updated only for the newly added member.
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(EXIT_FAILURE); } } while (0)

constexpr int BLOCK_SIZE = 256;

__global__ void buildCandidates(const Point* points, const unsigned char* clustered,
                                int n, double threshold, double* maxima,
                                int* members, int* sizes) {
    int seed = blockIdx.x;
    if (clustered[seed]) {
        if (threadIdx.x == 0) sizes[seed] = 0;
        return;
    }
    double* row = maxima + static_cast<size_t>(seed) * n;
    int* sequence = members + static_cast<size_t>(seed) * n;
    __shared__ double best_dist[BLOCK_SIZE];
    __shared__ int best_index[BLOCK_SIZE];
    __shared__ int last, count;
    if (threadIdx.x == 0) { last = seed; count = 1; sequence[0] = seed; }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        row[i] = (clustered[i] || i == seed) ? -1.0 : 0.0;
    __syncthreads();
    while (true) {
        double local_best = threshold;
        int local_index = -1;
        Point p = points[last];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double prev = row[i];
            if (prev < 0.0) continue;
            double dx = points[i].x - p.x, dy = points[i].y - p.y;
            double d = sqrt(dx * dx + dy * dy);
            double m = fmax(prev, d);
            row[i] = m;
            if (m < local_best || (m == local_best && local_index >= 0 && i < local_index)) {
                local_best = m; local_index = i;
            }
        }
        best_dist[threadIdx.x] = local_best;
        best_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                int other = best_index[threadIdx.x + offset];
                int own = best_index[threadIdx.x];
                double od = best_dist[threadIdx.x + offset];
                if (other >= 0 && (own < 0 || od < best_dist[threadIdx.x] ||
                    (od == best_dist[threadIdx.x] && other < own))) {
                    best_dist[threadIdx.x] = od; best_index[threadIdx.x] = other;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            last = best_index[0];
            if (last >= 0) { row[last] = -1.0; sequence[count++] = last; }
        }
        __syncthreads();
        if (last < 0) break;
    }
    if (threadIdx.x == 0) sizes[seed] = count;
}

__global__ void markCluster(unsigned char* clustered, const int* members, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    Point* d_points; unsigned char* d_clustered; double* d_maxima;
    int *d_members, *d_sizes;
    const size_t matrix_size = static_cast<size_t>(n) * n;
    CUDA_CHECK(cudaMalloc(&d_points, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_maxima, matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_members, matrix_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, n * sizeof(unsigned char)));
    std::vector<int> sizes(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        buildCandidates<<<n, BLOCK_SIZE>>>(d_points, d_clustered, n, threshold,
                                           d_maxima, d_members, d_sizes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes, n * sizeof(int), cudaMemcpyDeviceToHost));
        int seed = static_cast<int>(std::max_element(sizes.begin(), sizes.end()) - sizes.begin());
        int count = sizes[seed];
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(count);
        int* row = d_members + static_cast<size_t>(seed) * n;
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), row, count * sizeof(int), cudaMemcpyDeviceToHost));
        markCluster<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_clustered, row, count);
        CUDA_CHECK(cudaGetLastError());
        clusters.push_back(std::move(cluster));
        remaining -= count;
    }
    CUDA_CHECK(cudaFree(d_points)); CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_maxima)); CUDA_CHECK(cudaFree(d_members)); CUDA_CHECK(cudaFree(d_sizes));
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
