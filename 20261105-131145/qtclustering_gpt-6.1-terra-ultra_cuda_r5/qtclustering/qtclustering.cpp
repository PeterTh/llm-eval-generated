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

// Each seed is handled by a block. Its candidate distances are maintained
// incrementally: after adding a member only the distance to that member is new.
#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int BLOCK_SIZE = 256;

__global__ void makeDistances(const Point* points, double* distances, int n) {
    size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int a = static_cast<int>(i / n);
        int b = static_cast<int>(i % n);
        double dx = points[a].x - points[b].x;
        double dy = points[a].y - points[b].y;
        distances[i] = sqrt(dx * dx + dy * dy);
    }
}

__global__ void buildCandidates(const Point* points, const double* distances,
                                const int* clustered, int* sizes, int* members,
                                double* workspace, int n, double threshold,
                                int offset, bool use_shared) {
    int seed = offset + blockIdx.x;
    if (seed >= n || clustered[seed] || sizes[seed] >= 0) return;

    extern __shared__ double shared_max[];
    double* maxima = use_shared ? shared_max : workspace + static_cast<size_t>(blockIdx.x) * n;
    __shared__ double best_dist[BLOCK_SIZE / 32];
    __shared__ int best_index[BLOCK_SIZE / 32];
    __shared__ int last, count;
    int tid = threadIdx.x;
    int* row = members + static_cast<size_t>(seed) * n;
    for (int i = tid; i < n; i += blockDim.x)
        maxima[i] = (clustered[i] || i == seed) ? DBL_MAX : 0.0;
    if (tid == 0) {
        row[0] = seed;
        last = seed;
        count = 1;
    }
    __syncthreads();
    while (true) {
        double local_best = DBL_MAX;
        int local_index = -1;
        int member = last;
        for (int i = tid; i < n; i += blockDim.x) {
            double m = maxima[i];
            if (m == DBL_MAX) continue;
            double d;
            if (distances) {
                d = distances[static_cast<size_t>(member) * n + i];
            } else {
                double dx = points[i].x - points[member].x;
                double dy = points[i].y - points[member].y;
                d = sqrt(dx * dx + dy * dy);
            }
            // std::max in the original retains its first operand for NaN.
            if (d > m) m = d;
            // The maximum can only increase, so a failing candidate can
            // never become eligible later in this cluster.
            maxima[i] = (m < threshold) ? m : DBL_MAX;
            if (m < threshold && m < DBL_MAX && (m < local_best ||
                                  (m == local_best && (local_index < 0 || i < local_index)))) {
                local_best = m;
                local_index = i;
            }
        }
        // Reduce within each warp, then reduce the eight warp winners.
        for (int delta = 16; delta > 0; delta /= 2) {
            double other = __shfl_down_sync(0xffffffffu, local_best, delta);
            int oi = __shfl_down_sync(0xffffffffu, local_index, delta);
            if (oi >= 0 && (other < local_best ||
                (other == local_best && (local_index < 0 || oi < local_index)))) {
                local_best = other;
                local_index = oi;
            }
        }
        if ((tid & 31) == 0) {
            best_dist[tid / 32] = local_best;
            best_index[tid / 32] = local_index;
        }
        __syncthreads();
        if (tid < 32) {
            local_best = tid < BLOCK_SIZE / 32 ? best_dist[tid] : DBL_MAX;
            local_index = tid < BLOCK_SIZE / 32 ? best_index[tid] : -1;
            for (int delta = 16; delta > 0; delta /= 2) {
                double other = __shfl_down_sync(0xffffffffu, local_best, delta);
                int oi = __shfl_down_sync(0xffffffffu, local_index, delta);
                if (oi >= 0 && (other < local_best ||
                    (other == local_best && (local_index < 0 || oi < local_index)))) {
                    local_best = other;
                    local_index = oi;
                }
            }
            if (tid == 0) {
                last = local_index;
                if (last >= 0) {
                    maxima[last] = DBL_MAX;
                    row[count++] = last;
                }
            }
        }
        __syncthreads();
        if (last < 0 || count == n) break;
    }
    if (tid == 0) sizes[seed] = count;
}

__global__ void markCluster(const int* row, int size, int* clustered) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        clustered[row[i]] = 1;
}

// Removing points outside a candidate's member list cannot change any of its
// greedy choices. Only candidate clusters intersecting the removed set expire.
__global__ void invalidateCandidates(const int* clustered, int* sizes,
                                     const int* members, int n) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= n) return;
    if (clustered[seed]) { sizes[seed] = -1; return; }
    int size = sizes[seed];
    const int* row = members + static_cast<size_t>(seed) * n;
    for (int i = 0; i < size; ++i) {
        if (clustered[row[i]]) {
            sizes[seed] = -1;
            break;
        }
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t matrix_elements = static_cast<size_t>(n) * n;
    if (matrix_elements > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Point count is too large for CUDA workspace\n");
        std::exit(EXIT_FAILURE);
    }
    Point* d_points;
    int *d_clustered, *d_sizes, *d_members;
    double *d_distances = nullptr, *d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, matrix_elements * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_sizes, 0xff, static_cast<size_t>(n) * sizeof(int)));

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    // Leave room for the block reduction arrays as well as dynamic memory.
    bool use_shared = static_cast<size_t>(n) * sizeof(double) +
                      (BLOCK_SIZE / 32) * (sizeof(double) + sizeof(int)) + 64 <= prop.sharedMemPerBlock;
    int batch = use_shared ? n : std::min(n, 128);
    if (!use_shared)
        CUDA_CHECK(cudaMalloc(&d_workspace, static_cast<size_t>(batch) * n * sizeof(double)));

    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t distance_bytes = matrix_elements * sizeof(double);
    if (distance_bytes <= free_bytes / 2) {
        CUDA_CHECK(cudaMalloc(&d_distances, distance_bytes));
        int blocks = static_cast<int>(std::min<size_t>((matrix_elements + 255) / 256, 65535));
        makeDistances<<<blocks, 256>>>(d_points, d_distances, n);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<int> sizes(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        for (int offset = 0; offset < n; offset += batch) {
            int blocks = std::min(batch, n - offset);
            buildCandidates<<<blocks, BLOCK_SIZE, use_shared ? static_cast<size_t>(n) * sizeof(double) : 0>>>(
                d_points, d_distances, d_clustered, d_sizes, d_members, d_workspace,
                n, threshold, offset, use_shared);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
        int best_seed = -1, best_size = -1;
        for (int i = 0; i < n; ++i) {
            if (sizes[i] > best_size) {
                best_size = sizes[i];
                best_seed = i;
            }
        }
        if (best_seed < 0 || best_size <= 0) break;
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(best_size);
        const int* row = d_members + static_cast<size_t>(best_seed) * n;
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), row, static_cast<size_t>(best_size) * sizeof(int), cudaMemcpyDeviceToHost));
        markCluster<<<(best_size + 255) / 256, 256>>>(row, best_size, d_clustered);
        CUDA_CHECK(cudaGetLastError());
        remaining -= best_size;
        clusters.push_back(std::move(cluster));
        if (remaining) {
            invalidateCandidates<<<(n + 255) / 256, 256>>>(d_clustered, d_sizes, d_members, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));
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
