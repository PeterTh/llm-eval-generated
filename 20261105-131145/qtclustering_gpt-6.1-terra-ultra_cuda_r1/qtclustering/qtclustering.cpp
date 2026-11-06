// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
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

// Each seed is independent until the largest candidate cluster is committed.
// Store distances once and maintain the maximum distance to the current cluster
// incrementally, rather than scanning all of its members after every addition.
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr int BLOCK_SIZE = 256;

__global__ void computeDistances(const Point* points, double* distances, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    for (size_t pos = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pos < count; pos += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int a = static_cast<int>(pos / n);
        const int b = static_cast<int>(pos % n);
        const double dx = points[a].x - points[b].x;
        const double dy = points[a].y - points[b].y;
        distances[pos] = sqrt(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
    }
}

// Rows in maxima belong to seeds. A negative value encodes selection order;
// DBL_MAX marks an unavailable candidate. All other entries are running maxima.
__global__ void buildCandidates(const double* distances, double* maxima,
                                const int* membership, int* cardinalities,
                                int n, double threshold) {
    const int seed = blockIdx.x;
    const int tid = threadIdx.x;
    if (membership[seed] >= 0) {
        if (tid == 0) cardinalities[seed] = 0;
        return;
    }
    double* row = maxima + static_cast<size_t>(seed) * n;
    for (int i = tid; i < n; i += blockDim.x) {
        row[i] = i == seed ? -1.0 : (membership[i] >= 0 ? DBL_MAX : 0.0);
    }
    __syncthreads();

    __shared__ double bestDistances[BLOCK_SIZE / 32];
    __shared__ int bestIndices[BLOCK_SIZE / 32];
    __shared__ int next;
    int latest = seed;
    int count = 1;
    while (true) {
        double best = DBL_MAX;
        int bestIndex = INT_MAX;
        const double* distanceRow = distances + static_cast<size_t>(latest) * n;
        for (int i = tid; i < n; i += blockDim.x) {
            double d = row[i];
            if (d >= 0.0 && d < threshold) {
                d = fmax(d, distanceRow[i]);
                row[i] = d;
                if (d < threshold && d < DBL_MAX &&
                    (d < best || (d == best && i < bestIndex))) {
                    best = d;
                    bestIndex = i;
                }
            }
        }
        // First reduce within each warp, then reduce the eight warp winners.
        for (int offset = 16; offset > 0; offset /= 2) {
            const double other = __shfl_down_sync(0xffffffff, best, offset);
            const int otherIndex = __shfl_down_sync(0xffffffff, bestIndex, offset);
            if (other < best || (other == best && otherIndex < bestIndex)) {
                best = other;
                bestIndex = otherIndex;
            }
        }
        if ((tid & 31) == 0) {
            bestDistances[tid / 32] = best;
            bestIndices[tid / 32] = bestIndex;
        }
        __syncthreads();
        if (tid < 32) {
            best = tid < BLOCK_SIZE / 32 ? bestDistances[tid] : DBL_MAX;
            bestIndex = tid < BLOCK_SIZE / 32 ? bestIndices[tid] : INT_MAX;
            for (int offset = 16; offset > 0; offset /= 2) {
                const double other = __shfl_down_sync(0xffffffff, best, offset);
                const int otherIndex = __shfl_down_sync(0xffffffff, bestIndex, offset);
                if (other < best || (other == best && otherIndex < bestIndex)) {
                    best = other;
                    bestIndex = otherIndex;
                }
            }
            if (tid == 0) {
                next = bestIndex;
                if (next != INT_MAX) row[next] = -static_cast<double>(count + 1);
            }
        }
        __syncthreads();
        if (next == INT_MAX) break;
        latest = next;
        ++count;
    }
    if (tid == 0) cardinalities[seed] = count;
}

// Select the first seed with maximum cardinality and commit its members.
__global__ void commitBest(const double* maxima, const int* cardinalities,
                           int* membership, int* order, int* result, int n, int clusterId) {
    const int tid = threadIdx.x;
    int count = 0;
    int seed = INT_MAX;
    for (int i = tid; i < n; i += blockDim.x) {
        const int c = cardinalities[i];
        if (c > count || (c == count && i < seed)) {
            count = c;
            seed = i;
        }
    }
    __shared__ int counts[BLOCK_SIZE];
    __shared__ int seeds[BLOCK_SIZE];
    counts[tid] = count;
    seeds[tid] = seed;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride && (counts[tid + stride] > counts[tid] ||
            (counts[tid + stride] == counts[tid] && seeds[tid + stride] < seeds[tid]))) {
            counts[tid] = counts[tid + stride];
            seeds[tid] = seeds[tid + stride];
        }
        __syncthreads();
    }
    seed = seeds[0];
    // If even the largest candidate is a singleton, no two remaining points
    // can share a cluster. Commit the tail in the original seed order.
    if (counts[0] == 1) {
        if (tid == 0) {
            int tail = 0;
            for (int i = 0; i < n; ++i) {
                if (membership[i] < 0) {
                    membership[i] = clusterId + tail++;
                    order[i] = 0;
                }
            }
            result[0] = seed;
            result[1] = 1;
            result[2] = tail;
        }
        return;
    }
    if (tid == 0) {
        result[0] = seed;
        result[1] = counts[0];
        result[2] = 0;
    }
    if (counts[0] > 0) {
        const double* row = maxima + static_cast<size_t>(seed) * n;
        for (int i = tid; i < n; i += blockDim.x) {
            if (row[i] < 0.0) {
                membership[i] = clusterId;
                order[i] = -static_cast<int>(row[i]) - 1;
            }
        }
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    if (n == 0) return {};
    if (static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / sizeof(double) / static_cast<size_t>(n)) {
        fprintf(stderr, "Point count is too large for the distance matrix\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t matrixBytes = static_cast<size_t>(n) * n * sizeof(double);
    Point* devicePoints;
    double *distances, *maxima;
    int *membership, *order, *cardinalities, *result;
    CUDA_CHECK(cudaMalloc(&devicePoints, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&distances, matrixBytes));
    CUDA_CHECK(cudaMalloc(&maxima, matrixBytes));
    CUDA_CHECK(cudaMalloc(&membership, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&order, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cardinalities, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&result, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(devicePoints, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(membership, 0xff, static_cast<size_t>(n) * sizeof(int)));
    const size_t pairs = static_cast<size_t>(n) * n;
    const int blocks = static_cast<int>(std::min<size_t>((pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535));
    computeDistances<<<blocks, BLOCK_SIZE>>>(devicePoints, distances, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        buildCandidates<<<n, BLOCK_SIZE>>>(distances, maxima, membership, cardinalities, n, threshold);
        CUDA_CHECK(cudaGetLastError());
        commitBest<<<1, BLOCK_SIZE>>>(maxima, cardinalities, membership, order, result, n,
                                      static_cast<int>(clusters.size()));
        CUDA_CHECK(cudaGetLastError());
        int best[3];
        CUDA_CHECK(cudaMemcpy(best, result, sizeof(best), cudaMemcpyDeviceToHost));
        if (best[2] > 0) {
            clusters.resize(clusters.size() + best[2]);
            break;
        }
        Cluster cluster;
        cluster.seed_point = best[0];
        cluster.members.resize(best[1]);
        clusters.push_back(std::move(cluster));
        remaining -= best[1];
    }
    std::vector<int> labels(n), ordering(n);
    CUDA_CHECK(cudaMemcpy(labels.data(), membership, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ordering.data(), order, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        Cluster& cluster = clusters[labels[i]];
        if (cluster.members.empty()) {
            cluster.seed_point = i;
            cluster.members.resize(1);
        }
        cluster.members[ordering[i]] = i;
    }
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(cardinalities));
    CUDA_CHECK(cudaFree(order));
    CUDA_CHECK(cudaFree(membership));
    CUDA_CHECK(cudaFree(maxima));
    CUDA_CHECK(cudaFree(distances));
    CUDA_CHECK(cudaFree(devicePoints));
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
