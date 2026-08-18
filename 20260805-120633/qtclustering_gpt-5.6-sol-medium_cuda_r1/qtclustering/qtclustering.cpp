// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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

namespace {

constexpr int CUDA_BLOCK_SIZE = 256;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

// Distances are reused by every seed and every outer QT iteration.  Computing
// this matrix once removes the sqrt from the much hotter greedy search kernel.
__global__ void buildDistanceMatrix(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < total; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(k / n);
        const int j = static_cast<int>(k - static_cast<size_t>(i) * n);
        const double dx = points[i].x - points[j].x;
        const double dy = points[i].y - points[j].y;
        distances[k] = sqrt(dx * dx + dy * dy);
    }
}

__device__ __forceinline__ bool betterCandidate(double value, int index,
                                                 double best_value, int best_index) {
    return value < best_value || (value == best_value && index < best_index);
}

// One block constructs one seed's complete greedy candidate cluster. Threads
// cooperate across candidate points; all seed clusters execute concurrently.
__global__ void buildCandidateClusters(const double* __restrict__ distances,
                                       const unsigned char* __restrict__ clustered,
                                       double* __restrict__ max_distances,
                                       int* __restrict__ members,
                                       int* __restrict__ cardinalities,
                                       double threshold, int n) {
    const int seed = blockIdx.x;
    if (seed >= n || clustered[seed]) {
        if (seed < n && threadIdx.x == 0) cardinalities[seed] = 0;
        return;
    }

    double* seed_max = max_distances + static_cast<size_t>(seed) * n;
    int* seed_members = members + static_cast<size_t>(seed) * n;
    for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
        seed_max[candidate] = (clustered[candidate] || candidate == seed)
                                  ? DBL_MAX
                                  : 0.0;
    }
    if (threadIdx.x == 0) seed_members[0] = seed;
    __syncthreads();

    __shared__ double reduction_value[CUDA_BLOCK_SIZE];
    __shared__ int reduction_index[CUDA_BLOCK_SIZE];
    __shared__ int current_member;
    __shared__ int cluster_size;
    if (threadIdx.x == 0) {
        current_member = seed;
        cluster_size = 1;
    }
    __syncthreads();

    while (cluster_size < n) {
        double local_value = DBL_MAX;
        int local_index = INT_MAX;
        const size_t distance_row = static_cast<size_t>(current_member) * n;

        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            double value = seed_max[candidate];
            if (value != DBL_MAX) {
                value = fmax(value, distances[distance_row + candidate]);
                seed_max[candidate] = value;
                if (value < threshold &&
                    betterCandidate(value, candidate, local_value, local_index)) {
                    local_value = value;
                    local_index = candidate;
                }
            }
        }

        reduction_value[threadIdx.x] = local_value;
        reduction_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double other_value = reduction_value[threadIdx.x + stride];
                const int other_index = reduction_index[threadIdx.x + stride];
                if (betterCandidate(other_value, other_index,
                                    reduction_value[threadIdx.x],
                                    reduction_index[threadIdx.x])) {
                    reduction_value[threadIdx.x] = other_value;
                    reduction_index[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const int chosen = reduction_index[0];
            if (chosen == INT_MAX) {
                current_member = -1;
            } else {
                seed_max[chosen] = DBL_MAX;
                seed_members[cluster_size] = chosen;
                ++cluster_size;
                current_member = chosen;
            }
        }
        __syncthreads();
        if (current_member < 0) break;
    }

    if (threadIdx.x == 0) cardinalities[seed] = cluster_size;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
                  "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
private:
    T* pointer_ = nullptr;
};

} // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const size_t matrix_elements = static_cast<size_t>(N) * N;
    if (N != 0 && matrix_elements / static_cast<size_t>(N) != static_cast<size_t>(N)) {
        throw std::runtime_error("point count is too large");
    }

    DeviceBuffer<Point> device_points(N);
    DeviceBuffer<double> device_distances(matrix_elements);
    DeviceBuffer<double> device_max_distances(matrix_elements);
    DeviceBuffer<int> device_members(matrix_elements);
    DeviceBuffer<int> device_cardinalities(N);
    DeviceBuffer<unsigned char> device_clustered(N);

    cudaCheck(cudaMemcpy(device_points.get(), points.data(), N * sizeof(Point),
                         cudaMemcpyHostToDevice), "copy points to GPU");
    const int distance_blocks = static_cast<int>(std::min<size_t>(
        65535, (matrix_elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE));
    buildDistanceMatrix<<<distance_blocks, CUDA_BLOCK_SIZE>>>(
        device_points.get(), device_distances.get(), N);
    cudaCheck(cudaGetLastError(), "launch distance matrix kernel");

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> cardinalities(N);
    std::vector<Cluster> clusters;
    int remaining = N;

    while (remaining > 0) {
        cudaCheck(cudaMemcpy(device_clustered.get(), clustered.data(), N,
                             cudaMemcpyHostToDevice), "copy cluster mask to GPU");
        buildCandidateClusters<<<N, CUDA_BLOCK_SIZE>>>(
            device_distances.get(), device_clustered.get(),
            device_max_distances.get(), device_members.get(),
            device_cardinalities.get(), threshold, N);
        cudaCheck(cudaGetLastError(), "launch candidate cluster kernel");
        cudaCheck(cudaMemcpy(cardinalities.data(), device_cardinalities.get(),
                             N * sizeof(int), cudaMemcpyDeviceToHost),
                  "copy candidate cardinalities from GPU");

        int max_cardinality = -1;
        int best_seed = -1;
        for (int seed = 0; seed < N; ++seed) {
            if (!clustered[seed] && cardinalities[seed] > max_cardinality) {
                max_cardinality = cardinalities[seed];
                best_seed = seed;
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.resize(max_cardinality);
            cudaCheck(cudaMemcpy(
                          cluster.members.data(),
                          device_members.get() + static_cast<size_t>(best_seed) * N,
                          max_cardinality * sizeof(int), cudaMemcpyDeviceToHost),
                      "copy best cluster from GPU");
            for (int member : cluster.members) {
                if (!clustered[member]) {
                    clustered[member] = 1;
                    --remaining;
                }
            }
            clusters.push_back(std::move(cluster));
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
