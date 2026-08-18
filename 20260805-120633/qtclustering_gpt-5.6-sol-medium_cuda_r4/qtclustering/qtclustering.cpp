// QT Clustering Benchmark - CUDA implementation
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
#include <utility>
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
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) : ptr_(nullptr) {
        if (count != 0) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)),
                      "device allocation");
        }
    }

    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

private:
    T* ptr_;
};

// Distances are stored as distance[member][candidate].  Consequently every
// block reads a contiguous row when a new member is added to its cluster.
__global__ void computeDistanceMatrix(const Point* __restrict__ points,
                                      double* __restrict__ distances,
                                      int point_count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const int member = blockIdx.y * blockDim.y + threadIdx.y;
    if (candidate >= point_count || member >= point_count) return;

    const double dx = points[candidate].x - points[member].x;
    const double dy = points[candidate].y - points[member].y;
    distances[static_cast<size_t>(member) * point_count + candidate] =
        sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ bool betterCandidate(double lhs_distance, int lhs_index,
                                                 double rhs_distance, int rhs_index) {
    return lhs_distance < rhs_distance ||
           (lhs_distance == rhs_distance && lhs_index < rhs_index);
}

// One block constructs one candidate cluster. Candidate points are spread
// across all threads, and a deterministic lexicographic reduction retains the
// sequential implementation's lowest-index tie breaking.
__global__ void generateCandidateClusters(const int* __restrict__ seeds,
                                          int seed_count,
                                          const unsigned char* __restrict__ clustered,
                                          const double* __restrict__ distances,
                                          double* __restrict__ max_distances,
                                          double threshold,
                                          int point_count,
                                          int* __restrict__ cardinalities,
                                          int* __restrict__ output_members) {
    const int slot = blockIdx.x;
    if (slot >= seed_count) return;

    const int tid = threadIdx.x;
    const int seed = seeds[slot];
    double* const candidate_max =
        max_distances + static_cast<size_t>(slot) * point_count;

    for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
        candidate_max[candidate] =
            (clustered[candidate] || candidate == seed)
                ? DBL_MAX
                : distances[static_cast<size_t>(seed) * point_count + candidate];
    }

    __shared__ double reduction_distances[CUDA_BLOCK_SIZE];
    __shared__ int reduction_indices[CUDA_BLOCK_SIZE];
    __shared__ int selected;
    __syncthreads();

    int cardinality = 1;
    if (tid == 0 && output_members) output_members[0] = seed;

    while (cardinality < point_count) {
        double local_distance = DBL_MAX;
        int local_index = INT_MAX;
        for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
            const double value = candidate_max[candidate];
            if (value < threshold &&
                betterCandidate(value, candidate, local_distance, local_index)) {
                local_distance = value;
                local_index = candidate;
            }
        }

        reduction_distances[tid] = local_distance;
        reduction_indices[tid] = local_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset &&
                betterCandidate(reduction_distances[tid + offset],
                                reduction_indices[tid + offset],
                                reduction_distances[tid],
                                reduction_indices[tid])) {
                reduction_distances[tid] = reduction_distances[tid + offset];
                reduction_indices[tid] = reduction_indices[tid + offset];
            }
            __syncthreads();
        }

        if (tid == 0) {
            selected = reduction_indices[0];
            if (selected != INT_MAX) {
                candidate_max[selected] = DBL_MAX;
                if (output_members) output_members[cardinality] = selected;
            }
        }
        __syncthreads();

        if (selected == INT_MAX) break;
        ++cardinality;

        const double* const new_member_distances =
            distances + static_cast<size_t>(selected) * point_count;
        for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
            candidate_max[candidate] =
                fmax(candidate_max[candidate], new_member_distances[candidate]);
        }
        __syncthreads();
    }

    if (tid == 0) cardinalities[slot] = cardinality;
}

} // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    const size_t matrix_elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    if ((N != 0 && matrix_elements / static_cast<size_t>(N) != static_cast<size_t>(N)) ||
        matrix_elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Point count is too large\n");
        std::exit(EXIT_FAILURE);
    }

    DeviceBuffer<Point> device_points(N);
    DeviceBuffer<double> device_distances(matrix_elements);
    DeviceBuffer<double> device_max_distances(matrix_elements);
    DeviceBuffer<unsigned char> device_clustered(N);
    DeviceBuffer<int> device_seeds(N);
    DeviceBuffer<int> device_cardinalities(N);
    DeviceBuffer<int> device_members(N);

    cudaCheck(cudaMemcpy(device_points.get(), points.data(), N * sizeof(Point),
                         cudaMemcpyHostToDevice), "copying points to GPU");
    const dim3 distance_block(16, 16);
    const dim3 distance_grid((N + distance_block.x - 1) / distance_block.x,
                             (N + distance_block.y - 1) / distance_block.y);
    computeDistanceMatrix<<<distance_grid, distance_block>>>(
        device_points.get(), device_distances.get(), N);
    cudaCheck(cudaGetLastError(), "launching distance-matrix kernel");

    std::vector<int> cardinalities(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int active_count = static_cast<int>(unclustered_indices.size());
        cudaCheck(cudaMemcpy(device_seeds.get(), unclustered_indices.data(),
                             active_count * sizeof(int), cudaMemcpyHostToDevice),
                  "copying active seeds to GPU");
        cudaCheck(cudaMemcpy(device_clustered.get(), clustered.data(),
                             N * sizeof(unsigned char), cudaMemcpyHostToDevice),
                  "copying cluster mask to GPU");

        generateCandidateClusters<<<active_count, CUDA_BLOCK_SIZE>>>(
            device_seeds.get(), active_count, device_clustered.get(),
            device_distances.get(), device_max_distances.get(), threshold, N,
            device_cardinalities.get(), nullptr);
        cudaCheck(cudaGetLastError(), "launching candidate-cluster kernel");
        cudaCheck(cudaMemcpy(cardinalities.data(), device_cardinalities.get(),
                             active_count * sizeof(int), cudaMemcpyDeviceToHost),
                  "copying cluster cardinalities from GPU");

        int best_slot = 0;
        for (int slot = 1; slot < active_count; ++slot) {
            if (cardinalities[slot] > cardinalities[best_slot]) best_slot = slot;
        }

        const int best_seed = unclustered_indices[best_slot];
        const int max_cardinality = cardinalities[best_slot];

        // Reconstruct only the winning cluster. This avoids writing an N by N
        // membership matrix during the parallel search.
        cudaCheck(cudaMemcpy(device_seeds.get(), &best_seed, sizeof(int),
                             cudaMemcpyHostToDevice), "copying winning seed to GPU");
        generateCandidateClusters<<<1, CUDA_BLOCK_SIZE>>>(
            device_seeds.get(), 1, device_clustered.get(), device_distances.get(),
            device_max_distances.get(), threshold, N,
            device_cardinalities.get(), device_members.get());
        cudaCheck(cudaGetLastError(), "launching winning-cluster kernel");

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(max_cardinality);
        cudaCheck(cudaMemcpy(cluster.members.data(), device_members.get(),
                             max_cardinality * sizeof(int), cudaMemcpyDeviceToHost),
                  "copying winning cluster from GPU");

        for (int member : cluster.members) clustered[member] = 1;
        clusters.push_back(std::move(cluster));
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int index) { return clustered[index] != 0; }),
            unclustered_indices.end());
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

    // Create the CUDA context outside the measured region. Context startup is
    // process initialization rather than clustering work and otherwise dwarfs
    // small benchmark cases.
    cudaCheck(cudaFree(nullptr), "initializing CUDA context");
    
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
