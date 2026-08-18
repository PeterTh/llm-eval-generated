// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>
#include <string>
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

namespace {

constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int CUDA_WARP_SIZE = 32;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t count) {
        if (data_) cudaCheck(cudaFree(data_), "cudaFree");
        data_ = nullptr;
        if (count != 0) cudaCheck(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }
    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

// Each block independently grows one candidate cluster.  The row stores the
// running maximum distance from every point to the cluster, so adding a member
// only requires one parallel O(N) update instead of rescanning all members.
__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    int point_count,
                                    double* __restrict__ distances) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const int member = blockIdx.y * blockDim.y + threadIdx.y;
    if (candidate < point_count && member < point_count) {
        // Keep the operand ordering used by the original candidate/member loop.
        const double dx = points[candidate].x - points[member].x;
        const double dy = points[candidate].y - points[member].y;
        distances[static_cast<size_t>(member) * point_count + candidate] =
            sqrt(dx * dx + dy * dy);
    }
}

template <bool USE_DISTANCE_MATRIX>
__global__ void generateCandidateClusters(const Point* __restrict__ points,
                                           const double* __restrict__ distances,
                                           const unsigned char* __restrict__ clustered,
                                           const int* __restrict__ seeds,
                                           int seed_count,
                                           int point_count,
                                           double threshold,
                                           double* __restrict__ max_distances,
                                           int* __restrict__ cardinalities,
                                           int* __restrict__ recorded_members) {
    const int row_id = blockIdx.x;
    if (row_id >= seed_count) return;

    __shared__ double warp_distance[CUDA_BLOCK_SIZE / CUDA_WARP_SIZE];
    __shared__ int warp_index[CUDA_BLOCK_SIZE / CUDA_WARP_SIZE];
    __shared__ int selected;
    __shared__ int member_count;

    const int tid = threadIdx.x;
    const int seed = seeds[row_id];
    double* const row = max_distances + static_cast<size_t>(row_id) * point_count;
    const Point seed_point = points[seed];

    for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
        if (clustered[candidate] || candidate == seed) {
            row[candidate] = CUDART_INF;
        } else if constexpr (USE_DISTANCE_MATRIX) {
            row[candidate] = distances[static_cast<size_t>(seed) * point_count + candidate];
        } else {
            const double dx = points[candidate].x - seed_point.x;
            const double dy = points[candidate].y - seed_point.y;
            row[candidate] = sqrt(dx * dx + dy * dy);
        }
    }
    if (tid == 0) {
        member_count = 1;
        if (recorded_members) recorded_members[0] = seed;
    }
    __syncthreads();

    for (int iteration = 1; iteration < point_count; ++iteration) {
        double local_distance = CUDART_INF;
        int local_index = INT_MAX;
        for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
            const double candidate_distance = row[candidate];
            if (candidate_distance < threshold &&
                (candidate_distance < local_distance ||
                 (candidate_distance == local_distance && candidate < local_index))) {
                local_distance = candidate_distance;
                local_index = candidate;
            }
        }

        // A deterministic (distance, point-index) reduction exactly preserves
        // the sequential implementation's lowest-index tie breaking. Warp
        // shuffles avoid eight block-wide barriers per growth iteration.
        for (int offset = CUDA_WARP_SIZE / 2; offset != 0; offset /= 2) {
            const double other_distance = __shfl_down_sync(0xffffffff, local_distance, offset);
            const int other_index = __shfl_down_sync(0xffffffff, local_index, offset);
            if (other_distance < local_distance ||
                (other_distance == local_distance && other_index < local_index)) {
                local_distance = other_distance;
                local_index = other_index;
            }
        }
        const int lane = tid & (CUDA_WARP_SIZE - 1);
        const int warp = tid / CUDA_WARP_SIZE;
        if (lane == 0) {
            warp_distance[warp] = local_distance;
            warp_index[warp] = local_index;
        }
        __syncthreads();

        if (warp == 0) {
            local_distance = lane < CUDA_BLOCK_SIZE / CUDA_WARP_SIZE
                                 ? warp_distance[lane] : CUDART_INF;
            local_index = lane < CUDA_BLOCK_SIZE / CUDA_WARP_SIZE
                              ? warp_index[lane] : INT_MAX;
            for (int offset = CUDA_WARP_SIZE / 2; offset != 0; offset /= 2) {
                const double other_distance =
                    __shfl_down_sync(0xffffffff, local_distance, offset);
                const int other_index = __shfl_down_sync(0xffffffff, local_index, offset);
                if (other_distance < local_distance ||
                    (other_distance == local_distance && other_index < local_index)) {
                    local_distance = other_distance;
                    local_index = other_index;
                }
            }
            if (lane == 0) {
                selected = local_index == INT_MAX ? -1 : local_index;
                if (selected >= 0) {
                    if (recorded_members) recorded_members[member_count] = selected;
                    ++member_count;
                    row[selected] = CUDART_INF;
                }
            }
        }
        __syncthreads();
        if (selected < 0) break;

        const Point new_member = points[selected];
        for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
            const double old_maximum = row[candidate];
            if (old_maximum != CUDART_INF) {
                double new_distance;
                if constexpr (USE_DISTANCE_MATRIX) {
                    new_distance = distances[static_cast<size_t>(selected) * point_count +
                                             candidate];
                } else {
                    const double dx = points[candidate].x - new_member.x;
                    const double dy = points[candidate].y - new_member.y;
                    new_distance = sqrt(dx * dx + dy * dy);
                }
                if (new_distance > old_maximum) row[candidate] = new_distance;
            }
        }
        __syncthreads();
    }

    if (tid == 0) cardinalities[row_id] = member_count;
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

} // namespace

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
    
    DeviceBuffer<Point> device_points(N);
    DeviceBuffer<unsigned char> device_clustered(N);
    DeviceBuffer<int> device_seeds(N);
    DeviceBuffer<int> device_cardinalities(N);
    DeviceBuffer<int> device_members(N);
    cudaCheck(cudaMemcpy(device_points.get(), points.data(), N * sizeof(Point),
                         cudaMemcpyHostToDevice), "copy points to GPU");
    cudaCheck(cudaMemset(device_clustered.get(), 0, N * sizeof(unsigned char)),
              "initialize clustered flags");

    // Cache all pairwise distances when memory permits. They are reused by
    // every seed in every QT round and remove the dominant repeated sqrt work.
    DeviceBuffer<double> device_distances;
    bool use_distance_matrix = false;
    // Use most of the currently available memory to expose as many independent
    // seeds as possible, while retaining headroom for the CUDA runtime/driver.
    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaCheck(cudaMemGetInfo(&free_memory, &total_memory), "cudaMemGetInfo");
    const size_t matrix_elements = static_cast<size_t>(N) * N;
    const size_t matrix_bytes = matrix_elements * sizeof(double);
    if (matrix_bytes <= free_memory * 2 / 5) {
        try {
            device_distances.allocate(matrix_elements);
        } catch (const std::runtime_error&) {
            (void)cudaGetLastError();
            // The incremental CUDA path below needs only a batch of state rows.
        }
        if (device_distances.get()) {
            const dim3 threads(16, 16);
            const dim3 blocks((N + threads.x - 1) / threads.x,
                              (N + threads.y - 1) / threads.y);
            buildDistanceMatrix<<<blocks, threads>>>(device_points.get(), N,
                                                     device_distances.get());
            cudaCheck(cudaGetLastError(), "launch distance matrix kernel");
            cudaCheck(cudaDeviceSynchronize(), "build distance matrix");
            use_distance_matrix = true;
            cudaCheck(cudaMemGetInfo(&free_memory, &total_memory), "cudaMemGetInfo");
        }
    }
    const size_t bytes_per_row = static_cast<size_t>(N) * sizeof(double);
    size_t batch_size = std::min<size_t>(N, (free_memory * 3 / 5) / bytes_per_row);
    if (batch_size == 0) batch_size = 1;

    DeviceBuffer<double> device_max_distances;
    while (true) {
        // DeviceBuffer owns allocation, and retrying at smaller batches makes
        // the implementation robust to memory fragmentation.
        try {
            device_max_distances.allocate(batch_size * static_cast<size_t>(N));
            break;
        } catch (const std::runtime_error&) {
            if (batch_size == 1) throw;
            (void)cudaGetLastError();
            batch_size = std::max<size_t>(1, batch_size / 2);
        }
    }

    std::vector<int> cardinalities(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;

        // All currently unclustered seeds are independent. Process them in
        // memory-sized batches; ordering on the host preserves first-seed ties.
        for (size_t offset = 0; offset < unclustered_indices.size(); offset += batch_size) {
            const int count = static_cast<int>(std::min<size_t>(
                batch_size, unclustered_indices.size() - offset));
            cudaCheck(cudaMemcpy(device_seeds.get(), unclustered_indices.data() + offset,
                                 count * sizeof(int), cudaMemcpyHostToDevice),
                      "copy seed batch to GPU");
            if (use_distance_matrix) {
                generateCandidateClusters<true><<<count, CUDA_BLOCK_SIZE>>>(
                    device_points.get(), device_distances.get(), device_clustered.get(),
                    device_seeds.get(), count, N, threshold, device_max_distances.get(),
                    device_cardinalities.get(), nullptr);
            } else {
                generateCandidateClusters<false><<<count, CUDA_BLOCK_SIZE>>>(
                    device_points.get(), nullptr, device_clustered.get(), device_seeds.get(),
                    count, N, threshold, device_max_distances.get(),
                    device_cardinalities.get(), nullptr);
            }
            cudaCheck(cudaGetLastError(), "launch candidate cluster kernel");
            cudaCheck(cudaMemcpy(cardinalities.data() + offset, device_cardinalities.get(),
                                 count * sizeof(int), cudaMemcpyDeviceToHost),
                      "copy cardinalities from GPU");
        }

        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_seed = unclustered_indices[i];
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            cudaCheck(cudaMemcpy(device_seeds.get(), &best_seed, sizeof(int),
                                 cudaMemcpyHostToDevice), "copy winning seed to GPU");
            if (use_distance_matrix) {
                generateCandidateClusters<true><<<1, CUDA_BLOCK_SIZE>>>(
                    device_points.get(), device_distances.get(), device_clustered.get(),
                    device_seeds.get(), 1, N, threshold, device_max_distances.get(),
                    device_cardinalities.get(), device_members.get());
            } else {
                generateCandidateClusters<false><<<1, CUDA_BLOCK_SIZE>>>(
                    device_points.get(), nullptr, device_clustered.get(), device_seeds.get(),
                    1, N, threshold, device_max_distances.get(),
                    device_cardinalities.get(), device_members.get());
            }
            cudaCheck(cudaGetLastError(), "launch winning cluster replay kernel");

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.resize(max_cardinality);
            cudaCheck(cudaMemcpy(cluster.members.data(), device_members.get(),
                                 max_cardinality * sizeof(int), cudaMemcpyDeviceToHost),
                      "copy winning cluster from GPU");
            markClustered<<<(max_cardinality + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
                              CUDA_BLOCK_SIZE>>>(device_clustered.get(), device_members.get(),
                                                max_cardinality);
            cudaCheck(cudaGetLastError(), "launch clustered flag kernel");

            for (int member : cluster.members) clustered[member] = 1;
            clusters.push_back(std::move(cluster));

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    cudaCheck(cudaDeviceSynchronize(), "complete CUDA clustering");
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

    // Initialize the CUDA context outside the measured clustering region.
    cudaCheck(cudaFree(nullptr), "initialize CUDA runtime");
    
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
