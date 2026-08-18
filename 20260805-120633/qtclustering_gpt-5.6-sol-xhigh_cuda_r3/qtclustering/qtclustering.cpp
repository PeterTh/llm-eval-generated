// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <numeric>
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

namespace {

void checkCuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n",
                     file, line, expression, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count) {
        if (!tryAllocate(count)) {
            checkCuda(cudaErrorMemoryAllocation, "cudaMalloc", __FILE__, __LINE__);
        }
    }

    bool tryAllocate(std::size_t count) {
        release();
        if (count == 0) {
            return true;
        }
        const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&data_),
                                             count * sizeof(T));
        if (error != cudaSuccess) {
            data_ = nullptr;
            cudaGetLastError();
            return false;
        }
        return true;
    }

    void release() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
        }
    }

    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int CUDA_BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

__device__ __forceinline__ double pointDistance(const Point& first,
                                                const Point& second) {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    return sqrt(dx * dx + dy * dy);
}

// A dense matrix turns the repeatedly requested pair distances into coalesced
// reads.  Distances (rather than squared distances) retain the exact comparison
// semantics of the original implementation.
__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances,
                                    int point_count) {
    const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int member = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (candidate < point_count && member < point_count) {
        distances[static_cast<std::size_t>(member) * point_count + candidate] =
            pointDistance(points[candidate], points[member]);
    }
}

__device__ __forceinline__ bool precedes(double lhs_distance, int lhs_index,
                                         double rhs_distance, int rhs_index) {
    return lhs_distance < rhs_distance ||
           (lhs_distance == rhs_distance && lhs_index < rhs_index);
}

template <bool USE_DISTANCE_MATRIX>
__device__ __forceinline__ double loadDistance(
    int candidate, int member, int point_count,
    const Point* __restrict__ points,
    const double* __restrict__ distances) {
    if constexpr (USE_DISTANCE_MATRIX) {
        return distances[static_cast<std::size_t>(member) * point_count + candidate];
    } else {
        return pointDistance(points[candidate], points[member]);
    }
}

// Each warp grows one candidate cluster.  Seeds are independent within a QT
// round, while lanes cooperate over candidate points.  A running maximum makes
// every growth step O(N), replacing the sequential implementation's repeated
// O(N * cluster_size) diameter scans without changing the greedy order.
template <bool USE_DISTANCE_MATRIX>
__global__ void evaluateSeeds(
    const Point* __restrict__ points,
    const double* __restrict__ distances,
    const unsigned char* __restrict__ clustered,
    const int* __restrict__ active_seeds,
    int seed_offset,
    int seed_count,
    int point_count,
    double threshold,
    double* __restrict__ diameter_rows,
    unsigned long long* best_cluster) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int local_seed = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;

    if (local_seed >= seed_count) {
        return;
    }

    const int seed = active_seeds[seed_offset + local_seed];
    double* const row = diameter_rows +
        static_cast<std::size_t>(local_seed) * point_count;

    for (int candidate = lane; candidate < point_count; candidate += WARP_SIZE) {
        row[candidate] = (clustered[candidate] || candidate == seed) ? DBL_MAX : 0.0;
    }

    int newest_member = seed;
    int cardinality = 1;

    while (cardinality < point_count) {
        double local_distance = DBL_MAX;
        int local_index = INT_MAX;

        for (int candidate = lane; candidate < point_count; candidate += WARP_SIZE) {
            double candidate_diameter = row[candidate];
            if (candidate == newest_member) {
                candidate_diameter = DBL_MAX;
                row[candidate] = DBL_MAX;
            } else if (candidate_diameter != DBL_MAX) {
                const double distance = loadDistance<USE_DISTANCE_MATRIX>(
                    candidate, newest_member, point_count, points, distances);
                if (candidate_diameter < distance) {
                    candidate_diameter = distance;
                    row[candidate] = distance;
                }
                if (candidate_diameter < threshold &&
                    precedes(candidate_diameter, candidate,
                             local_distance, local_index)) {
                    local_distance = candidate_diameter;
                    local_index = candidate;
                }
            }
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            const double other_distance = __shfl_down_sync(
                0xffffffffu, local_distance, offset);
            const int other_index = __shfl_down_sync(
                0xffffffffu, local_index, offset);
            if (precedes(other_distance, other_index,
                         local_distance, local_index)) {
                local_distance = other_distance;
                local_index = other_index;
            }
        }

        const int closest = __shfl_sync(0xffffffffu, local_index, 0);
        if (closest == INT_MAX) {
            break;
        }
        newest_member = closest;
        ++cardinality;
    }

    if (lane == 0) {
        // atomicMax simultaneously maximizes cardinality and implements the
        // original first-seed tie break (active seeds are in point order).
        const unsigned long long packed =
            (static_cast<unsigned long long>(static_cast<unsigned int>(cardinality)) << 32) |
            (0xffffffffull - static_cast<unsigned int>(seed));
        atomicMax(best_cluster, packed);
    }
}

template <bool USE_DISTANCE_MATRIX>
__global__ void materializeCluster(
    const Point* __restrict__ points,
    const double* __restrict__ distances,
    unsigned char* __restrict__ clustered,
    int seed,
    int point_count,
    double threshold,
    double* __restrict__ diameter_row,
    int* __restrict__ members) {
    const int lane = threadIdx.x;
    for (int candidate = lane; candidate < point_count; candidate += WARP_SIZE) {
        diameter_row[candidate] =
            (clustered[candidate] || candidate == seed) ? DBL_MAX : 0.0;
    }

    if (lane == 0) {
        members[0] = seed;
    }

    int newest_member = seed;
    int cardinality = 1;
    while (cardinality < point_count) {
        double local_distance = DBL_MAX;
        int local_index = INT_MAX;

        for (int candidate = lane; candidate < point_count; candidate += WARP_SIZE) {
            double candidate_diameter = diameter_row[candidate];
            if (candidate == newest_member) {
                candidate_diameter = DBL_MAX;
                diameter_row[candidate] = DBL_MAX;
            } else if (candidate_diameter != DBL_MAX) {
                const double distance = loadDistance<USE_DISTANCE_MATRIX>(
                    candidate, newest_member, point_count, points, distances);
                if (candidate_diameter < distance) {
                    candidate_diameter = distance;
                    diameter_row[candidate] = distance;
                }
                if (candidate_diameter < threshold &&
                    precedes(candidate_diameter, candidate,
                             local_distance, local_index)) {
                    local_distance = candidate_diameter;
                    local_index = candidate;
                }
            }
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            const double other_distance = __shfl_down_sync(
                0xffffffffu, local_distance, offset);
            const int other_index = __shfl_down_sync(
                0xffffffffu, local_index, offset);
            if (precedes(other_distance, other_index,
                         local_distance, local_index)) {
                local_distance = other_distance;
                local_index = other_index;
            }
        }

        const int closest = __shfl_sync(0xffffffffu, local_index, 0);
        if (closest == INT_MAX) {
            break;
        }
        newest_member = closest;
        if (lane == 0) {
            members[cardinality] = closest;
        }
        ++cardinality;
    }

    __syncwarp();
    for (int index = lane; index < cardinality; index += WARP_SIZE) {
        clustered[members[index]] = 1;
    }
}

}  // namespace

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

// Main QT clustering algorithm.  The outer rounds retain their required
// sequential dependency, while every candidate cluster in a round is evaluated
// concurrently on the GPU.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const std::size_t point_count = static_cast<std::size_t>(N);

    DeviceBuffer<Point> device_points(point_count);
    DeviceBuffer<unsigned char> device_clustered(point_count);
    DeviceBuffer<int> device_active_seeds(point_count);
    DeviceBuffer<int> device_members(point_count);
    DeviceBuffer<unsigned long long> device_best_cluster(1);

    CUDA_CHECK(cudaMemcpy(device_points.get(), points.data(),
                          point_count * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered.get(), 0,
                          point_count * sizeof(unsigned char)));

    // Use a precomputed matrix whenever it leaves ample room for the per-seed
    // diameter state.  Large inputs automatically switch to direct GPU distance
    // evaluation rather than failing because of the O(N^2) cache.
    DeviceBuffer<double> device_distances;
    bool use_distance_matrix = false;
    std::size_t matrix_elements = 0;
    if (point_count <= SIZE_MAX / point_count) {
        matrix_elements = point_count * point_count;
        if (matrix_elements <= SIZE_MAX / sizeof(double)) {
            std::size_t free_memory = 0;
            std::size_t total_memory = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
            const std::size_t matrix_bytes = matrix_elements * sizeof(double);
            if (matrix_bytes <= free_memory / 3 &&
                device_distances.tryAllocate(matrix_elements)) {
                use_distance_matrix = true;
            }
        }
    }

    if (use_distance_matrix) {
        constexpr int TILE = 16;
        const dim3 block(TILE, TILE);
        const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
        buildDistanceMatrix<<<grid, block>>>(device_points.get(),
                                             device_distances.get(), N);
        CUDA_CHECK(cudaGetLastError());
    }

    std::size_t free_memory = 0;
    std::size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    const std::size_t row_bytes = point_count * sizeof(double);
    std::size_t rows_per_batch = std::min(
        point_count, (free_memory - free_memory / 5) / row_bytes);
    rows_per_batch = std::max<std::size_t>(rows_per_batch, 1);

    DeviceBuffer<double> device_diameter_rows;
    while (!device_diameter_rows.tryAllocate(rows_per_batch * point_count)) {
        if (rows_per_batch == 1) {
            std::fprintf(stderr,
                         "CUDA error: insufficient memory for one clustering seed\n");
            std::exit(EXIT_FAILURE);
        }
        rows_per_batch = std::max<std::size_t>(rows_per_batch / 2, 1);
    }

    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> active_seeds(point_count);
    std::iota(active_seeds.begin(), active_seeds.end(), 0);
    std::vector<Cluster> clusters;

    while (!active_seeds.empty()) {
        CUDA_CHECK(cudaMemcpy(device_active_seeds.get(), active_seeds.data(),
                              active_seeds.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_best_cluster.get(), 0,
                              sizeof(unsigned long long)));

        for (std::size_t offset = 0; offset < active_seeds.size();
             offset += rows_per_batch) {
            const int batch_size = static_cast<int>(std::min(
                rows_per_batch, active_seeds.size() - offset));
            const int block_count =
                (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            if (use_distance_matrix) {
                evaluateSeeds<true><<<block_count, CUDA_BLOCK_SIZE>>>(
                    device_points.get(), device_distances.get(),
                    device_clustered.get(), device_active_seeds.get(),
                    static_cast<int>(offset), batch_size, N, threshold,
                    device_diameter_rows.get(), device_best_cluster.get());
            } else {
                evaluateSeeds<false><<<block_count, CUDA_BLOCK_SIZE>>>(
                    device_points.get(), nullptr,
                    device_clustered.get(), device_active_seeds.get(),
                    static_cast<int>(offset), batch_size, N, threshold,
                    device_diameter_rows.get(), device_best_cluster.get());
            }
            CUDA_CHECK(cudaGetLastError());
        }

        unsigned long long packed_best = 0;
        CUDA_CHECK(cudaMemcpy(&packed_best, device_best_cluster.get(),
                              sizeof(packed_best), cudaMemcpyDeviceToHost));
        const int expected_cardinality = static_cast<int>(packed_best >> 32);
        const int best_seed = static_cast<int>(
            0xffffffffu - static_cast<unsigned int>(packed_best));
        if (expected_cardinality <= 0 || best_seed < 0 || best_seed >= N) {
            std::fprintf(stderr, "CUDA clustering failed to select an active seed\n");
            std::exit(EXIT_FAILURE);
        }

        if (use_distance_matrix) {
            materializeCluster<true><<<1, WARP_SIZE>>>(
                device_points.get(), device_distances.get(),
                device_clustered.get(), best_seed, N, threshold,
                device_diameter_rows.get(), device_members.get());
        } else {
            materializeCluster<false><<<1, WARP_SIZE>>>(
                device_points.get(), nullptr,
                device_clustered.get(), best_seed, N, threshold,
                device_diameter_rows.get(), device_members.get());
        }
        CUDA_CHECK(cudaGetLastError());

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(expected_cardinality);
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), device_members.get(),
                              static_cast<std::size_t>(expected_cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        for (const int member : cluster.members) {
            clustered[member] = 1;
        }
        clusters.push_back(std::move(cluster));
        active_seeds.erase(
            std::remove_if(active_seeds.begin(), active_seeds.end(),
                           [&clustered](int index) { return clustered[index] != 0; }),
            active_seeds.end());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Initialize the CUDA context before the timed region.  Context creation is
    // a process startup cost, not part of the clustering algorithm.
    CUDA_CHECK(cudaFree(nullptr));
    
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
