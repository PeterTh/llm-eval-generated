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
#include <cstdint>
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

[[noreturn]] static void cudaFailure(const cudaError_t error,
                                     const char* expression,
                                     const char* file,
                                     const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(const size_t count) {
        allocate(count);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (pointer_) {
                cudaFree(pointer_);
            }
            pointer_ = std::exchange(other.pointer_, nullptr);
        }
        return *this;
    }

    ~DeviceBuffer() {
        if (pointer_) {
            cudaFree(pointer_);
        }
    }

    void allocate(const size_t count) {
        if (count == 0) {
            return;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                              count * sizeof(T)));
    }

    T* get() { return pointer_; }
    const T* get() const { return pointer_; }

private:
    T* pointer_ = nullptr;
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
        // For N <= 30 the original expression always truncates to zero,
        // preventing data generation from making progress.
        if (N <= 30) {
            group_cnt = 1;
        }
        
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

// Each warp compares (distance, point index) pairs.  The explicit index tie
// break reproduces the left-to-right scan performed by the sequential code.
__device__ __forceinline__ void takeCloser(const double candidate_distance,
                                           const int candidate_index,
                                           double& best_distance,
                                           int& best_index) {
    if (candidate_index >= 0 &&
        (best_index < 0 || candidate_distance < best_distance ||
         (candidate_distance == best_distance && candidate_index < best_index))) {
        best_distance = candidate_distance;
        best_index = candidate_index;
    }
}

// Store a dense row-major distance matrix.  A warp spans one matrix row, so
// both point loads and the much more important later row reads are coalesced.
__global__ void computeDistancesKernel(const Point* __restrict__ points,
                                       double* __restrict__ distances,
                                       const int point_count) {
    const int column = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
    const int row = static_cast<int>(blockIdx.y) * 8 + threadIdx.y;

    if (row < point_count && column < point_count) {
        const double dx = points[row].x - points[column].x;
        const double dy = points[row].y - points[column].y;
        distances[static_cast<size_t>(row) * point_count + column] =
            sqrt(dx * dx + dy * dy);
    }
}

// One block grows one seed's candidate cluster.  Seeds are independent and
// therefore occupy the GPU concurrently.  Within a seed, all candidate points
// are scanned and updated in parallel.  Maintaining each candidate's current
// maximum distance changes repeated O(k*N) rescans into one O(N) update per
// greedy step without changing the chosen point.
__global__ void generateCandidateClustersKernel(
        const int* __restrict__ active_seeds,
        const std::uint8_t* __restrict__ clustered,
        const double* __restrict__ distances,
        const int point_count,
        const double threshold,
        double* __restrict__ global_workspace,
        int* __restrict__ cardinalities,
        int* __restrict__ output_members) {
    extern __shared__ double shared_max_distances[];
    __shared__ double warp_distances[8];
    __shared__ int warp_indices[8];
    __shared__ int selected_point;

    const int local_seed = static_cast<int>(blockIdx.x);
    const int seed = active_seeds[local_seed];
    double* max_distances = global_workspace
        ? global_workspace + static_cast<size_t>(local_seed) * point_count
        : shared_max_distances;

    for (int candidate = threadIdx.x; candidate < point_count;
         candidate += blockDim.x) {
        double candidate_distance = threshold;
        if (!clustered[candidate] && candidate != seed) {
            const double seed_distance =
                distances[static_cast<size_t>(seed) * point_count + candidate];
            if (seed_distance < threshold) {
                candidate_distance = seed_distance;
            }
        }
        max_distances[candidate] = candidate_distance;
    }

    int member_count = 1;
    if (output_members && threadIdx.x == 0) {
        output_members[static_cast<size_t>(local_seed) * point_count] = seed;
    }
    __syncthreads();

    while (member_count < point_count) {
        double best_distance = threshold;
        int best_index = -1;

        for (int candidate = threadIdx.x; candidate < point_count;
             candidate += blockDim.x) {
            const double candidate_distance = max_distances[candidate];
            if (candidate_distance < threshold) {
                takeCloser(candidate_distance, candidate, best_distance, best_index);
            }
        }

        const unsigned int full_warp = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other_distance =
                __shfl_down_sync(full_warp, best_distance, offset);
            const int other_index = __shfl_down_sync(full_warp, best_index, offset);
            takeCloser(other_distance, other_index, best_distance, best_index);
        }

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) {
            warp_distances[warp] = best_distance;
            warp_indices[warp] = best_index;
        }
        __syncthreads();

        if (warp == 0) {
            if (lane < blockDim.x / 32) {
                best_distance = warp_distances[lane];
                best_index = warp_indices[lane];
            } else {
                best_distance = threshold;
                best_index = -1;
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                const double other_distance =
                    __shfl_down_sync(full_warp, best_distance, offset);
                const int other_index =
                    __shfl_down_sync(full_warp, best_index, offset);
                takeCloser(other_distance, other_index,
                           best_distance, best_index);
            }
            if (lane == 0) {
                selected_point = best_index;
            }
        }
        __syncthreads();

        const int selected = selected_point;
        if (selected < 0) {
            break;
        }

        if (output_members && threadIdx.x == 0) {
            output_members[static_cast<size_t>(local_seed) * point_count +
                           member_count] = selected;
        }
        ++member_count;

        const double* __restrict__ selected_distances =
            distances + static_cast<size_t>(selected) * point_count;
        for (int candidate = threadIdx.x; candidate < point_count;
             candidate += blockDim.x) {
            double current_maximum = max_distances[candidate];
            if (candidate == selected) {
                max_distances[candidate] = threshold;
            } else if (current_maximum < threshold) {
                const double new_distance = selected_distances[candidate];
                if (current_maximum < new_distance) {
                    current_maximum = new_distance;
                }
                // Once over the threshold, the candidate can never become
                // eligible again.  The threshold itself is a compact sentinel.
                max_distances[candidate] =
                    current_maximum < threshold ? current_maximum : threshold;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        cardinalities[local_seed] = member_count;
    }
}

__device__ __forceinline__ void takeLargerCluster(const int candidate_count,
                                                  const int candidate_position,
                                                  int& best_count,
                                                  int& best_position) {
    if (candidate_count > best_count ||
        (candidate_count == best_count && candidate_position < best_position)) {
        best_count = candidate_count;
        best_position = candidate_position;
    }
}

// Reducing cardinalities on-device avoids copying an O(N) result vector to the
// CPU after every cluster.  active_seeds remains sorted, so position order also
// implements the original lowest-seed tie break.
__global__ void selectBestClusterKernel(const int* __restrict__ cardinalities,
                                        const int active_count,
                                        int2* __restrict__ result) {
    __shared__ int warp_counts[8];
    __shared__ int warp_positions[8];

    int best_count = -1;
    int best_position = INT_MAX;
    for (int position = threadIdx.x; position < active_count;
         position += blockDim.x) {
        takeLargerCluster(cardinalities[position], position,
                          best_count, best_position);
    }

    const unsigned int full_warp = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other_count = __shfl_down_sync(full_warp, best_count, offset);
        const int other_position =
            __shfl_down_sync(full_warp, best_position, offset);
        takeLargerCluster(other_count, other_position,
                          best_count, best_position);
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_counts[warp] = best_count;
        warp_positions[warp] = best_position;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < blockDim.x / 32) {
            best_count = warp_counts[lane];
            best_position = warp_positions[lane];
        } else {
            best_count = -1;
            best_position = INT_MAX;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const int other_count =
                __shfl_down_sync(full_warp, best_count, offset);
            const int other_position =
                __shfl_down_sync(full_warp, best_position, offset);
            takeLargerCluster(other_count, other_position,
                              best_count, best_position);
        }
        if (lane == 0) {
            *result = make_int2(best_count, best_position);
        }
    }
}

__global__ void markClusteredKernel(const int* __restrict__ members,
                                    const int member_count,
                                    std::uint8_t* __restrict__ clustered) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < member_count;
         i += blockDim.x * gridDim.x) {
        clustered[members[i]] = 1;
    }
}

static cudaDeviceProp configureCandidateKernel(const size_t shared_bytes,
                                                bool& use_shared_workspace) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     generateCandidateClustersKernel));
    const size_t opt_in_limit = properties.sharedMemPerBlockOptin > 0
        ? properties.sharedMemPerBlockOptin
        : properties.sharedMemPerBlock;
    const size_t available_dynamic = opt_in_limit > attributes.sharedSizeBytes
        ? opt_in_limit - attributes.sharedSizeBytes
        : 0;
    use_shared_workspace = shared_bytes <= available_dynamic;

    if (use_shared_workspace) {
        CUDA_CHECK(cudaFuncSetAttribute(
            generateCandidateClustersKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            generateCandidateClustersKernel,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    return properties;
}

static void initializeCudaRuntime() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("CUDA device: %s (%d SMs)\n",
                properties.name, properties.multiProcessorCount);
}

// Main QT clustering algorithm.  All candidate construction, cardinality
// comparison, and membership marking is performed by CUDA kernels.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    constexpr int candidate_threads = 256;
    const int point_count = static_cast<int>(points.size());
    const size_t matrix_elements =
        static_cast<size_t>(point_count) * static_cast<size_t>(point_count);
    const size_t shared_bytes = static_cast<size_t>(point_count) * sizeof(double);

    bool use_shared_workspace = false;
    const cudaDeviceProp properties =
        configureCandidateKernel(shared_bytes, use_shared_workspace);

    DeviceBuffer<Point> device_points(points.size());
    DeviceBuffer<double> device_distances(matrix_elements);
    DeviceBuffer<int> device_active(points.size());
    DeviceBuffer<std::uint8_t> device_clustered(points.size());
    DeviceBuffer<int> device_cardinalities(points.size());
    DeviceBuffer<int> device_best_members(points.size());
    DeviceBuffer<int2> device_best_result(1);
    DeviceBuffer<double> device_workspace;
    int workspace_batch_capacity = point_count;
    if (!use_shared_workspace) {
        int resident_blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks_per_sm, generateCandidateClustersKernel,
            candidate_threads, 0));
        // Several resident waves avoid launch-tail underutilization while bounding
        // workspace at O(N*SM-count), rather than allocating another O(N^2)
        // array for seeds that cannot be resident simultaneously.
        const int efficient_batch = std::max(
            1, 8 * properties.multiProcessorCount * resident_blocks_per_sm);
        workspace_batch_capacity = std::min(point_count, efficient_batch);
        device_workspace.allocate(static_cast<size_t>(workspace_batch_capacity) *
                                  point_count);
    }

    CUDA_CHECK(cudaMemcpy(device_points.get(), points.data(),
                          points.size() * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered.get(), 0,
                          points.size() * sizeof(std::uint8_t)));

    const dim3 distance_threads(32, 8);
    const dim3 distance_blocks((point_count + 31) / 32,
                               (point_count + 7) / 8);
    computeDistancesKernel<<<distance_blocks, distance_threads>>>(
        device_points.get(), device_distances.get(), point_count);
    CUDA_CHECK(cudaGetLastError());

    std::vector<std::uint8_t> clustered(point_count, 0);
    std::vector<int> active_indices(point_count);
    for (int i = 0; i < point_count; ++i) {
        active_indices[i] = i;
    }

    std::vector<Cluster> clusters;
    clusters.reserve(point_count);

    while (!active_indices.empty()) {
        const int active_count = static_cast<int>(active_indices.size());
        CUDA_CHECK(cudaMemcpy(device_active.get(), active_indices.data(),
                              active_indices.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        if (use_shared_workspace) {
            generateCandidateClustersKernel<<<active_count, candidate_threads,
                shared_bytes>>>(
                    device_active.get(), device_clustered.get(),
                    device_distances.get(), point_count, threshold, nullptr,
                    device_cardinalities.get(), nullptr);
            CUDA_CHECK(cudaGetLastError());
        } else {
            for (int offset = 0; offset < active_count;
                 offset += workspace_batch_capacity) {
                const int batch_size = std::min(
                    workspace_batch_capacity, active_count - offset);
                generateCandidateClustersKernel<<<batch_size,
                    candidate_threads>>>(
                        device_active.get() + offset, device_clustered.get(),
                        device_distances.get(), point_count, threshold,
                        device_workspace.get(),
                        device_cardinalities.get() + offset, nullptr);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        selectBestClusterKernel<<<1, candidate_threads>>>(
            device_cardinalities.get(), active_count,
            device_best_result.get());
        CUDA_CHECK(cudaGetLastError());

        int2 best_result{};
        CUDA_CHECK(cudaMemcpy(&best_result, device_best_result.get(),
                              sizeof(best_result), cudaMemcpyDeviceToHost));
        const int best_cardinality = best_result.x;
        const int best_position = best_result.y;
        if (best_position < 0 || best_position >= active_count ||
            best_cardinality <= 0) {
            break;
        }

        // Recompute only the winning seed to materialize its ordered member
        // list.  This avoids O(N^2) member output from all losing seeds.
        generateCandidateClustersKernel<<<1, candidate_threads,
            use_shared_workspace ? shared_bytes : 0>>>(
                device_active.get() + best_position, device_clustered.get(),
                device_distances.get(), point_count, threshold,
                device_workspace.get(), device_cardinalities.get(),
                device_best_members.get());
        CUDA_CHECK(cudaGetLastError());

        Cluster best_cluster;
        best_cluster.seed_point = active_indices[best_position];
        best_cluster.members.resize(best_cardinality);
        CUDA_CHECK(cudaMemcpy(best_cluster.members.data(),
                              device_best_members.get(),
                              static_cast<size_t>(best_cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        const int mark_blocks = std::min(
            (best_cardinality + candidate_threads - 1) / candidate_threads, 256);
        markClusteredKernel<<<mark_blocks, candidate_threads>>>(
            device_best_members.get(), best_cardinality,
            device_clustered.get());
        CUDA_CHECK(cudaGetLastError());

        for (const int member : best_cluster.members) {
            clustered[member] = 1;
        }
        clusters.push_back(std::move(best_cluster));

        active_indices.erase(
            std::remove_if(active_indices.begin(), active_indices.end(),
                           [&clustered](const int index) {
                               return clustered[index] != 0;
                           }),
            active_indices.end());
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

    // Runtime/context creation is a one-time process cost rather than part of
    // clustering, so initialize it before starting the benchmark timer.
    initializeCudaRuntime();
    
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
