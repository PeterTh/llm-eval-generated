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
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct alignas(16) Point {
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

constexpr int CUDA_REDUCTION_BLOCK_SIZE = 256;

int candidateBlockSize(const int active_count) {
    // Smaller blocks schedule more independent seeds concurrently for the
    // common benchmark sizes; larger active sets benefit from more threads
    // cooperating on each candidate scan.
    return active_count <= 1024 ? 128 : 256;
}

void checkCuda(const cudaError_t result, const char* expression,
               const char* file, const int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                     file, line, expression, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(const size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(const size_t count) {
        if (count == 0) return;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                              count * sizeof(T)));
    }

    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

struct DistanceIndex {
    double value;
    int index;
};

struct BestResult {
    int cardinality;
    int seed_slot;
};

__device__ __forceinline__ DistanceIndex chooseCloser(
    const DistanceIndex a, const DistanceIndex b) {
    return (b.value < a.value ||
            (b.value == a.value && b.index < a.index)) ? b : a;
}

__device__ __forceinline__ DistanceIndex warpMinimum(DistanceIndex value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        DistanceIndex other;
        other.value = __shfl_down_sync(0xffffffffu, value.value, offset);
        other.index = __shfl_down_sync(0xffffffffu, value.index, offset);
        value = chooseCloser(value, other);
    }
    return value;
}

// All block threads call this routine.  The reduction compares the original
// point slots as a secondary key, reproducing the sequential scan's tie rule.
__device__ __forceinline__ int findClosestSlot(
    const double* max_distances, const int active_count, const double threshold,
    double* warp_values, int* warp_indices, int* selected_slot) {
    DistanceIndex local{DBL_MAX, INT_MAX};
    for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
        const double value = max_distances[slot];
        if (value < threshold) {
            local = chooseCloser(local, DistanceIndex{value, slot});
        }
    }

    local = warpMinimum(local);
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_values[warp] = local.value;
        warp_indices[warp] = local.index;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        DistanceIndex block_value = lane < warp_count
            ? DistanceIndex{warp_values[lane], warp_indices[lane]}
            : DistanceIndex{DBL_MAX, INT_MAX};
        block_value = warpMinimum(block_value);
        if (lane == 0) *selected_slot = block_value.index;
    }
    __syncthreads();
    return *selected_slot;
}

__global__ void buildDistanceMatrixKernel(const Point* __restrict__ points,
                                          double* __restrict__ distances,
                                          const int point_count) {
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= point_count || column >= point_count) return;

    const Point a = points[row];
    const Point b = points[column];
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    distances[static_cast<size_t>(row) * point_count + column] =
        sqrt(dx * dx + dy * dy);
}

// One block grows one seed cluster.  Every candidate's running maximum
// distance is retained in shared memory, eliminating the sequential code's
// repeated scan over all existing members.
__global__ void candidateCardinalitySharedKernel(
    const double* __restrict__ distances, const int* __restrict__ active,
    int* __restrict__ cardinalities, int* __restrict__ sequences,
    const int point_count, const int active_count, const double threshold) {
    const int seed_slot = static_cast<int>(blockIdx.x);
    if (seed_slot >= active_count) return;

    extern __shared__ double max_distances[];
    __shared__ double warp_values[32];
    __shared__ int warp_indices[32];
    __shared__ int selected_slot;

    const int seed = active[seed_slot];
    int* const sequence = sequences == nullptr
        ? nullptr
        : sequences + static_cast<size_t>(seed_slot) * point_count;
    for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
        max_distances[slot] = slot == seed_slot
            ? DBL_MAX
            : distances[static_cast<size_t>(seed) * point_count + active[slot]];
    }
    if (threadIdx.x == 0 && sequence != nullptr) sequence[0] = seed;
    __syncthreads();

    int cardinality = 1;
    while (cardinality < active_count) {
        const int selected = findClosestSlot(max_distances, active_count,
                                             threshold, warp_values,
                                             warp_indices, &selected_slot);
        if (selected == INT_MAX) break;

        const int selected_point = active[selected];
        if (threadIdx.x == 0 && sequence != nullptr) {
            sequence[cardinality] = selected_point;
        }
        ++cardinality;
        for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
            if (slot == selected) {
                max_distances[slot] = DBL_MAX;
            } else {
                const double next = distances[
                    static_cast<size_t>(selected_point) * point_count + active[slot]];
                const double current = max_distances[slot];
                max_distances[slot] = current < next ? next : current;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) cardinalities[seed_slot] = cardinality;
}

// Large inputs use global state when one seed's state no longer fits in a
// block's shared memory.  Seeds remain fully parallel in this path.
__global__ void candidateCardinalityGlobalKernel(
    const double* __restrict__ distances, const int* __restrict__ active,
    double* __restrict__ state, int* __restrict__ cardinalities,
    const int point_count, const int active_count, const double threshold) {
    const int seed_slot = static_cast<int>(blockIdx.x);
    if (seed_slot >= active_count) return;

    __shared__ double warp_values[32];
    __shared__ int warp_indices[32];
    __shared__ int selected_slot;
    double* const max_distances =
        state + static_cast<size_t>(seed_slot) * active_count;
    const int seed = active[seed_slot];

    for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
        max_distances[slot] = slot == seed_slot
            ? DBL_MAX
            : distances[static_cast<size_t>(seed) * point_count + active[slot]];
    }
    __syncthreads();

    int cardinality = 1;
    while (cardinality < active_count) {
        const int selected = findClosestSlot(max_distances, active_count,
                                             threshold, warp_values,
                                             warp_indices, &selected_slot);
        if (selected == INT_MAX) break;

        const int selected_point = active[selected];
        ++cardinality;
        for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
            if (slot == selected) {
                max_distances[slot] = DBL_MAX;
            } else {
                const double next = distances[
                    static_cast<size_t>(selected_point) * point_count + active[slot]];
                const double current = max_distances[slot];
                max_distances[slot] = current < next ? next : current;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) cardinalities[seed_slot] = cardinality;
}

__device__ __forceinline__ BestResult chooseLarger(const BestResult a,
                                                   const BestResult b) {
    return (b.cardinality > a.cardinality ||
            (b.cardinality == a.cardinality && b.seed_slot < a.seed_slot))
        ? b : a;
}

__device__ __forceinline__ BestResult warpMaximum(BestResult value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        BestResult other;
        other.cardinality =
            __shfl_down_sync(0xffffffffu, value.cardinality, offset);
        other.seed_slot =
            __shfl_down_sync(0xffffffffu, value.seed_slot, offset);
        value = chooseLarger(value, other);
    }
    return value;
}

__global__ void bestSeedKernel(const int* __restrict__ cardinalities,
                               const int active_count,
                               BestResult* __restrict__ result) {
    BestResult local{-1, INT_MAX};
    for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
        local = chooseLarger(local, BestResult{cardinalities[slot], slot});
    }
    local = warpMaximum(local);

    __shared__ int warp_cardinalities[32];
    __shared__ int warp_slots[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_cardinalities[warp] = local.cardinality;
        warp_slots[warp] = local.seed_slot;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        BestResult block_value = lane < warp_count
            ? BestResult{warp_cardinalities[lane], warp_slots[lane]}
            : BestResult{-1, INT_MAX};
        block_value = warpMaximum(block_value);
        if (lane == 0) *result = block_value;
    }
}

__global__ void reconstructGlobalKernel(
    const double* __restrict__ distances, const int* __restrict__ active,
    double* __restrict__ max_distances, int* __restrict__ members,
    const int point_count, const int active_count, const int seed_slot,
    const double threshold) {
    __shared__ double warp_values[32];
    __shared__ int warp_indices[32];
    __shared__ int selected_slot;

    const int seed = active[seed_slot];
    for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
        max_distances[slot] = slot == seed_slot
            ? DBL_MAX
            : distances[static_cast<size_t>(seed) * point_count + active[slot]];
    }
    if (threadIdx.x == 0) members[0] = seed;
    __syncthreads();

    int cardinality = 1;
    while (cardinality < active_count) {
        const int selected = findClosestSlot(max_distances, active_count,
                                             threshold, warp_values,
                                             warp_indices, &selected_slot);
        if (selected == INT_MAX) break;
        const int selected_point = active[selected];
        if (threadIdx.x == 0) members[cardinality] = selected_point;
        ++cardinality;

        for (int slot = threadIdx.x; slot < active_count; slot += blockDim.x) {
            if (slot == selected) {
                max_distances[slot] = DBL_MAX;
            } else {
                const double next = distances[
                    static_cast<size_t>(selected_point) * point_count + active[slot]];
                const double current = max_distances[slot];
                max_distances[slot] = current < next ? next : current;
            }
        }
        __syncthreads();
    }
}

} // namespace

// Main QT clustering algorithm.  Candidate seeds are always evaluated on the
// GPU; there is deliberately no sequential clustering fallback.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int point_count = static_cast<int>(points.size());
    const size_t count = static_cast<size_t>(point_count);
    if (count > std::numeric_limits<size_t>::max() / count ||
        count * count > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Point count is too large for the distance matrix\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t matrix_elements = count * count;

    DeviceBuffer<Point> device_points(count);
    DeviceBuffer<double> device_distances(matrix_elements);
    DeviceBuffer<int> device_active(count);
    DeviceBuffer<int> device_cardinalities(count);
    DeviceBuffer<int> device_members(count);
    DeviceBuffer<BestResult> device_best(1);

    CUDA_CHECK(cudaMemcpy(device_points.get(), points.data(),
                          count * sizeof(Point), cudaMemcpyHostToDevice));
    const dim3 distance_block(32, 8);
    const dim3 distance_grid(
        (point_count + distance_block.x - 1) / distance_block.x,
        (point_count + distance_block.y - 1) / distance_block.y);
    buildDistanceMatrixKernel<<<distance_grid, distance_block>>>(
        device_points.get(), device_distances.get(), point_count);
    CUDA_CHECK(cudaGetLastError());

    cudaFuncAttributes shared_kernel_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&shared_kernel_attributes,
                                     candidateCardinalitySharedKernel));
    int device = 0;
    int opt_in_shared_bytes = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&opt_in_shared_bytes,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device));
    // Leave a small alignment reserve because the reported opt-in limit also
    // includes the kernel's statically allocated reduction scratch space.
    const int max_dynamic_shared_bytes = std::max(
        0, opt_in_shared_bytes -
            static_cast<int>(shared_kernel_attributes.sharedSizeBytes) - 256);
    CUDA_CHECK(cudaFuncSetAttribute(
        candidateCardinalitySharedKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_dynamic_shared_bytes));
    const int shared_state_capacity = std::max(
        1, max_dynamic_shared_bytes / static_cast<int>(sizeof(double)));

    DeviceBuffer<double> device_global_state;
    DeviceBuffer<int> device_sequences;
    if (point_count > shared_state_capacity) {
        device_global_state.allocate(matrix_elements);
    } else {
        // Recording all seed sequences avoids serially rebuilding the winning
        // cluster.  This is bounded to the shared-memory path, where the extra
        // matrix is modest and the saved kernel launches matter most.
        device_sequences.allocate(matrix_elements);
    }

    std::vector<int> active(count);
    std::iota(active.begin(), active.end(), 0);
    std::vector<unsigned char> clustered(count, 0);
    std::vector<Cluster> clusters;
    std::vector<int> selected_members;

    while (!active.empty()) {
        const int active_count = static_cast<int>(active.size());
        const int candidate_block_size = candidateBlockSize(active_count);
        const size_t active_bytes = active.size() * sizeof(int);
        CUDA_CHECK(cudaMemcpy(device_active.get(), active.data(), active_bytes,
                              cudaMemcpyHostToDevice));

        if (active_count <= shared_state_capacity &&
            device_sequences.get() != nullptr) {
            const size_t shared_bytes = active.size() * sizeof(double);
            candidateCardinalitySharedKernel<<<active_count, candidate_block_size,
                                               shared_bytes>>>(
                device_distances.get(), device_active.get(),
                device_cardinalities.get(), device_sequences.get(),
                point_count, active_count, threshold);
        } else {
            candidateCardinalityGlobalKernel<<<active_count,
                                               candidate_block_size>>>(
                device_distances.get(), device_active.get(),
                device_global_state.get(), device_cardinalities.get(),
                point_count, active_count, threshold);
        }
        CUDA_CHECK(cudaGetLastError());

        bestSeedKernel<<<1, CUDA_REDUCTION_BLOCK_SIZE>>>(
            device_cardinalities.get(), active_count, device_best.get());
        CUDA_CHECK(cudaGetLastError());
        BestResult best{};
        CUDA_CHECK(cudaMemcpy(&best, device_best.get(), sizeof(best),
                              cudaMemcpyDeviceToHost));

        // If every candidate is a singleton, all future iterations are fixed:
        // the sequential algorithm emits the remaining points in index order.
        if (best.cardinality == 1) {
            clusters.reserve(clusters.size() + active.size());
            for (const int point : active) {
                clusters.push_back(Cluster{{point}, point});
            }
            break;
        }
        if (best.cardinality <= 0 || best.seed_slot < 0 ||
            best.seed_slot >= active_count) {
            std::fprintf(stderr, "CUDA clustering produced an invalid seed\n");
            std::exit(EXIT_FAILURE);
        }

        selected_members.resize(best.cardinality);
        if (active_count <= shared_state_capacity &&
            device_sequences.get() != nullptr) {
            const int* const winning_sequence = device_sequences.get() +
                static_cast<size_t>(best.seed_slot) * point_count;
            CUDA_CHECK(cudaMemcpy(selected_members.data(), winning_sequence,
                                  selected_members.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        } else {
            reconstructGlobalKernel<<<1, candidate_block_size>>>(
                device_distances.get(), device_active.get(),
                device_global_state.get(), device_members.get(), point_count,
                active_count, best.seed_slot, threshold);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(selected_members.data(), device_members.get(),
                                  selected_members.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }

        Cluster cluster;
        cluster.seed_point = active[best.seed_slot];
        cluster.members = selected_members;
        clusters.push_back(std::move(cluster));

        for (const int member : selected_members) clustered[member] = 1;
        active.erase(std::remove_if(active.begin(), active.end(),
                                    [&clustered](const int point) {
                                        return clustered[point] != 0;
                                    }),
                     active.end());
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

    // CUDA context creation is a one-time process startup cost rather than
    // clustering work, so initialize it before the measured region.
    CUDA_CHECK(cudaFree(nullptr));
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> cluster_time =
        cluster_end - cluster_start;
    
    printf("Clustering time: %.3f ms\n", cluster_time.count());
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
