// QT Clustering Benchmark - CUDA implementation
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

// CUDA calls are deliberately fatal: this benchmark has no sequential fallback.
// A CUDA failure therefore cannot silently change the parallelization approach.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_status_ = (call);                                \
        if (cuda_status_ != cudaSuccess) {                                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(cuda_status_));            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
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
constexpr double INVALID_DISTANCE = DBL_MAX;

// Precomputing exact Euclidean distances turns the repeatedly evaluated square
// roots in QT's inner loop into coalesced reads. The full matrix is intentional:
// selected-member rows are then contiguous for every candidate-cluster block.
__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances,
                                    const int point_count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const int member = blockIdx.y * blockDim.y + threadIdx.y;
    if (candidate >= point_count || member >= point_count) {
        return;
    }

    const double dx = points[candidate].x - points[member].x;
    const double dy = points[candidate].y - points[member].y;
    distances[static_cast<size_t>(member) * point_count + candidate] =
        sqrt(dx * dx + dy * dy);
}

struct CandidateChoice {
    double diameter;
    int index;
};

__device__ __forceinline__ void takeBetter(const double other_diameter,
                                           const int other_index,
                                           CandidateChoice& choice) {
    if (other_diameter < choice.diameter ||
        (other_diameter == choice.diameter && other_index < choice.index)) {
        choice.diameter = other_diameter;
        choice.index = other_index;
    }
}

// Deterministic (diameter, point-index) reduction. Using the index as the
// secondary key exactly preserves the sequential scan's first-point tie break.
template <int BLOCK_SIZE>
__device__ __forceinline__ CandidateChoice blockArgMin(
    CandidateChoice choice, double* shared_diameters, int* shared_indices) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_COUNT = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        const double other_diameter =
            __shfl_down_sync(0xffffffffu, choice.diameter, offset);
        const int other_index =
            __shfl_down_sync(0xffffffffu, choice.index, offset);
        takeBetter(other_diameter, other_index, choice);
    }

    if (lane == 0) {
        shared_diameters[warp] = choice.diameter;
        shared_indices[warp] = choice.index;
    }
    __syncthreads();

    if (warp == 0) {
        CandidateChoice warp_choice{INVALID_DISTANCE, INT_MAX};
        if (lane < WARP_COUNT) {
            warp_choice.diameter = shared_diameters[lane];
            warp_choice.index = shared_indices[lane];
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            const double other_diameter =
                __shfl_down_sync(0xffffffffu, warp_choice.diameter, offset);
            const int other_index =
                __shfl_down_sync(0xffffffffu, warp_choice.index, offset);
            takeBetter(other_diameter, other_index, warp_choice);
        }
        if (lane == 0) {
            shared_diameters[0] = warp_choice.diameter;
            shared_indices[0] = warp_choice.index;
        }
    }
    __syncthreads();

    return CandidateChoice{shared_diameters[0], shared_indices[0]};
}

// One block independently grows one seed cluster. ITEMS candidates per thread
// retain their running maximum diameter in registers across growth iterations.
// This removes both the original repeated member scan and all working-matrix
// traffic for the normal benchmark sizes.
template <int ITEMS>
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void growCandidateClusters(const double* __restrict__ distances,
                           unsigned char* __restrict__ clustered,
                           const int* __restrict__ active_seeds,
                           const int point_count,
                           const double threshold,
                           unsigned long long* __restrict__ best_cluster,
                           int* __restrict__ candidate_members) {
    const int active_index = blockIdx.x;
    const int seed = active_seeds[active_index];
    if (clustered[seed]) {
        return;
    }
    const int tid = threadIdx.x;
    int* const seed_members =
        candidate_members + static_cast<size_t>(seed) * point_count;

    __shared__ double shared_diameters[32];
    __shared__ int shared_indices[32];

    double max_diameters[ITEMS];
    CandidateChoice local_choice{INVALID_DISTANCE, INT_MAX};

#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int candidate = tid + item * CUDA_BLOCK_SIZE;
        double candidate_diameter = INVALID_DISTANCE;
        if (candidate < point_count && candidate != seed && !clustered[candidate]) {
            const double seed_distance =
                distances[static_cast<size_t>(seed) * point_count + candidate];
            if (seed_distance < threshold) {
                candidate_diameter = seed_distance;
                takeBetter(seed_distance, candidate, local_choice);
            }
        }
        max_diameters[item] = candidate_diameter;
    }

    if (tid == 0) {
        seed_members[0] = seed;
    }

    CandidateChoice selected = blockArgMin<CUDA_BLOCK_SIZE>(
        local_choice, shared_diameters, shared_indices);
    int cardinality = 1;

    while (selected.index != INT_MAX) {
        const int newest_member = selected.index;
        if (tid == 0) {
            seed_members[cardinality] = newest_member;
        }
        ++cardinality;

        local_choice = CandidateChoice{INVALID_DISTANCE, INT_MAX};
#pragma unroll
        for (int item = 0; item < ITEMS; ++item) {
            const int candidate = tid + item * CUDA_BLOCK_SIZE;
            double candidate_diameter = max_diameters[item];
            if (candidate == newest_member) {
                candidate_diameter = INVALID_DISTANCE;
            } else if (candidate_diameter < threshold) {
                const double new_distance =
                    distances[static_cast<size_t>(newest_member) * point_count + candidate];
                candidate_diameter = max(candidate_diameter, new_distance);
                // Diameter only increases, so an invalid candidate can be
                // permanently discarded from all subsequent iterations.
                if (candidate_diameter >= threshold) {
                    candidate_diameter = INVALID_DISTANCE;
                } else {
                    takeBetter(candidate_diameter, candidate, local_choice);
                }
            }
            max_diameters[item] = candidate_diameter;
        }

        selected = blockArgMin<CUDA_BLOCK_SIZE>(
            local_choice, shared_diameters, shared_indices);
    }

    if (tid == 0) {
        // Higher cardinality wins; bitwise-inverted seed makes the lower seed
        // win ties under a single deterministic atomicMax.
        const unsigned long long packed =
            (static_cast<unsigned long long>(cardinality) << 32) |
            (0xffffffffull - static_cast<unsigned int>(seed));
        atomicMax(best_cluster, packed);
    }
}

// Arbitrary-size fallback. It retains exactly the same GPU algorithm, placing
// the per-seed running maxima in global memory once register tiling is no longer
// practical. There is intentionally no CPU execution path.
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void growCandidateClustersGlobal(const double* __restrict__ distances,
                                 double* __restrict__ working_diameters,
                                 unsigned char* __restrict__ clustered,
                                 const int* __restrict__ active_seeds,
                                 const int point_count,
                                 const double threshold,
                                 unsigned long long* __restrict__ best_cluster,
                                 int* __restrict__ candidate_members) {
    const int active_index = blockIdx.x;
    const int seed = active_seeds[active_index];
    if (clustered[seed]) {
        return;
    }
    const int tid = threadIdx.x;
    int* const seed_members =
        candidate_members + static_cast<size_t>(seed) * point_count;
    double* const seed_diameters =
        working_diameters + static_cast<size_t>(active_index) * point_count;

    __shared__ double shared_diameters[32];
    __shared__ int shared_indices[32];

    CandidateChoice local_choice{INVALID_DISTANCE, INT_MAX};
    for (int candidate = tid; candidate < point_count;
         candidate += CUDA_BLOCK_SIZE) {
        double candidate_diameter = INVALID_DISTANCE;
        if (candidate != seed && !clustered[candidate]) {
            const double seed_distance =
                distances[static_cast<size_t>(seed) * point_count + candidate];
            if (seed_distance < threshold) {
                candidate_diameter = seed_distance;
                takeBetter(seed_distance, candidate, local_choice);
            }
        }
        seed_diameters[candidate] = candidate_diameter;
    }

    if (tid == 0) {
        seed_members[0] = seed;
    }

    CandidateChoice selected = blockArgMin<CUDA_BLOCK_SIZE>(
        local_choice, shared_diameters, shared_indices);
    int cardinality = 1;

    while (selected.index != INT_MAX) {
        const int newest_member = selected.index;
        if (tid == 0) {
            seed_members[cardinality] = newest_member;
        }
        ++cardinality;

        local_choice = CandidateChoice{INVALID_DISTANCE, INT_MAX};
        for (int candidate = tid; candidate < point_count;
             candidate += CUDA_BLOCK_SIZE) {
            double candidate_diameter = seed_diameters[candidate];
            if (candidate == newest_member) {
                candidate_diameter = INVALID_DISTANCE;
            } else if (candidate_diameter < threshold) {
                const double new_distance =
                    distances[static_cast<size_t>(newest_member) * point_count + candidate];
                candidate_diameter = max(candidate_diameter, new_distance);
                if (candidate_diameter >= threshold) {
                    candidate_diameter = INVALID_DISTANCE;
                } else {
                    takeBetter(candidate_diameter, candidate, local_choice);
                }
            }
            seed_diameters[candidate] = candidate_diameter;
        }

        selected = blockArgMin<CUDA_BLOCK_SIZE>(
            local_choice, shared_diameters, shared_indices);
    }

    if (tid == 0) {
        const unsigned long long packed =
            (static_cast<unsigned long long>(cardinality) << 32) |
            (0xffffffffull - static_cast<unsigned int>(seed));
        atomicMax(best_cluster, packed);
    }
}

// Commit the selected row in parallel. All candidate member orders already
// exist on-device, so the winning seed never needs to be grown a second time.
__global__ void commitBestCluster(const unsigned long long* __restrict__ best_cluster,
                                  const int* __restrict__ candidate_members,
                                  const int point_count,
                                  unsigned char* __restrict__ clustered,
                                  int* __restrict__ output_members) {
    const unsigned long long packed_best = *best_cluster;
    const int cardinality = static_cast<int>(packed_best >> 32);
    const int seed = static_cast<int>(
        0xffffffffu - static_cast<unsigned int>(packed_best));
    const int* const seed_members =
        candidate_members + static_cast<size_t>(seed) * point_count;

    for (int i = threadIdx.x; i < cardinality; i += blockDim.x) {
        const int member = seed_members[i];
        output_members[i] = member;
        clustered[member] = 1;
    }
}

void launchCandidateKernel(const int active_count,
                           const int point_count,
                           const double threshold,
                           const double* distances,
                           double* working_diameters,
                           unsigned char* clustered,
                           const int* active_seeds,
                           unsigned long long* best_cluster,
                           int* members) {
    const dim3 grid(active_count);
    const dim3 block(CUDA_BLOCK_SIZE);

    if (point_count <= CUDA_BLOCK_SIZE) {
        growCandidateClusters<1><<<grid, block>>>(
            distances, clustered, active_seeds, point_count, threshold,
            best_cluster, members);
    } else if (point_count <= CUDA_BLOCK_SIZE * 2) {
        growCandidateClusters<2><<<grid, block>>>(
            distances, clustered, active_seeds, point_count, threshold,
            best_cluster, members);
    } else if (point_count <= CUDA_BLOCK_SIZE * 4) {
        growCandidateClusters<4><<<grid, block>>>(
            distances, clustered, active_seeds, point_count, threshold,
            best_cluster, members);
    } else if (point_count <= CUDA_BLOCK_SIZE * 8) {
        growCandidateClusters<8><<<grid, block>>>(
            distances, clustered, active_seeds, point_count, threshold,
            best_cluster, members);
    } else if (point_count <= CUDA_BLOCK_SIZE * 16) {
        growCandidateClusters<16><<<grid, block>>>(
            distances, clustered, active_seeds, point_count, threshold,
            best_cluster, members);
    } else {
        growCandidateClustersGlobal<<<grid, block>>>(
            distances, working_diameters, clustered, active_seeds,
            point_count, threshold, best_cluster, members);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// Main QT clustering algorithm. Each outer iteration retains the original
// commit ordering, while every active seed and all of its candidate scoring
// work execute on the GPU.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int point_count = static_cast<int>(points.size());
    const size_t matrix_elements =
        static_cast<size_t>(point_count) * static_cast<size_t>(point_count);

    DeviceBuffer<Point> device_points(points.size());
    DeviceBuffer<double> device_distances(matrix_elements);
    DeviceBuffer<unsigned char> device_clustered(points.size());
    DeviceBuffer<int> device_active_seeds(points.size());
    DeviceBuffer<int> device_candidate_members(matrix_elements);
    DeviceBuffer<int> device_output_members(points.size());
    DeviceBuffer<unsigned long long> device_best_cluster(1);

    // Above 4096 points register tiling gives way to an N-by-N GPU workspace.
    // Allocate a one-element dummy otherwise so kernel dispatch stays simple.
    const bool use_global_workspace = point_count > CUDA_BLOCK_SIZE * 16;
    DeviceBuffer<double> device_working_diameters(
        use_global_workspace ? matrix_elements : 1);

    CUDA_CHECK(cudaMemcpy(device_points.get(), points.data(),
                          points.size() * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered.get(), 0,
                          points.size() * sizeof(unsigned char)));

    const dim3 distance_block(32, 8);
    const dim3 distance_grid(
        (point_count + distance_block.x - 1) / distance_block.x,
        (point_count + distance_block.y - 1) / distance_block.y);
    buildDistanceMatrix<<<distance_grid, distance_block>>>(
        device_points.get(), device_distances.get(), point_count);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> active_seeds(points.size());
    std::iota(active_seeds.begin(), active_seeds.end(), 0);
    CUDA_CHECK(cudaMemcpy(device_active_seeds.get(), active_seeds.data(),
                          active_seeds.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    std::vector<int> host_members(points.size());
    std::vector<int> cluster_seeds;
    std::vector<int> cluster_cardinalities;
    cluster_seeds.reserve(points.size());
    cluster_cardinalities.reserve(points.size());

    int remaining_points = point_count;
    int member_offset = 0;
    while (remaining_points > 0) {
        CUDA_CHECK(cudaMemset(device_best_cluster.get(), 0,
                              sizeof(unsigned long long)));

        launchCandidateKernel(
            point_count, point_count, threshold,
            device_distances.get(), device_working_diameters.get(),
            device_clustered.get(), device_active_seeds.get(),
            device_best_cluster.get(), device_candidate_members.get());

        commitBestCluster<<<1, CUDA_BLOCK_SIZE>>>(
            device_best_cluster.get(), device_candidate_members.get(), point_count,
            device_clustered.get(), device_output_members.get() + member_offset);
        CUDA_CHECK(cudaGetLastError());

        unsigned long long packed_best = 0;
        CUDA_CHECK(cudaMemcpy(&packed_best, device_best_cluster.get(),
                              sizeof(packed_best), cudaMemcpyDeviceToHost));
        const int max_cardinality = static_cast<int>(packed_best >> 32);
        const int best_seed = static_cast<int>(
            0xffffffffu - static_cast<unsigned int>(packed_best));

        cluster_seeds.push_back(best_seed);
        cluster_cardinalities.push_back(max_cardinality);
        member_offset += max_cardinality;
        remaining_points -= max_cardinality;
    }

    CUDA_CHECK(cudaMemcpy(host_members.data(), device_output_members.get(),
                          points.size() * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<Cluster> clusters;
    clusters.reserve(cluster_seeds.size());
    member_offset = 0;
    for (size_t i = 0; i < cluster_seeds.size(); ++i) {
        Cluster cluster;
        cluster.seed_point = cluster_seeds[i];
        const int cardinality = cluster_cardinalities[i];
        cluster.members.assign(host_members.begin() + member_offset,
                               host_members.begin() + member_offset + cardinality);
        member_offset += cardinality;
        clusters.push_back(std::move(cluster));
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

    // Create the CUDA context before the measured clustering region. Context
    // construction is runtime initialization, not part of the QT algorithm.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
    
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
