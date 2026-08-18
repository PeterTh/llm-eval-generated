// QT Clustering Benchmark
//
// Every still-unclustered point is an independent QT candidate in one round.
// The CUDA implementation exploits that independence: one block grows one
// candidate cluster, while its threads evaluate candidate points in parallel.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS_PER_BLOCK = 256;
static constexpr int MAX_SEEDS_PER_GPU_BATCH = 8192;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

struct DeviceState {
    int device = 0;
    int multiprocessor_count = 1;
    int batch_capacity = 0;
    bool has_distance_cache = false;
    Point* points = nullptr;
    double* distances = nullptr;
    unsigned char* clustered = nullptr;
    int* seeds = nullptr;
    int* members = nullptr;
    int* selected_members = nullptr;
    unsigned long long* best_packed = nullptr;
};

[[noreturn]] static void cudaFail(cudaError_t error, const char* expression,
                                  const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_error__ = (expression);                          \
        if (cuda_error__ != cudaSuccess) cudaFail(cuda_error__, #expression,    \
                                                    __FILE__, __LINE__);          \
    } while (false)

// Generate synthetic 2D point data in clusters.
void generateSyntheticData(std::vector<Point>& points, const int N,
                           unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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
            ++count;
            --group_cnt;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances,
                                    const int point_count) {
    const std::size_t total = static_cast<std::size_t>(point_count) * point_count;
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (; index < total; index += stride) {
        const int row = static_cast<int>(index / point_count);
        const int column = static_cast<int>(index - static_cast<std::size_t>(row) * point_count);
        const double dx = points[row].x - points[column].x;
        const double dy = points[row].y - points[column].y;
        distances[index] = sqrt(dx * dx + dy * dy);
    }
}

__device__ __forceinline__ bool isBetterCandidate(const double distance_a,
                                                   const int index_a,
                                                   const double distance_b,
                                                   const int index_b) {
    // The original serial scan retains the lowest point index on an exact tie.
    return distance_a < distance_b ||
           (distance_a == distance_b && index_a < index_b);
}

template <bool UseDistanceCache>
__global__ void generateCandidateClusters(
    const Point* __restrict__ points, const double* __restrict__ distances,
    const unsigned char* __restrict__ clustered, const int* __restrict__ seeds,
    const int seed_count, const double threshold, const int point_count,
    int* __restrict__ members, unsigned long long* __restrict__ best_packed) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= seed_count) return;

    const int seed = seeds[row];
    if (clustered[seed]) return;

    const std::size_t row_offset = static_cast<std::size_t>(row) * point_count;
    __shared__ double reduction_distance[CUDA_THREADS_PER_BLOCK];
    __shared__ int reduction_index[CUDA_THREADS_PER_BLOCK];
    __shared__ int member_count;
    __shared__ int finished;

    if (thread == 0) {
        members[row_offset] = seed;
        member_count = 1;
        finished = 0;
    }
    __syncthreads();

    while (true) {
        // DBL_MAX is also the serial implementation's initial minimum.
        double local_best_distance = DBL_MAX;
        int local_best_index = point_count;

        for (int candidate = thread; candidate < point_count;
             candidate += blockDim.x) {
            if (clustered[candidate]) continue;

            double max_distance = 0.0;
            bool already_in_cluster = false;
            if constexpr (UseDistanceCache) {
                // member-major lookup gives adjacent threads adjacent loads.
                for (int member_position = 0; member_position < member_count;
                     ++member_position) {
                    const int member = members[row_offset + member_position];
                    if (candidate == member) {
                        already_in_cluster = true;
                        break;
                    }
                    const double candidate_distance =
                        distances[static_cast<std::size_t>(member) * point_count + candidate];
                    if (candidate_distance > max_distance) {
                        max_distance = candidate_distance;
                    }
                }
            } else {
                const Point candidate_point = points[candidate];
                for (int member_position = 0; member_position < member_count;
                     ++member_position) {
                    const int member = members[row_offset + member_position];
                    if (candidate == member) {
                        already_in_cluster = true;
                        break;
                    }
                    const Point member_point = points[member];
                    const double dx = candidate_point.x - member_point.x;
                    const double dy = candidate_point.y - member_point.y;
                    const double candidate_distance = sqrt(dx * dx + dy * dy);
                    if (candidate_distance > max_distance) {
                        max_distance = candidate_distance;
                    }
                }
            }

            if (!already_in_cluster && max_distance < threshold &&
                isBetterCandidate(max_distance, candidate, local_best_distance,
                                  local_best_index)) {
                local_best_distance = max_distance;
                local_best_index = candidate;
            }
        }

        reduction_distance[thread] = local_best_distance;
        reduction_index[thread] = local_best_index;
        __syncthreads();

        for (int offset = CUDA_THREADS_PER_BLOCK / 2; offset > 0; offset /= 2) {
            if (thread < offset &&
                isBetterCandidate(reduction_distance[thread + offset],
                                  reduction_index[thread + offset],
                                  reduction_distance[thread], reduction_index[thread])) {
                reduction_distance[thread] = reduction_distance[thread + offset];
                reduction_index[thread] = reduction_index[thread + offset];
            }
            __syncthreads();
        }

        if (thread == 0) {
            const int closest = reduction_index[0];
            if (closest == point_count) {
                finished = 1;
            } else {
                members[row_offset + member_count] = closest;
                ++member_count;
                finished = (member_count == point_count);
            }

            if (finished) {
                // Higher cardinality wins; for equal cardinality lower seed wins.
                const unsigned long long packed =
                    (static_cast<unsigned long long>(member_count) << 32) |
                    static_cast<unsigned int>(0xffffffffu - static_cast<unsigned int>(seed));
                atomicMax(best_packed, packed);
            }
        }
        __syncthreads();
        if (finished) return;
    }
}

__global__ void markClustered(unsigned char* clustered,
                              const int* selected_members, const int member_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < member_count) clustered[selected_members[index]] = 1;
}

static void launchDistanceMatrix(DeviceState& state, const int point_count) {
    constexpr int threads = 256;
    const std::size_t total = static_cast<std::size_t>(point_count) * point_count;
    const std::size_t block_count = (total + threads - 1) / threads;
    buildDistanceMatrix<<<static_cast<unsigned int>(block_count), threads>>>(
        state.points, state.distances, point_count);
    CUDA_CHECK(cudaGetLastError());
}

static void releaseDeviceState(DeviceState& state) {
    if (state.device < 0) return;
    cudaSetDevice(state.device);
    if (state.best_packed) cudaFree(state.best_packed);
    if (state.selected_members) cudaFree(state.selected_members);
    if (state.members) cudaFree(state.members);
    if (state.seeds) cudaFree(state.seeds);
    if (state.clustered) cudaFree(state.clustered);
    if (state.distances) cudaFree(state.distances);
    if (state.points) cudaFree(state.points);
    state = {};
    state.device = -1;
}

static DeviceState createDeviceState(const int device, const std::vector<Point>& points) {
    const int point_count = static_cast<int>(points.size());
    const std::size_t point_count_size = static_cast<std::size_t>(point_count);
    const std::size_t pair_count = point_count_size * point_count_size;
    DeviceState state;
    state.device = device;

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    state.multiprocessor_count = properties.multiProcessorCount;

    CUDA_CHECK(cudaMalloc(&state.points, point_count_size * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&state.clustered, point_count_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&state.selected_members, point_count_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.best_packed, sizeof(unsigned long long)));

    std::size_t free_memory = 0;
    std::size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    const bool distance_size_fits =
        pair_count <= std::numeric_limits<std::size_t>::max() / sizeof(double);
    const std::size_t distance_bytes = distance_size_fits ?
        pair_count * sizeof(double) : 0;
    // Keeping at least half the free memory available leaves room for batches and
    // avoids turning a useful cache into an allocation failure on larger inputs.
    if (distance_size_fits && distance_bytes <= free_memory / 2) {
        const cudaError_t cache_status = cudaMalloc(&state.distances, distance_bytes);
        if (cache_status == cudaSuccess) {
            state.has_distance_cache = true;
        } else if (cache_status != cudaErrorMemoryAllocation) {
            cudaFail(cache_status, "cudaMalloc(&state.distances, distance_bytes)",
                     __FILE__, __LINE__);
        }
    }

    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    constexpr std::size_t safety_reserve = 512ull * 1024ull * 1024ull;
    const std::size_t usable_memory =
        free_memory > safety_reserve ? free_memory - safety_reserve : free_memory / 2;
    const std::size_t bytes_per_seed = point_count_size * sizeof(int);
    const std::size_t memory_limited_capacity = std::max<std::size_t>(
        1, usable_memory / std::max<std::size_t>(bytes_per_seed, 1));
    state.batch_capacity = static_cast<int>(std::min<std::size_t>(
        {point_count_size, static_cast<std::size_t>(MAX_SEEDS_PER_GPU_BATCH),
         memory_limited_capacity}));

    const std::size_t batch_elements = point_count_size * state.batch_capacity;
    CUDA_CHECK(cudaMalloc(&state.seeds, state.batch_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.members, batch_elements * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.points, points.data(), point_count_size * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.clustered, 0, point_count_size * sizeof(unsigned char)));

    if (state.has_distance_cache) launchDistanceMatrix(state, point_count);
    return state;
}

static void launchCandidateGeneration(DeviceState& state, const int seed_count,
                                      const double threshold, const int point_count) {
    CUDA_CHECK(cudaMemsetAsync(state.best_packed, 0, sizeof(unsigned long long)));
    if (state.has_distance_cache) {
        generateCandidateClusters<true><<<seed_count, CUDA_THREADS_PER_BLOCK>>>(
            state.points, state.distances, state.clustered, state.seeds, seed_count,
            threshold, point_count, state.members, state.best_packed);
    } else {
        generateCandidateClusters<false><<<seed_count, CUDA_THREADS_PER_BLOCK>>>(
            state.points, nullptr, state.clustered, state.seeds, seed_count, threshold,
            point_count, state.members, state.best_packed);
    }
    CUDA_CHECK(cudaGetLastError());
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         const double threshold) {
    const int point_count = static_cast<int>(points.size());
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "CUDA error: no CUDA-capable device is available\n");
        std::exit(EXIT_FAILURE);
    }

    cudaDeviceProp first_device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&first_device_properties, 0));
    const int initial_blocks_per_device = std::max(
        1, first_device_properties.multiProcessorCount * 8);
    // Do not create idle CUDA contexts: a later round cannot have more active
    // seeds than this first one, so it cannot benefit from more devices either.
    device_count = std::min<int>(device_count, std::max(1, static_cast<int>(
        (static_cast<std::size_t>(point_count) + initial_blocks_per_device - 1) /
        initial_blocks_per_device)));

    std::vector<DeviceState> devices;
    devices.reserve(device_count);
    for (int device = 0; device < device_count; ++device) {
        devices.push_back(createDeviceState(device, points));
    }

    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    for (int point = 0; point < point_count; ++point) unclustered_indices[point] = point;

    std::vector<Cluster> clusters;
    clusters.reserve(point_count);
    std::vector<int> selected_members(point_count);

    while (!unclustered_indices.empty()) {
        // A few resident blocks per SM are needed before another device improves
        // throughput. This retains strong single-GPU occupancy for small inputs.
        const int blocks_per_device = std::max(
            1, devices.front().multiprocessor_count * 8);
        const int active_device_count = std::min<int>(
            device_count, std::max(1, static_cast<int>(
                (unclustered_indices.size() + blocks_per_device - 1) / blocks_per_device)));

        unsigned long long winning_packed = 0;
        int winning_device = -1;
        int winning_size = 0;
        std::size_t next_seed = 0;

        while (next_seed < unclustered_indices.size()) {
            struct PendingBatch {
                int device_index;
                std::size_t begin;
                int count;
            };
            std::vector<PendingBatch> pending_batches;
            pending_batches.reserve(active_device_count);

            for (int device_index = 0;
                 device_index < active_device_count && next_seed < unclustered_indices.size();
                 ++device_index) {
                DeviceState& state = devices[device_index];
                const std::size_t remaining = unclustered_indices.size() - next_seed;
                const int devices_remaining = active_device_count - device_index;
                const std::size_t balanced_count =
                    (remaining + devices_remaining - 1) / devices_remaining;
                const int count = static_cast<int>(std::min<std::size_t>(
                    {remaining, balanced_count,
                     static_cast<std::size_t>(state.batch_capacity)}));

                CUDA_CHECK(cudaSetDevice(state.device));
                CUDA_CHECK(cudaMemcpyAsync(state.seeds, unclustered_indices.data() + next_seed,
                                           count * sizeof(int), cudaMemcpyHostToDevice));
                launchCandidateGeneration(state, count, threshold, point_count);
                pending_batches.push_back({device_index, next_seed, count});
                next_seed += count;
            }

            // All devices have been dispatched before any result is awaited.
            for (const PendingBatch& batch : pending_batches) {
                DeviceState& state = devices[batch.device_index];
                unsigned long long packed = 0;
                CUDA_CHECK(cudaSetDevice(state.device));
                CUDA_CHECK(cudaMemcpy(&packed, state.best_packed, sizeof(packed),
                                      cudaMemcpyDeviceToHost));
                if (packed > winning_packed) {
                    const int candidate_size = static_cast<int>(packed >> 32);
                    const int candidate_seed = static_cast<int>(
                        0xffffffffu - static_cast<unsigned int>(packed));
                    const auto first = unclustered_indices.begin() + batch.begin;
                    const auto last = first + batch.count;
                    const auto seed_position = std::lower_bound(first, last, candidate_seed);
                    const int row = static_cast<int>(seed_position - first);

                    CUDA_CHECK(cudaMemcpyAsync(state.selected_members,
                                               state.members +
                                                   static_cast<std::size_t>(row) * point_count,
                                               candidate_size * sizeof(int),
                                               cudaMemcpyDeviceToDevice));
                    winning_packed = packed;
                    winning_device = batch.device_index;
                    winning_size = candidate_size;
                }
            }
        }

        if (winning_device < 0 || winning_size <= 0) {
            std::fprintf(stderr, "CUDA error: no QT candidate was generated\n");
            for (DeviceState& state : devices) releaseDeviceState(state);
            std::exit(EXIT_FAILURE);
        }

        DeviceState& winner = devices[winning_device];
        CUDA_CHECK(cudaSetDevice(winner.device));
        CUDA_CHECK(cudaMemcpy(selected_members.data(), winner.selected_members,
                              winning_size * sizeof(int), cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = static_cast<int>(
            0xffffffffu - static_cast<unsigned int>(winning_packed));
        cluster.members.assign(selected_members.begin(),
                               selected_members.begin() + winning_size);
        clusters.push_back(std::move(cluster));

        for (int member_index = 0; member_index < winning_size; ++member_index) {
            clustered[selected_members[member_index]] = 1;
        }

        const int marker_blocks = (winning_size + CUDA_THREADS_PER_BLOCK - 1) /
                                  CUDA_THREADS_PER_BLOCK;
        for (DeviceState& state : devices) {
            CUDA_CHECK(cudaSetDevice(state.device));
            CUDA_CHECK(cudaMemcpyAsync(state.selected_members, selected_members.data(),
                                       winning_size * sizeof(int), cudaMemcpyHostToDevice));
            markClustered<<<marker_blocks, CUDA_THREADS_PER_BLOCK>>>(
                state.clustered, state.selected_members, winning_size);
            CUDA_CHECK(cudaGetLastError());
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int point) { return clustered[point] != 0; }),
            unclustered_indices.end());
    }

    for (DeviceState& state : devices) {
        CUDA_CHECK(cudaSetDevice(state.device));
        CUDA_CHECK(cudaDeviceSynchronize());
        releaseDeviceState(state);
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points, const double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    for (std::size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (std::size_t i = 0; i < cluster.members.size(); ++i) {
            for (std::size_t j = i + 1; j < cluster.members.size(); ++j) {
                max_diameter = std::max(max_diameter,
                    distance(points[cluster.members[i]], points[cluster.members[j]]));
            }
        }

        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c,
                        cluster.members.size(), cluster.seed_point, max_diameter);
        }
        if (max_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c,
                        max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (std::size_t c = 0; c < clusters.size(); ++c) {
        for (const int member : clusters[c].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
    for (const int owner : membership) {
        if (owner >= 0) ++clustered_count;
    }
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
                clustered_count, points.size() - clustered_count);
    return valid;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                    num_points, threshold);
        return 1;
    }

    std::printf("QT Clustering Benchmark\n");
    std::printf("Number of points: %d\n", num_points);
    std::printf("Distance threshold: %.2f\n", threshold);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    const auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    std::printf("Clustering time: %ld ms\n", cluster_time.count());
    std::printf("Clusters found: %zu\n", clusters.size());

    int total_clustered = 0;
    int max_cluster_size = 0;
    for (const Cluster& cluster : clusters) {
        const int size = static_cast<int>(cluster.members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }

    const double avg_cluster_size = clusters.empty() ? 0.0 :
        static_cast<double>(total_clustered) / clusters.size();
    std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
                100.0 * total_clustered / num_points);
    std::printf("Average cluster size: %.2f\n", avg_cluster_size);
    std::printf("Maximum cluster size: %d\n", max_cluster_size);

    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    std::printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec,
                points_per_sec);

    if (printResults) {
        std::vector<double> membership_data;
        membership_data.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (std::size_t c = 0; c < clusters.size(); ++c) {
            for (const int member : clusters[c].members) {
                membership[member] = static_cast<int>(c);
            }
        }
        for (const int member : membership) {
            membership_data.push_back(static_cast<double>(member));
        }
        print_results(membership_data, "ClusterMembership");
    }

    if (validate) {
        if (validateClusters(clusters, points, threshold)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
