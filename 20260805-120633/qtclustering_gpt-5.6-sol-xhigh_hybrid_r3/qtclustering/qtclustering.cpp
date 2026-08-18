// QT Clustering Benchmark - hybrid MPI/OpenMP/CUDA implementation
//
// Candidate clusters are independent during each QT iteration.  MPI ranks
// divide the seeds, one CUDA block greedily grows each candidate, and MPI
// MAXLOC selects the largest candidate (and lowest seed on ties).  OpenMP is
// used for host-side transformations, bookkeeping, statistics, and checking.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double MAX_WIDTH = 20.0;
constexpr double MAX_HEIGHT = 20.0;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Point {
    double x;
    double y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

void checkCuda(cudaError_t status, const char* expression,
               const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    int rank = -1;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "MPI rank %d: CUDA failure at %s:%d: %s: %s\n",
                 rank, file, line, expression, cudaGetErrorString(status));
    std::fflush(stderr);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::abort();
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

// Generate the same synthetic point set as the sequential benchmark.
void generateSyntheticData(std::vector<Point>& points, const int count,
                           unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int generated = 0;

    while (generated < count) {
        const double center_x = frand() * MAX_WIDTH;
        const double center_y = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (count / 30.0));

        // The original distribution is unchanged for its terminating domain.
        // For <= 30 points its expression always truncates to zero, so supply
        // the otherwise missing progress guarantee for small correctness runs.
        if (count <= 30) {
            group_count = 1;
        }

        if (group_count > count - generated) {
            group_count = count - generated;
        }

        while (group_count > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = center_x + dx;
            const double y = center_y + dy;

            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT) {
                continue;
            }

            points[generated++] = {x, y};
            --group_count;
        }
    }
}

inline double pointDistance(const Point& first, const Point& second) {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    return std::sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ bool betterCandidate(double lhs_distance,
                                                 int lhs_index,
                                                 double rhs_distance,
                                                 int rhs_index) {
    return lhs_distance < rhs_distance ||
           (lhs_distance == rhs_distance && lhs_index < rhs_index);
}

__device__ __forceinline__ void warpReduceBest(double& distance, int& index) {
    constexpr unsigned int full_warp = 0xffffffffU;
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        const double other_distance =
            __shfl_down_sync(full_warp, distance, offset);
        const int other_index = __shfl_down_sync(full_warp, index, offset);
        if ((threadIdx.x & (warpSize - 1)) + offset < warpSize &&
            betterCandidate(other_distance, other_index, distance, index)) {
            distance = other_distance;
            index = other_index;
        }
    }
}

// Grow one candidate per CUDA block.  Storing only the running maximum
// squared distance for every point turns the original repeated O(k) diameter
// scan into one O(N) update per added member.  Squaring is monotone for these
// non-negative distances, so choices and the strict threshold are unchanged.
template <bool SAVE_MEMBERS>
__global__ void growCandidatesKernel(const double* __restrict__ point_x,
                                     const double* __restrict__ point_y,
                                     const unsigned char* __restrict__ clustered,
                                     const int* __restrict__ seeds,
                                     int* __restrict__ cardinalities,
                                     int* __restrict__ saved_members,
                                     double* __restrict__ max_distance_squared,
                                     int point_count,
                                     double threshold_squared) {
    const int row = static_cast<int>(blockIdx.x);
    const int seed = seeds[row];
    double* const distances = max_distance_squared +
                              static_cast<std::size_t>(row) * point_count;

    constexpr int warp_count = CUDA_BLOCK_SIZE / 32;
    __shared__ double warp_distance[warp_count];
    __shared__ int warp_index[warp_count];
    __shared__ int selected;
    __shared__ int cardinality;

    for (int point = threadIdx.x; point < point_count; point += blockDim.x) {
        if (clustered[point] || point == seed) {
            distances[point] = CUDART_INF;
            continue;
        }

        const double dx = point_x[point] - point_x[seed];
        const double dy = point_y[point] - point_y[seed];
        const double distance_squared = dx * dx + dy * dy;
        // A rejected point can never become eligible: its running maximum
        // only increases as the candidate cluster grows.
        distances[point] = distance_squared < threshold_squared
                               ? distance_squared
                               : CUDART_INF;
    }

    if (threadIdx.x == 0) {
        cardinality = 1;
        if constexpr (SAVE_MEMBERS) {
            saved_members[0] = seed;
        }
    }
    __syncthreads();

    while (true) {
        double local_distance = CUDART_INF;
        int local_index = INT_MAX;

        for (int point = threadIdx.x; point < point_count; point += blockDim.x) {
            const double distance_squared = distances[point];
            if (distance_squared < threshold_squared &&
                betterCandidate(distance_squared, point,
                                local_distance, local_index)) {
                local_distance = distance_squared;
                local_index = point;
            }
        }

        warpReduceBest(local_distance, local_index);
        const int lane = threadIdx.x & (warpSize - 1);
        const int warp = threadIdx.x / warpSize;
        if (lane == 0) {
            warp_distance[warp] = local_distance;
            warp_index[warp] = local_index;
        }
        __syncthreads();

        if (warp == 0) {
            local_distance = lane < warp_count ? warp_distance[lane] : CUDART_INF;
            local_index = lane < warp_count ? warp_index[lane] : INT_MAX;
            warpReduceBest(local_distance, local_index);
            if (lane == 0) {
                selected = local_index == INT_MAX ? -1 : local_index;
                if (selected >= 0) {
                    if constexpr (SAVE_MEMBERS) {
                        saved_members[cardinality] = selected;
                    }
                    ++cardinality;
                }
            }
        }
        __syncthreads();

        if (selected < 0) {
            break;
        }

        for (int point = threadIdx.x; point < point_count; point += blockDim.x) {
            double old_distance = distances[point];
            if (point == selected) {
                distances[point] = CUDART_INF;
            } else if (old_distance < threshold_squared) {
                const double dx = point_x[point] - point_x[selected];
                const double dy = point_y[point] - point_y[selected];
                const double new_distance = dx * dx + dy * dy;
                old_distance = new_distance > old_distance
                                   ? new_distance
                                   : old_distance;
                distances[point] = old_distance < threshold_squared
                                       ? old_distance
                                       : CUDART_INF;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        cardinalities[row] = cardinality;
    }
}

__global__ void markClusteredKernel(unsigned char* clustered,
                                    const int* members, int member_count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < member_count) {
        clustered[members[index]] = 1;
    }
}

class GpuCandidateEngine {
public:
    GpuCandidateEngine(const std::vector<double>& point_x,
                       const std::vector<double>& point_y,
                       int maximum_seed_count,
                       int ranks_sharing_gpu)
        : point_count_(static_cast<int>(point_x.size())) {
        const std::size_t point_bytes =
            static_cast<std::size_t>(point_count_) * sizeof(double);
        const std::size_t member_bytes =
            static_cast<std::size_t>(point_count_) * sizeof(int);

        CUDA_CHECK(cudaMalloc(&device_x_, point_bytes));
        CUDA_CHECK(cudaMalloc(&device_y_, point_bytes));
        CUDA_CHECK(cudaMalloc(&device_clustered_,
                              static_cast<std::size_t>(point_count_)));
        CUDA_CHECK(cudaMalloc(&device_members_, member_bytes));
        CUDA_CHECK(cudaMemcpy(device_x_, point_x.data(), point_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_y_, point_y.data(), point_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_clustered_, 0,
                              static_cast<std::size_t>(point_count_)));

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;

        const std::size_t row_bytes =
            static_cast<std::size_t>(point_count_) * sizeof(double);
        // Leave room for the CUDA context, MPI buffers, and concurrent ranks
        // in case users intentionally place more than one rank on a GPU.
        const std::size_t usable_bytes =
            (free_bytes / 5U * 4U) /
            static_cast<std::size_t>(std::max(1, ranks_sharing_gpu));
        const int requested_capacity = std::max(1, maximum_seed_count);
        const std::size_t memory_capacity =
            std::max<std::size_t>(1, usable_bytes / row_bytes);
        batch_capacity_ = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(requested_capacity), memory_capacity));

        // Fragmented or concurrently allocated device memory can make the
        // estimate optimistic.  Retrying smaller batches keeps memory usage
        // bounded without changing the algorithm.
        while (batch_capacity_ > 0) {
            const std::size_t workspace_bytes =
                static_cast<std::size_t>(batch_capacity_) * row_bytes;
            const cudaError_t status =
                cudaMalloc(&device_max_distances_, workspace_bytes);
            if (status == cudaSuccess) {
                break;
            }
            cudaGetLastError();
            batch_capacity_ /= 2;
        }
        if (batch_capacity_ == 0) {
            CUDA_CHECK(cudaErrorMemoryAllocation);
        }

        CUDA_CHECK(cudaMalloc(&device_seeds_,
                              static_cast<std::size_t>(batch_capacity_) *
                                  sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_cardinalities_,
                              static_cast<std::size_t>(batch_capacity_) *
                                  sizeof(int)));
        host_cardinalities_.resize(batch_capacity_);
    }

    GpuCandidateEngine(const GpuCandidateEngine&) = delete;
    GpuCandidateEngine& operator=(const GpuCandidateEngine&) = delete;

    ~GpuCandidateEngine() {
        cudaFree(device_cardinalities_);
        cudaFree(device_seeds_);
        cudaFree(device_max_distances_);
        cudaFree(device_members_);
        cudaFree(device_clustered_);
        cudaFree(device_y_);
        cudaFree(device_x_);
    }

    int batchCapacity() const { return batch_capacity_; }

    std::pair<int, int> evaluate(const std::vector<int>& seeds,
                                 double threshold_squared) {
        int best_cardinality = -1;
        int best_seed = INT_MAX;

        for (std::size_t offset = 0; offset < seeds.size();
             offset += static_cast<std::size_t>(batch_capacity_)) {
            const int batch_size = static_cast<int>(std::min<std::size_t>(
                static_cast<std::size_t>(batch_capacity_),
                seeds.size() - offset));
            CUDA_CHECK(cudaMemcpy(device_seeds_, seeds.data() + offset,
                                  static_cast<std::size_t>(batch_size) * sizeof(int),
                                  cudaMemcpyHostToDevice));

            growCandidatesKernel<false>
                <<<batch_size, CUDA_BLOCK_SIZE>>>(
                    device_x_, device_y_, device_clustered_, device_seeds_,
                    device_cardinalities_, nullptr, device_max_distances_,
                    point_count_, threshold_squared);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host_cardinalities_.data(),
                                  device_cardinalities_,
                                  static_cast<std::size_t>(batch_size) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            for (int index = 0; index < batch_size; ++index) {
                const int seed = seeds[offset + static_cast<std::size_t>(index)];
                const int cardinality = host_cardinalities_[index];
                if (cardinality > best_cardinality ||
                    (cardinality == best_cardinality && seed < best_seed)) {
                    best_cardinality = cardinality;
                    best_seed = seed;
                }
            }
        }

        return {best_cardinality, best_seed};
    }

    std::vector<int> materialize(int seed, int expected_cardinality,
                                 double threshold_squared) {
        CUDA_CHECK(cudaMemcpy(device_seeds_, &seed, sizeof(int),
                              cudaMemcpyHostToDevice));
        growCandidatesKernel<true><<<1, CUDA_BLOCK_SIZE>>>(
            device_x_, device_y_, device_clustered_, device_seeds_,
            device_cardinalities_, device_members_, device_max_distances_,
            point_count_, threshold_squared);
        CUDA_CHECK(cudaGetLastError());

        int cardinality = 0;
        CUDA_CHECK(cudaMemcpy(&cardinality, device_cardinalities_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (cardinality != expected_cardinality) {
            int rank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::fprintf(stderr,
                         "MPI rank %d: regenerated seed %d with size %d, "
                         "expected %d\n",
                         rank, seed, cardinality, expected_cardinality);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        std::vector<int> members(static_cast<std::size_t>(cardinality));
        CUDA_CHECK(cudaMemcpy(members.data(), device_members_,
                              static_cast<std::size_t>(cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        return members;
    }

    void markClustered(const std::vector<int>& members) {
        const int member_count = static_cast<int>(members.size());
        CUDA_CHECK(cudaMemcpy(device_members_, members.data(),
                              static_cast<std::size_t>(member_count) * sizeof(int),
                              cudaMemcpyHostToDevice));
        const int block_count =
            (member_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        markClusteredKernel<<<block_count, CUDA_BLOCK_SIZE>>>(
            device_clustered_, device_members_, member_count);
        CUDA_CHECK(cudaGetLastError());
    }

private:
    int point_count_ = 0;
    int batch_capacity_ = 0;
    double* device_x_ = nullptr;
    double* device_y_ = nullptr;
    double* device_max_distances_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    int* device_seeds_ = nullptr;
    int* device_cardinalities_ = nullptr;
    int* device_members_ = nullptr;
    std::vector<int> host_cardinalities_;
};

std::vector<Cluster> qtClusteringHybrid(const std::vector<Point>& points,
                                        double threshold,
                                        GpuCandidateEngine& gpu,
                                        int rank, int world_size) {
    const int point_count = static_cast<int>(points.size());
    const double threshold_squared = threshold * threshold;
    std::vector<unsigned char> clustered(static_cast<std::size_t>(point_count), 0);
    std::vector<Cluster> clusters;
    std::vector<int> local_seeds;
    local_seeds.reserve(
        static_cast<std::size_t>((point_count - 1) / world_size + 1));
    int remaining = point_count;

    while (remaining > 0) {
        local_seeds.clear();
        for (int seed = rank; seed < point_count; seed += world_size) {
            if (!clustered[seed]) {
                local_seeds.push_back(seed);
            }
        }

        const auto [local_cardinality, local_seed] =
            gpu.evaluate(local_seeds, threshold_squared);

        struct {
            int cardinality;
            int seed;
        } local_best{local_cardinality, local_seed}, global_best{-1, INT_MAX};

        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT,
                      MPI_MAXLOC, MPI_COMM_WORLD);
        if (global_best.seed == INT_MAX || global_best.cardinality <= 0) {
            break;
        }

        // If the largest candidate is a singleton, every remaining candidate
        // is a singleton.  Removing points cannot make a later candidate grow,
        // so emit the rest in the original ascending-seed order without one
        // global synchronization per point.
        if (global_best.cardinality == 1) {
            if (rank == 0) {
                for (int seed = 0; seed < point_count; ++seed) {
                    if (!clustered[seed]) {
                        clusters.push_back(Cluster{{seed}, seed});
                    }
                }
            }
            remaining = 0;
            break;
        }

        const int owner = global_best.seed % world_size;
        std::vector<int> best_members;
        if (rank == owner) {
            best_members = gpu.materialize(global_best.seed,
                                           global_best.cardinality,
                                           threshold_squared);
        } else {
            best_members.resize(
                static_cast<std::size_t>(global_best.cardinality));
        }
        MPI_Bcast(best_members.data(), global_best.cardinality, MPI_INT,
                  owner, MPI_COMM_WORLD);

#pragma omp parallel for schedule(static) if(global_best.cardinality >= 4096)
        for (int index = 0; index < global_best.cardinality; ++index) {
            clustered[best_members[static_cast<std::size_t>(index)]] = 1;
        }
        gpu.markClustered(best_members);
        remaining -= global_best.cardinality;

        if (rank == 0) {
            clusters.push_back(Cluster{std::move(best_members), global_best.seed});
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    std::vector<double> diameters(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t cluster_index = 0;
         cluster_index < static_cast<std::ptrdiff_t>(clusters.size());
         ++cluster_index) {
        const Cluster& cluster = clusters[static_cast<std::size_t>(cluster_index)];
        double max_diameter = 0.0;
        for (std::size_t first = 0; first < cluster.members.size(); ++first) {
            for (std::size_t second = first + 1;
                 second < cluster.members.size(); ++second) {
                const double distance = pointDistance(
                    points[cluster.members[first]], points[cluster.members[second]]);
                max_diameter = std::max(max_diameter, distance);
            }
        }
        diameters[static_cast<std::size_t>(cluster_index)] = max_diameter;
    }

    for (std::size_t cluster_index = 0;
         cluster_index < clusters.size(); ++cluster_index) {
        const Cluster& cluster = clusters[cluster_index];
        const double max_diameter = diameters[cluster_index];
        if (cluster_index < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        cluster_index, cluster.members.size(),
                        cluster.seed_point, max_diameter);
        }
        if (max_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        cluster_index, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (std::size_t cluster_index = 0;
         cluster_index < clusters.size(); ++cluster_index) {
        for (const int member : clusters[cluster_index].members) {
            if (membership[member] >= 0) {
                std::printf(
                    "ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                    member, membership[member], cluster_index);
                valid = false;
            }
            membership[member] = static_cast<int>(cluster_index);
        }
    }

    int clustered_count = 0;
#pragma omp parallel for reduction(+ : clustered_count) schedule(static)
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(membership.size()); ++index) {
        clustered_count += membership[static_cast<std::size_t>(index)] >= 0;
    }

    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count,
                points.size() - static_cast<std::size_t>(clustered_count));
    return valid;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
    bool argument_error = false;
    bool help_requested = false;

    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "-n") == 0 && index + 1 < argc) {
            num_points = std::atoi(argv[++index]);
        } else if (std::strcmp(argv[index], "-t") == 0 && index + 1 < argc) {
            threshold = std::atof(argv[++index]);
        } else if (std::strcmp(argv[index], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[index], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[index], "-h") == 0) {
            help_requested = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[index]);
            }
            argument_error = true;
        }
    }

    if (help_requested || argument_error || num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            if (!help_requested && !argument_error) {
                std::printf(
                    "Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                    num_points, threshold);
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (help_requested && !argument_error) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    MPI_Comm node_communicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &node_communicator);
    int local_rank = 0;
    MPI_Comm_rank(node_communicator, &local_rank);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA-capable GPU is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    // CUDA_VISIBLE_DEVICES commonly renumbers every assigned GPU to device 0.
    // Group ranks by the physical PCI address so independently assigned GPUs
    // do not unnecessarily divide their memory budget by the node rank count.
    int pci_domain = 0;
    int pci_bus = 0;
    int pci_device = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&pci_domain, cudaDevAttrPciDomainId, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&pci_bus, cudaDevAttrPciBusId, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&pci_device, cudaDevAttrPciDeviceId, device));
    const int physical_gpu_color =
        ((pci_domain & 0xffff) << 13) |
        ((pci_bus & 0xff) << 5) |
        (pci_device & 0x1f);

    MPI_Comm gpu_communicator = MPI_COMM_NULL;
    MPI_Comm_split(node_communicator, physical_gpu_color, local_rank,
                   &gpu_communicator);
    int ranks_sharing_gpu = 1;
    MPI_Comm_size(gpu_communicator, &ranks_sharing_gpu);

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid parallelism: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, %d CUDA GPU(s)/node\n",
                    world_size, omp_get_max_threads(), device_count);
    }

    std::vector<Point> points(static_cast<std::size_t>(num_points));
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    static_assert(std::is_standard_layout_v<Point> && sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype mpi_point_type = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_point_type);
    MPI_Type_commit(&mpi_point_type);
    MPI_Bcast(points.data(), num_points, mpi_point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&mpi_point_type);

    std::vector<double> point_x(static_cast<std::size_t>(num_points));
    std::vector<double> point_y(static_cast<std::size_t>(num_points));
#pragma omp parallel for schedule(static)
    for (int point = 0; point < num_points; ++point) {
        point_x[static_cast<std::size_t>(point)] =
            points[static_cast<std::size_t>(point)].x;
        point_y[static_cast<std::size_t>(point)] =
            points[static_cast<std::size_t>(point)].y;
    }

    int return_code = EXIT_SUCCESS;
    {
        const int maximum_local_seeds =
            (num_points - 1) / world_size + 1;
        GpuCandidateEngine gpu(point_x, point_y, maximum_local_seeds,
                               ranks_sharing_gpu);

        int minimum_batch = 0;
        int maximum_batch = 0;
        const int local_batch = gpu.batchCapacity();
        MPI_Reduce(&local_batch, &minimum_batch, 1, MPI_INT, MPI_MIN,
                   0, MPI_COMM_WORLD);
        MPI_Reduce(&local_batch, &maximum_batch, 1, MPI_INT, MPI_MAX,
                   0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("CUDA seed batch capacity/rank: %d..%d\n",
                        minimum_batch, maximum_batch);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const auto cluster_start = std::chrono::steady_clock::now();
        std::vector<Cluster> clusters =
            qtClusteringHybrid(points, threshold, gpu, rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        const auto cluster_end = std::chrono::steady_clock::now();

        if (rank == 0) {
            const auto cluster_milliseconds =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    cluster_end - cluster_start);
            const double cluster_seconds =
                std::chrono::duration<double>(cluster_end - cluster_start).count();

            std::printf("Clustering time: %lld ms\n",
                        static_cast<long long>(cluster_milliseconds.count()));
            std::printf("Clusters found: %zu\n", clusters.size());

            int total_clustered = 0;
            int max_cluster_size = 0;
#pragma omp parallel for reduction(+ : total_clustered) \
    reduction(max : max_cluster_size) schedule(static)
            for (std::ptrdiff_t index = 0;
                 index < static_cast<std::ptrdiff_t>(clusters.size()); ++index) {
                const int size = static_cast<int>(
                    clusters[static_cast<std::size_t>(index)].members.size());
                total_clustered += size;
                max_cluster_size = std::max(max_cluster_size, size);
            }

            const double average_cluster_size = clusters.empty()
                ? 0.0
                : static_cast<double>(total_clustered) / clusters.size();
            std::printf("Points clustered: %d / %d (%.1f%%)\n",
                        total_clustered, num_points,
                        100.0 * total_clustered / num_points);
            std::printf("Average cluster size: %.2f\n", average_cluster_size);
            std::printf("Maximum cluster size: %d\n", max_cluster_size);
            std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                        clusters.size() / cluster_seconds,
                        num_points / cluster_seconds);

            if (print_results_requested) {
                std::vector<int> membership(static_cast<std::size_t>(num_points), -1);
                for (std::size_t cluster_index = 0;
                     cluster_index < clusters.size(); ++cluster_index) {
                    for (const int member : clusters[cluster_index].members) {
                        membership[member] = static_cast<int>(cluster_index);
                    }
                }
                std::vector<double> membership_data(
                    static_cast<std::size_t>(num_points));
#pragma omp parallel for schedule(static)
                for (int point = 0; point < num_points; ++point) {
                    membership_data[static_cast<std::size_t>(point)] =
                        static_cast<double>(
                            membership[static_cast<std::size_t>(point)]);
                }
                print_results(membership_data, "ClusterMembership");
            }

            if (validate) {
                const bool valid = validateClusters(clusters, points, threshold);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                return_code = valid ? EXIT_SUCCESS : EXIT_FAILURE;
            }
        }
    }

    MPI_Bcast(&return_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&gpu_communicator);
    MPI_Comm_free(&node_communicator);
    MPI_Finalize();
    return return_code;
}
