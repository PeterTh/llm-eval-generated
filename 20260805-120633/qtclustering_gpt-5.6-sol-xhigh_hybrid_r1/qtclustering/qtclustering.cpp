// QT Clustering Benchmark - hybrid MPI/OpenMP/CUDA implementation
//
// MPI distributes independent seed clusters across accelerators.  A CUDA
// block constructs one candidate cluster, and OpenMP handles host reductions
// and bookkeeping.  The winning cluster is synchronized after every QT
// iteration so the result is identical for every MPI rank.

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
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_BLOCK_SIZE = 256;

static int mpi_rank = 0;

[[noreturn]] static void cudaFailure(const cudaError_t error, const char* expression,
                                     const char* file, const int line) {
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d while evaluating %s: %s\n",
                 mpi_rank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

struct DeviceChoice {
    double diameter_squared;
    int point;
};

static_assert(sizeof(Point) == 2 * sizeof(double), "Point must be tightly packed");

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
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (N / 30.0));

        // The original distribution formula truncates every group to zero
        // when N < 30.  There is no valid sequential result in that case (the
        // generator never terminates), so retain the distribution while
        // ensuring the documented positive input range is usable.
        if (N < 30 && group_count == 0) {
            group_count = 1;
        }

        if (group_count > N - count) {
            group_count = N - count;
        }

        while (group_count > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;

            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT) {
                continue;
            }

            points[count++] = {x, y};
            --group_count;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ DeviceChoice betterChoice(const DeviceChoice a,
                                                      const DeviceChoice b) {
    if (b.diameter_squared < a.diameter_squared ||
        (b.diameter_squared == a.diameter_squared && b.point < a.point)) {
        return b;
    }
    return a;
}

template <int BlockSize>
__device__ __forceinline__ DeviceChoice blockMinimum(DeviceChoice value,
                                                      DeviceChoice* shared) {
    static_assert(BlockSize % 32 == 0, "CUDA block must contain complete warps");
    constexpr unsigned int warp_mask = 0xffffffffU;
    constexpr int warp_count = BlockSize / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        const DeviceChoice other{
            __shfl_down_sync(warp_mask, value.diameter_squared, offset),
            __shfl_down_sync(warp_mask, value.point, offset)};
        if (lane < offset) {
            value = betterChoice(value, other);
        }
    }
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < warp_count ? shared[lane]
                                  : DeviceChoice{DBL_MAX, INT_MAX};
        for (int offset = 16; offset > 0; offset >>= 1) {
            const DeviceChoice other{
                __shfl_down_sync(warp_mask, value.diameter_squared, offset),
                __shfl_down_sync(warp_mask, value.point, offset)};
            if (lane < offset) {
                value = betterChoice(value, other);
            }
        }
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ double pointDistanceSquared(const Point a,
                                                        const Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Each block independently grows one seed cluster.  scratch stores, for every
// point, the maximum squared distance to members already selected.  Updating
// it with only the newly selected member is equivalent to rescanning every
// member, but changes a cubic inner operation into a quadratic one.
template <bool SaveMembers>
__global__ void candidateKernel(const Point* __restrict__ points,
                                const unsigned char* __restrict__ clustered,
                                const int* __restrict__ seeds,
                                const int seed_count, const int point_count,
                                const double threshold_squared,
                                double* __restrict__ scratch,
                                int* __restrict__ cardinalities,
                                int* __restrict__ saved_members) {
    const int seed_slot = static_cast<int>(blockIdx.x);
    if (seed_slot >= seed_count) {
        return;
    }

    __shared__ DeviceChoice shared_choices[CUDA_BLOCK_SIZE / 32];
    const int seed = seeds[seed_slot];
    const Point seed_point = points[seed];
    double* const row = scratch + static_cast<size_t>(seed_slot) * point_count;

    DeviceChoice local{DBL_MAX, INT_MAX};
    for (int candidate = static_cast<int>(threadIdx.x); candidate < point_count;
         candidate += CUDA_BLOCK_SIZE) {
        double value = DBL_MAX;
        if (clustered[candidate] == 0 && candidate != seed) {
            const double d2 = pointDistanceSquared(points[candidate], seed_point);
            if (d2 < threshold_squared) {
                value = d2;
                local = betterChoice(local, DeviceChoice{value, candidate});
            }
        }
        row[candidate] = value;
    }

    DeviceChoice selected = blockMinimum<CUDA_BLOCK_SIZE>(local, shared_choices);
    int cardinality = 1;
    if constexpr (SaveMembers) {
        if (threadIdx.x == 0) {
            saved_members[0] = seed;
        }
    }

    while (selected.point != INT_MAX) {
        const int added = selected.point;
        if constexpr (SaveMembers) {
            if (threadIdx.x == 0) {
                saved_members[cardinality] = added;
            }
        }
        ++cardinality;

        local = DeviceChoice{DBL_MAX, INT_MAX};
        const Point added_point = points[added];
        for (int candidate = static_cast<int>(threadIdx.x);
             candidate < point_count; candidate += CUDA_BLOCK_SIZE) {
            double value = row[candidate];
            if (candidate == added) {
                value = DBL_MAX;
            } else if (value < DBL_MAX) {
                const double d2 = pointDistanceSquared(points[candidate], added_point);
                value = value > d2 ? value : d2;
                if (value >= threshold_squared) {
                    value = DBL_MAX;
                } else {
                    local = betterChoice(local, DeviceChoice{value, candidate});
                }
            }
            row[candidate] = value;
        }
        selected = blockMinimum<CUDA_BLOCK_SIZE>(local, shared_choices);
    }

    if (threadIdx.x == 0) {
        cardinalities[seed_slot] = cardinality;
    }
}

__global__ void markClusteredKernel(unsigned char* clustered,
                                    const int* members, const int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        clustered[members[index]] = 1;
    }
}

struct HostChoice {
    int cardinality = -1;
    int seed = INT_MAX;
};

static inline bool betterHostChoice(const HostChoice& candidate,
                                    const HostChoice& current) {
    return candidate.cardinality > current.cardinality ||
           (candidate.cardinality == current.cardinality &&
            candidate.seed < current.seed);
}

class CudaQtEngine {
public:
    CudaQtEngine(const std::vector<Point>& points, const int maximum_local_seeds,
                 const int device)
        : point_count_(static_cast<int>(points.size())), device_(device) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points_),
                              points.size() * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered_),
                              points.size() * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_members_),
                              points.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(device_points_, points.data(),
                              points.size() * sizeof(Point), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_clustered_, 0,
                              points.size() * sizeof(unsigned char)));

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;

        const size_t bytes_per_seed = points.size() * sizeof(double);
        const size_t scratch_budget = free_bytes / 2;
        const size_t memory_capacity =
            std::max<size_t>(1, scratch_budget / std::max<size_t>(1, bytes_per_seed));
        batch_capacity_ = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(std::max(1, maximum_local_seeds)), memory_capacity));

        // Allocations may race with another MPI rank when a GPU is
        // oversubscribed.  Back off instead of failing immediately.
        while (true) {
            const cudaError_t status = cudaMalloc(
                reinterpret_cast<void**>(&device_scratch_),
                static_cast<size_t>(batch_capacity_) * bytes_per_seed);
            if (status == cudaSuccess) {
                break;
            }
            cudaGetLastError();
            if (batch_capacity_ == 1) {
                cudaFailure(status, "cudaMalloc(device_scratch_)", __FILE__, __LINE__);
            }
            batch_capacity_ = std::max(1, batch_capacity_ / 2);
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_seeds_),
                              static_cast<size_t>(batch_capacity_) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cardinalities_),
                              static_cast<size_t>(batch_capacity_) * sizeof(int)));
        host_cardinalities_.resize(batch_capacity_);
    }

    CudaQtEngine(const CudaQtEngine&) = delete;
    CudaQtEngine& operator=(const CudaQtEngine&) = delete;

    ~CudaQtEngine() {
        cudaSetDevice(device_);
        cudaFree(device_cardinalities_);
        cudaFree(device_seeds_);
        cudaFree(device_scratch_);
        cudaFree(device_members_);
        cudaFree(device_clustered_);
        cudaFree(device_points_);
    }

    HostChoice findBest(const std::vector<int>& seeds,
                        const double threshold_squared) {
        CUDA_CHECK(cudaSetDevice(device_));
        HostChoice best;
        for (size_t begin = 0; begin < seeds.size();
             begin += static_cast<size_t>(batch_capacity_)) {
            const int count = static_cast<int>(std::min<size_t>(
                batch_capacity_, seeds.size() - begin));
            CUDA_CHECK(cudaMemcpy(device_seeds_, seeds.data() + begin,
                                  static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyHostToDevice));
            candidateKernel<false><<<count, CUDA_BLOCK_SIZE>>>(
                device_points_, device_clustered_, device_seeds_, count,
                point_count_, threshold_squared, device_scratch_,
                device_cardinalities_, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host_cardinalities_.data(), device_cardinalities_,
                                  static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            HostChoice batch_best;
            // Large batches benefit from a CPU team while short batches stay
            // serial to avoid repeatedly launching a team in late QT rounds.
#pragma omp parallel if(count >= 1024)
            {
                HostChoice thread_best;
#pragma omp for nowait schedule(static)
                for (int i = 0; i < count; ++i) {
                    const HostChoice candidate{host_cardinalities_[i],
                                               seeds[begin + i]};
                    if (betterHostChoice(candidate, thread_best)) {
                        thread_best = candidate;
                    }
                }
#pragma omp critical
                {
                    if (betterHostChoice(thread_best, batch_best)) {
                        batch_best = thread_best;
                    }
                }
            }
            if (betterHostChoice(batch_best, best)) {
                best = batch_best;
            }
        }
        return best;
    }

    std::vector<int> buildMembers(const int seed, const int expected_cardinality,
                                  const double threshold_squared) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaMemcpy(device_seeds_, &seed, sizeof(int),
                              cudaMemcpyHostToDevice));
        candidateKernel<true><<<1, CUDA_BLOCK_SIZE>>>(
            device_points_, device_clustered_, device_seeds_, 1, point_count_,
            threshold_squared, device_scratch_, device_cardinalities_,
            device_members_);
        CUDA_CHECK(cudaGetLastError());

        int cardinality = 0;
        CUDA_CHECK(cudaMemcpy(&cardinality, device_cardinalities_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (cardinality != expected_cardinality) {
            std::fprintf(stderr,
                         "Rank %d: candidate %d changed cardinality (%d != %d)\n",
                         mpi_rank, seed, cardinality, expected_cardinality);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        std::vector<int> members(cardinality);
        CUDA_CHECK(cudaMemcpy(members.data(), device_members_,
                              static_cast<size_t>(cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        return members;
    }

    void markClustered(const std::vector<int>& members) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaMemcpy(device_members_, members.data(),
                              members.size() * sizeof(int), cudaMemcpyHostToDevice));
        const int blocks = static_cast<int>((members.size() + CUDA_BLOCK_SIZE - 1) /
                                            CUDA_BLOCK_SIZE);
        markClusteredKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            device_clustered_, device_members_, static_cast<int>(members.size()));
        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int batchCapacity() const { return batch_capacity_; }
    int device() const { return device_; }

private:
    int point_count_ = 0;
    int device_ = 0;
    int batch_capacity_ = 1;
    Point* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    int* device_members_ = nullptr;
    int* device_seeds_ = nullptr;
    int* device_cardinalities_ = nullptr;
    double* device_scratch_ = nullptr;
    std::vector<int> host_cardinalities_;
};

// One MPI process may be assigned several accelerators.  OpenMP drives those
// CUDA contexts concurrently; with the common one-rank-per-GPU launch policy
// this naturally degenerates to one engine without changing the algorithm.
class HybridCudaQtEngine {
public:
    HybridCudaQtEngine(const std::vector<Point>& points,
                       const int maximum_local_seeds,
                       const std::vector<int>& devices) {
        engines_.resize(devices.size(), nullptr);
        seed_partitions_.resize(devices.size());
        choices_.resize(devices.size());
        const int seeds_per_device = std::max(
            1, (maximum_local_seeds + static_cast<int>(devices.size()) - 1) /
                   static_cast<int>(devices.size()));

#pragma omp parallel for schedule(static) num_threads(devices.size()) if(devices.size() > 1)
        for (size_t i = 0; i < devices.size(); ++i) {
            engines_[i] = new CudaQtEngine(points, seeds_per_device, devices[i]);
            seed_partitions_[i].reserve(seeds_per_device);
        }
    }

    HybridCudaQtEngine(const HybridCudaQtEngine&) = delete;
    HybridCudaQtEngine& operator=(const HybridCudaQtEngine&) = delete;

    ~HybridCudaQtEngine() {
#pragma omp parallel for schedule(static) num_threads(engines_.size()) if(engines_.size() > 1)
        for (size_t i = 0; i < engines_.size(); ++i) {
            delete engines_[i];
        }
    }

    HostChoice findBest(const std::vector<int>& seeds,
                        const double threshold_squared) {
        for (std::vector<int>& partition : seed_partitions_) {
            partition.clear();
        }
        for (size_t i = 0; i < seeds.size(); ++i) {
            seed_partitions_[i % seed_partitions_.size()].push_back(seeds[i]);
        }

#pragma omp parallel for schedule(static) num_threads(engines_.size()) if(engines_.size() > 1)
        for (size_t i = 0; i < engines_.size(); ++i) {
            choices_[i] = engines_[i]->findBest(seed_partitions_[i],
                                                threshold_squared);
        }

        HostChoice best;
        for (const HostChoice& choice : choices_) {
            if (betterHostChoice(choice, best)) {
                best = choice;
            }
        }
        return best;
    }

    std::vector<int> buildMembers(const int seed, const int expected_cardinality,
                                  const double threshold_squared) {
        return engines_.front()->buildMembers(seed, expected_cardinality,
                                              threshold_squared);
    }

    void markClustered(const std::vector<int>& members) {
#pragma omp parallel for schedule(static) num_threads(engines_.size()) if(engines_.size() > 1)
        for (size_t i = 0; i < engines_.size(); ++i) {
            engines_[i]->markClustered(members);
        }
    }

    void synchronize() {
#pragma omp parallel for schedule(static) num_threads(engines_.size()) if(engines_.size() > 1)
        for (size_t i = 0; i < engines_.size(); ++i) {
            engines_[i]->synchronize();
        }
    }

    int minimumBatchCapacity() const {
        int capacity = INT_MAX;
        for (const CudaQtEngine* engine : engines_) {
            capacity = std::min(capacity, engine->batchCapacity());
        }
        return capacity;
    }

    size_t deviceCount() const { return engines_.size(); }

private:
    std::vector<CudaQtEngine*> engines_;
    std::vector<std::vector<int>> seed_partitions_;
    std::vector<HostChoice> choices_;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, const int world_size,
                                  const int world_rank, HybridCudaQtEngine& engine) {
    const int point_count = static_cast<int>(points.size());
    const double threshold_squared = threshold * threshold;
    std::vector<unsigned char> clustered(point_count);
    std::vector<Cluster> clusters;
    int remaining = point_count;

    // This parallel initialization is intentionally unconditional: every MPI
    // process employs its host cores as part of the hybrid execution.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < point_count; ++i) {
        clustered[i] = 0;
    }

    std::vector<int> local_seeds;
    local_seeds.reserve((point_count + world_size - 1) / world_size);

    while (remaining > 0) {
        local_seeds.clear();
        for (int seed = world_rank; seed < point_count; seed += world_size) {
            if (clustered[seed] == 0) {
                local_seeds.push_back(seed);
            }
        }

        const HostChoice local_best = engine.findBest(local_seeds,
                                                      threshold_squared);
        struct {
            int value;
            int index;
        } local_pair{local_best.cardinality, local_best.seed}, global_pair{-1,
                                                                           INT_MAX};
        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        if (global_pair.index == INT_MAX || global_pair.value <= 0) {
            break;
        }

        const int owner = global_pair.index % world_size;
        std::vector<int> members;
        if (world_rank == owner) {
            members = engine.buildMembers(global_pair.index, global_pair.value,
                                          threshold_squared);
        } else {
            members.resize(global_pair.value);
        }
        MPI_Bcast(members.data(), global_pair.value, MPI_INT, owner,
                  MPI_COMM_WORLD);

        engine.markClustered(members);
#pragma omp parallel for schedule(static) if(global_pair.value >= 256)
        for (int i = 0; i < global_pair.value; ++i) {
            clustered[members[i]] = 1;
        }
        remaining -= global_pair.value;
        clusters.push_back(Cluster{std::move(members), global_pair.index});
    }

    engine.synchronize();
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points, const double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const Cluster& cluster = clusters[c];
        double max_diameter = 0.0;
        const long long count = static_cast<long long>(cluster.members.size());

#pragma omp parallel for reduction(max : max_diameter) schedule(static) if(count >= 64)
        for (long long i = 0; i < count; ++i) {
            double row_max = 0.0;
            for (long long j = i + 1; j < count; ++j) {
                row_max = std::max(
                    row_max,
                    distance(points[cluster.members[i]], points[cluster.members[j]]));
            }
            max_diameter = std::max(max_diameter, row_max);
        }

        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c,
                        cluster.members.size(), cluster.seed_point, max_diameter);
        }
        if (max_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        c, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (const int member : clusters[c].members) {
            if (membership[member] >= 0) {
                std::printf(
                    "ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                    member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
#pragma omp parallel for reduction(+ : clustered_count) schedule(static)
    for (size_t i = 0; i < membership.size(); ++i) {
        clustered_count += membership[i] >= 0;
    }
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count, points.size() - clustered_count);
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

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "MPI implementation lacks MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    omp_set_dynamic(0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
    bool help = false;
    bool arguments_valid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (mpi_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
            break;
        }
    }

    if (help || !arguments_valid) {
        if (mpi_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank,
                        MPI_INFO_NULL, &local_communicator);
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(local_communicator, &local_rank);
    MPI_Comm_size(local_communicator, &local_size);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "The hybrid benchmark requires a CUDA GPU\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> assigned_devices;
    if (local_size <= device_count) {
        for (int device = local_rank; device < device_count; device += local_size) {
            assigned_devices.push_back(device);
        }
    } else {
        assigned_devices.push_back(local_rank % device_count);
    }
    CUDA_CHECK(cudaSetDevice(assigned_devices.front()));
    CUDA_CHECK(cudaFree(nullptr));

    if (mpi_rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, %zu CUDA GPU(s) on rank 0\n",
                    world_size, omp_get_max_threads(), assigned_devices.size());
    }

    std::vector<Point> points(num_points);
    if (mpi_rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(reinterpret_cast<double*>(points.data()), 2 * num_points, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

    const int maximum_local_seeds =
        (num_points + world_size - 1) / world_size;
    HybridCudaQtEngine engine(points, maximum_local_seeds, assigned_devices);
    if (mpi_rank == 0) {
        std::printf("CUDA seed batch capacity: %d per GPU (rank 0)\n",
                    engine.minimumBatchCapacity());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, world_size, mpi_rank, engine);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(cluster_end - cluster_start).count();
    double cluster_seconds = 0.0;
    MPI_Reduce(&local_seconds, &cluster_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int result = 0;
    if (mpi_rank == 0) {
        const long long cluster_milliseconds =
            static_cast<long long>(cluster_seconds * 1000.0);
        std::printf("Clustering time: %lld ms\n", cluster_milliseconds);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int maximum_cluster_size = 0;
#pragma omp parallel for reduction(+ : total_clustered) reduction(max : maximum_cluster_size) schedule(static)
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            maximum_cluster_size = std::max(maximum_cluster_size, size);
        }

        const double average_cluster_size =
            clusters.empty()
                ? 0.0
                : static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
                    num_points, 100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", maximum_cluster_size);

        const double clusters_per_second =
            cluster_seconds > 0.0 ? clusters.size() / cluster_seconds : 0.0;
        const double points_per_second =
            cluster_seconds > 0.0 ? num_points / cluster_seconds : 0.0;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters_per_second, points_per_second);

        if (print_results_requested) {
            std::vector<double> membership_data(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
#pragma omp parallel for schedule(static)
            for (int i = 0; i < num_points; ++i) {
                membership_data[i] = static_cast<double>(membership[i]);
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&local_communicator);
    MPI_Finalize();
    return result;
}
