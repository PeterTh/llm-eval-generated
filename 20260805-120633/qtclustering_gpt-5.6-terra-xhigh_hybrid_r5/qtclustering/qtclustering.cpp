// QT clustering benchmark -- MPI + OpenMP + CUDA implementation.
//
// The QT choice made in every round is inherently sequential: selecting a
// cluster changes the candidate set for the following round.  Within a round,
// however, the candidate cluster for every remaining seed is independent.
// MPI distributes those seeds over GPUs, CUDA constructs them in batches, and
// OpenMP updates/compacts the replicated host-side candidate set.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;
static constexpr int MAX_SEEDS_PER_BATCH = 256;
static constexpr int PROGRESS_CHECK_INTERVAL = 8;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

struct SeedChoice {
    int cardinality;
    int seed;
};

// MPI_MAX chooses the largest cardinality first and, within that cardinality,
// the complemented seed value.  The latter makes the smallest original seed
// win without a per-rank gather.
static std::uint64_t packSeedChoice(const SeedChoice choice) {
    if (choice.cardinality < 0) {
        return 0;
    }
    const auto cardinality = static_cast<std::uint32_t>(choice.cardinality);
    const auto inverse_seed = UINT32_MAX - static_cast<std::uint32_t>(choice.seed);
    return (static_cast<std::uint64_t>(cardinality) << 32) | inverse_seed;
}

static inline int divUp(const int value, const int divisor) {
    return (value + divisor - 1) / divisor;
}

static void mpiAbort(const int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

static void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

// Generate the exact deterministic data set used by the original benchmark.
static void generateSyntheticData(std::vector<Point>& points, const int N,
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
            points[count++] = {x, y};
            --group_cnt;
        }
    }
}

static inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Each score is the maximum distance of a candidate from the current cluster.
// Appending one member only requires a max update, which is algebraically
// equivalent to the original rescan of all current members.
__global__ void initializeScoresKernel(double* scores, const Point* points,
                                       const unsigned char* clustered,
                                       const int* seeds, int point_count) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.y;
    if (point >= point_count) {
        return;
    }

    const int seed = seeds[slot];
    double score = CUDART_INF;
    if (!clustered[point] && point != seed) {
        const double dx = points[point].x - points[seed].x;
        const double dy = points[point].y - points[seed].y;
        score = sqrt(dx * dx + dy * dy);
    }
    scores[static_cast<size_t>(slot) * point_count + point] = score;
}

__device__ __forceinline__ bool isBetterCandidate(const double value, const int index,
                                                   const double best_value,
                                                   const int best_index) {
    return value < best_value || (value == best_value && index < best_index);
}

// One block reduces the candidates for one seed.  Index is part of the
// reduction so equal diameters retain the first candidate in original order.
__global__ void selectClosestKernel(const double* scores, int point_count,
                                    double* best_values, int* best_indices) {
    const int slot = blockIdx.x;
    const double* seed_scores = scores + static_cast<size_t>(slot) * point_count;

    double local_value = CUDART_INF;
    int local_index = INT_MAX;
    for (int point = threadIdx.x; point < point_count; point += blockDim.x) {
        const double value = seed_scores[point];
        if (isBetterCandidate(value, point, local_value, local_index)) {
            local_value = value;
            local_index = point;
        }
    }

    extern __shared__ unsigned char reduction_storage[];
    double* shared_values = reinterpret_cast<double*>(reduction_storage);
    int* shared_indices = reinterpret_cast<int*>(shared_values + blockDim.x);
    shared_values[threadIdx.x] = local_value;
    shared_indices[threadIdx.x] = local_index;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset &&
            isBetterCandidate(shared_values[threadIdx.x + offset],
                              shared_indices[threadIdx.x + offset],
                              shared_values[threadIdx.x], shared_indices[threadIdx.x])) {
            shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
            shared_indices[threadIdx.x] = shared_indices[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        best_values[slot] = shared_values[0];
        best_indices[slot] = shared_indices[0];
    }
}

__global__ void appendAndUpdateKernel(double* scores, const Point* points,
                                      int point_count, const double threshold,
                                      const double* best_values,
                                      const int* best_indices, int* cardinalities,
                                      int* members) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.y;
    const double best_value = best_values[slot];
    if (!(best_value < threshold)) {
        return;
    }

    const int selected = best_indices[slot];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int position = cardinalities[slot];
        if (members != nullptr) {
            members[static_cast<size_t>(slot) * point_count + position] = selected;
        }
        cardinalities[slot] = position + 1;
    }

    if (point >= point_count) {
        return;
    }

    double* const score = scores + static_cast<size_t>(slot) * point_count + point;
    if (point == selected) {
        *score = CUDART_INF;
        return;
    }

    const double old_value = *score;
    if (old_value != CUDART_INF) {
        const double dx = points[point].x - points[selected].x;
        const double dy = points[point].y - points[selected].y;
        const double new_distance = sqrt(dx * dx + dy * dy);
        *score = fmax(old_value, new_distance);
    }
}

class CudaCandidateEvaluator {
public:
    CudaCandidateEvaluator(const std::vector<Point>& points, const int mpi_rank)
        : point_count_(static_cast<int>(points.size())), rank_(mpi_rank) {
        int device_count = 0;
        checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank_);
        if (device_count == 0) {
            mpiAbort(rank_, "no CUDA accelerator is available");
        }

        MPI_Comm node_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL,
                            &node_comm);
        int local_rank = 0;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);

        // Accelerator-cluster launchers conventionally place one MPI rank per GPU.
        // Modulo keeps the program correct if a node is intentionally oversubscribed.
        checkCuda(cudaSetDevice(local_rank % device_count), "cudaSetDevice", rank_);

        size_t free_memory = 0;
        size_t total_memory = 0;
        checkCuda(cudaMemGetInfo(&free_memory, &total_memory), "cudaMemGetInfo", rank_);
        (void)total_memory;

        const size_t score_bytes_per_seed =
            static_cast<size_t>(point_count_) * sizeof(double);
        const size_t usable_score_memory = free_memory / 2;
        const size_t memory_limited_batch = score_bytes_per_seed == 0 ? 1 :
            usable_score_memory / score_bytes_per_seed;
        if (memory_limited_batch == 0) {
            mpiAbort(rank_, "not enough GPU memory for one candidate cluster");
        }
        batch_capacity_ = static_cast<int>(std::min<size_t>(
            {static_cast<size_t>(MAX_SEEDS_PER_BATCH), memory_limited_batch,
             static_cast<size_t>(std::numeric_limits<int>::max())}));

        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags", rank_);
        checkCuda(cudaMalloc(&d_points_, static_cast<size_t>(point_count_) * sizeof(Point)),
                  "cudaMalloc(points)", rank_);
        checkCuda(cudaMalloc(&d_clustered_, static_cast<size_t>(point_count_)),
                  "cudaMalloc(clustered)", rank_);
        checkCuda(cudaMalloc(&d_scores_, static_cast<size_t>(batch_capacity_) * point_count_ *
                                          sizeof(double)),
                  "cudaMalloc(scores)", rank_);
        checkCuda(cudaMalloc(&d_seeds_, static_cast<size_t>(batch_capacity_) * sizeof(int)),
                  "cudaMalloc(seeds)", rank_);
        checkCuda(cudaMalloc(&d_best_values_, static_cast<size_t>(batch_capacity_) *
                                                sizeof(double)),
                  "cudaMalloc(best values)", rank_);
        checkCuda(cudaMalloc(&d_best_indices_, static_cast<size_t>(batch_capacity_) *
                                                 sizeof(int)),
                  "cudaMalloc(best indices)", rank_);
        checkCuda(cudaMalloc(&d_cardinalities_, static_cast<size_t>(batch_capacity_) *
                                                   sizeof(int)),
                  "cudaMalloc(cardinalities)", rank_);
        checkCuda(cudaMalloc(&d_members_, static_cast<size_t>(point_count_) * sizeof(int)),
                  "cudaMalloc(members)", rank_);

        checkCuda(cudaMemcpyAsync(d_points_, points.data(),
                                  static_cast<size_t>(point_count_) * sizeof(Point),
                                  cudaMemcpyHostToDevice, stream_),
                  "cudaMemcpyAsync(points)", rank_);
        checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(points)", rank_);
        host_best_values_.resize(batch_capacity_);
        host_cardinalities_.resize(batch_capacity_);
    }

    CudaCandidateEvaluator(const CudaCandidateEvaluator&) = delete;
    CudaCandidateEvaluator& operator=(const CudaCandidateEvaluator&) = delete;

    ~CudaCandidateEvaluator() {
        if (d_members_) cudaFree(d_members_);
        if (d_cardinalities_) cudaFree(d_cardinalities_);
        if (d_best_indices_) cudaFree(d_best_indices_);
        if (d_best_values_) cudaFree(d_best_values_);
        if (d_seeds_) cudaFree(d_seeds_);
        if (d_scores_) cudaFree(d_scores_);
        if (d_clustered_) cudaFree(d_clustered_);
        if (d_points_) cudaFree(d_points_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void setClustered(const std::vector<unsigned char>& clustered) {
        checkCuda(cudaMemcpyAsync(d_clustered_, clustered.data(),
                                  static_cast<size_t>(point_count_),
                                  cudaMemcpyHostToDevice, stream_),
                  "cudaMemcpyAsync(clustered)", rank_);
    }

    SeedChoice chooseBestSeed(const int* seeds, const int seed_count,
                               const double threshold) {
        SeedChoice choice{-1, -1};
        for (int first = 0; first < seed_count; first += batch_capacity_) {
            const int batch_size = std::min(batch_capacity_, seed_count - first);
            runBatch(seeds + first, batch_size, threshold, nullptr);
            for (int slot = 0; slot < batch_size; ++slot) {
                if (host_cardinalities_[slot] > choice.cardinality) {
                    choice = {host_cardinalities_[slot], seeds[first + slot]};
                }
            }
        }
        return choice;
    }

    std::vector<int> buildCluster(const int seed, const double threshold) {
        runBatch(&seed, 1, threshold, d_members_);
        const int size = host_cardinalities_[0];
        std::vector<int> members(size);
        if (size > 0) {
            members[0] = seed;
        }
        if (size > 1) {
            checkCuda(cudaMemcpyAsync(members.data() + 1, d_members_ + 1,
                                      static_cast<size_t>(size - 1) * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_),
                      "cudaMemcpyAsync(members)", rank_);
            checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(members)", rank_);
        }
        return members;
    }

private:
    void runBatch(const int* host_seeds, const int batch_size, const double threshold,
                  int* output_members) {
        checkCuda(cudaMemcpyAsync(d_seeds_, host_seeds,
                                  static_cast<size_t>(batch_size) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_),
                  "cudaMemcpyAsync(seeds)", rank_);
        const dim3 candidate_grid(divUp(point_count_, CUDA_THREADS), batch_size);
        initializeScoresKernel<<<candidate_grid, CUDA_THREADS, 0, stream_>>>(
            d_scores_, d_points_, d_clustered_, d_seeds_, point_count_);
        checkCuda(cudaPeekAtLastError(), "initializeScoresKernel launch", rank_);

        // A seed is always the first member, just as in the sequential version.
        // Initialize every slot; batches contain independent candidate seeds.
        std::fill_n(host_cardinalities_.begin(), batch_size, 1);
        checkCuda(cudaMemcpyAsync(d_cardinalities_, host_cardinalities_.data(),
                                  static_cast<size_t>(batch_size) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_),
                  "cudaMemcpyAsync(initial cardinalities)", rank_);

        const size_t reduction_bytes = CUDA_THREADS * (sizeof(double) + sizeof(int));
        for (int step = 1; step < point_count_; ++step) {
            selectClosestKernel<<<batch_size, CUDA_THREADS, reduction_bytes, stream_>>>(
                d_scores_, point_count_, d_best_values_, d_best_indices_);
            checkCuda(cudaPeekAtLastError(), "selectClosestKernel launch", rank_);

            appendAndUpdateKernel<<<candidate_grid, CUDA_THREADS, 0, stream_>>>(
                d_scores_, d_points_, point_count_, threshold, d_best_values_, d_best_indices_,
                d_cardinalities_, output_members);
            checkCuda(cudaPeekAtLastError(), "appendAndUpdateKernel launch", rank_);

            // Checking at intervals amortizes host/device synchronization.  The extra
            // no-op iterations after a finished cluster touch no point data.
            if ((step % PROGRESS_CHECK_INTERVAL) == 0 || step + 1 == point_count_) {
                checkCuda(cudaMemcpyAsync(host_best_values_.data(), d_best_values_,
                                          static_cast<size_t>(batch_size) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_),
                          "cudaMemcpyAsync(best values)", rank_);
                checkCuda(cudaStreamSynchronize(stream_),
                          "cudaStreamSynchronize(candidate batch)", rank_);
                bool has_active_candidate = false;
                for (int slot = 0; slot < batch_size; ++slot) {
                    has_active_candidate |= host_best_values_[slot] < threshold;
                }
                if (!has_active_candidate) {
                    break;
                }
            }
        }

        checkCuda(cudaMemcpyAsync(host_cardinalities_.data(), d_cardinalities_,
                                  static_cast<size_t>(batch_size) * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream_),
                  "cudaMemcpyAsync(cardinalities)", rank_);
        checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(cardinalities)", rank_);
    }

    int point_count_;
    int rank_;
    int batch_capacity_ = 1;
    cudaStream_t stream_ = nullptr;
    Point* d_points_ = nullptr;
    unsigned char* d_clustered_ = nullptr;
    double* d_scores_ = nullptr;
    int* d_seeds_ = nullptr;
    double* d_best_values_ = nullptr;
    int* d_best_indices_ = nullptr;
    int* d_cardinalities_ = nullptr;
    int* d_members_ = nullptr;
    std::vector<double> host_best_values_;
    std::vector<int> host_cardinalities_;
};

static void markAndCompact(std::vector<unsigned char>& clustered,
                           const std::vector<int>& new_members,
                           std::vector<int>& unclustered_indices,
                           std::vector<int>& compaction_buffer) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(new_members.size()); ++i) {
        clustered[new_members[i]] = 1;
    }

    // Preserve ascending seed order while using all CPU threads.  This matters
    // for the original first-seed tie breaking rule.
    const int thread_count = omp_get_max_threads();
    std::vector<int> offsets(thread_count + 1, 0);
    const int old_size = static_cast<int>(unclustered_indices.size());
    // Use a distinct reusable buffer so compaction cannot overwrite input that
    // another OpenMP thread has not scanned yet.
    compaction_buffer.resize(old_size);
    #pragma omp parallel num_threads(thread_count)
    {
        const int tid = omp_get_thread_num();
        const int begin = static_cast<int>((static_cast<long long>(old_size) * tid) /
                                           thread_count);
        const int end = static_cast<int>((static_cast<long long>(old_size) * (tid + 1)) /
                                         thread_count);
        int local_count = 0;
        for (int i = begin; i < end; ++i) {
            local_count += !clustered[unclustered_indices[i]];
        }
        offsets[tid + 1] = local_count;
        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= thread_count; ++i) {
                offsets[i] += offsets[i - 1];
            }
        }
        #pragma omp barrier
        int output = offsets[tid];
        for (int i = begin; i < end; ++i) {
            const int point = unclustered_indices[i];
            if (!clustered[point]) {
                compaction_buffer[output++] = point;
            }
        }
    }
    compaction_buffer.resize(offsets.back());
    unclustered_indices.swap(compaction_buffer);
}

static std::vector<Cluster> qtClusteringHybrid(const std::vector<Point>& points,
                                                const double threshold,
                                                const int mpi_rank,
                                                const int mpi_size) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    for (int point = 0; point < point_count; ++point) {
        unclustered_indices[point] = point;
    }

    CudaCandidateEvaluator evaluator(points, mpi_rank);
    std::vector<Cluster> clusters;
    std::vector<int> compaction_buffer;
    compaction_buffer.reserve(point_count);

    while (!unclustered_indices.empty()) {
        evaluator.setClustered(clustered);
        const int local_begin = static_cast<int>(
            static_cast<long long>(unclustered_indices.size()) * mpi_rank / mpi_size);
        const int local_end = static_cast<int>(
            static_cast<long long>(unclustered_indices.size()) * (mpi_rank + 1) / mpi_size);
        const int local_seed_count = local_end - local_begin;
        const SeedChoice local_choice = evaluator.chooseBestSeed(
            unclustered_indices.data() + local_begin, local_seed_count, threshold);

        const std::uint64_t local_choice_key = packSeedChoice(local_choice);
        std::uint64_t global_choice_key = 0;
        MPI_Allreduce(&local_choice_key, &global_choice_key, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);
        const SeedChoice global_choice{
            static_cast<int>(global_choice_key >> 32),
            static_cast<int>(UINT32_MAX - static_cast<std::uint32_t>(global_choice_key))};
        const int local_owner = local_choice.seed == global_choice.seed ? mpi_rank : INT_MAX;
        int owner_rank = INT_MAX;
        MPI_Allreduce(&local_owner, &owner_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (global_choice.seed < 0 || global_choice.cardinality <= 0) {
            mpiAbort(mpi_rank, "could not select a QT cluster");
        }

        std::vector<int> winning_members;
        if (mpi_rank == owner_rank) {
            winning_members = evaluator.buildCluster(global_choice.seed, threshold);
            if (static_cast<int>(winning_members.size()) != global_choice.cardinality) {
                mpiAbort(mpi_rank, "CUDA candidate reconstruction disagrees with selection");
            }
        }
        int member_count = static_cast<int>(winning_members.size());
        MPI_Bcast(&member_count, 1, MPI_INT, owner_rank, MPI_COMM_WORLD);
        if (mpi_rank != owner_rank) {
            winning_members.resize(member_count);
        }
        MPI_Bcast(winning_members.data(), member_count, MPI_INT, owner_rank, MPI_COMM_WORLD);

        markAndCompact(clustered, winning_members, unclustered_indices, compaction_buffer);
        if (mpi_rank == 0) {
            clusters.push_back({std::move(winning_members), global_choice.seed});
        }
    }
    return clusters;
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points,
                             const double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const Cluster& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
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
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (const int member : clusters[c].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    const int clustered_count = static_cast<int>(std::count_if(
        membership.begin(), membership.end(), [](const int value) { return value >= 0; }));
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
                clustered_count, points.size() - clustered_count);
    return valid;
}

static void printUsage(const char* prog_name) {
    std::printf("Usage: %s [options]\n", prog_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int mpi_rank = 0;
    int mpi_size = 1;
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        mpiAbort(mpi_rank, "MPI does not provide MPI_THREAD_FUNNELED support");
    }
    omp_set_dynamic(0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_flag = false;
    bool show_help = false;
    bool bad_option = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_flag = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (mpi_rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            bad_option = true;
        }
    }
    if (show_help || bad_option) {
        if (mpi_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad_option ? 1 : 0;
    }
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks, CUDA per rank, OpenMP host work\n",
                    mpi_size);
    }

    std::vector<Point> points(num_points);
    if (mpi_rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), static_cast<int>(points.size() * sizeof(Point)), MPI_BYTE, 0,
              MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters =
        qtClusteringHybrid(points, threshold, mpi_rank, mpi_size);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    const long long local_cluster_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            cluster_end - cluster_start).count();
    long long cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_LONG_LONG_INT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::printf("Clustering time: %lld ms\n", cluster_time);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
        const double average_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
                    100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);
        const double time_sec = cluster_time / 1000.0;
        const double clusters_per_sec = time_sec > 0.0 ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = time_sec > 0.0 ? num_points / time_sec : 0.0;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec,
                    points_per_sec);

        if (print_results_flag) {
            std::vector<double> membership_data;
            membership_data.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
            for (const int membership_id : membership) {
                membership_data.push_back(static_cast<double>(membership_id));
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
