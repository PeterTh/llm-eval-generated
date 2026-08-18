// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA implementation
//
// MPI distributes independent seed clusters across accelerators.  A CUDA
// block constructs one candidate cluster, using all of its threads to find
// the next member.  OpenMP handles the host-side batch preparation and
// reduction.  The greedy choices and tie breaking are identical to the
// original sequential implementation.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_BLOCK_SIZE = 256;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

static_assert(std::is_standard_layout_v<Point>);
static_assert(sizeof(Point) == 2 * sizeof(double));

[[noreturn]] void abortCuda(const cudaError_t error, const char* expression,
                            const char* file, const int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d for %s: %s\n",
                 rank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

[[noreturn]] void abortMpi(const int error, const char* expression,
                           const char* file, const int line) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: MPI failure at %s:%d for %s: %.*s\n",
                 rank, file, line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            abortCuda(cuda_check_error, #expression, __FILE__, __LINE__);        \
        }                                                                        \
    } while (false)

#define MPI_CHECK(expression)                                                    \
    do {                                                                         \
        const int mpi_check_error = (expression);                                \
        if (mpi_check_error != MPI_SUCCESS) {                                    \
            abortMpi(mpi_check_error, #expression, __FILE__, __LINE__);          \
        }                                                                        \
    } while (false)

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

__device__ __forceinline__ double deviceDistance(const Point& p1,
                                                  const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances,
                                    const int point_count,
                                    const size_t element_count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    while (index < element_count) {
        const int row = static_cast<int>(index / point_count);
        const int column = static_cast<int>(index -
                                            static_cast<size_t>(row) * point_count);
        distances[index] = deviceDistance(points[row], points[column]);
        index += stride;
    }
}

// Each block builds a complete candidate cluster.  max_distances stores the
// current diameter that would result from adding each point.  Updating it
// when a member is added is equivalent to rescanning all existing members,
// but changes the work per seed from cubic to quadratic in the point count.
__global__ void buildCandidateClusters(
    const Point* __restrict__ points,
    const double* __restrict__ distance_matrix,
    const unsigned char* __restrict__ clustered,
    const int* __restrict__ seeds,
    int* __restrict__ cardinalities,
    int* __restrict__ member_output,
    double* __restrict__ max_distances,
    const double threshold,
    const int point_count) {
    __shared__ double reduction_values[CUDA_BLOCK_SIZE];
    __shared__ int reduction_indices[CUDA_BLOCK_SIZE];
    __shared__ int cardinality;

    const int tid = threadIdx.x;
    const int seed = seeds[blockIdx.x];
    double* const seed_distances =
        max_distances + static_cast<size_t>(blockIdx.x) * point_count;

    if (tid == 0) {
        cardinality = 1;
        if (member_output != nullptr) {
            member_output[0] = seed;
        }
    }

    for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
        double candidate_distance = CUDART_INF;
        if (!clustered[candidate] && candidate != seed) {
            candidate_distance = distance_matrix != nullptr
                ? distance_matrix[static_cast<size_t>(seed) * point_count + candidate]
                : deviceDistance(points[seed], points[candidate]);
            // A diameter only increases as members are added.  Permanently
            // discard a point as soon as it violates the strict threshold.
            if (!(candidate_distance < threshold)) {
                candidate_distance = CUDART_INF;
            }
        }
        seed_distances[candidate] = candidate_distance;
    }
    __syncthreads();

    while (true) {
        double local_value = CUDART_INF;
        int local_index = -1;

        for (int candidate = tid; candidate < point_count;
             candidate += blockDim.x) {
            const double value = seed_distances[candidate];
            if (value < local_value ||
                (value == local_value && value < CUDART_INF &&
                 (local_index < 0 || candidate < local_index))) {
                local_value = value;
                local_index = candidate;
            }
        }

        reduction_values[tid] = local_value;
        reduction_indices[tid] = local_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                const double other_value = reduction_values[tid + offset];
                const int other_index = reduction_indices[tid + offset];
                const int current_index = reduction_indices[tid];
                if (other_index >= 0 &&
                    (current_index < 0 || other_value < reduction_values[tid] ||
                     (other_value == reduction_values[tid] &&
                      other_index < current_index))) {
                    reduction_values[tid] = other_value;
                    reduction_indices[tid] = other_index;
                }
            }
            __syncthreads();
        }

        const int closest = reduction_indices[0];
        if (closest < 0) {
            break;
        }

        if (tid == 0) {
            if (member_output != nullptr) {
                member_output[cardinality] = closest;
            }
            ++cardinality;
        }

        for (int candidate = tid; candidate < point_count;
             candidate += blockDim.x) {
            double current = seed_distances[candidate];
            if (candidate == closest) {
                current = CUDART_INF;
            } else if (current < CUDART_INF) {
                const double next_distance = distance_matrix != nullptr
                    ? distance_matrix[static_cast<size_t>(closest) * point_count +
                                      candidate]
                    : deviceDistance(points[closest], points[candidate]);
                if (!(next_distance < threshold)) {
                    current = CUDART_INF;
                } else if (next_distance > current) {
                    current = next_distance;
                }
            }
            seed_distances[candidate] = current;
        }
        __syncthreads();
    }

    if (tid == 0) {
        cardinalities[blockIdx.x] = cardinality;
    }
}

class CudaCandidateEngine {
public:
    CudaCandidateEngine(const std::vector<Point>& points, const int max_local_seeds)
        : point_count_(static_cast<int>(points.size())) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points_),
                              points.size() * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_clustered_),
                              points.size() * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_best_members_),
                              points.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_points_, points.data(), points.size() * sizeof(Point),
                              cudaMemcpyHostToDevice));

        allocateDistanceMatrix();

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;

        const size_t row_bytes = points.size() * sizeof(double);
        size_t requested_capacity = std::min<size_t>(
            static_cast<size_t>(max_local_seeds),
            std::max<size_t>(1, (free_bytes * 3 / 4) / row_bytes));

        cudaError_t allocation_error = cudaErrorMemoryAllocation;
        while (requested_capacity > 0) {
            allocation_error = cudaMalloc(
                reinterpret_cast<void**>(&d_max_distances_),
                requested_capacity * row_bytes);
            if (allocation_error == cudaSuccess) {
                break;
            }
            cudaGetLastError();
            requested_capacity /= 2;
        }
        if (allocation_error != cudaSuccess || requested_capacity == 0) {
            abortCuda(allocation_error, "cudaMalloc(candidate workspace)",
                      __FILE__, __LINE__);
        }
        batch_capacity_ = static_cast<int>(requested_capacity);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_seeds_),
                              requested_capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cardinalities_),
                              requested_capacity * sizeof(int)));
    }

    CudaCandidateEngine(const CudaCandidateEngine&) = delete;
    CudaCandidateEngine& operator=(const CudaCandidateEngine&) = delete;

    ~CudaCandidateEngine() {
        cudaFree(d_cardinalities_);
        cudaFree(d_seeds_);
        cudaFree(d_max_distances_);
        cudaFree(d_distance_matrix_);
        cudaFree(d_best_members_);
        cudaFree(d_clustered_);
        cudaFree(d_points_);
    }

    int batchCapacity() const { return batch_capacity_; }

    void setClustered(const std::vector<unsigned char>& clustered) {
        CUDA_CHECK(cudaMemcpy(d_clustered_, clustered.data(),
                              clustered.size() * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    void evaluate(const int* seeds, const int count, const double threshold,
                  std::vector<int>& cardinalities) {
        if (count <= 0 || count > batch_capacity_) {
            return;
        }
        CUDA_CHECK(cudaMemcpy(d_seeds_, seeds, static_cast<size_t>(count) * sizeof(int),
                              cudaMemcpyHostToDevice));
        buildCandidateClusters<<<count, CUDA_BLOCK_SIZE>>>(
            d_points_, d_distance_matrix_, d_clustered_, d_seeds_,
            d_cardinalities_, nullptr, d_max_distances_, threshold, point_count_);
        CUDA_CHECK(cudaGetLastError());

        cardinalities.resize(count);
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_cardinalities_,
                              static_cast<size_t>(count) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    void buildMembers(const int seed, const int expected_cardinality,
                      const double threshold, std::vector<int>& members) {
        CUDA_CHECK(cudaMemcpy(d_seeds_, &seed, sizeof(int), cudaMemcpyHostToDevice));
        buildCandidateClusters<<<1, CUDA_BLOCK_SIZE>>>(
            d_points_, d_distance_matrix_, d_clustered_, d_seeds_,
            d_cardinalities_, d_best_members_, d_max_distances_, threshold,
            point_count_);
        CUDA_CHECK(cudaGetLastError());

        int cardinality = 0;
        CUDA_CHECK(cudaMemcpy(&cardinality, d_cardinalities_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (cardinality != expected_cardinality) {
            int rank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::fprintf(stderr,
                         "Rank %d: non-deterministic candidate size for seed %d "
                         "(%d != %d)\n",
                         rank, seed, cardinality, expected_cardinality);
            MPI_Abort(MPI_COMM_WORLD, 2);
            std::abort();
        }

        members.resize(cardinality);
        CUDA_CHECK(cudaMemcpy(members.data(), d_best_members_,
                              static_cast<size_t>(cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

private:
    void allocateDistanceMatrix() {
        const size_t n = static_cast<size_t>(point_count_);
        if (n == 0 || n > std::numeric_limits<size_t>::max() / n ||
            n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
            return;
        }

        const size_t elements = n * n;
        const size_t bytes = elements * sizeof(double);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;

        // Keep enough memory for many concurrent seed workspaces.  If the
        // all-pairs cache is too large, distances are computed on demand.
        if (bytes > free_bytes * 2 / 3) {
            return;
        }

        const cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&d_distance_matrix_), bytes);
        if (error != cudaSuccess) {
            d_distance_matrix_ = nullptr;
            cudaGetLastError();
            return;
        }

        const size_t required_blocks =
            (elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<size_t>(required_blocks, 65535));
        buildDistanceMatrix<<<blocks, CUDA_BLOCK_SIZE>>>(
            d_points_, d_distance_matrix_, point_count_, elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int point_count_ = 0;
    int batch_capacity_ = 0;
    Point* d_points_ = nullptr;
    unsigned char* d_clustered_ = nullptr;
    double* d_distance_matrix_ = nullptr;
    double* d_max_distances_ = nullptr;
    int* d_seeds_ = nullptr;
    int* d_cardinalities_ = nullptr;
    int* d_best_members_ = nullptr;
};

struct SeedChoice {
    int cardinality;
    int seed;
};

static_assert(sizeof(SeedChoice) == 2 * sizeof(int));

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, MPI_Comm communicator) {
    int rank = 0;
    int rank_count = 1;
    MPI_CHECK(MPI_Comm_rank(communicator, &rank));
    MPI_CHECK(MPI_Comm_size(communicator, &rank_count));

    const int point_count = static_cast<int>(points.size());
    const int max_local_seeds = (point_count + rank_count - 1) / rank_count;
    CudaCandidateEngine engine(points, max_local_seeds);

    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    std::vector<Cluster> clusters;
    std::vector<int> local_seeds;
    std::vector<int> batch_cardinalities;
    local_seeds.reserve(max_local_seeds);
    batch_cardinalities.reserve(engine.batchCapacity());

    #pragma omp parallel for schedule(static) if(point_count >= 1024)
    for (int i = 0; i < point_count; ++i) {
        unclustered_indices[i] = i;
    }

    while (!unclustered_indices.empty()) {
        const size_t active_count = unclustered_indices.size();
        const size_t local_count = active_count <= static_cast<size_t>(rank)
            ? 0
            : 1 + (active_count - 1 - static_cast<size_t>(rank)) /
                      static_cast<size_t>(rank_count);
        local_seeds.resize(local_count);

        #pragma omp parallel for schedule(static) if(local_count >= 512)
        for (size_t i = 0; i < local_count; ++i) {
            local_seeds[i] = unclustered_indices[
                static_cast<size_t>(rank) + i * static_cast<size_t>(rank_count)];
        }

        engine.setClustered(clustered);
        SeedChoice local_choice{-1, INT_MAX};

        for (size_t offset = 0; offset < local_count;
             offset += static_cast<size_t>(engine.batchCapacity())) {
            const int batch_count = static_cast<int>(std::min<size_t>(
                static_cast<size_t>(engine.batchCapacity()), local_count - offset));
            engine.evaluate(local_seeds.data() + offset, batch_count, threshold,
                            batch_cardinalities);

            int batch_max = -1;
            #pragma omp parallel for reduction(max : batch_max) schedule(static) \
                if(batch_count >= 256)
            for (int i = 0; i < batch_count; ++i) {
                batch_max = std::max(batch_max, batch_cardinalities[i]);
            }

            if (batch_max > local_choice.cardinality) {
                for (int i = 0; i < batch_count; ++i) {
                    if (batch_cardinalities[i] == batch_max) {
                        local_choice = {batch_max,
                                        local_seeds[offset + static_cast<size_t>(i)]};
                        break;
                    }
                }
            }
        }

        SeedChoice global_choice{-1, INT_MAX};
        MPI_CHECK(MPI_Allreduce(&local_choice, &global_choice, 1, MPI_2INT,
                                MPI_MAXLOC, communicator));
        if (global_choice.seed == INT_MAX || global_choice.cardinality <= 0) {
            break;
        }

        const auto position_it = std::lower_bound(unclustered_indices.begin(),
                                                  unclustered_indices.end(),
                                                  global_choice.seed);
        if (position_it == unclustered_indices.end() ||
            *position_it != global_choice.seed) {
            std::fprintf(stderr, "Rank %d: selected seed %d is not active\n",
                         rank, global_choice.seed);
            MPI_Abort(communicator, 3);
            std::abort();
        }
        const size_t position = static_cast<size_t>(
            std::distance(unclustered_indices.begin(), position_it));
        const int owner = static_cast<int>(position % static_cast<size_t>(rank_count));

        std::vector<int> best_members(global_choice.cardinality);
        if (rank == owner) {
            engine.buildMembers(global_choice.seed, global_choice.cardinality,
                                threshold, best_members);
        }
        MPI_CHECK(MPI_Bcast(best_members.data(), global_choice.cardinality, MPI_INT,
                            owner, communicator));

        for (const int member : best_members) {
            clustered[member] = 1;
        }
        clusters.push_back({std::move(best_members), global_choice.seed});
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int index) {
                               return clustered[index] != 0;
                           }),
            unclustered_indices.end());
    }

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        #pragma omp parallel for reduction(max : max_diameter) schedule(static) \
            if(cluster.members.size() >= 64)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                             points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        c, cluster.members.size(), cluster.seed_point, max_diameter);
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
    #pragma omp parallel for reduction(+ : clustered_count) schedule(static) \
        if(membership.size() >= 1024)
    for (size_t i = 0; i < membership.size(); ++i) {
        clustered_count += membership[i] >= 0 ? 1 : 0;
    }

    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count,
                points.size() - static_cast<size_t>(clustered_count));
    return valid;
}

void printUsage(const char* prog_name) {
    std::printf("Usage: %s [options]\n", prog_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int init_error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                                           &provided_thread_level);
    if (init_error != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return 1;
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    int rank = 0;
    int rank_count = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &rank_count));
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 4);
        return 1;
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_enabled = false;
    bool show_help = false;
    bool arguments_valid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_enabled = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
            break;
        }
    }

    if (show_help || !arguments_valid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return show_help ? 0 : 1;
    }
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local_communicator));
    int local_rank = 0;
    int local_rank_count = 1;
    MPI_CHECK(MPI_Comm_rank(local_communicator, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_communicator, &local_rank_count));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (local_rank == 0) {
            std::fprintf(stderr, "No CUDA accelerator is available on this node\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 5);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    CUDA_CHECK(cudaFree(nullptr));

    if (local_rank == 0 && local_rank_count > device_count) {
        std::fprintf(stderr,
                     "Warning: %d MPI ranks share %d CUDA devices on a node; "
                     "one rank per device is recommended\n",
                     local_rank_count, device_count);
    }

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, CUDA enabled\n",
                    rank_count, omp_get_max_threads());
    }

    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Datatype point_type = MPI_DATATYPE_NULL;
    MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &point_type));
    MPI_CHECK(MPI_Type_commit(&point_type));
    MPI_CHECK(MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_free(&point_type));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_CHECK(MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD));

    int result = 0;
    if (rank == 0) {
        const long long cluster_milliseconds =
            static_cast<long long>(cluster_time * 1000.0);
        std::printf("Clustering time: %lld ms\n", cluster_milliseconds);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        #pragma omp parallel for reduction(+ : total_clustered) \
            reduction(max : max_cluster_size) schedule(static) \
            if(clusters.size() >= 128)
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double average_cluster_size = clusters.empty()
            ? 0.0
            : static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
                    num_points, 100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);

        const double safe_time = std::max(cluster_time,
                                          std::numeric_limits<double>::min());
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / safe_time, num_points / safe_time);

        if (print_results_enabled) {
            std::vector<double> membership_data;
            membership_data.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
            for (const int membership_index : membership) {
                membership_data.push_back(static_cast<double>(membership_index));
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }

    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_free(&local_communicator));
    MPI_CHECK(MPI_Finalize());
    return result;
}
