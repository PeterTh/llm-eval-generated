// Hybrid MPI/OpenMP/CUDA QT clustering benchmark.
//
// A QT round evaluates every currently unclustered point as a possible seed.
// Those evaluations are independent, so MPI distributes seeds among ranks and
// CUDA evaluates many seeds concurrently.  A deterministic MPI_MAXLOC selects
// the same winner (including the original lowest-seed tie break) on every rank.
// OpenMP handles rank-local reductions and the optional O(N^2) validation.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

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

struct BestCandidate {
    int cardinality;
    int seed;
};

static bool isBetter(const BestCandidate& a, const BestCandidate& b) {
    return a.cardinality > b.cardinality ||
           (a.cardinality == b.cardinality && a.seed < b.seed);
}

[[noreturn]] static void fatalError(const char* kind, const char* expression,
                                    const char* detail, const char* file, int line) {
    int initialized = 0;
    int rank = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "Rank %d: %s failure at %s:%d: %s: %s\n",
                 rank, kind, file, line, expression, detail);
    std::fflush(stderr);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::abort();
}

static void checkCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
    if (status != cudaSuccess) {
        fatalError("CUDA", expression, cudaGetErrorString(status), file, line);
    }
}

static void checkMpi(int status, const char* expression,
                     const char* file, int line) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(status, message, &length);
        fatalError("MPI", expression, message, file, line);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

// Generate exactly the same data stream as the original sequential benchmark.
static void generateSyntheticData(std::vector<Point>& points, const int n,
                                  unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double center_x = frand() * MAX_WIDTH;
        const double center_y = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (n / 30.0));
        // For n <= 30 the original expression is always zero and never makes
        // progress.  This only repairs that otherwise non-terminating corner.
        if (n <= 30) group_count = 1;
        group_count = std::min(group_count, n - count);

        while (group_count > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = center_x + dx;
            const double y = center_y + dy;

            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT) {
                continue;
            }
            points[count++] = {x, y};
            --group_count;
        }
    }
}

__global__ static void buildDistanceMatrix(const Point* __restrict__ points,
                                           double* __restrict__ distances,
                                           int n, size_t total) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int row = static_cast<int>(index / static_cast<size_t>(n));
        const int column = static_cast<int>(index - static_cast<size_t>(row) * n);
        const double dx = points[row].x - points[column].x;
        const double dy = points[row].y - points[column].y;
        distances[index] = sqrt(dx * dx + dy * dy);
    }
}

// One CUDA block evaluates one seed.  Maintaining each candidate's running
// maximum distance is equivalent to rescanning all current cluster members,
// but avoids the extra factor of cluster cardinality in the original code.
__global__ static void evaluateSeeds(const Point* __restrict__ points,
                                     const double* __restrict__ distances,
                                     const unsigned char* __restrict__ clustered,
                                     const int* __restrict__ seeds,
                                     int* __restrict__ cardinalities,
                                     int* __restrict__ output_members,
                                     double* __restrict__ candidate_maxima,
                                     int n, double threshold) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int seed = seeds[row];
    double* maxima = candidate_maxima + static_cast<size_t>(row) * n;

    __shared__ double reduction_distance[CUDA_BLOCK_SIZE];
    __shared__ int reduction_index[CUDA_BLOCK_SIZE];
    __shared__ double member_x;
    __shared__ double member_y;
    __shared__ int last_member;
    __shared__ int member_count;
    __shared__ int keep_going;

    for (int candidate = tid; candidate < n; candidate += blockDim.x) {
        maxima[candidate] = (clustered[candidate] || candidate == seed)
                                 ? CUDART_INF
                                 : 0.0;
    }

    if (tid == 0) {
        last_member = seed;
        member_count = 1;
        keep_going = 1;
        if (output_members != nullptr) {
            output_members[0] = seed;
        }
    }
    __syncthreads();

    while (keep_going) {
        if (tid == 0 && distances == nullptr) {
            member_x = points[last_member].x;
            member_y = points[last_member].y;
        }
        __syncthreads();

        double thread_best_distance = CUDART_INF;
        int thread_best_index = INT_MAX;

        for (int candidate = tid; candidate < n; candidate += blockDim.x) {
            double maximum = maxima[candidate];
            if (maximum == CUDART_INF) {
                continue;
            }

            double new_distance;
            if (distances != nullptr) {
                new_distance = distances[static_cast<size_t>(last_member) * n + candidate];
            } else {
                const double dx = member_x - points[candidate].x;
                const double dy = member_y - points[candidate].y;
                new_distance = sqrt(dx * dx + dy * dy);
            }
            maximum = maximum < new_distance ? new_distance : maximum;

            // A maximum distance can only increase as the cluster grows.
            // Permanently retire candidates that have crossed the threshold.
            if (!(maximum < threshold)) {
                maxima[candidate] = CUDART_INF;
                continue;
            }
            maxima[candidate] = maximum;

            if (maximum < thread_best_distance ||
                (maximum == thread_best_distance && candidate < thread_best_index)) {
                thread_best_distance = maximum;
                thread_best_index = candidate;
            }
        }

        reduction_distance[tid] = thread_best_distance;
        reduction_index[tid] = thread_best_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                const double other_distance = reduction_distance[tid + offset];
                const int other_index = reduction_index[tid + offset];
                if (other_distance < reduction_distance[tid] ||
                    (other_distance == reduction_distance[tid] &&
                     other_index < reduction_index[tid])) {
                    reduction_distance[tid] = other_distance;
                    reduction_index[tid] = other_index;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int selected = reduction_index[0];
            if (selected == INT_MAX) {
                keep_going = 0;
                cardinalities[row] = member_count;
            } else {
                maxima[selected] = CUDART_INF;
                last_member = selected;
                if (output_members != nullptr) {
                    output_members[member_count] = selected;
                }
                ++member_count;
            }
        }
        __syncthreads();
    }
}

__global__ static void markClustered(unsigned char* clustered,
                                     const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += blockDim.x * gridDim.x) {
        clustered[members[i]] = 1;
    }
}

class CudaEvaluator {
public:
    CudaEvaluator(const std::vector<Point>& points, int maximum_local_seeds)
        : n_(static_cast<int>(points.size())) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&device_points_, points.size() * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&device_clustered_, points.size() * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&device_updates_, points.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(device_points_, points.data(), points.size() * sizeof(Point),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemsetAsync(device_clustered_, 0,
                                   points.size() * sizeof(unsigned char), stream_));

        if (points.size() > std::numeric_limits<size_t>::max() / points.size() ||
            points.size() * points.size() >
                std::numeric_limits<size_t>::max() / sizeof(double)) {
            fatalError("input", "distance matrix size", "size_t overflow", __FILE__, __LINE__);
        }

        const size_t matrix_elements = points.size() * points.size();
        const size_t matrix_bytes = matrix_elements * sizeof(double);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        // The matrix removes nearly all square roots from iterative clustering.
        // Retain enough memory for at least one seed row and CUDA/runtime headroom.
        const size_t row_bytes = points.size() * sizeof(double);
        const size_t headroom = std::max<size_t>(64ULL << 20, row_bytes * 2);
        if (matrix_bytes < free_bytes && matrix_bytes + headroom < free_bytes * 3 / 4) {
            const cudaError_t allocation = cudaMalloc(&device_distances_, matrix_bytes);
            if (allocation == cudaSuccess) {
                const size_t required_blocks =
                    (matrix_elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
                const int blocks = static_cast<int>(std::min<size_t>(required_blocks, 65535));
                buildDistanceMatrix<<<blocks, CUDA_BLOCK_SIZE, 0, stream_>>>(
                    device_points_, device_distances_, n_, matrix_elements);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // Clear the allocation error and use on-the-fly distances below.
                cudaGetLastError();
                device_distances_ = nullptr;
            }
        }

        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t usable_bytes = free_bytes * 3 / 4;
        size_t rows_by_memory = row_bytes == 0 ? 1 : usable_bytes / row_bytes;
        rows_by_memory = std::max<size_t>(1, rows_by_memory);
        batch_capacity_ = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(std::max(1, maximum_local_seeds)), rows_by_memory));

        // Allocation can still fail due to fragmentation.  Halving preserves a
        // functional bounded-memory path without changing the computation.
        while (true) {
            const size_t workspace_bytes =
                static_cast<size_t>(batch_capacity_) * row_bytes;
            const cudaError_t allocation = cudaMalloc(&device_maxima_, workspace_bytes);
            if (allocation == cudaSuccess) {
                break;
            }
            cudaGetLastError();
            if (batch_capacity_ == 1) {
                checkCuda(allocation, "cudaMalloc(device_maxima_)", __FILE__, __LINE__);
            }
            batch_capacity_ = std::max(1, batch_capacity_ / 2);
        }

        CUDA_CHECK(cudaMalloc(&device_seeds_,
                              static_cast<size_t>(batch_capacity_) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_cardinalities_,
                              static_cast<size_t>(batch_capacity_) * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&host_cardinalities_,
                                  static_cast<size_t>(batch_capacity_) * sizeof(int)));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    ~CudaEvaluator() {
        if (host_cardinalities_ != nullptr) cudaFreeHost(host_cardinalities_);
        if (device_cardinalities_ != nullptr) cudaFree(device_cardinalities_);
        if (device_seeds_ != nullptr) cudaFree(device_seeds_);
        if (device_maxima_ != nullptr) cudaFree(device_maxima_);
        if (device_distances_ != nullptr) cudaFree(device_distances_);
        if (device_updates_ != nullptr) cudaFree(device_updates_);
        if (device_clustered_ != nullptr) cudaFree(device_clustered_);
        if (device_points_ != nullptr) cudaFree(device_points_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    CudaEvaluator(const CudaEvaluator&) = delete;
    CudaEvaluator& operator=(const CudaEvaluator&) = delete;

    BestCandidate findLocalBest(const std::vector<int>& seeds) {
        BestCandidate local_best{-1, INT_MAX};
        for (size_t begin = 0; begin < seeds.size(); begin += batch_capacity_) {
            const int count = static_cast<int>(std::min<size_t>(
                batch_capacity_, seeds.size() - begin));
            CUDA_CHECK(cudaMemcpyAsync(device_seeds_, seeds.data() + begin,
                                       static_cast<size_t>(count) * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_));
            evaluateSeeds<<<count, CUDA_BLOCK_SIZE, 0, stream_>>>(
                device_points_, device_distances_, device_clustered_, device_seeds_,
                device_cardinalities_, nullptr, device_maxima_, n_, threshold_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(host_cardinalities_, device_cardinalities_,
                                       static_cast<size_t>(count) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            BestCandidate batch_best{-1, INT_MAX};
#pragma omp parallel
            {
                BestCandidate thread_best{-1, INT_MAX};
#pragma omp for nowait schedule(static)
                for (int i = 0; i < count; ++i) {
                    const BestCandidate candidate{host_cardinalities_[i], seeds[begin + i]};
                    if (isBetter(candidate, thread_best)) {
                        thread_best = candidate;
                    }
                }
#pragma omp critical(qt_local_best)
                {
                    if (isBetter(thread_best, batch_best)) {
                        batch_best = thread_best;
                    }
                }
            }
            if (isBetter(batch_best, local_best)) {
                local_best = batch_best;
            }
        }
        return local_best;
    }

    std::vector<int> generateMembers(int seed, double threshold) {
        threshold_ = threshold;
        CUDA_CHECK(cudaMemcpyAsync(device_seeds_, &seed, sizeof(int),
                                   cudaMemcpyHostToDevice, stream_));
        evaluateSeeds<<<1, CUDA_BLOCK_SIZE, 0, stream_>>>(
            device_points_, device_distances_, device_clustered_, device_seeds_,
            device_cardinalities_, device_updates_, device_maxima_, n_, threshold_);
        CUDA_CHECK(cudaGetLastError());

        int count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&count, device_cardinalities_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::vector<int> members(count);
        CUDA_CHECK(cudaMemcpyAsync(members.data(), device_updates_,
                                   static_cast<size_t>(count) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return members;
    }

    void setThreshold(double threshold) {
        threshold_ = threshold;
    }

    void addCluster(const std::vector<int>& members) {
        if (members.empty()) return;
        CUDA_CHECK(cudaMemcpyAsync(device_updates_, members.data(),
                                   members.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, stream_));
        const int blocks = std::min<int>(
            (static_cast<int>(members.size()) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
            65535);
        markClustered<<<blocks, CUDA_BLOCK_SIZE, 0, stream_>>>(
            device_clustered_, device_updates_, static_cast<int>(members.size()));
        CUDA_CHECK(cudaGetLastError());
    }

    bool hasDistanceMatrix() const { return device_distances_ != nullptr; }
    int batchCapacity() const { return batch_capacity_; }

private:
    int n_ = 0;
    int batch_capacity_ = 1;
    double threshold_ = 0.0;
    cudaStream_t stream_ = nullptr;
    Point* device_points_ = nullptr;
    double* device_distances_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    int* device_updates_ = nullptr;
    int* device_seeds_ = nullptr;
    int* device_cardinalities_ = nullptr;
    double* device_maxima_ = nullptr;
    int* host_cardinalities_ = nullptr;
};

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         double threshold, int rank, int ranks,
                                         CudaEvaluator& evaluator) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> local_seeds;
    local_seeds.reserve((n + ranks - 1) / ranks);
    for (int seed = rank; seed < n; seed += ranks) {
        local_seeds.push_back(seed);
    }

    evaluator.setThreshold(threshold);
    std::vector<Cluster> clusters;

    while (true) {
        const BestCandidate local_best = evaluator.findLocalBest(local_seeds);
        int local_pair[2] = {local_best.cardinality, local_best.seed};
        int global_pair[2] = {-1, INT_MAX};
        MPI_CHECK(MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT,
                                MPI_MAXLOC, MPI_COMM_WORLD));

        const int best_cardinality = global_pair[0];
        const int best_seed = global_pair[1];
        if (best_cardinality <= 0 || best_seed == INT_MAX) {
            break;
        }

        const int owner = best_seed % ranks;
        std::vector<int> members;
        int member_count = 0;
        if (rank == owner) {
            members = evaluator.generateMembers(best_seed, threshold);
            member_count = static_cast<int>(members.size());
        }
        MPI_CHECK(MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (rank != owner) {
            members.resize(member_count);
        }
        MPI_CHECK(MPI_Bcast(members.data(), member_count, MPI_INT,
                            owner, MPI_COMM_WORLD));

        // Cardinality was evaluated from precisely the same state.  This also
        // catches device or synchronization errors before ranks can diverge.
        if (member_count != best_cardinality) {
            fatalError("consistency", "winning cardinality",
                       "candidate regeneration produced a different size",
                       __FILE__, __LINE__);
        }

        clusters.push_back(Cluster{members, best_seed});
        for (const int member : members) {
            clustered[member] = 1;
        }
        evaluator.addCluster(members);

        local_seeds.erase(
            std::remove_if(local_seeds.begin(), local_seeds.end(),
                           [&clustered](int seed) { return clustered[seed] != 0; }),
            local_seeds.end());
    }
    return clusters;
}

static inline double pointDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points,
                             double threshold) {
    std::vector<double> diameters(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < static_cast<long long>(clusters.size()); ++c) {
        double maximum = 0.0;
        const std::vector<int>& members = clusters[static_cast<size_t>(c)].members;
        for (size_t i = 0; i < members.size(); ++i) {
            for (size_t j = i + 1; j < members.size(); ++j) {
                maximum = std::max(maximum,
                    pointDistance(points[members[i]], points[members[j]]));
            }
        }
        diameters[static_cast<size_t>(c)] = maximum;
    }

    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        c, clusters[c].members.size(), clusters[c].seed_point,
                        diameters[c]);
        }
        if (diameters[c] > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        c, diameters[c], threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (const int member : clusters[c].members) {
            if (member < 0 || static_cast<size_t>(member) >= points.size()) {
                std::printf("ERROR: Invalid point index %d in cluster %zu\n", member, c);
                valid = false;
                continue;
            }
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    const int clustered_count = static_cast<int>(std::count_if(
        membership.begin(), membership.end(), [](int value) { return value >= 0; }));
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count, points.size() - clustered_count);
    if (clustered_count != static_cast<int>(points.size())) {
        std::printf("ERROR: Not all points were assigned to a cluster\n");
        valid = false;
    }
    return valid;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        fatalError("MPI", "MPI_Init_thread", "MPI_THREAD_FUNNELED unavailable",
                   __FILE__, __LINE__);
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
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
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            arguments_valid = false;
        }
    }

    if (show_help || !arguments_valid || num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            if (!show_help && (num_points <= 0 || threshold <= 0.0)) {
                std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                            num_points, threshold);
            }
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return show_help ? 0 : 1;
    }

    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local_communicator));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_communicator, &local_rank));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fatalError("CUDA", "cudaGetDeviceCount", "no CUDA accelerator found",
                   __FILE__, __LINE__);
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    // Force context creation before the measured clustering region.
    CUDA_CHECK(cudaFree(nullptr));

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), %d OpenMP thread(s)/rank, "
                    "%d CUDA device(s)/node\n",
                    ranks, omp_get_max_threads(), device_count);
    }

    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must be two packed doubles");
    MPI_CHECK(MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const int maximum_local_seeds = (num_points + ranks - 1) / ranks;
    auto evaluator = std::make_unique<CudaEvaluator>(points, maximum_local_seeds);
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, ranks, *evaluator);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD));

    int exit_code = 0;
    if (rank == 0) {
        const long long elapsed_ms = static_cast<long long>(elapsed * 1000.0);
        std::printf("Clustering time: %lld ms\n", elapsed_ms);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int maximum_cluster_size = 0;
#pragma omp parallel for reduction(+:total_clustered) reduction(max:maximum_cluster_size)
        for (long long i = 0; i < static_cast<long long>(clusters.size()); ++i) {
            const int size = static_cast<int>(clusters[static_cast<size_t>(i)].members.size());
            total_clustered += size;
            maximum_cluster_size = std::max(maximum_cluster_size, size);
        }
        const double average_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n",
                    total_clustered, num_points,
                    100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", maximum_cluster_size);

        const double safe_elapsed = std::max(elapsed, 1.0e-12);
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / safe_elapsed, num_points / safe_elapsed);

        if (print_results_requested) {
            std::vector<double> membership_data(num_points, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership_data[member] = static_cast<double>(c);
                }
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }

    MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    evaluator.reset();
    MPI_CHECK(MPI_Comm_free(&local_communicator));
    MPI_CHECK(MPI_Finalize());
    return exit_code;
}
