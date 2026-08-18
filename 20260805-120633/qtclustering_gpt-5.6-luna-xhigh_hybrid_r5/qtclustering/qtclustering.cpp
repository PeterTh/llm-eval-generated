// QT Clustering Benchmark - hybrid MPI/OpenMP/CUDA implementation
//
// Candidate clusters for the current set of seeds are independent.  MPI
// distributes those seeds between ranks, OpenMP handles the host-side work,
// and CUDA evaluates all candidates in a seed batch in parallel.  The greedy
// growth of one candidate remains sequential, as required by QT clustering.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cfloat>
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

// Structure to represent a point in 2D space.  It is deliberately POD so it
// can be copied directly to the CUDA device and broadcast as two doubles.
struct Point {
    double x, y;
};

// Structure to represent a cluster.
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

namespace {

[[noreturn]] void cudaFailure(const cudaError_t error,
                              const char* expression,
                              const char* file,
                              const int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d (%s): %s\n",
                 file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cuda_status = (expression); \
        if (cuda_status != cudaSuccess) { \
            cudaFailure(cuda_status, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

// Generate synthetic 2D point data in clusters.  The generator is retained
// as the original serial rand_r stream so the benchmark data and result hash
// remain unchanged for the normal N >= 30 input range.
void generateSyntheticData(std::vector<Point>& points,
                           const int point_count,
                           unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < point_count) {
        const double center_x = frand() * MAX_WIDTH;
        const double center_y = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (point_count / 30.0));

        // The original benchmark loops forever for N < 30 because every
        // generated group has size zero.  A singleton is the only sensible
        // continuation for that otherwise valid input.
        if (group_count == 0 && point_count <= 30) {
            group_count = 1;
        }
        group_count = std::min(group_count, point_count - count);

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

            points[count] = {x, y};
            ++count;
            --group_count;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// CUDA uses DBL_MAX as the invalid/ineligible marker.  A valid distance is
// always strictly below the positive threshold, so this marker is unambiguous.
__device__ __forceinline__ double deviceDistance(const Point& p1,
                                                 const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void initializeNearestKernel(const Point* points,
                                        const int* seeds,
                                        const unsigned char* clustered,
                                        const double threshold,
                                        const int seed_count,
                                        const int point_count,
                                        double* nearest) {
    const size_t work_item = static_cast<size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
    const size_t total = static_cast<size_t>(seed_count) * point_count;
    if (work_item >= total) {
        return;
    }

    const int seed_row = static_cast<int>(work_item / point_count);
    const int candidate = static_cast<int>(work_item % point_count);
    const int seed = seeds[seed_row];

    if (candidate == seed || clustered[candidate] != 0) {
        nearest[work_item] = DBL_MAX;
        return;
    }

    const double candidate_distance = deviceDistance(points[candidate],
                                                      points[seed]);
    nearest[work_item] = candidate_distance < threshold
                        ? candidate_distance
                        : DBL_MAX;
}

// One CUDA block owns one seed row.  The explicit pair comparison preserves
// the original serial rule: equal distances select the lowest point index.
__global__ void selectNearestKernel(const double* nearest,
                                    const double threshold,
                                    const int seed_count,
                                    const int point_count,
                                    int* selected,
                                    int* cardinalities,
                                    int* active_count) {
    __shared__ double best_distances[CUDA_BLOCK_SIZE];
    __shared__ int best_indices[CUDA_BLOCK_SIZE];

    const int seed_row = static_cast<int>(blockIdx.x);
    if (seed_row >= seed_count) {
        return;
    }

    double best_distance = DBL_MAX;
    int best_index = INT_MAX;
    const size_t row_offset = static_cast<size_t>(seed_row) * point_count;

    for (int candidate = threadIdx.x;
         candidate < point_count;
         candidate += blockDim.x) {
        const double candidate_distance = nearest[row_offset + candidate];
        if (candidate_distance < DBL_MAX && candidate_distance < threshold
            && (candidate_distance < best_distance
                || (candidate_distance == best_distance
                    && candidate < best_index))) {
            best_distance = candidate_distance;
            best_index = candidate;
        }
    }

    best_distances[threadIdx.x] = best_distance;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const double other_distance = best_distances[threadIdx.x + stride];
            const int other_index = best_indices[threadIdx.x + stride];
            if (other_distance < best_distances[threadIdx.x]
                || (other_distance == best_distances[threadIdx.x]
                    && other_index < best_indices[threadIdx.x])) {
                best_distances[threadIdx.x] = other_distance;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int chosen = (best_indices[0] == INT_MAX)
                         ? -1
                         : best_indices[0];
        selected[seed_row] = chosen;
        if (chosen >= 0) {
            ++cardinalities[seed_row];
            atomicAdd(active_count, 1);
        }
    }
}

// Incremental diameter update.  The original implementation recomputed the
// distance to every existing member for every candidate.  Keeping the current
// maximum is mathematically equivalent and reduces each growth step to one
// distance per candidate while retaining the same greedy choices.
__global__ void updateNearestKernel(const Point* points,
                                    const int* selected,
                                    const double threshold,
                                    const int seed_count,
                                    const int point_count,
                                    double* nearest) {
    const size_t work_item = static_cast<size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
    const size_t total = static_cast<size_t>(seed_count) * point_count;
    if (work_item >= total) {
        return;
    }

    const int seed_row = static_cast<int>(work_item / point_count);
    const int candidate = static_cast<int>(work_item % point_count);
    const int chosen = selected[seed_row];
    if (chosen < 0) {
        return;
    }

    const size_t row_offset = static_cast<size_t>(seed_row) * point_count;
    if (candidate == chosen) {
        nearest[work_item] = DBL_MAX;
        return;
    }

    const double current_distance = nearest[work_item];
    if (current_distance == DBL_MAX || current_distance >= threshold) {
        return;
    }

    const double added_distance = deviceDistance(points[candidate],
                                                 points[chosen]);
    const double diameter = current_distance > added_distance
                          ? current_distance
                          : added_distance;
    nearest[row_offset + candidate] = diameter < threshold
                                    ? diameter
                                    : DBL_MAX;
}

inline int cudaBlockCount(const size_t work_items) {
    const size_t blocks = (work_items + CUDA_BLOCK_SIZE - 1)
                        / CUDA_BLOCK_SIZE;
    return static_cast<int>(std::max<size_t>(1, blocks));
}

class CudaCandidateEngine {
public:
    CudaCandidateEngine(const std::vector<Point>& points, const int device)
        : point_count_(static_cast<int>(points.size())) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points_),
                              points.size() * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered_),
                              points.size() * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemcpy(device_points_, points.data(),
                              points.size() * sizeof(Point),
                              cudaMemcpyHostToDevice));
    }

    CudaCandidateEngine(const CudaCandidateEngine&) = delete;
    CudaCandidateEngine& operator=(const CudaCandidateEngine&) = delete;

    ~CudaCandidateEngine() {
        cudaFree(device_active_count_);
        cudaFree(device_cardinalities_);
        cudaFree(device_selected_);
        cudaFree(device_nearest_);
        cudaFree(device_seeds_);
        cudaFree(device_clustered_);
        cudaFree(device_points_);
    }

    void beginRound(const std::vector<unsigned char>& clustered) {
        CUDA_CHECK(cudaMemcpy(device_clustered_, clustered.data(),
                              clustered.size() * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    int chooseBatchSize(const int requested) const {
        if (requested <= 0) {
            return 0;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;

        // The nearest-distance matrix is the dominant allocation.  Leave
        // headroom for the CUDA runtime and other rank-local allocations.
        const size_t bytes_per_seed =
            static_cast<size_t>(point_count_) * sizeof(double)
            + 2 * sizeof(int);
        const size_t usable_bytes = (free_bytes / 10) * 8;
        size_t possible = usable_bytes / std::max<size_t>(1, bytes_per_seed);
        possible = std::max<size_t>(1, possible);
        possible = std::min<size_t>(possible, INT_MAX);
        return std::min(requested, static_cast<int>(possible));
    }

    // Evaluate all candidate clusters in one batch.  If captured_members is
    // supplied, the batch is expected to contain one seed and its exact greedy
    // member sequence is returned.
    void runBatch(const std::vector<int>& seeds,
                  const double threshold,
                  std::vector<int>& cardinalities,
                  std::vector<int>* captured_members = nullptr) {
        const int seed_count = static_cast<int>(seeds.size());
        if (seed_count == 0) {
            cardinalities.clear();
            return;
        }

        ensureCapacity(seed_count);
        CUDA_CHECK(cudaMemcpy(device_seeds_, seeds.data(),
                              seeds.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_cardinalities_, 0,
                              seeds.size() * sizeof(int)));

        const size_t work_items = static_cast<size_t>(seed_count)
                                 * point_count_;
        initializeNearestKernel<<<cudaBlockCount(work_items),
                                  CUDA_BLOCK_SIZE>>>(
            device_points_, device_seeds_, device_clustered_, threshold,
            seed_count, point_count_, device_nearest_);
        CUDA_CHECK(cudaGetLastError());

        if (captured_members != nullptr) {
            captured_members->clear();
            captured_members->push_back(seeds[0]);
        }

        int active = 0;
        do {
            CUDA_CHECK(cudaMemset(device_active_count_, 0, sizeof(int)));
            selectNearestKernel<<<seed_count, CUDA_BLOCK_SIZE>>>(
                device_nearest_, threshold, seed_count, point_count_,
                device_selected_, device_cardinalities_,
                device_active_count_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&active, device_active_count_, sizeof(int),
                                  cudaMemcpyDeviceToHost));

            if (captured_members != nullptr && active > 0) {
                int selected = -1;
                CUDA_CHECK(cudaMemcpy(&selected, device_selected_, sizeof(int),
                                      cudaMemcpyDeviceToHost));
                if (selected >= 0) {
                    captured_members->push_back(selected);
                }
            }

            if (active > 0) {
                updateNearestKernel<<<cudaBlockCount(work_items),
                                      CUDA_BLOCK_SIZE>>>(
                    device_points_, device_selected_,
                    threshold, seed_count, point_count_, device_nearest_);
                CUDA_CHECK(cudaGetLastError());
            }
        } while (active > 0);

        cardinalities.resize(seed_count);
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), device_cardinalities_,
                              seeds.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Every candidate contains its seed even when no additional point is
        // eligible, matching generateCandidateCluster exactly.
        for (int& cardinality : cardinalities) {
            ++cardinality;
        }
    }

private:
    void ensureCapacity(const int requested) {
        if (requested <= capacity_) {
            return;
        }

        cudaFree(device_cardinalities_);
        cudaFree(device_selected_);
        cudaFree(device_nearest_);
        cudaFree(device_seeds_);
        cudaFree(device_active_count_);
        device_cardinalities_ = nullptr;
        device_selected_ = nullptr;
        device_nearest_ = nullptr;
        device_seeds_ = nullptr;
        device_active_count_ = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_seeds_),
                              static_cast<size_t>(requested) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_nearest_),
                              static_cast<size_t>(requested)
                              * point_count_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_selected_),
                              static_cast<size_t>(requested) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cardinalities_),
                              static_cast<size_t>(requested) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_active_count_),
                              sizeof(int)));
        capacity_ = requested;
    }

    int point_count_;
    int capacity_ = 0;
    Point* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    int* device_seeds_ = nullptr;
    double* device_nearest_ = nullptr;
    int* device_selected_ = nullptr;
    int* device_cardinalities_ = nullptr;
    int* device_active_count_ = nullptr;
};

void buildLocalSeeds(const std::vector<int>& unclustered_indices,
                     const int rank,
                     const int world_size,
                     std::vector<int>& local_seeds) {
    const int remaining = static_cast<int>(unclustered_indices.size());
    const int local_count = remaining <= rank
                          ? 0
                          : (remaining - rank + world_size - 1) / world_size;
    local_seeds.resize(local_count);

    // Cyclic ownership balances variable candidate-cluster cardinalities while
    // preserving ascending order within each rank for deterministic tie logic.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < local_count; ++i) {
        local_seeds[i] = unclustered_indices[rank + i * world_size];
    }
}

bool betterCandidate(const int candidate_size,
                     const int candidate_seed,
                     const int current_size,
                     const int current_seed) {
    return candidate_size > current_size
        || (candidate_size == current_size && candidate_seed < current_seed);
}

// Main QT clustering algorithm.  All ranks execute the same outer greedy
// loop; MPI only synchronizes the globally best candidate for each round.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int world_size,
                                  const int local_device) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < point_count; ++i) {
        unclustered_indices[i] = i;
    }

    std::vector<Cluster> clusters;
    CudaCandidateEngine cuda_engine(points, local_device);
    std::vector<int> local_seeds;
    std::vector<int> cardinalities;

    while (!unclustered_indices.empty()) {
        buildLocalSeeds(unclustered_indices, rank, world_size, local_seeds);
        cuda_engine.beginRound(clustered);

        int local_best_size = -1;
        int local_best_seed = INT_MAX;
        const int batch_limit = cuda_engine.chooseBatchSize(
            static_cast<int>(local_seeds.size()));

        for (int begin = 0; begin < static_cast<int>(local_seeds.size());
             begin += batch_limit) {
            const int batch_count = std::min(
                batch_limit, static_cast<int>(local_seeds.size()) - begin);
            std::vector<int> batch_seeds(local_seeds.begin() + begin,
                                         local_seeds.begin()
                                         + begin + batch_count);
            cuda_engine.runBatch(batch_seeds, threshold, cardinalities);

            int batch_best_size = -1;
            int batch_best_seed = INT_MAX;

            // The GPU computes cardinalities.  This reduction selects the
            // local winner and is intentionally OpenMP-parallel as well.
#pragma omp parallel
            {
                int thread_best_size = -1;
                int thread_best_seed = INT_MAX;

#pragma omp for nowait schedule(static)
                for (int i = 0; i < batch_count; ++i) {
                    const int seed = batch_seeds[i];
                    if (betterCandidate(cardinalities[i], seed,
                                        thread_best_size, thread_best_seed)) {
                        thread_best_size = cardinalities[i];
                        thread_best_seed = seed;
                    }
                }

#pragma omp critical
                {
                    if (betterCandidate(thread_best_size, thread_best_seed,
                                        batch_best_size, batch_best_seed)) {
                        batch_best_size = thread_best_size;
                        batch_best_seed = thread_best_seed;
                    }
                }
            }

            if (betterCandidate(batch_best_size, batch_best_seed,
                                local_best_size, local_best_seed)) {
                local_best_size = batch_best_size;
                local_best_seed = batch_best_seed;
            }
        }

        int global_best_size = -1;
        MPI_Allreduce(&local_best_size, &global_best_size, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD);

        const int local_tied_seed = (local_best_size == global_best_size)
                                  ? local_best_seed
                                  : INT_MAX;
        int global_best_seed = INT_MAX;
        MPI_Allreduce(&local_tied_seed, &global_best_seed, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);

        if (global_best_size <= 0 || global_best_seed == INT_MAX) {
            break;
        }

        const int local_winner_rank = (local_best_seed == global_best_seed)
                                    ? rank
                                    : INT_MAX;
        int winner_rank = INT_MAX;
        MPI_Allreduce(&local_winner_rank, &winner_rank, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);

        // Re-run only the globally selected seed to obtain its exact member
        // sequence.  This avoids storing an O(number_of_seeds * N) history on
        // either host or device during the much larger cardinality search.
        std::vector<int> best_members;
        if (rank == winner_rank) {
            std::vector<int> winner_seed(1, global_best_seed);
            cuda_engine.runBatch(winner_seed, threshold, cardinalities,
                                 &best_members);
            if (static_cast<int>(best_members.size()) != global_best_size) {
                std::fprintf(stderr,
                             "Rank %d generated an inconsistent best cluster\n",
                             rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        best_members.resize(global_best_size);
        MPI_Bcast(best_members.data(), global_best_size, MPI_INT, winner_rank,
                  MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < global_best_size; ++i) {
            clustered[best_members[i]] = 1;
        }

        std::vector<int> next_unclustered;
        next_unclustered.resize(unclustered_indices.size());
        int next_count = 0;
        // This compaction is ordered to exactly match remove_if on the
        // ascending unclustered list.  The outer loop is synchronization
        // light compared with the CUDA candidate search.
        for (const int index : unclustered_indices) {
            if (clustered[index] == 0) {
                next_unclustered[next_count++] = index;
            }
        }
        next_unclustered.resize(next_count);
        unclustered_indices.swap(next_unclustered);
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties.
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    std::vector<double> diameters(clusters.size(), 0.0);

#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < static_cast<long long>(clusters.size()); ++c) {
        const auto& cluster = clusters[static_cast<size_t>(c)];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                max_diameter = std::max(
                    max_diameter,
                    distance(points[cluster.members[i]],
                             points[cluster.members[j]]));
            }
        }
        diameters[static_cast<size_t>(c)] = max_diameter;
    }

    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        c, cluster.members.size(), cluster.seed_point,
                        max_diameter);
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
                std::printf("ERROR: Point %d appears in multiple clusters (%d "
                            "and %zu)\n",
                            member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
    for (const int member : membership) {
        if (member >= 0) {
            ++clustered_count;
        }
    }

    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count,
                points.size() - static_cast<size_t>(clustered_count));
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

} // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_communicator);
    int local_rank = 0;
    MPI_Comm_rank(local_communicator, &local_rank);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_flag = false;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Comm_free(&local_communicator);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Comm_free(&local_communicator);
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0
        || num_points > INT_MAX / 2) {
        if (rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        MPI_Comm_free(&local_communicator);
        MPI_Finalize();
        return 1;
    }

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if (device_query != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "MPI rank %d found no usable CUDA device: %s\n",
                     rank, cudaGetErrorString(device_query));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: MPI ranks=%d, OpenMP threads=%d, CUDA device=%d\n",
                    world_size, omp_get_max_threads(), device);
    }

    std::vector<Point> points(static_cast<size_t>(num_points));
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, world_size, device);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        const auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            cluster_end - cluster_start);
        std::printf("Clustering time: %ld ms\n", cluster_time.count());
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
#pragma omp parallel for reduction(+:total_clustered) reduction(max:max_cluster_size)
        for (long long i = 0; i < static_cast<long long>(clusters.size()); ++i) {
            const int size = static_cast<int>(clusters[static_cast<size_t>(i)].members.size());
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

        const double time_sec = std::max(
            cluster_time.count() / 1000.0, std::numeric_limits<double>::min());
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters_per_sec, points_per_sec);

        if (print_results_flag) {
            std::vector<double> membership_data;
            membership_data.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
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
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                MPI_Comm_free(&local_communicator);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Comm_free(&local_communicator);
    MPI_Finalize();
    return 0;
}
