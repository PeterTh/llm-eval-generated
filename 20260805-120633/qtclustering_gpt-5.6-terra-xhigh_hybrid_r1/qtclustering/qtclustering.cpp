// QT clustering benchmark -- distributed CUDA/OpenMP implementation.
//
// Candidate clusters are independent within one QT iteration.  MPI assigns
// seeds cyclically to ranks, CUDA builds those candidate clusters in batches,
// and OpenMP reduces the batch results and updates the committed cluster.
// The global MPI reduction uses (negative cardinality, seed) MINLOC, which is
// exactly the cardinality-first, first-seed tie breaking of the reference.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr double MAX_WIDTH = 20.0;
constexpr double MAX_HEIGHT = 20.0;
constexpr int CUDA_THREADS = 256;

struct Point {
    double x;
    double y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

struct BestCandidate {
    int cardinality = -1;
    int seed = INT_MAX;
};

struct HybridContext {
    int rank = 0;
    int size = 1;
    int local_rank = 0;
    int device = 0;
    MPI_Comm node_comm = MPI_COMM_NULL;
};

struct DeviceWorkspace {
    Point* points = nullptr;
    double* distances = nullptr;
    unsigned char* clustered = nullptr;
    int* seeds = nullptr;
    double* candidate_max = nullptr;
    unsigned char* in_cluster = nullptr;
    int* cardinalities = nullptr;
    int* members = nullptr;
    int point_count = 0;
    int batch_capacity = 0;

    void release() {
        cudaFree(members);
        cudaFree(cardinalities);
        cudaFree(in_cluster);
        cudaFree(candidate_max);
        cudaFree(seeds);
        cudaFree(clustered);
        cudaFree(distances);
        cudaFree(points);
        members = nullptr;
        cardinalities = nullptr;
        in_cluster = nullptr;
        candidate_max = nullptr;
        seeds = nullptr;
        clustered = nullptr;
        distances = nullptr;
        points = nullptr;
    }
};

[[noreturn]] void abortWithMessage(const HybridContext& context, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", context.rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression, const HybridContext& context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", context.rank,
                     expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(context, call) checkCuda((call), #call, (context))

bool isBetter(const BestCandidate& candidate, const BestCandidate& current) {
    return candidate.cardinality > current.cardinality ||
           (candidate.cardinality == current.cardinality && candidate.seed < current.seed);
}

// Generate the same deterministic synthetic input on rank zero as the
// sequential benchmark, then broadcast it to all ranks.
void generateSyntheticData(std::vector<Point>& points, const int count, unsigned int seed = 42) {
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
        // The reference expression is zero for every N < 30, which would
        // otherwise make valid small benchmark inputs loop forever.  Keep the
        // original random sequence exactly for all regular benchmark sizes.
        if (count < 30) {
            group_count = 1;
        }

        if (group_count > (count - generated)) {
            group_count = count - generated;
        }

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

            points[generated] = {x, y};
            ++generated;
            --group_count;
        }
    }
}

inline double distance(const Point& first, const Point& second) {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Each CUDA block computes one candidate cluster.  candidate_max[c] is kept
// incrementally: after a member is selected it becomes the maximum distance
// from c to every member selected so far.  This gives the same value as the
// reference's repeated scan of all cluster members, but avoids recomputing it.
__global__ void candidateClusterKernel(const int* __restrict__ seeds,
                                       const int seed_count,
                                       const unsigned char* __restrict__ clustered,
                                       const double* __restrict__ distances,
                                       const int point_count,
                                       const double threshold,
                                       double* __restrict__ candidate_max,
                                       unsigned char* __restrict__ in_cluster,
                                       int* __restrict__ cardinalities,
                                       int* __restrict__ members) {
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= seed_count) {
        return;
    }

    const int thread = static_cast<int>(threadIdx.x);
    const int seed = seeds[slot];
    const size_t base = static_cast<size_t>(slot) * point_count;

    __shared__ double best_distances[CUDA_THREADS];
    __shared__ int best_indices[CUDA_THREADS];
    __shared__ int selected;
    __shared__ int cluster_size;
    __shared__ int finished;

    for (int candidate = thread; candidate < point_count; candidate += CUDA_THREADS) {
        candidate_max[base + candidate] = distances[static_cast<size_t>(seed) * point_count + candidate];
        in_cluster[base + candidate] = static_cast<unsigned char>(candidate == seed);
    }
    __syncthreads();

    if (thread == 0) {
        cluster_size = 1;
        selected = seed;
        finished = 0;
        if (members != nullptr) {
            members[base] = seed;
        }
    }
    __syncthreads();

    for (int iteration = 1; iteration < point_count; ++iteration) {
        double local_distance = DBL_MAX;
        int local_index = -1;

        for (int candidate = thread; candidate < point_count; candidate += CUDA_THREADS) {
            const double candidate_distance = candidate_max[base + candidate];
            if (clustered[candidate] == 0 && in_cluster[base + candidate] == 0 &&
                candidate_distance < threshold &&
                (local_index < 0 || candidate_distance < local_distance ||
                 (candidate_distance == local_distance && candidate < local_index))) {
                local_distance = candidate_distance;
                local_index = candidate;
            }
        }

        best_distances[thread] = local_distance;
        best_indices[thread] = local_index;
        __syncthreads();

        for (int stride = CUDA_THREADS / 2; stride > 0; stride /= 2) {
            if (thread < stride) {
                const int other_index = best_indices[thread + stride];
                const double other_distance = best_distances[thread + stride];
                const int this_index = best_indices[thread];
                const double this_distance = best_distances[thread];
                if (other_index >= 0 &&
                    (this_index < 0 || other_distance < this_distance ||
                     (other_distance == this_distance && other_index < this_index))) {
                    best_distances[thread] = other_distance;
                    best_indices[thread] = other_index;
                }
            }
            __syncthreads();
        }

        if (thread == 0) {
            selected = best_indices[0];
            finished = (selected < 0);
            if (!finished) {
                in_cluster[base + selected] = 1;
                if (members != nullptr) {
                    members[base + cluster_size] = selected;
                }
                ++cluster_size;
            }
        }
        __syncthreads();

        if (finished) {
            break;
        }

        for (int candidate = thread; candidate < point_count; candidate += CUDA_THREADS) {
            if (clustered[candidate] == 0 && in_cluster[base + candidate] == 0) {
                const double old_maximum = candidate_max[base + candidate];
                const double next_distance =
                    distances[static_cast<size_t>(candidate) * point_count + selected];
                candidate_max[base + candidate] =
                    (next_distance > old_maximum) ? next_distance : old_maximum;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        cardinalities[slot] = cluster_size;
    }
}

// Distances are partitioned by rows across ranks.  They are then all-gathered
// so every GPU can evaluate its assigned seeds without communication in the
// candidate-building hot loop.
__global__ void distanceRowsKernel(const Point* __restrict__ points,
                                   double* __restrict__ local_distances,
                                   const int first_row,
                                   const int row_count,
                                   const int point_count) {
    const int column = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int local_row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (local_row >= row_count || column >= point_count) {
        return;
    }

    const Point first = points[first_row + local_row];
    const Point second = points[column];
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    local_distances[static_cast<size_t>(local_row) * point_count + column] =
        sqrt(dx * dx + dy * dy);
}

void selectDevice(const HybridContext& context) {
    int device_count = 0;
    CUDA_CHECK(context, cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        abortWithMessage(context, "no CUDA device is visible to this MPI rank");
    }
    CUDA_CHECK(context, cudaSetDevice(context.device % device_count));
    CUDA_CHECK(context, cudaFree(nullptr)); // Establish the selected device context now.
}

void buildDeviceWorkspace(DeviceWorkspace& workspace,
                          const std::vector<Point>& points,
                          const HybridContext& context) {
    const int point_count = static_cast<int>(points.size());
    const int first_row = static_cast<int>((static_cast<long long>(point_count) * context.rank) /
                                           context.size);
    const int last_row = static_cast<int>((static_cast<long long>(point_count) * (context.rank + 1)) /
                                          context.size);
    const int local_rows = last_row - first_row;

    const size_t matrix_elements = static_cast<size_t>(point_count) * point_count;
    if (matrix_elements > static_cast<size_t>(INT_MAX)) {
        abortWithMessage(context,
                         "point count is too large for the required MPI distance all-gather");
    }

    workspace.point_count = point_count;
    CUDA_CHECK(context, cudaMalloc(&workspace.points, static_cast<size_t>(point_count) * sizeof(Point)));
    CUDA_CHECK(context, cudaMemcpy(workspace.points, points.data(),
                                   static_cast<size_t>(point_count) * sizeof(Point),
                                   cudaMemcpyHostToDevice));

    std::vector<double> local_distances(static_cast<size_t>(local_rows) * point_count);
    double* device_local_distances = nullptr;
    if (local_rows > 0) {
        CUDA_CHECK(context, cudaMalloc(&device_local_distances,
                                       static_cast<size_t>(local_rows) * point_count * sizeof(double)));
        const dim3 threads(16, 16);
        const dim3 blocks((point_count + threads.x - 1) / threads.x,
                          (local_rows + threads.y - 1) / threads.y);
        distanceRowsKernel<<<blocks, threads>>>(workspace.points, device_local_distances,
                                                 first_row, local_rows, point_count);
        CUDA_CHECK(context, cudaGetLastError());
        CUDA_CHECK(context, cudaMemcpy(local_distances.data(), device_local_distances,
                                       static_cast<size_t>(local_rows) * point_count * sizeof(double),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK(context, cudaFree(device_local_distances));
    }

    std::vector<int> receive_counts(context.size);
    std::vector<int> displacements(context.size);
    for (int rank = 0; rank < context.size; ++rank) {
        const int begin = static_cast<int>((static_cast<long long>(point_count) * rank) / context.size);
        const int end = static_cast<int>((static_cast<long long>(point_count) * (rank + 1)) / context.size);
        receive_counts[rank] = (end - begin) * point_count;
        displacements[rank] = begin * point_count;
    }

    std::vector<double> all_distances(matrix_elements);
    MPI_Allgatherv(local_rows > 0 ? local_distances.data() : nullptr, local_rows * point_count,
                   MPI_DOUBLE, all_distances.data(), receive_counts.data(), displacements.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    CUDA_CHECK(context, cudaMalloc(&workspace.distances, matrix_elements * sizeof(double)));
    CUDA_CHECK(context, cudaMemcpy(workspace.distances, all_distances.data(),
                                   matrix_elements * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double>().swap(all_distances);

    CUDA_CHECK(context, cudaMalloc(&workspace.clustered,
                                   static_cast<size_t>(point_count) * sizeof(unsigned char)));

    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(context, cudaMemGetInfo(&free_memory, &total_memory));
    const size_t final_cluster_bytes = static_cast<size_t>(point_count) * sizeof(int);
    const size_t bytes_per_seed = static_cast<size_t>(point_count) *
                                      (sizeof(double) + sizeof(unsigned char)) +
                                  sizeof(int);
    // Keep 30% free for the CUDA runtime, MPI's device registration, and the
    // final winning cluster buffer.  Batches avoid a quadratic per-rank heap.
    const size_t usable_memory = free_memory > final_cluster_bytes
                                     ? (free_memory - final_cluster_bytes) * 7 / 10
                                     : 0;
    const size_t memory_capacity = usable_memory / bytes_per_seed;
    // A rank will never own more than this many cyclically assigned seeds.
    // Capping the allocation here is important on large-memory GPUs: a small
    // benchmark should not reserve the entire accelerator merely because it
    // could fit a much larger batch.
    const size_t owned_seed_count = context.rank < point_count
                                        ? static_cast<size_t>((point_count - 1 - context.rank) /
                                                              context.size + 1)
                                        : 0;
    const size_t capacity = std::min(memory_capacity, std::max<size_t>(1, owned_seed_count));
    if (capacity == 0 || capacity > static_cast<size_t>(INT_MAX)) {
        abortWithMessage(context, "insufficient GPU memory for one QT candidate cluster");
    }
    workspace.batch_capacity = static_cast<int>(capacity);

    const size_t batch_points = static_cast<size_t>(workspace.batch_capacity) * point_count;
    CUDA_CHECK(context, cudaMalloc(&workspace.seeds,
                                   static_cast<size_t>(workspace.batch_capacity) * sizeof(int)));
    CUDA_CHECK(context, cudaMalloc(&workspace.candidate_max, batch_points * sizeof(double)));
    CUDA_CHECK(context, cudaMalloc(&workspace.in_cluster, batch_points * sizeof(unsigned char)));
    CUDA_CHECK(context, cudaMalloc(&workspace.cardinalities,
                                   static_cast<size_t>(workspace.batch_capacity) * sizeof(int)));
    CUDA_CHECK(context, cudaMalloc(&workspace.members, final_cluster_bytes));
}

BestCandidate reduceBatchOnHost(const std::vector<int>& seeds,
                                const std::vector<int>& cardinalities) {
    const int thread_count = omp_get_max_threads();
    std::vector<BestCandidate> thread_best(static_cast<size_t>(thread_count));

#pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        BestCandidate best;
#pragma omp for schedule(static) nowait
        for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
            const BestCandidate candidate{cardinalities[i], seeds[i]};
            if (isBetter(candidate, best)) {
                best = candidate;
            }
        }
        thread_best[thread] = best;
    }

    BestCandidate result;
    for (const BestCandidate& candidate : thread_best) {
        if (isBetter(candidate, result)) {
            result = candidate;
        }
    }
    return result;
}

BestCandidate evaluateLocalSeeds(DeviceWorkspace& workspace,
                                 const std::vector<unsigned char>& clustered,
                                 const double threshold,
                                 const HybridContext& context) {
    const int point_count = workspace.point_count;
    const int local_thread_count = omp_get_max_threads();
    std::vector<std::vector<int>> seeds_by_thread(static_cast<size_t>(local_thread_count));

    // Cyclic seed ownership spreads adjacent seeds, which commonly belong to
    // the same synthetic group, across GPU ranks.  The final reduction still
    // uses the original global seed number for deterministic ties.
#pragma omp parallel
    {
        std::vector<int>& private_seeds = seeds_by_thread[omp_get_thread_num()];
#pragma omp for schedule(static)
        for (int seed = 0; seed < point_count; ++seed) {
            if (clustered[seed] == 0 && seed % context.size == context.rank) {
                private_seeds.push_back(seed);
            }
        }
    }

    std::vector<int> local_seeds;
    for (const auto& private_seeds : seeds_by_thread) {
        local_seeds.insert(local_seeds.end(), private_seeds.begin(), private_seeds.end());
    }

    CUDA_CHECK(context, cudaMemcpy(workspace.clustered, clustered.data(),
                                   static_cast<size_t>(point_count) * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice));

    BestCandidate local_best;
    std::vector<int> host_cardinalities;
    std::vector<int> batch_seeds;
    for (size_t offset = 0; offset < local_seeds.size();
         offset += static_cast<size_t>(workspace.batch_capacity)) {
        const int batch_size = static_cast<int>(std::min(
            static_cast<size_t>(workspace.batch_capacity), local_seeds.size() - offset));
        CUDA_CHECK(context, cudaMemcpy(workspace.seeds, local_seeds.data() + offset,
                                       static_cast<size_t>(batch_size) * sizeof(int),
                                       cudaMemcpyHostToDevice));
        candidateClusterKernel<<<batch_size, CUDA_THREADS>>>(
            workspace.seeds, batch_size, workspace.clustered, workspace.distances, point_count,
            threshold, workspace.candidate_max, workspace.in_cluster, workspace.cardinalities, nullptr);
        CUDA_CHECK(context, cudaGetLastError());
        host_cardinalities.resize(batch_size);
        CUDA_CHECK(context, cudaMemcpy(host_cardinalities.data(), workspace.cardinalities,
                                       static_cast<size_t>(batch_size) * sizeof(int),
                                       cudaMemcpyDeviceToHost));
        batch_seeds.assign(local_seeds.begin() + static_cast<std::ptrdiff_t>(offset),
                           local_seeds.begin() + static_cast<std::ptrdiff_t>(offset + batch_size));
        const BestCandidate batch_best = reduceBatchOnHost(batch_seeds, host_cardinalities);
        if (isBetter(batch_best, local_best)) {
            local_best = batch_best;
        }
    }
    return local_best;
}

std::vector<int> materializeWinningCluster(DeviceWorkspace& workspace,
                                           const std::vector<unsigned char>& clustered,
                                           const int seed,
                                           const int expected_size,
                                           const double threshold,
                                           const HybridContext& context) {
    CUDA_CHECK(context, cudaMemcpy(workspace.clustered, clustered.data(),
                                   static_cast<size_t>(workspace.point_count) * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK(context, cudaMemcpy(workspace.seeds, &seed, sizeof(seed), cudaMemcpyHostToDevice));
    candidateClusterKernel<<<1, CUDA_THREADS>>>(
        workspace.seeds, 1, workspace.clustered, workspace.distances, workspace.point_count,
        threshold, workspace.candidate_max, workspace.in_cluster, workspace.cardinalities,
        workspace.members);
    CUDA_CHECK(context, cudaGetLastError());

    int actual_size = 0;
    CUDA_CHECK(context, cudaMemcpy(&actual_size, workspace.cardinalities, sizeof(actual_size),
                                   cudaMemcpyDeviceToHost));
    if (actual_size != expected_size) {
        abortWithMessage(context, "non-deterministic CUDA candidate reconstruction");
    }

    std::vector<int> members(static_cast<size_t>(actual_size));
    CUDA_CHECK(context, cudaMemcpy(members.data(), workspace.members,
                                   static_cast<size_t>(actual_size) * sizeof(int),
                                   cudaMemcpyDeviceToHost));
    return members;
}

std::vector<Cluster> qtClusteringHybrid(const std::vector<Point>& points,
                                        const double threshold,
                                        const HybridContext& context) {
    DeviceWorkspace workspace;
    buildDeviceWorkspace(workspace, points, context);

    const int point_count = static_cast<int>(points.size());
    int remaining = point_count;
    std::vector<unsigned char> clustered(static_cast<size_t>(point_count), 0);
    std::vector<Cluster> clusters;

    while (remaining > 0) {
        const BestCandidate local_best = evaluateLocalSeeds(workspace, clustered, threshold, context);
        const int local_result[2] = {-local_best.cardinality, local_best.seed};
        int global_result[2] = {INT_MAX, INT_MAX};
        MPI_Allreduce(local_result, global_result, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        const int winning_size = -global_result[0];
        const int winning_seed = global_result[1];
        if (winning_seed < 0 || winning_seed >= point_count || winning_size <= 0) {
            abortWithMessage(context, "MPI seed reduction did not find a valid QT cluster");
        }

        const int owner = winning_seed % context.size;
        std::vector<int> winning_members;
        if (context.rank == owner) {
            winning_members = materializeWinningCluster(workspace, clustered, winning_seed,
                                                        winning_size, threshold, context);
        } else {
            winning_members.resize(static_cast<size_t>(winning_size));
        }
        MPI_Bcast(winning_members.data(), winning_size, MPI_INT, owner, MPI_COMM_WORLD);

        // QT candidates contain no duplicate member.  Each iteration therefore
        // writes unique bytes, so this OpenMP update has no data races.
#pragma omp parallel for schedule(static)
        for (int member_index = 0; member_index < winning_size; ++member_index) {
            clustered[winning_members[member_index]] = 1;
        }
        remaining -= winning_size;

        if (context.rank == 0) {
            clusters.push_back({std::move(winning_members), winning_seed});
        }
    }

    workspace.release();
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
        const Cluster& cluster = clusters[cluster_index];
        double max_diameter = 0.0;
        for (size_t first = 0; first < cluster.members.size(); ++first) {
            for (size_t second = first + 1; second < cluster.members.size(); ++second) {
                max_diameter = std::max(max_diameter,
                    distance(points[cluster.members[first]], points[cluster.members[second]]));
            }
        }

        if (cluster_index < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", cluster_index,
                        cluster.members.size(), cluster.seed_point, max_diameter);
        }
        if (max_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        cluster_index, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
        for (int member : clusters[cluster_index].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], cluster_index);
                valid = false;
            }
            membership[member] = static_cast<int>(cluster_index);
        }
    }

    int clustered_count = 0;
    for (const int cluster_index : membership) {
        clustered_count += (cluster_index >= 0);
    }
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
                clustered_count, points.size() - static_cast<size_t>(clustered_count));
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
    HybridContext context;
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &context.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &context.size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        abortWithMessage(context, "MPI does not provide the MPI_THREAD_FUNNELED level required by OpenMP");
    }
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, context.rank, MPI_INFO_NULL,
                        &context.node_comm);
    MPI_Comm_rank(context.node_comm, &context.local_rank);
    context.device = context.local_rank;

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    bool parse_error = false;
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
            num_points = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-t") == 0 && argument + 1 < argc) {
            threshold = std::atof(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            show_help = true;
        } else {
            parse_error = true;
        }
    }

    if (show_help || parse_error || num_points <= 0 || threshold <= 0.0) {
        if (context.rank == 0) {
            if (parse_error) {
                std::printf("Unknown or incomplete option\n");
            } else if (!show_help) {
                std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                            num_points, threshold);
            }
            printUsage(argv[0]);
        }
        MPI_Comm_free(&context.node_comm);
        MPI_Finalize();
        return (show_help && !parse_error) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    selectDevice(context);
    if (context.rank == 0) {
        std::printf("QT Clustering Benchmark (MPI + OpenMP + CUDA)\n");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", context.size, omp_get_max_threads());
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    static_assert(std::is_standard_layout<Point>::value && sizeof(Point) == 2 * sizeof(double),
                  "Point must be two contiguous doubles for MPI broadcasting");
    std::vector<Point> points(static_cast<size_t>(num_points));
    if (context.rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_start = std::chrono::steady_clock::now();
    const std::vector<Cluster> clusters = qtClusteringHybrid(points, threshold, context);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_end = std::chrono::steady_clock::now();
    const long local_cluster_time = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start).count());
    long cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (context.rank == 0) {
        std::printf("Clustering time: %ld ms\n", cluster_time);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int cluster_size = static_cast<int>(cluster.members.size());
            total_clustered += cluster_size;
            max_cluster_size = std::max(max_cluster_size, cluster_size);
        }
        const double average_cluster_size = clusters.empty()
                                                ? 0.0
                                                : static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
                    100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);

        const double seconds = cluster_time / 1000.0;
        const double rate_denominator = (seconds > 0.0) ? seconds : 1.0e-9;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / rate_denominator, num_points / rate_denominator);

        if (print_results_requested) {
            std::vector<double> membership_data;
            membership_data.reserve(static_cast<size_t>(num_points));
            std::vector<int> membership(static_cast<size_t>(num_points), -1);
            for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
                for (int member : clusters[cluster_index].members) {
                    membership[member] = static_cast<int>(cluster_index);
                }
            }
            for (int membership_value : membership) {
                membership_data.push_back(static_cast<double>(membership_value));
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate && !validateClusters(clusters, points, threshold)) {
            MPI_Comm_free(&context.node_comm);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        if (validate) {
            std::printf("Validation: PASSED\n");
        }
    }

    MPI_Comm_free(&context.node_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
