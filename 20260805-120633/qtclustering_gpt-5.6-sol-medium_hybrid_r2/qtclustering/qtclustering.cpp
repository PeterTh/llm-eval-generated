// Hybrid MPI + OpenMP + CUDA QT clustering benchmark.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

static void mpiAbortWithCudaError(cudaError_t error, const char* expression,
                                  const char* file, int line) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA error at %s:%d for %s: %s\n", rank,
                 file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 2);
}

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        const cudaError_t cuda_check_error = (expr);                            \
        if (cuda_check_error != cudaSuccess)                                   \
            mpiAbortWithCudaError(cuda_check_error, #expr, __FILE__, __LINE__);\
    } while (false)

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
        group_count = std::min(group_count, count - generated);

        while (group_count > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = center_x + dx;
            const double y = center_y + dy;
            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT)
                continue;
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

__device__ __forceinline__ bool betterCandidate(double lhs_distance, int lhs,
                                                 double rhs_distance, int rhs) {
    return lhs_distance < rhs_distance ||
           (lhs_distance == rhs_distance && lhs < rhs);
}

// A block builds one candidate cluster. Maintaining the maximum distance of
// every point incrementally reduces candidate construction from cubic to
// quadratic work. Threads cooperate on distance updates and deterministic
// minimum selection; independent seeds fill the GPU concurrently.
__global__ void buildCandidateClusters(const Point* __restrict__ points,
                                       const unsigned char* __restrict__ clustered,
                                       const int* __restrict__ seeds,
                                       int seed_count, int point_count,
                                       double threshold,
                                       double* __restrict__ max_distances,
                                       int* __restrict__ cardinalities,
                                       int* __restrict__ output_members) {
    const int seed_slot = blockIdx.x;
    if (seed_slot >= seed_count)
        return;

    __shared__ double reduction_distance[CUDA_THREADS];
    __shared__ int reduction_index[CUDA_THREADS];
    __shared__ int selected;
    __shared__ int cardinality;

    const int thread = threadIdx.x;
    const int seed = seeds[seed_slot];
    double* const seed_distances =
        max_distances + static_cast<size_t>(seed_slot) * point_count;

    for (int candidate = thread; candidate < point_count;
         candidate += blockDim.x) {
        if (clustered[candidate] || candidate == seed) {
            seed_distances[candidate] = DBL_MAX;
        } else {
            const double dx = points[candidate].x - points[seed].x;
            const double dy = points[candidate].y - points[seed].y;
            seed_distances[candidate] = sqrt(dx * dx + dy * dy);
        }
    }
    if (thread == 0) {
        cardinality = 1;
        if (output_members)
            output_members[0] = seed;
    }
    __syncthreads();

    while (cardinality < point_count) {
        double local_distance = DBL_MAX;
        int local_index = INT_MAX;
        for (int candidate = thread; candidate < point_count;
             candidate += blockDim.x) {
            const double candidate_distance = seed_distances[candidate];
            if (candidate_distance < threshold &&
                betterCandidate(candidate_distance, candidate,
                                local_distance, local_index)) {
                local_distance = candidate_distance;
                local_index = candidate;
            }
        }
        reduction_distance[thread] = local_distance;
        reduction_index[thread] = local_index;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread < stride &&
                betterCandidate(reduction_distance[thread + stride],
                                reduction_index[thread + stride],
                                reduction_distance[thread],
                                reduction_index[thread])) {
                reduction_distance[thread] = reduction_distance[thread + stride];
                reduction_index[thread] = reduction_index[thread + stride];
            }
            __syncthreads();
        }

        if (thread == 0) {
            selected = reduction_index[0] == INT_MAX ? -1 : reduction_index[0];
            if (selected >= 0) {
                if (output_members)
                    output_members[cardinality] = selected;
                ++cardinality;
                seed_distances[selected] = DBL_MAX;
            }
        }
        __syncthreads();
        if (selected < 0)
            break;

        const Point added = points[selected];
        for (int candidate = thread; candidate < point_count;
             candidate += blockDim.x) {
            const double old_distance = seed_distances[candidate];
            if (old_distance != DBL_MAX) {
                const double dx = points[candidate].x - added.x;
                const double dy = points[candidate].y - added.y;
                const double new_distance = sqrt(dx * dx + dy * dy);
                seed_distances[candidate] =
                    new_distance > old_distance ? new_distance : old_distance;
            }
        }
        __syncthreads();
    }

    if (thread == 0)
        cardinalities[seed_slot] = cardinality;
}

class CudaCandidateEngine {
public:
    CudaCandidateEngine(const std::vector<Point>& points, int maximum_seed_count)
        : point_count_(static_cast<int>(points.size())) {
        CUDA_CHECK(cudaMalloc(&device_points_, points.size() * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&device_clustered_, points.size()));
        CUDA_CHECK(cudaMemcpy(device_points_, points.data(),
                              points.size() * sizeof(Point),
                              cudaMemcpyHostToDevice));

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;
        const size_t bytes_per_seed = points.size() * sizeof(double);
        const size_t memory_limited = bytes_per_seed == 0
                                          ? 1
                                          : (free_bytes / 2) / bytes_per_seed;
        capacity_ = std::max<size_t>(1, std::min<size_t>(
            static_cast<size_t>(maximum_seed_count),
            std::min<size_t>(memory_limited, 65535)));

        // Fragmentation can make cudaMemGetInfo optimistic. Back off cleanly.
        cudaError_t allocation_error = cudaErrorMemoryAllocation;
        while (capacity_ > 0) {
            allocation_error = cudaMalloc(
                &device_max_distances_, capacity_ * bytes_per_seed);
            if (allocation_error == cudaSuccess)
                break;
            cudaGetLastError();
            capacity_ /= 2;
        }
        if (allocation_error != cudaSuccess || capacity_ == 0)
            CUDA_CHECK(allocation_error);
        CUDA_CHECK(cudaMalloc(&device_seeds_, capacity_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_cardinalities_, capacity_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_members_, points.size() * sizeof(int)));
    }

    ~CudaCandidateEngine() {
        cudaFree(device_members_);
        cudaFree(device_cardinalities_);
        cudaFree(device_seeds_);
        cudaFree(device_max_distances_);
        cudaFree(device_clustered_);
        cudaFree(device_points_);
    }

    CudaCandidateEngine(const CudaCandidateEngine&) = delete;
    CudaCandidateEngine& operator=(const CudaCandidateEngine&) = delete;

    std::pair<int, int> bestLocalSeed(const std::vector<int>& seeds,
                                      const std::vector<unsigned char>& clustered,
                                      double threshold) {
        CUDA_CHECK(cudaMemcpy(device_clustered_, clustered.data(),
                              clustered.size(), cudaMemcpyHostToDevice));
        int best_cardinality = -1;
        int best_seed = INT_MAX;

        for (size_t offset = 0; offset < seeds.size(); offset += capacity_) {
            const int batch_count = static_cast<int>(
                std::min(capacity_, seeds.size() - offset));
            CUDA_CHECK(cudaMemcpy(device_seeds_, seeds.data() + offset,
                                  batch_count * sizeof(int),
                                  cudaMemcpyHostToDevice));
            buildCandidateClusters<<<batch_count, CUDA_THREADS>>>(
                device_points_, device_clustered_, device_seeds_, batch_count,
                point_count_, threshold, device_max_distances_,
                device_cardinalities_, nullptr);
            CUDA_CHECK(cudaGetLastError());

            host_cardinalities_.resize(batch_count);
            CUDA_CHECK(cudaMemcpy(host_cardinalities_.data(),
                                  device_cardinalities_,
                                  batch_count * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            int batch_best_cardinality = -1;
            int batch_best_seed = INT_MAX;
#pragma omp parallel
            {
                int thread_cardinality = -1;
                int thread_seed = INT_MAX;
#pragma omp for nowait schedule(static)
                for (int i = 0; i < batch_count; ++i) {
                    const int seed = seeds[offset + static_cast<size_t>(i)];
                    const int cardinality = host_cardinalities_[i];
                    if (cardinality > thread_cardinality ||
                        (cardinality == thread_cardinality && seed < thread_seed)) {
                        thread_cardinality = cardinality;
                        thread_seed = seed;
                    }
                }
#pragma omp critical
                {
                    if (thread_cardinality > batch_best_cardinality ||
                        (thread_cardinality == batch_best_cardinality &&
                         thread_seed < batch_best_seed)) {
                        batch_best_cardinality = thread_cardinality;
                        batch_best_seed = thread_seed;
                    }
                }
            }
            if (batch_best_cardinality > best_cardinality ||
                (batch_best_cardinality == best_cardinality &&
                 batch_best_seed < best_seed)) {
                best_cardinality = batch_best_cardinality;
                best_seed = batch_best_seed;
            }
        }
        return {best_cardinality, best_seed};
    }

    std::vector<int> materializeSeed(int seed, int expected_cardinality,
                                     double threshold) {
        CUDA_CHECK(cudaMemcpy(device_seeds_, &seed, sizeof(int),
                              cudaMemcpyHostToDevice));
        buildCandidateClusters<<<1, CUDA_THREADS>>>(
            device_points_, device_clustered_, device_seeds_, 1, point_count_,
            threshold, device_max_distances_, device_cardinalities_,
            device_members_);
        CUDA_CHECK(cudaGetLastError());
        int cardinality = 0;
        CUDA_CHECK(cudaMemcpy(&cardinality, device_cardinalities_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (cardinality != expected_cardinality) {
            std::fprintf(stderr,
                         "CUDA candidate cardinality changed from %d to %d\n",
                         expected_cardinality, cardinality);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        std::vector<int> members(cardinality);
        CUDA_CHECK(cudaMemcpy(members.data(), device_members_,
                              cardinality * sizeof(int),
                              cudaMemcpyDeviceToHost));
        return members;
    }

private:
    int point_count_ = 0;
    size_t capacity_ = 0;
    Point* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    int* device_seeds_ = nullptr;
    int* device_cardinalities_ = nullptr;
    int* device_members_ = nullptr;
    double* device_max_distances_ = nullptr;
    std::vector<int> host_cardinalities_;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold, int rank, int world_size) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered(point_count);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < point_count; ++i)
        unclustered[i] = i;

    const int maximum_local_seeds =
        std::max(1, (point_count + world_size - 1) / world_size);
    CudaCandidateEngine engine(points, maximum_local_seeds);
    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        std::vector<int> local_seeds;
        local_seeds.reserve((unclustered.size() + world_size - 1) / world_size);
        for (size_t i = rank; i < unclustered.size(); i += world_size)
            local_seeds.push_back(unclustered[i]);

        const auto local_best =
            engine.bestLocalSeed(local_seeds, clustered, threshold);
        int local_pair[2] = {local_best.first, local_best.second};
        int global_pair[2] = {-1, INT_MAX};
        MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        const int best_cardinality = global_pair[0];
        const int best_seed = global_pair[1];
        if (best_cardinality <= 0 || best_seed == INT_MAX)
            break;

        const auto position = std::lower_bound(unclustered.begin(),
                                               unclustered.end(), best_seed);
        const int owner = static_cast<int>(position - unclustered.begin()) %
                          world_size;
        std::vector<int> best_members(best_cardinality);
        if (rank == owner)
            best_members = engine.materializeSeed(best_seed, best_cardinality,
                                                  threshold);
        MPI_Bcast(best_members.data(), best_cardinality, MPI_INT, owner,
                  MPI_COMM_WORLD);

        clusters.push_back({best_members, best_seed});
#pragma omp parallel for schedule(static)
        for (int i = 0; i < best_cardinality; ++i)
            clustered[best_members[i]] = 1;

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int index) { return clustered[index]; }),
            unclustered.end());
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points, double threshold) {
    std::vector<double> diameters(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < clusters.size(); ++c) {
        double maximum = 0.0;
        const auto& members = clusters[c].members;
        for (size_t i = 0; i < members.size(); ++i)
            for (size_t j = i + 1; j < members.size(); ++j)
                maximum = std::max(
                    maximum,
                    pointDistance(points[members[i]], points[members[j]]));
        diameters[c] = maximum;
    }

    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        if (c < 10)
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        c, clusters[c].members.size(), clusters[c].seed_point,
                        diameters[c]);
        if (diameters[c] > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        c, diameters[c], threshold);
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
        membership.begin(), membership.end(), [](int value) { return value >= 0; }));
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
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rank = 0;
    MPI_Comm shared_communicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &shared_communicator);
    MPI_Comm_rank(shared_communicator, &local_rank);
    MPI_Comm_free(&shared_communicator);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0)
            std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            num_points = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            threshold = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (std::strcmp(argv[i], "-r") == 0)
            print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0)
            parse_status = 2;
        else {
            if (rank == 0)
                std::printf("Unknown option: %s\n", argv[i]);
            parse_status = 1;
        }
    }
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0)
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        parse_status = 1;
    }
    if (parse_status != 0) {
        if (rank == 0)
            printUsage(argv[0]);
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI ranks, %d OpenMP threads/rank, CUDA\n",
                    world_size, omp_get_max_threads());
    }

    std::vector<Point> points(num_points);
    if (rank == 0)
        generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, world_size);
    const double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int exit_status = 0;
    if (rank == 0) {
        const long elapsed_ms = static_cast<long>(elapsed * 1000.0);
        std::printf("Clustering time: %ld ms\n", elapsed_ms);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int maximum_cluster_size = 0;
#pragma omp parallel for reduction(+ : total_clustered) reduction(max : maximum_cluster_size)
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            maximum_cluster_size = std::max(maximum_cluster_size, size);
        }
        const double average_cluster_size = clusters.empty()
                                                ? 0.0
                                                : static_cast<double>(total_clustered) /
                                                      clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
                    num_points, 100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", maximum_cluster_size);
        const double safe_elapsed = std::max(elapsed, 1.0e-12);
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / safe_elapsed, num_points / safe_elapsed);

        if (print_results_requested) {
            std::vector<double> membership_data(num_points, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (const int member : clusters[c].members)
                    membership_data[member] = static_cast<double>(c);
            print_results(membership_data, "ClusterMembership");
        }
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_status;
}
