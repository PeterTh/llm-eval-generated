// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA implementation

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
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

struct CandidateResult {
    int cardinality;
    int seed;
};

static void mpiCheck(const int error, const char* expression,
                     const char* file, const int line) {
    if (error == MPI_SUCCESS) return;

    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "MPI error on rank %d at %s:%d (%s): %.*s\n",
                 rank, file, line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define MPI_CHECK(call) mpiCheck((call), #call, __FILE__, __LINE__)

static void cudaCheck(const cudaError_t error, const char* expression,
                      const char* file, const int line) {
    if (error == cudaSuccess) return;

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "CUDA error on rank %d at %s:%d (%s): %s\n",
                 rank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

void generateSyntheticData(std::vector<Point>& points, const int point_count,
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
        group_count = std::min(group_count, point_count - count);

        while (group_count > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = center_x + dx;
            const double y = center_y + dy;

            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y};
            --group_count;
        }
    }
}

inline double pointDistance(const Point& first, const Point& second) {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    return std::sqrt(dx * dx + dy * dy);
}

/*
 * One persistent block evaluates one seed at a time.  Each block owns a row
 * of candidate diameters and obtains new seeds through a global work queue.
 * This keeps exactly the original greedy ordering while avoiding the serial
 * implementation's repeated scan over every member already in the cluster.
 */
template <bool store_members>
__global__ void candidateClustersKernel(const Point* __restrict__ points,
                                        const unsigned char* __restrict__ active,
                                        const int* __restrict__ seeds,
                                        const int seed_count,
                                        const int point_count,
                                        const double threshold,
                                        double* __restrict__ workspace,
                                        int* __restrict__ cardinalities,
                                        int* __restrict__ members,
                                        int* __restrict__ next_seed) {
    __shared__ double reduction_distance[CUDA_THREADS];
    __shared__ int reduction_index[CUDA_THREADS];
    __shared__ int work_index;

    const int thread = threadIdx.x;
    double* const candidate_diameter =
        workspace + static_cast<size_t>(blockIdx.x) * point_count;

    while (true) {
        if (thread == 0) work_index = atomicAdd(next_seed, 1);
        __syncthreads();

        const int seed_slot = work_index;
        if (seed_slot >= seed_count) return;

        const int seed = seeds[seed_slot];
        for (int candidate = thread; candidate < point_count;
             candidate += blockDim.x) {
            candidate_diameter[candidate] =
                (active[candidate] != 0 && candidate != seed) ? 0.0 : DBL_MAX;
        }
        if constexpr (store_members) {
            if (thread == 0) {
                members[static_cast<size_t>(seed_slot) * point_count] = seed;
            }
        }
        __syncthreads();

        int cardinality = 1;
        int last_member = seed;
        while (true) {
            double local_distance = DBL_MAX;
            int local_index = INT_MAX;
            const Point member_point = points[last_member];

            // Update each candidate's maximum distance and reduce by the
            // lexicographic key (diameter, point index).  The point-index tie
            // break is the sequential candidate-loop rule.
            for (int candidate = thread; candidate < point_count;
                 candidate += blockDim.x) {
                double diameter = candidate_diameter[candidate];
                if (diameter == DBL_MAX) continue;

                const double dx = points[candidate].x - member_point.x;
                const double dy = points[candidate].y - member_point.y;
                const double candidate_distance = sqrt(dx * dx + dy * dy);
                if (candidate_distance > diameter) diameter = candidate_distance;
                candidate_diameter[candidate] = diameter;

                if (diameter < threshold &&
                    (diameter < local_distance ||
                     (diameter == local_distance && candidate < local_index))) {
                    local_distance = diameter;
                    local_index = candidate;
                }
            }

            reduction_distance[thread] = local_distance;
            reduction_index[thread] = local_index;
            __syncthreads();

            for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
                if (thread < offset) {
                    const double other_distance = reduction_distance[thread + offset];
                    const int other_index = reduction_index[thread + offset];
                    if (other_distance < reduction_distance[thread] ||
                        (other_distance == reduction_distance[thread] &&
                         other_index < reduction_index[thread])) {
                        reduction_distance[thread] = other_distance;
                        reduction_index[thread] = other_index;
                    }
                }
                __syncthreads();
            }

            const int selected = reduction_index[0];
            if (selected == INT_MAX) break;

            if (thread == 0) {
                candidate_diameter[selected] = DBL_MAX;
                if constexpr (store_members) {
                    members[static_cast<size_t>(seed_slot) * point_count +
                            cardinality] = selected;
                }
            }
            ++cardinality;
            last_member = selected;
            __syncthreads();
        }

        if (thread == 0) cardinalities[seed_slot] = cardinality;
        __syncthreads();
    }
}

__global__ void markInactiveKernel(unsigned char* __restrict__ active,
                                   const int* __restrict__ members,
                                   const int member_count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < member_count) active[members[index]] = 0;
}

class CudaCandidateEngine {
public:
    CudaCandidateEngine(const std::vector<Point>& points, const int device)
        : point_count_(static_cast<int>(points.size())), device_(device) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaGetDeviceProperties(&properties_, device_));

        const size_t point_bytes = points.size() * sizeof(Point);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points_), point_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_active_),
                              points.size() * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_seeds_),
                              points.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_cardinalities_),
                              points.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_members_),
                              points.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_next_seed_), sizeof(int)));
        CUDA_CHECK(cudaMemcpy(device_points_, points.data(), point_bytes,
                              cudaMemcpyHostToDevice));

        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, candidateClustersKernel<false>, CUDA_THREADS, 0));
        const int occupancy_workers =
            std::max(1, blocks_per_sm * properties_.multiProcessorCount);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;
        const size_t bytes_per_worker = points.size() * sizeof(double);
        const size_t workspace_budget = free_bytes * 3 / 4;
        const size_t memory_workers = workspace_budget / bytes_per_worker;
        if (memory_workers == 0) {
            std::fprintf(stderr,
                         "CUDA device %d does not have enough memory for one QT "
                         "candidate workspace (%zu bytes required)\n",
                         device_, bytes_per_worker);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            std::abort();
        }

        worker_capacity_ = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(std::min(point_count_, occupancy_workers)),
            memory_workers));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_workspace_),
                              static_cast<size_t>(worker_capacity_) *
                                  bytes_per_worker));
    }

    CudaCandidateEngine(const CudaCandidateEngine&) = delete;
    CudaCandidateEngine& operator=(const CudaCandidateEngine&) = delete;

    ~CudaCandidateEngine() {
        cudaFree(device_workspace_);
        cudaFree(device_next_seed_);
        cudaFree(device_members_);
        cudaFree(device_cardinalities_);
        cudaFree(device_seeds_);
        cudaFree(device_active_);
        cudaFree(device_points_);
    }

    void setActive(const std::vector<unsigned char>& active) {
        CUDA_CHECK(cudaMemcpy(device_active_, active.data(),
                              active.size() * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    void removeMembers(const std::vector<int>& members) {
        CUDA_CHECK(cudaMemcpy(device_members_, members.data(),
                              members.size() * sizeof(int), cudaMemcpyHostToDevice));
        const int member_count = static_cast<int>(members.size());
        const int blocks = (member_count + CUDA_THREADS - 1) / CUDA_THREADS;
        markInactiveKernel<<<blocks, CUDA_THREADS>>>(
            device_active_, device_members_, member_count);
        CUDA_CHECK(cudaGetLastError());
    }

    CandidateResult evaluate(const std::vector<int>& seeds,
                             const double threshold) {
        if (seeds.empty()) return {-1, INT_MAX};

        CUDA_CHECK(cudaMemcpy(device_seeds_, seeds.data(),
                              seeds.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_next_seed_, 0, sizeof(int)));
        const int workers =
            std::min(worker_capacity_, static_cast<int>(seeds.size()));
        candidateClustersKernel<false><<<workers, CUDA_THREADS>>>(
            device_points_, device_active_, device_seeds_,
            static_cast<int>(seeds.size()), point_count_, threshold,
            device_workspace_, device_cardinalities_, nullptr, device_next_seed_);
        CUDA_CHECK(cudaGetLastError());

        host_cardinalities_.resize(seeds.size());
        CUDA_CHECK(cudaMemcpy(host_cardinalities_.data(), device_cardinalities_,
                              seeds.size() * sizeof(int), cudaMemcpyDeviceToHost));

        CandidateResult best{-1, INT_MAX};
        for (size_t index = 0; index < seeds.size(); ++index) {
            const int cardinality = host_cardinalities_[index];
            if (cardinality > best.cardinality ||
                (cardinality == best.cardinality && seeds[index] < best.seed)) {
                best = {cardinality, seeds[index]};
            }
        }
        return best;
    }

    std::vector<int> generateMembers(const int seed, const double threshold) {
        CUDA_CHECK(cudaMemcpy(device_seeds_, &seed, sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_next_seed_, 0, sizeof(int)));
        candidateClustersKernel<true><<<1, CUDA_THREADS>>>(
            device_points_, device_active_, device_seeds_, 1, point_count_,
            threshold, device_workspace_, device_cardinalities_, device_members_,
            device_next_seed_);
        CUDA_CHECK(cudaGetLastError());

        int cardinality = 0;
        CUDA_CHECK(cudaMemcpy(&cardinality, device_cardinalities_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        std::vector<int> members(cardinality);
        CUDA_CHECK(cudaMemcpy(members.data(), device_members_,
                              members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        return members;
    }

    const char* deviceName() const { return properties_.name; }
    int workerCapacity() const { return worker_capacity_; }

private:
    int point_count_ = 0;
    int device_ = 0;
    int worker_capacity_ = 0;
    cudaDeviceProp properties_{};
    Point* device_points_ = nullptr;
    unsigned char* device_active_ = nullptr;
    int* device_seeds_ = nullptr;
    int* device_cardinalities_ = nullptr;
    int* device_members_ = nullptr;
    int* device_next_seed_ = nullptr;
    double* device_workspace_ = nullptr;
    std::vector<int> host_cardinalities_;
};

static int selectCudaDevice() {
    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_communicator));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_communicator, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_communicator));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "MPI rank %d found no CUDA-capable device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }

    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  CudaCandidateEngine& engine,
                                  const int rank, const int world_size) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> active(point_count);
    std::vector<int> active_indices(point_count);

#pragma omp parallel for schedule(static) if (point_count >= 16384)
    for (int point = 0; point < point_count; ++point) {
        active[point] = 1;
        active_indices[point] = point;
    }

    std::vector<Cluster> clusters;
    std::vector<int> local_seeds;
    local_seeds.reserve((point_count + world_size - 1) / world_size);
    engine.setActive(active);

    while (!active_indices.empty()) {
        local_seeds.clear();
        for (size_t position = static_cast<size_t>(rank);
             position < active_indices.size(); position += world_size) {
            local_seeds.push_back(active_indices[position]);
        }

        const CandidateResult local_best = engine.evaluate(local_seeds, threshold);
        struct {
            int cardinality;
            int seed;
        } local_pair{local_best.cardinality, local_best.seed}, global_pair{};
        MPI_CHECK(MPI_Allreduce(&local_pair, &global_pair, 1, MPI_2INT,
                                MPI_MAXLOC, MPI_COMM_WORLD));

        if (global_pair.cardinality <= 0 || global_pair.seed == INT_MAX) break;

        // active_indices is sorted and identically replicated, so the cyclic
        // partition tells every rank the owner without another collective.
        const auto winner_position =
            std::lower_bound(active_indices.begin(), active_indices.end(),
                             global_pair.seed);
        const int owner = static_cast<int>(
            std::distance(active_indices.begin(), winner_position) % world_size);

        std::vector<int> members;
        if (rank == owner) {
            members = engine.generateMembers(global_pair.seed, threshold);
            if (static_cast<int>(members.size()) != global_pair.cardinality) {
                std::fprintf(stderr,
                             "Rank %d generated inconsistent cardinality for seed %d "
                             "(%zu instead of %d)\n",
                             rank, global_pair.seed, members.size(),
                             global_pair.cardinality);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                std::abort();
            }
        } else {
            members.resize(global_pair.cardinality);
        }
        MPI_CHECK(MPI_Bcast(members.data(), global_pair.cardinality, MPI_INT,
                            owner, MPI_COMM_WORLD));
        engine.removeMembers(members);

        if (rank == 0) clusters.push_back({members, global_pair.seed});

#pragma omp parallel for schedule(static) if (global_pair.cardinality >= 4096)
        for (int index = 0; index < global_pair.cardinality; ++index) {
            active[members[index]] = 0;
        }
        active_indices.erase(
            std::remove_if(active_indices.begin(), active_indices.end(),
                           [&active](const int point) { return active[point] == 0; }),
            active_indices.end());
    }

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    for (size_t cluster_index = 0; cluster_index < clusters.size();
         ++cluster_index) {
        const Cluster& cluster = clusters[cluster_index];
        double max_diameter = 0.0;

#pragma omp parallel for reduction(max : max_diameter) schedule(static) \
    if (cluster.members.size() >= 256)
        for (long long first = 0;
             first < static_cast<long long>(cluster.members.size()); ++first) {
            for (size_t second = static_cast<size_t>(first) + 1;
                 second < cluster.members.size(); ++second) {
                max_diameter =
                    std::max(max_diameter,
                             pointDistance(points[cluster.members[first]],
                                           points[cluster.members[second]]));
            }
        }

        if (cluster_index < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        cluster_index, cluster.members.size(), cluster.seed_point,
                        max_diameter);
        }
        if (max_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        cluster_index, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t cluster_index = 0; cluster_index < clusters.size();
         ++cluster_index) {
        for (const int member : clusters[cluster_index].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], cluster_index);
                valid = false;
            }
            membership[member] = static_cast<int>(cluster_index);
        }
    }

    int clustered_count = 0;
#pragma omp parallel for reduction(+ : clustered_count) schedule(static) \
    if (membership.size() >= 16384)
    for (long long point = 0; point < static_cast<long long>(membership.size());
         ++point) {
        clustered_count += membership[point] >= 0;
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
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int init_error =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (init_error != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required FUNNELED thread level\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    bool arguments_valid = true;

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
            show_help = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[index]);
            arguments_valid = false;
            break;
        }
    }

    if (show_help || !arguments_valid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return arguments_valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);

    static_assert(std::is_standard_layout_v<Point> && sizeof(Point) == 2 * sizeof(double) &&
                  offsetof(Point, y) == sizeof(double));
    MPI_Datatype point_type = MPI_DATATYPE_NULL;
    MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &point_type));
    MPI_CHECK(MPI_Type_commit(&point_type));
    MPI_CHECK(MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_free(&point_type));

    const int cuda_device = selectCudaDevice();
    int exit_code = EXIT_SUCCESS;
    {
        CudaCandidateEngine engine(points, cuda_device);
        if (rank == 0) {
            std::printf("Hybrid execution: %d MPI rank(s), %d OpenMP thread(s)/rank, "
                        "CUDA device %d (%s), %d resident candidate block(s)\n",
                        world_size, omp_get_max_threads(), cuda_device,
                        engine.deviceName(), engine.workerCapacity());
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double clustering_start = MPI_Wtime();
        const std::vector<Cluster> clusters =
            qtClustering(points, threshold, engine, rank, world_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double local_elapsed = MPI_Wtime() - clustering_start;
        double elapsed = 0.0;
        MPI_CHECK(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                             MPI_COMM_WORLD));

        if (rank == 0) {
            const long elapsed_ms = static_cast<long>(elapsed * 1000.0);
            std::printf("Clustering time: %ld ms\n", elapsed_ms);
            std::printf("Clusters found: %zu\n", clusters.size());

            int total_clustered = 0;
            int max_cluster_size = 0;
            for (const Cluster& cluster : clusters) {
                const int size = static_cast<int>(cluster.members.size());
                total_clustered += size;
                max_cluster_size = std::max(max_cluster_size, size);
            }
            const double average_cluster_size =
                clusters.empty()
                    ? 0.0
                    : static_cast<double>(total_clustered) / clusters.size();

            std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
                        num_points, 100.0 * total_clustered / num_points);
            std::printf("Average cluster size: %.2f\n", average_cluster_size);
            std::printf("Maximum cluster size: %d\n", max_cluster_size);
            const double measured_seconds = std::max(elapsed, 1.0e-12);
            std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                        clusters.size() / measured_seconds,
                        num_points / measured_seconds);

            if (print_results_requested) {
                std::vector<int> membership(num_points, -1);
                for (size_t cluster_index = 0; cluster_index < clusters.size();
                     ++cluster_index) {
                    for (const int member : clusters[cluster_index].members) {
                        membership[member] = static_cast<int>(cluster_index);
                    }
                }
                std::vector<double> membership_data(num_points);
#pragma omp parallel for schedule(static) if (num_points >= 16384)
                for (int point = 0; point < num_points; ++point) {
                    membership_data[point] = static_cast<double>(membership[point]);
                }
                print_results(membership_data, "ClusterMembership");
            }

            if (validate) {
                const bool valid = validateClusters(clusters, points, threshold);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                if (!valid) exit_code = EXIT_FAILURE;
            }
        }

        MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    MPI_CHECK(MPI_Finalize());
    return exit_code;
}
