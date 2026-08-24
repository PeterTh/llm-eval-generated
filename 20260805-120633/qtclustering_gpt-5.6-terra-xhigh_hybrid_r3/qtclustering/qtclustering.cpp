// QT Clustering Benchmark
//
// Hybrid MPI + OpenMP + CUDA implementation.  QT's greedy outer loop is
// necessarily sequential, but the candidate clusters for every possible seed
// in one iteration are independent.  MPI partitions those seeds, CUDA expands
// a rank's candidates in parallel, and OpenMP performs the small host-side
// state updates and reductions.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
// The benchmark uses MPI's C API.  Suppress Open MPI's deprecated C++ binding,
// which otherwise makes NVCC compile third-party callback casts unnecessarily.
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;
static constexpr double CUDA_MAX_DOUBLE = 1.7976931348623157e+308;
static constexpr int CUDA_MAX_INT = 2147483647;

struct Point {
    double x;
    double y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

[[noreturn]] void cudaFail(cudaError_t error, const char* expression,
                           const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file,
                 line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(expression)                                                    \
    do {                                                                          \
        const cudaError_t cuda_error_ = (expression);                            \
        if (cuda_error_ != cudaSuccess) {                                        \
            cudaFail(cuda_error_, #expression, __FILE__, __LINE__);             \
        }                                                                         \
    } while (false)

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

        if (group_count > point_count - count) {
            group_count = point_count - count;
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
            points[count++] = {x, y};
            --group_count;
        }
    }
}

inline double distance(const Point& first, const Point& second) {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    return std::sqrt(dx * dx + dy * dy);
}

// The matrix is stored as distance[member * N + candidate].  Consecutive CUDA
// threads inspect consecutive candidates for a fixed member, making the hot
// loads coalesced.
__global__ void buildDistanceMatrix(const Point* points, double* distances,
                                    int point_count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const int member = blockIdx.y * blockDim.y + threadIdx.y;
    if (candidate >= point_count || member >= point_count) {
        return;
    }

    const double dx = points[candidate].x - points[member].x;
    const double dy = points[candidate].y - points[member].y;
    distances[static_cast<size_t>(member) * point_count + candidate] =
        sqrt(dx * dx + dy * dy);
}

__device__ inline bool isBetterCandidate(double distance_a, int index_a,
                                         double distance_b, int index_b) {
    return distance_a < distance_b ||
           (distance_a == distance_b && index_a < index_b);
}

// One block expands one seed candidate.  Every thread examines a stripe of
// possible points, then the block reduces to the exact serial tie-break:
// smaller maximum diameter and, on a tie, the lower point index.
__global__ void findClosestCandidates(
    const double* distances, const unsigned char* clustered,
    const unsigned char* candidate_memberships, const int* member_lists,
    const int* member_counts, const unsigned char* active, int* next_members,
    int point_count, double threshold) {
    const int local_seed = blockIdx.x;
    if (active[local_seed] == 0) {
        if (threadIdx.x == 0) {
            next_members[local_seed] = -1;
        }
        return;
    }

    const unsigned char* members =
        candidate_memberships + static_cast<size_t>(local_seed) * point_count;
    const int* member_list =
        member_lists + static_cast<size_t>(local_seed) * point_count;
    const int member_count = member_counts[local_seed];
    double best_diameter = CUDA_MAX_DOUBLE;
    int best_index = CUDA_MAX_INT;

    for (int candidate = threadIdx.x; candidate < point_count;
         candidate += blockDim.x) {
        if (clustered[candidate] != 0 || members[candidate] != 0) {
            continue;
        }

        double maximum_distance = 0.0;
        for (int member_index = 0; member_index < member_count; ++member_index) {
            const int member = member_list[member_index];
            maximum_distance = fmax(
                maximum_distance,
                distances[static_cast<size_t>(member) * point_count + candidate]);
        }

        if (maximum_distance < threshold &&
            isBetterCandidate(maximum_distance, candidate, best_diameter,
                              best_index)) {
            best_diameter = maximum_distance;
            best_index = candidate;
        }
    }

    __shared__ double reduced_diameter[CUDA_THREADS];
    __shared__ int reduced_index[CUDA_THREADS];
    reduced_diameter[threadIdx.x] = best_diameter;
    reduced_index[threadIdx.x] = best_index;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset &&
            isBetterCandidate(reduced_diameter[threadIdx.x + offset],
                              reduced_index[threadIdx.x + offset],
                              reduced_diameter[threadIdx.x],
                              reduced_index[threadIdx.x])) {
            reduced_diameter[threadIdx.x] = reduced_diameter[threadIdx.x + offset];
            reduced_index[threadIdx.x] = reduced_index[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        next_members[local_seed] =
            reduced_index[0] == CUDA_MAX_INT ? -1 : reduced_index[0];
    }
}

__global__ void initializeCandidateMemberships(
    unsigned char* candidate_memberships, int* member_lists, int* member_counts,
    unsigned char* active, const unsigned char* clustered, int local_seed_count,
    int point_count, int seed_begin) {
    const int local_seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_seed >= local_seed_count) {
        return;
    }

    const int seed = seed_begin + local_seed;
    const bool usable = clustered[seed] == 0;
    active[local_seed] = usable ? 1 : 0;
    if (usable) {
        const size_t offset = static_cast<size_t>(local_seed) * point_count;
        candidate_memberships[offset + seed] = 1;
        member_lists[offset] = seed;
        member_counts[local_seed] = 1;
    }
}

__global__ void appendClosestCandidates(
    unsigned char* candidate_memberships, int* member_lists, int* member_counts,
    const int* next_members, int local_seed_count, int point_count) {
    const int local_seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_seed >= local_seed_count) {
        return;
    }

    const int next = next_members[local_seed];
    if (next >= 0) {
        const size_t offset = static_cast<size_t>(local_seed) * point_count;
        const int member_count = member_counts[local_seed];
        candidate_memberships[offset + next] = 1;
        member_lists[offset + member_count] = next;
        member_counts[local_seed] = member_count + 1;
    }
}

struct CandidateChoice {
    int cardinality = -1;
    int seed = std::numeric_limits<int>::max();
    int local_seed = -1;
};

bool betterChoice(const CandidateChoice& first, const CandidateChoice& second) {
    return first.cardinality > second.cardinality ||
           (first.cardinality == second.cardinality && first.seed < second.seed);
}

class CudaCandidateEvaluator {
public:
    explicit CudaCandidateEvaluator(const std::vector<Point>& points)
        : point_count_(static_cast<int>(points.size())) {
        const size_t distance_count = static_cast<size_t>(point_count_) * point_count_;
        if (point_count_ <= 0 ||
            distance_count > std::numeric_limits<size_t>::max() / sizeof(double)) {
            throw std::runtime_error("point count is too large for CUDA distance matrix");
        }

        CUDA_CHECK(cudaMalloc(&device_points_, point_count_ * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&device_distances_, distance_count * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&device_clustered_, point_count_ * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemcpy(device_points_, points.data(),
                              point_count_ * sizeof(Point),
                              cudaMemcpyHostToDevice));

        const dim3 block(16, 16);
        const dim3 grid((point_count_ + block.x - 1) / block.x,
                        (point_count_ + block.y - 1) / block.y);
        buildDistanceMatrix<<<grid, block>>>(device_points_, device_distances_,
                                             point_count_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CudaCandidateEvaluator(const CudaCandidateEvaluator&) = delete;
    CudaCandidateEvaluator& operator=(const CudaCandidateEvaluator&) = delete;

    ~CudaCandidateEvaluator() {
        cudaFree(device_active_);
        cudaFree(device_next_members_);
        cudaFree(device_member_counts_);
        cudaFree(device_member_lists_);
        cudaFree(device_memberships_);
        cudaFree(device_clustered_);
        cudaFree(device_distances_);
        cudaFree(device_points_);
    }

    CandidateChoice evaluate(const std::vector<unsigned char>& clustered,
                             int seed_begin, int seed_end,
                             std::vector<unsigned char>& winning_members) {
        const int local_seed_count = seed_end - seed_begin;
        CandidateChoice choice;
        if (local_seed_count == 0) {
            return choice;
        }

        reserveCandidates(local_seed_count);
        CUDA_CHECK(cudaMemcpy(device_clustered_, clustered.data(),
                              point_count_ * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(device_memberships_, 0,
                              static_cast<size_t>(local_seed_count) * point_count_));

        const int one_dimensional_grid =
            (local_seed_count + CUDA_THREADS - 1) / CUDA_THREADS;
        initializeCandidateMemberships<<<one_dimensional_grid, CUDA_THREADS>>>(
            device_memberships_, device_member_lists_, device_member_counts_,
            device_active_, device_clustered_, local_seed_count, point_count_, seed_begin);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> cardinalities(local_seed_count, 0);
        std::vector<unsigned char> active(local_seed_count, 0);
#pragma omp parallel for schedule(static)
        for (int local_seed = 0; local_seed < local_seed_count; ++local_seed) {
            if (clustered[seed_begin + local_seed] == 0) {
                cardinalities[local_seed] = 1;
                active[local_seed] = 1;
            }
        }

        int have_active_candidates = 1;
        while (have_active_candidates != 0) {
            findClosestCandidates<<<local_seed_count, CUDA_THREADS>>>(
                device_distances_, device_clustered_, device_memberships_,
                device_member_lists_, device_member_counts_, device_active_,
                device_next_members_, point_count_, threshold_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host_next_members_.data(), device_next_members_,
                                  local_seed_count * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            have_active_candidates = 0;
#pragma omp parallel for reduction(| : have_active_candidates) schedule(static)
            for (int local_seed = 0; local_seed < local_seed_count; ++local_seed) {
                const int next = host_next_members_[local_seed];
                if (next >= 0) {
                    ++cardinalities[local_seed];
                    active[local_seed] = 1;
                    have_active_candidates = 1;
                } else {
                    active[local_seed] = 0;
                }
            }

            if (have_active_candidates != 0) {
                appendClosestCandidates<<<one_dimensional_grid, CUDA_THREADS>>>(
                    device_memberships_, device_member_lists_, device_member_counts_,
                    device_next_members_, local_seed_count, point_count_);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(device_active_, active.data(), local_seed_count,
                                      cudaMemcpyHostToDevice));
            }
        }

#pragma omp parallel
        {
            CandidateChoice thread_choice;
#pragma omp for nowait schedule(static)
            for (int local_seed = 0; local_seed < local_seed_count; ++local_seed) {
                CandidateChoice current{cardinalities[local_seed],
                                        seed_begin + local_seed, local_seed};
                if (betterChoice(current, thread_choice)) {
                    thread_choice = current;
                }
            }
#pragma omp critical
            {
                if (betterChoice(thread_choice, choice)) {
                    choice = thread_choice;
                }
            }
        }

        if (choice.local_seed >= 0) {
            winning_members.resize(point_count_);
            CUDA_CHECK(cudaMemcpy(
                winning_members.data(),
                device_memberships_ +
                    static_cast<size_t>(choice.local_seed) * point_count_,
                point_count_ * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        }
        return choice;
    }

    void setThreshold(double threshold) { threshold_ = threshold; }

private:
    void reserveCandidates(int local_seed_count) {
        if (local_seed_count <= capacity_) {
            return;
        }
        cudaFree(device_active_);
        cudaFree(device_next_members_);
        cudaFree(device_member_counts_);
        cudaFree(device_member_lists_);
        cudaFree(device_memberships_);
        device_active_ = nullptr;
        device_next_members_ = nullptr;
        device_member_counts_ = nullptr;
        device_member_lists_ = nullptr;
        device_memberships_ = nullptr;

        const size_t membership_count =
            static_cast<size_t>(local_seed_count) * point_count_;
        CUDA_CHECK(cudaMalloc(&device_memberships_, membership_count));
        CUDA_CHECK(cudaMalloc(&device_member_lists_, membership_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_member_counts_, local_seed_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_active_, local_seed_count));
        CUDA_CHECK(cudaMalloc(&device_next_members_, local_seed_count * sizeof(int)));
        host_next_members_.resize(local_seed_count);
        capacity_ = local_seed_count;
    }

    int point_count_;
    int capacity_ = 0;
    double threshold_ = 0.0;
    Point* device_points_ = nullptr;
    double* device_distances_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    unsigned char* device_memberships_ = nullptr;
    int* device_member_lists_ = nullptr;
    int* device_member_counts_ = nullptr;
    unsigned char* device_active_ = nullptr;
    int* device_next_members_ = nullptr;
    std::vector<int> host_next_members_;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int mpi_rank, int mpi_size) {
    const int point_count = static_cast<int>(points.size());
    const int seed_begin = static_cast<int>(
        static_cast<long long>(point_count) * mpi_rank / mpi_size);
    const int seed_end = static_cast<int>(
        static_cast<long long>(point_count) * (mpi_rank + 1) / mpi_size);
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<unsigned char> winning_members(point_count, 0);
    std::vector<Cluster> clusters;

    CudaCandidateEvaluator evaluator(points);
    evaluator.setThreshold(threshold);

    int clustered_count = 0;
    std::vector<int> all_pairs(static_cast<size_t>(mpi_size) * 2);
    while (clustered_count < point_count) {
        const CandidateChoice local_choice =
            evaluator.evaluate(clustered, seed_begin, seed_end, winning_members);

        int local_pair[2] = {local_choice.cardinality, local_choice.seed};
        MPI_Allgather(local_pair, 2, MPI_INT, all_pairs.data(), 2, MPI_INT,
                      MPI_COMM_WORLD);

        CandidateChoice global_choice;
        int owner = -1;
        for (int rank = 0; rank < mpi_size; ++rank) {
            CandidateChoice current{all_pairs[2 * rank], all_pairs[2 * rank + 1], -1};
            if (betterChoice(current, global_choice)) {
                global_choice = current;
                owner = rank;
            }
        }
        if (owner < 0 || global_choice.cardinality <= 0) {
            throw std::runtime_error("QT clustering failed to select a candidate");
        }

        MPI_Bcast(winning_members.data(), point_count, MPI_UNSIGNED_CHAR, owner,
                  MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global_choice.seed;
        cluster.members.reserve(global_choice.cardinality);
        for (int point = 0; point < point_count; ++point) {
            if (winning_members[point] != 0) {
                clustered[point] = 1;
                ++clustered_count;
                if (mpi_rank == 0) {
                    cluster.members.push_back(point);
                }
            }
        }
        if (mpi_rank == 0) {
            clusters.push_back(std::move(cluster));
        }
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points, double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
        const Cluster& cluster = clusters[cluster_index];
        double maximum_diameter = 0.0;
        for (size_t first = 0; first < cluster.members.size(); ++first) {
            for (size_t second = first + 1; second < cluster.members.size(); ++second) {
                maximum_diameter = std::max(
                    maximum_diameter,
                    distance(points[cluster.members[first]], points[cluster.members[second]]));
            }
        }

        if (cluster_index < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        cluster_index, cluster.members.size(), cluster.seed_point,
                        maximum_diameter);
        }
        if (maximum_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        cluster_index, maximum_diameter, threshold);
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
    for (int point_membership : membership) {
        clustered_count += point_membership >= 0;
    }
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
                clustered_count, points.size() - clustered_count);
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
    int mpi_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_level);
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
    int exit_code = 0;

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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                std::printf("Unknown option: %s\n", argv[argument]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    if (mpi_thread_level < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Comm node_communicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL,
                        &node_communicator);
    int local_rank = 0;
    MPI_Comm_rank(node_communicator, &local_rank);
    int gpu_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
    if (gpu_count <= 0) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "No CUDA device is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(local_rank % gpu_count));
    MPI_Comm_free(&node_communicator);

    if (mpi_rank == 0) {
        std::printf("QT Clustering Benchmark (MPI ranks: %d, OpenMP threads/rank: %d)\n",
                    mpi_size, omp_get_max_threads());
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points(num_points);
    if (mpi_rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)), MPI_BYTE, 0,
              MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double clustering_start = MPI_Wtime();
    std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size);
    MPI_Barrier(MPI_COMM_WORLD);
    const double clustering_time = MPI_Wtime() - clustering_start;
    double global_clustering_time = 0.0;
    MPI_Reduce(&clustering_time, &global_clustering_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::printf("Clustering time: %.3f ms\n", global_clustering_time * 1000.0);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int maximum_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            maximum_cluster_size = std::max(maximum_cluster_size, size);
        }
        const double average_cluster_size =
            clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
                    100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", maximum_cluster_size);

        const double elapsed_seconds = std::max(global_clustering_time, 1.0e-9);
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / elapsed_seconds, num_points / elapsed_seconds);

        if (print_results_requested) {
            std::vector<double> membership_data;
            membership_data.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
                for (int member : clusters[cluster_index].members) {
                    membership[member] = static_cast<int>(cluster_index);
                }
            }
            for (int point_membership : membership) {
                membership_data.push_back(static_cast<double>(point_membership));
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate && !validateClusters(clusters, points, threshold)) {
            exit_code = 1;
        } else if (validate) {
            std::printf("Validation: PASSED\n");
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
