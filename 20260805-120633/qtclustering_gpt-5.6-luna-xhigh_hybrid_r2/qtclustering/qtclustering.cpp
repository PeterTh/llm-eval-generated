// QT Clustering Benchmark - hybrid MPI/OpenMP/CUDA implementation
//
// QT (Quality Threshold) clustering builds a cluster from a seed by
// repeatedly adding the closest point whose distance from every member keeps
// the cluster diameter below a threshold.  Cluster rounds are inherently
// sequential, but all seed trials in a round are independent.  This
// implementation distributes those trials across MPI ranks, evaluates them
// concurrently with OpenMP, and evaluates each trial's candidate distances on
// the rank-local CUDA device.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static const int CUDA_THREADS = 256;
static constexpr double SCORE_INFINITY = 1.7976931348623157e+308;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// CUDA and host use the same expression and operation order.  No fast-math
// flags are enabled, since threshold comparisons are part of the semantics.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

struct CandidateScore {
    double score;
    int candidate;
};

// Each thread updates the score of one candidate.  scores[candidate] stores
// the maximum distance seen for this seed so far.  Marking a selected point
// with infinity makes it permanently ineligible without a second per-seed
// membership array on the device.
__global__ void evaluateCandidatesKernel(const Point* points,
                                         const unsigned char* clustered,
                                         double* scores,
                                         const int current_member,
                                         const double threshold,
                                         const int point_count,
                                         CandidateScore* block_best) {
    extern __shared__ CandidateScore reduction[];

    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    CandidateScore best;
    best.score = SCORE_INFINITY;
    best.candidate = INT_MAX;

    if (candidate < point_count && clustered[candidate] == 0) {
        const double previous = scores[candidate];
        if (previous != SCORE_INFINITY) {
            const double candidate_distance = distance(points[candidate],
                                                       points[current_member]);
            const double score = previous > candidate_distance
                               ? previous : candidate_distance;
            scores[candidate] = score;

            if (score < threshold) {
                best.score = score;
                best.candidate = candidate;
            }
        }
    }

    reduction[threadIdx.x] = best;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const CandidateScore other = reduction[threadIdx.x + offset];
            if (other.score < reduction[threadIdx.x].score ||
                (other.score == reduction[threadIdx.x].score &&
                 other.candidate < reduction[threadIdx.x].candidate)) {
                reduction[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_best[blockIdx.x] = reduction[0];
    }
}

[[noreturn]] void cudaFailure(const cudaError_t error,
                              const char* expression,
                              const char* file,
                              const int line) {
    fprintf(stderr, "CUDA failure at %s:%d (%s): %s\n",
            file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cuda_status = (expression); \
        if (cuda_status != cudaSuccess) { \
            cudaFailure(cuda_status, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

class CudaWorkspace {
  public:
    CudaWorkspace(const int point_count,
                  const int block_count,
                  const Point* device_points,
                  const unsigned char* device_clustered)
        : point_count_(point_count),
          block_count_(block_count),
          device_points_(device_points),
          device_clustered_(device_clustered),
          host_block_best_(static_cast<size_t>(block_count)) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_scores_),
                              static_cast<size_t>(point_count_) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_block_best_),
                              static_cast<size_t>(block_count_) * sizeof(CandidateScore)));
    }

    ~CudaWorkspace() {
        if (device_scores_ != nullptr) {
            cudaFree(device_scores_);
        }
        if (device_block_best_ != nullptr) {
            cudaFree(device_block_best_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;

    std::vector<int> generate(const int seed_point,
                               const double threshold) {
        std::vector<int> members;
        members.reserve(static_cast<size_t>(point_count_));
        members.push_back(seed_point);

        CUDA_CHECK(cudaMemsetAsync(device_scores_, 0,
                                   static_cast<size_t>(point_count_) * sizeof(double),
                                   stream_));

        const double infinity = SCORE_INFINITY;
        CUDA_CHECK(cudaMemcpyAsync(device_scores_ + seed_point,
                                   &infinity, sizeof(double),
                                   cudaMemcpyHostToDevice, stream_));

        while (static_cast<int>(members.size()) < point_count_) {
            const int current_member = members.back();
            evaluateCandidatesKernel<<<block_count_, CUDA_THREADS,
                                       CUDA_THREADS * sizeof(CandidateScore),
                                       stream_>>>(
                device_points_, device_clustered_, device_scores_,
                current_member, threshold, point_count_, device_block_best_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(host_block_best_.data(),
                                       device_block_best_,
                                       static_cast<size_t>(block_count_) *
                                           sizeof(CandidateScore),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            CandidateScore best;
            best.score = SCORE_INFINITY;
            best.candidate = INT_MAX;
            for (const CandidateScore block_best : host_block_best_) {
                if (block_best.score < best.score ||
                    (block_best.score == best.score &&
                     block_best.candidate < best.candidate)) {
                    best = block_best;
                }
            }

            if (best.candidate == INT_MAX) {
                break;
            }

            members.push_back(best.candidate);
            CUDA_CHECK(cudaMemcpyAsync(device_scores_ + best.candidate,
                                       &infinity, sizeof(double),
                                       cudaMemcpyHostToDevice, stream_));
        }

        return members;
    }

  private:
    int point_count_;
    int block_count_;
    const Point* device_points_;
    const unsigned char* device_clustered_;
    cudaStream_t stream_ = nullptr;
    double* device_scores_ = nullptr;
    CandidateScore* device_block_best_ = nullptr;
    std::vector<CandidateScore> host_block_best_;
};

class CudaEngine {
  public:
    CudaEngine(const std::vector<Point>& points,
               const int local_rank)
        : point_count_(static_cast<int>(points.size())),
          block_count_((point_count_ + CUDA_THREADS - 1) / CUDA_THREADS) {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            fprintf(stderr, "No CUDA device is available for the MPI rank.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        device_id_ = local_rank % device_count;
        CUDA_CHECK(cudaSetDevice(device_id_));
        CUDA_CHECK(cudaFree(0));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points_),
                              static_cast<size_t>(point_count_) * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered_),
                              static_cast<size_t>(point_count_) *
                                  sizeof(unsigned char)));
        CUDA_CHECK(cudaMemcpy(device_points_, points.data(),
                              static_cast<size_t>(point_count_) * sizeof(Point),
                              cudaMemcpyHostToDevice));

        int requested_threads = omp_get_max_threads();
        if (requested_threads < 1) {
            requested_threads = 1;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        (void)total_bytes;

        const size_t bytes_per_workspace =
            static_cast<size_t>(point_count_) * sizeof(double) +
            static_cast<size_t>(block_count_) * sizeof(CandidateScore);
        // Leave half of the currently free memory available for CUDA runtime
        // allocations and concurrent MPI/OpenMP activity.
        size_t capacity = bytes_per_workspace == 0
                        ? 1
                        : (free_bytes / 2) / bytes_per_workspace;
        if (capacity < 1) {
            capacity = 1;
        }
        const int worker_count = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(requested_threads), capacity));

        omp_set_dynamic(0);
        omp_set_num_threads(worker_count);
        workspaces_.reserve(static_cast<size_t>(worker_count));
        for (int i = 0; i < worker_count; ++i) {
            workspaces_.emplace_back(std::make_unique<CudaWorkspace>(
                point_count_, block_count_, device_points_, device_clustered_));
        }
        worker_count_ = worker_count;
    }

    ~CudaEngine() {
        workspaces_.clear();
        if (device_points_ != nullptr) {
            cudaFree(device_points_);
        }
        if (device_clustered_ != nullptr) {
            cudaFree(device_clustered_);
        }
    }

    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    int workerCount() const { return worker_count_; }

    void activate() const {
        CUDA_CHECK(cudaSetDevice(device_id_));
    }

    void uploadClustered(const std::vector<unsigned char>& clustered) const {
        activate();
        CUDA_CHECK(cudaMemcpy(device_clustered_, clustered.data(),
                              static_cast<size_t>(point_count_) *
                                  sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    void synchronize() const {
        activate();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<int> generate(const int seed_point,
                              const double threshold,
                              const int worker_id) const {
        activate();
        return workspaces_[static_cast<size_t>(worker_id)]->generate(seed_point,
                                                                      threshold);
    }

  private:
    int point_count_;
    int block_count_;
    int device_id_ = 0;
    int worker_count_ = 1;
    Point* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    std::vector<std::unique_ptr<CudaWorkspace>> workspaces_;
};

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
        // Preserve the original stream of generated values for the normal
        // benchmark sizes.  The lower bound only prevents the original
        // generator from looping forever for point counts smaller than 30.
        int group_count = static_cast<int>(frand() * (point_count / 30.0));
        if (point_count <= 30) {
            group_count = 1;
        }

        if (group_count > (point_count - count)) {
            group_count = point_count - count;
        }

        while (group_count > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = sqrt(r * r - dx * dx) * sign;
            const double x = center_x + dx;
            const double y = center_y + dy;

            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }

            points[count] = {x, y};
            ++count;
            --group_count;
        }
    }
}

struct SeedResult {
    int seed = -1;
    int cardinality = -1;
    std::vector<int> members;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  CudaEngine& cuda_engine,
                                  const int mpi_rank,
                                  const int mpi_size) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(static_cast<size_t>(point_count), 0);
    std::vector<int> unclustered_indices(static_cast<size_t>(point_count));
    std::vector<Cluster> clusters;

    for (int i = 0; i < point_count; ++i) {
        unclustered_indices[static_cast<size_t>(i)] = i;
    }

    while (!unclustered_indices.empty()) {
        cuda_engine.uploadClustered(clustered);

        std::vector<int> local_seeds;
        local_seeds.reserve((unclustered_indices.size() +
                             static_cast<size_t>(mpi_size) - 1) /
                            static_cast<size_t>(mpi_size));
        for (size_t i = static_cast<size_t>(mpi_rank);
             i < unclustered_indices.size();
             i += static_cast<size_t>(mpi_size)) {
            local_seeds.push_back(unclustered_indices[i]);
        }

        std::vector<SeedResult> thread_best(
            static_cast<size_t>(cuda_engine.workerCount()));

        // Every OpenMP worker owns one CUDA stream and score workspace.  A
        // dynamic schedule balances seeds whose clusters have different
        // cardinalities.
#pragma omp parallel num_threads(cuda_engine.workerCount())
        {
            const int worker_id = omp_get_thread_num();
            SeedResult& best = thread_best[static_cast<size_t>(worker_id)];

#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < static_cast<int>(local_seeds.size()); ++i) {
                const int seed = local_seeds[static_cast<size_t>(i)];
                std::vector<int> members = cuda_engine.generate(
                    seed, threshold, worker_id);
                const int cardinality = static_cast<int>(members.size());

                if (cardinality > best.cardinality ||
                    (cardinality == best.cardinality &&
                     (best.seed < 0 || seed < best.seed))) {
                    best.seed = seed;
                    best.cardinality = cardinality;
                    best.members = std::move(members);
                }
            }
        }

        cuda_engine.synchronize();

        SeedResult local_best;
        for (SeedResult& candidate : thread_best) {
            if (candidate.cardinality > local_best.cardinality ||
                (candidate.cardinality == local_best.cardinality &&
                 (local_best.seed < 0 || candidate.seed < local_best.seed))) {
                local_best = std::move(candidate);
            }
        }

        int global_cardinality = -1;
        MPI_Allreduce(&local_best.cardinality, &global_cardinality, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int local_seed_for_global = INT_MAX;
        if (local_best.cardinality == global_cardinality) {
            local_seed_for_global = local_best.seed;
        }
        int global_seed = INT_MAX;
        MPI_Allreduce(&local_seed_for_global, &global_seed, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);

        if (global_seed == INT_MAX || global_cardinality <= 0) {
            break;
        }

        int local_winner_rank = mpi_size;
        if (local_best.cardinality == global_cardinality &&
            local_best.seed == global_seed) {
            local_winner_rank = mpi_rank;
        }
        int winner_rank = mpi_size;
        MPI_Allreduce(&local_winner_rank, &winner_rank, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);

        int best_member_count = 0;
        if (mpi_rank == winner_rank) {
            best_member_count = static_cast<int>(local_best.members.size());
        }
        MPI_Bcast(&best_member_count, 1, MPI_INT, winner_rank,
                  MPI_COMM_WORLD);

        std::vector<int> best_members(static_cast<size_t>(best_member_count));
        if (mpi_rank == winner_rank) {
            best_members = std::move(local_best.members);
        }
        MPI_Bcast(best_members.data(), best_member_count, MPI_INT,
                  winner_rank, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global_seed;
        cluster.members = std::move(best_members);
        for (const int member : cluster.members) {
            clustered[static_cast<size_t>(member)] = 1;
        }
        clusters.push_back(std::move(cluster));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(),
                           unclustered_indices.end(),
                           [&clustered](const int index) {
                               return clustered[static_cast<size_t>(index)] != 0;
                           }),
            unclustered_indices.end());
    }

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const Cluster& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[
                    static_cast<size_t>(cluster.members[i])], points[
                    static_cast<size_t>(cluster.members[j])]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (const int member : clusters[c].members) {
            if (membership[static_cast<size_t>(member)] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[static_cast<size_t>(member)], c);
                valid = false;
            }
            membership[static_cast<size_t>(member)] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
    for (const int member_cluster : membership) {
        if (member_cluster >= 0) {
            ++clustered_count;
        }
    }

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count,
           points.size() - static_cast<size_t>(clustered_count));

    return valid;
}

void printUsage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_flag = false;
    bool parse_error = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            print_results_flag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            parse_error = true;
        }
    }

    if (parse_error || num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            if (parse_error) {
                printf("Unknown or incomplete option.\n");
            } else {
                printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                       num_points, threshold);
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points(static_cast<size_t>(num_points));
    if (mpi_rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_communicator);
    int local_rank = 0;
    MPI_Comm_rank(local_communicator, &local_rank);

    std::vector<Cluster> clusters;
    double local_elapsed_seconds = 0.0;
    {
        CudaEngine cuda_engine(points, local_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        const auto cluster_start = std::chrono::steady_clock::now();
        clusters = qtClustering(points, threshold, cuda_engine,
                                mpi_rank, mpi_size);
        cuda_engine.synchronize();
        const auto cluster_end = std::chrono::steady_clock::now();
        local_elapsed_seconds = std::chrono::duration<double>(
            cluster_end - cluster_start).count();
    }
    MPI_Comm_free(&local_communicator);

    double cluster_seconds = 0.0;
    MPI_Reduce(&local_elapsed_seconds, &cluster_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    int validation_result = 0;
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n",
               static_cast<long>(cluster_seconds * 1000.0));
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double average_cluster_size = clusters.empty()
            ? 0.0
            : static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", average_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double clusters_per_second = cluster_seconds > 0.0
            ? clusters.size() / cluster_seconds : 0.0;
        const double points_per_second = cluster_seconds > 0.0
            ? num_points / cluster_seconds : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_second, points_per_second);

        if (print_results_flag) {
            std::vector<double> membership_data;
            membership_data.reserve(static_cast<size_t>(num_points));
            std::vector<int> membership(static_cast<size_t>(num_points), -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[static_cast<size_t>(member)] = static_cast<int>(c);
                }
            }
            for (const int member_cluster : membership) {
                membership_data.push_back(static_cast<double>(member_cluster));
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate) {
            validation_result = validateClusters(clusters, points, threshold)
                              ? 0 : 1;
            printf("Validation: %s\n",
                   validation_result == 0 ? "PASSED" : "FAILED");
        }
    }

    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validation_result;
}
