// QT Clustering Benchmark - MPI + OpenMP + CUDA implementation
//
// QT (Quality Threshold) clustering builds one cluster at a time.  The
// greedy expansion of a single candidate cluster is inherently sequential,
// but all seed candidates for one expansion are independent.  MPI distributes
// those seeds between ranks, CUDA evaluates the candidates in parallel, and
// OpenMP performs the small host-side reductions and validation work.

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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// A score is valid when candidate >= 0.  The candidate field is part of the
// ordering so reduction order cannot change the original lowest-index tie
// breaking rule.
struct CandidateScore {
    double max_distance;
    int candidate;
};

__host__ __device__ inline CandidateScore betterScore(const CandidateScore a,
                                                       const CandidateScore b) {
    if (a.candidate < 0) return b;
    if (b.candidate < 0) return a;
    if (a.max_distance < b.max_distance) return a;
    if (b.max_distance < a.max_distance) return b;
    return (a.candidate < b.candidate) ? a : b;
}

__host__ __device__ inline CandidateScore invalidScore() {
    return {DBL_MAX, -1};
}

// One block evaluates a contiguous range of candidate points and leaves its
// best candidate in block_scores.  The distance calculation intentionally
// uses sqrt and no fast-math approximation to preserve the sequential
// implementation's threshold and ordering semantics.
__global__ void findClosestCandidatesKernel(const Point* points,
                                             const unsigned char* clustered,
                                             const unsigned char* in_cluster,
                                             const int* members,
                                             const int member_count,
                                             const int point_count,
                                             const double threshold,
                                             CandidateScore* block_scores) {
    extern __shared__ CandidateScore shared_scores[];

    const int local_index = static_cast<int>(threadIdx.x);
    const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    CandidateScore score = invalidScore();

    if (candidate < point_count && !clustered[candidate] && !in_cluster[candidate]) {
        const Point candidate_point = points[candidate];
        double max_distance = 0.0;

        for (int i = 0; i < member_count; ++i) {
            const Point member_point = points[members[i]];
            const double dx = candidate_point.x - member_point.x;
            const double dy = candidate_point.y - member_point.y;
            const double current_distance = sqrt(dx * dx + dy * dy);
            if (current_distance > max_distance) {
                max_distance = current_distance;
            }
        }

        if (max_distance < threshold) {
            score = {max_distance, candidate};
        }
    }

    shared_scores[local_index] = score;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            shared_scores[local_index] =
                betterScore(shared_scores[local_index], shared_scores[local_index + stride]);
        }
        __syncthreads();
    }

    if (local_index == 0) {
        block_scores[blockIdx.x] = shared_scores[0];
    }
}

[[noreturn]] void abortCuda(const cudaError_t error,
                            const char* expression,
                            const char* file,
                            const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",
                 file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cuda_check_error = (expression); \
        if (cuda_check_error != cudaSuccess) { \
            abortCuda(cuda_check_error, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

int localMpiRank() {
    // These are the local-rank variables used by the common MPI launchers.
    // CUDA_VISIBLE_DEVICES makes the selected ordinal local to the process.
    const char* names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID"
    };

    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            char* end = nullptr;
            const long rank = std::strtol(value, &end, 10);
            if (end != value && rank >= 0 && rank <= std::numeric_limits<int>::max()) {
                return static_cast<int>(rank);
            }
        }
    }

    // With no launcher-specific local rank, MPI's global rank is the most
    // useful deterministic mapping for a single-node run.
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

class CudaCandidateSearch {
  public:
    CudaCandidateSearch(const std::vector<Point>& points, const int mpi_rank)
        : point_count_(static_cast<int>(points.size())),
          block_count_((point_count_ + CUDA_THREADS - 1) / CUDA_THREADS),
          block_scores_(static_cast<size_t>(block_count_)) {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            std::fprintf(stderr, "No CUDA device is visible to MPI rank %d\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, 2);
            std::abort();
        }

        const int device = localMpiRank() % device_count;
        CUDA_CHECK(cudaSetDevice(device));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_points_),
                              points.size() * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_clustered_),
                              points.size() * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_in_cluster_),
                              points.size() * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_members_),
                              points.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_block_scores_),
                              static_cast<size_t>(block_count_) * sizeof(CandidateScore)));

        CUDA_CHECK(cudaMemcpy(device_points_, points.data(), points.size() * sizeof(Point),
                              cudaMemcpyHostToDevice));
    }

    CudaCandidateSearch(const CudaCandidateSearch&) = delete;
    CudaCandidateSearch& operator=(const CudaCandidateSearch&) = delete;

    ~CudaCandidateSearch() {
        cudaFree(device_block_scores_);
        cudaFree(device_members_);
        cudaFree(device_in_cluster_);
        cudaFree(device_clustered_);
        cudaFree(device_points_);
    }

    void setClustered(const std::vector<unsigned char>& clustered) {
        CUDA_CHECK(cudaMemcpy(device_clustered_, clustered.data(),
                              clustered.size() * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    int generateCandidateCluster(const int seed_point,
                                 const double threshold,
                                 std::vector<int>& cluster_members) {
        cluster_members.clear();
        cluster_members.reserve(static_cast<size_t>(point_count_));

        CUDA_CHECK(cudaMemset(device_in_cluster_, 0,
                              static_cast<size_t>(point_count_) * sizeof(unsigned char)));
        const unsigned char seed_flag = 1;
        CUDA_CHECK(cudaMemcpy(device_in_cluster_ + seed_point, &seed_flag,
                              sizeof(seed_flag), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_members_, &seed_point, sizeof(seed_point),
                              cudaMemcpyHostToDevice));
        cluster_members.push_back(seed_point);

        while (static_cast<int>(cluster_members.size()) < point_count_) {
            findClosestCandidatesKernel<<<block_count_, CUDA_THREADS,
                                          CUDA_THREADS * sizeof(CandidateScore)>>>(
                device_points_, device_clustered_, device_in_cluster_, device_members_,
                static_cast<int>(cluster_members.size()), point_count_, threshold,
                device_block_scores_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(block_scores_.data(), device_block_scores_,
                                  static_cast<size_t>(block_count_) * sizeof(CandidateScore),
                                  cudaMemcpyDeviceToHost));

            const CandidateScore best = reduceBlockScoresWithOpenMP();
            if (best.candidate < 0) {
                break;
            }

            const int next_member = best.candidate;
            cluster_members.push_back(next_member);

            const unsigned char member_flag = 1;
            CUDA_CHECK(cudaMemcpy(device_in_cluster_ + next_member, &member_flag,
                                  sizeof(member_flag), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(device_members_ + cluster_members.size() - 1,
                                  &next_member, sizeof(next_member), cudaMemcpyHostToDevice));
        }

        return static_cast<int>(cluster_members.size());
    }

  private:
    CandidateScore reduceBlockScoresWithOpenMP() const {
        const int workers = std::max(1, omp_get_max_threads());
        std::vector<CandidateScore> per_worker(static_cast<size_t>(workers), invalidScore());

#pragma omp parallel
        {
            const int worker = omp_get_thread_num();
            CandidateScore local_best = invalidScore();

#pragma omp for schedule(static)
            for (int block = 0; block < block_count_; ++block) {
                local_best = betterScore(local_best, block_scores_[block]);
            }

            per_worker[static_cast<size_t>(worker)] = local_best;
        }

        CandidateScore best = invalidScore();
        for (const CandidateScore score : per_worker) {
            best = betterScore(best, score);
        }
        return best;
    }

    int point_count_;
    int block_count_;
    std::vector<CandidateScore> block_scores_;
    Point* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    unsigned char* device_in_cluster_ = nullptr;
    int* device_members_ = nullptr;
    CandidateScore* device_block_scores_ = nullptr;
};

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };

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
            count++;
            group_cnt--;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// MPI distributes seed candidates, while the CUDA engine above performs the
// greedy expansion of each seed.  The reduction uses max cardinality and then
// minimum seed, exactly matching the original ascending-seed tie break.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int mpi_rank,
                                  const int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(static_cast<size_t>(N), 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<Cluster> clusters;
    CudaCandidateSearch cuda_search(points, mpi_rank);

    while (!unclustered_indices.empty()) {
        cuda_search.setClustered(clustered);

        int local_result[2] = {-1, std::numeric_limits<int>::max()};
        std::vector<int> local_best_members;

        for (const int seed : unclustered_indices) {
            if (seed % mpi_size != mpi_rank) {
                continue;
            }

            std::vector<int> candidate_members;
            const int cardinality = cuda_search.generateCandidateCluster(
                seed, threshold, candidate_members);
            if (cardinality > local_result[0] ||
                (cardinality == local_result[0] && seed < local_result[1])) {
                local_result[0] = cardinality;
                local_result[1] = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        int global_result[2] = {-1, std::numeric_limits<int>::max()};
        MPI_Allreduce(local_result, global_result, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        if (global_result[0] <= 0) {
            break;
        }

        const int winning_seed = global_result[1];
        const int winner_rank = winning_seed % mpi_size;
        std::vector<int> winning_members;

        if (mpi_rank == winner_rank) {
            if (local_result[1] == winning_seed) {
                winning_members = std::move(local_best_members);
            } else {
                // The owner may have evaluated the winning seed but discarded
                // its members after finding a better local seed.  Recompute it
                // once on that owner so only the selected cluster is broadcast.
                cuda_search.generateCandidateCluster(winning_seed, threshold, winning_members);
            }
        }

        int winning_size = global_result[0];
        MPI_Bcast(&winning_size, 1, MPI_INT, winner_rank, MPI_COMM_WORLD);
        if (mpi_rank != winner_rank) {
            winning_members.resize(static_cast<size_t>(winning_size));
        }
        MPI_Bcast(winning_members.data(), winning_size, MPI_INT, winner_rank,
                  MPI_COMM_WORLD);

        for (const int member : winning_members) {
            clustered[member] = 1;
        }

        if (mpi_rank == 0) {
            clusters.push_back({winning_members, winning_seed});
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int index) { return clustered[index] != 0; }),
            unclustered_indices.end());
    }

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    std::vector<double> diameters(clusters.size(), 0.0);
    std::vector<unsigned char> diameter_valid(clusters.size(), 1);

#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < static_cast<long long>(clusters.size()); ++c) {
        const auto& cluster = clusters[static_cast<size_t>(c)];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                             points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        diameters[static_cast<size_t>(c)] = max_diameter;
        if (max_diameter > threshold * 1.001) {
            diameter_valid[static_cast<size_t>(c)] = 0;
        }
    }

    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        c, cluster.members.size(), cluster.seed_point, diameters[c]);
        }
        if (diameter_valid[c] == 0) {
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

    int clustered_count = 0;
#pragma omp parallel for reduction(+:clustered_count)
    for (long long i = 0; i < static_cast<long long>(membership.size()); ++i) {
        if (membership[static_cast<size_t>(i)] >= 0) {
            clustered_count++;
        }
    }

    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count, points.size() - clustered_count);
    return valid;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    bool show_help = false;
    bool parse_error = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            parse_error = true;
        }
    }

    if (show_help) {
        if (mpi_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    if (parse_error || num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (mpi_rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points(static_cast<size_t>(num_points));
    generateSyntheticData(points, num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, mpi_rank, mpi_size);
    const double local_cluster_time = MPI_Wtime() - cluster_start;

    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        const long cluster_time_ms = static_cast<long>(cluster_time * 1000.0);
        std::printf("Clustering time: %ld ms\n", cluster_time_ms);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
#pragma omp parallel for reduction(+:total_clustered) reduction(max:max_cluster_size)
        for (long long i = 0; i < static_cast<long long>(clusters.size()); ++i) {
            const int size = static_cast<int>(clusters[static_cast<size_t>(i)].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double avg_cluster_size = clusters.empty()
            ? 0.0
            : static_cast<double>(total_clustered) / clusters.size();

        std::printf("Points clustered: %d / %d (%.1f%%)\n",
                    total_clustered, num_points,
                    100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", avg_cluster_size);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = cluster_time > 0.0 ? cluster_time : std::numeric_limits<double>::min();
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(static_cast<size_t>(num_points));
            std::vector<int> membership(static_cast<size_t>(num_points), -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
            for (const int member : membership) {
                membershipData.push_back(static_cast<double>(member));
            }
            print_results(membershipData, "ClusterMembership");
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
