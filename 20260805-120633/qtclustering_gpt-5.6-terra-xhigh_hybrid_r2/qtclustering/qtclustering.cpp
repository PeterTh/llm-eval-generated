// QT Clustering Benchmark
//
// The commit order of QT clustering is inherently serial: a chosen cluster
// changes every later candidate.  This implementation keeps that order, while
// evaluating the independent candidate seeds with MPI ranks and OpenMP worker
// threads.  A CUDA stream per OpenMP worker evaluates the greedy additions for
// one candidate entirely on its local GPU.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_BLOCK_SIZE = 256;

// Structure to represent a point in 2D space.
struct Point {
    double x;
    double y;
};

// Structure to represent a cluster.
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// The state is intentionally byte-addressable, unlike vector<bool>, so it is
// inexpensive to transfer to and manipulate on the GPU.
enum PointState : unsigned char {
    AVAILABLE = 0,
    ALREADY_CLUSTERED = 1,
    IN_CANDIDATE = 2,
};

struct CandidateChoice {
    double diameter;
    int index;
};

struct CandidateResult {
    int cardinality;
    int seed;
};

static int mpi_rank = 0;

[[noreturn]] void cudaFail(const char* operation, const cudaError_t status) {
    std::fprintf(stderr, "MPI rank %d: CUDA %s failed: %s\n", mpi_rank,
                 operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

inline void cudaCheck(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFail(operation, status);
    }
}

inline void cudaCheckLaunch(const char* operation) {
    cudaCheck(cudaGetLastError(), operation);
}

// Generate synthetic 2D point data in clusters.  Generation remains on rank
// zero and is broadcast verbatim, preserving the original deterministic data.
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
        const double radius = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }

        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
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

inline bool betterChoice(const CandidateChoice& left,
                         const CandidateChoice& right) {
    return left.index >= 0 &&
           (right.index < 0 || left.diameter < right.diameter ||
            (left.diameter == right.diameter && left.index < right.index));
}

inline bool betterResult(const CandidateResult& left,
                         const CandidateResult& right) {
    return left.cardinality > right.cardinality ||
           (left.cardinality == right.cardinality && left.cardinality >= 0 &&
            left.seed < right.seed);
}

// Set candidate state from the current globally committed membership.  The
// candidate seed is already a member before its first distance update.
__global__ void initializeCandidateKernel(const unsigned char* clustered,
                                          unsigned char* state,
                                          double* max_distances,
                                          const int seed,
                                          const int point_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < point_count) {
        state[index] = clustered[index] ? ALREADY_CLUSTERED : AVAILABLE;
        if (index == seed) {
            state[index] = IN_CANDIDATE;
        }
        max_distances[index] = 0.0;
    }
}

// QT's diameter test is incremental: after adding a member, the maximum
// distance for every viable point is max(previous maximum, distance to the new
// member).  This is algebraically the same as recomputing the maximum against
// the complete candidate cluster on every greedy step.
__global__ void updateMaxDistancesKernel(const Point* points,
                                         const unsigned char* state,
                                         double* max_distances,
                                         const int last_member,
                                         const int point_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < point_count && state[index] == AVAILABLE) {
        const double dx = points[index].x - points[last_member].x;
        const double dy = points[index].y - points[last_member].y;
        const double distance_to_last = sqrt(dx * dx + dy * dy);
        if (distance_to_last > max_distances[index]) {
            max_distances[index] = distance_to_last;
        }
    }
}

__device__ __forceinline__ bool deviceBetterChoice(const CandidateChoice left,
                                                    const CandidateChoice right) {
    return left.index >= 0 &&
           (right.index < 0 || left.diameter < right.diameter ||
            (left.diameter == right.diameter && left.index < right.index));
}

// Each block returns its exact lexicographic minimum (diameter, point index).
// The small final reduction is on the host, avoiding non-deterministic atomic
// floating point ordering while transferring only O(N / block_size) values.
__global__ void reduceCandidateChoicesKernel(const unsigned char* state,
                                             const double* max_distances,
                                             const double threshold,
                                             const int point_count,
                                             CandidateChoice* block_choices) {
    __shared__ CandidateChoice choices[CUDA_BLOCK_SIZE];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    CandidateChoice choice{DBL_MAX, -1};
    if (index < point_count && state[index] == AVAILABLE &&
        max_distances[index] < threshold) {
        choice = {max_distances[index], index};
    }
    choices[threadIdx.x] = choice;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset &&
            deviceBetterChoice(choices[threadIdx.x + offset],
                               choices[threadIdx.x])) {
            choices[threadIdx.x] = choices[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_choices[blockIdx.x] = choices[0];
    }
}

__global__ void selectCandidateMemberKernel(unsigned char* state,
                                            const int selected_member) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        state[selected_member] = IN_CANDIDATE;
    }
}

// Workspace is deliberately per OpenMP worker.  It owns a CUDA stream and
// buffers, allowing independent seed candidates to overlap on each GPU.
class CandidateWorkspace {
public:
    explicit CandidateWorkspace(const int point_count)
        : point_count_(point_count),
          block_count_((point_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE) {
        cudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "stream creation");
        cudaCheck(cudaMalloc(&device_clustered_, point_count_ * sizeof(*device_clustered_)),
                  "clustered-state allocation");
        cudaCheck(cudaMalloc(&device_state_, point_count_ * sizeof(*device_state_)),
                  "candidate-state allocation");
        cudaCheck(cudaMalloc(&device_max_distances_,
                             point_count_ * sizeof(*device_max_distances_)),
                  "distance-buffer allocation");
        cudaCheck(cudaMalloc(&device_block_choices_,
                             block_count_ * sizeof(*device_block_choices_)),
                  "reduction-buffer allocation");
        cudaCheck(cudaMallocHost(&host_block_choices_,
                                 block_count_ * sizeof(*host_block_choices_)),
                  "pinned reduction-buffer allocation");
    }

    CandidateWorkspace(const CandidateWorkspace&) = delete;
    CandidateWorkspace& operator=(const CandidateWorkspace&) = delete;

    ~CandidateWorkspace() {
        // CUDA resources may be destroyed after a normal CUDA error.  There is
        // no useful recovery path at that point, so cleanup intentionally does
        // not mask the original error.
        if (host_block_choices_ != nullptr) cudaFreeHost(host_block_choices_);
        if (device_block_choices_ != nullptr) cudaFree(device_block_choices_);
        if (device_max_distances_ != nullptr) cudaFree(device_max_distances_);
        if (device_state_ != nullptr) cudaFree(device_state_);
        if (device_clustered_ != nullptr) cudaFree(device_clustered_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    void setClustered(const std::vector<unsigned char>& clustered) {
        cudaCheck(cudaMemcpyAsync(device_clustered_, clustered.data(),
                                  point_count_ * sizeof(*device_clustered_),
                                  cudaMemcpyHostToDevice, stream_),
                  "clustered-state transfer");
    }

    int buildCandidate(const int seed, const Point* device_points,
                       const double threshold,
                       std::vector<int>* members = nullptr) {
        const dim3 blocks(block_count_);
        const dim3 threads(CUDA_BLOCK_SIZE);
        initializeCandidateKernel<<<blocks, threads, 0, stream_>>>(
            device_clustered_, device_state_, device_max_distances_, seed, point_count_);
        cudaCheckLaunch("candidate initialization kernel launch");

        int cardinality = 1;
        int last_member = seed;
        if (members != nullptr) {
            members->clear();
            members->reserve(point_count_);
            members->push_back(seed);
        }
        while (cardinality < point_count_) {
            updateMaxDistancesKernel<<<blocks, threads, 0, stream_>>>(
                device_points, device_state_, device_max_distances_, last_member, point_count_);
            cudaCheckLaunch("distance-update kernel launch");
            reduceCandidateChoicesKernel<<<blocks, threads, 0, stream_>>>(
                device_state_, device_max_distances_, threshold, point_count_,
                device_block_choices_);
            cudaCheckLaunch("candidate-reduction kernel launch");
            cudaCheck(cudaMemcpyAsync(host_block_choices_, device_block_choices_,
                                      block_count_ * sizeof(*host_block_choices_),
                                      cudaMemcpyDeviceToHost, stream_),
                      "candidate-reduction transfer");
            cudaCheck(cudaStreamSynchronize(stream_), "candidate-reduction synchronization");

            CandidateChoice next{std::numeric_limits<double>::max(), -1};
            for (int block = 0; block < block_count_; ++block) {
                if (betterChoice(host_block_choices_[block], next)) {
                    next = host_block_choices_[block];
                }
            }
            if (next.index < 0) {
                break;
            }

            selectCandidateMemberKernel<<<1, 1, 0, stream_>>>(device_state_, next.index);
            cudaCheckLaunch("candidate-member selection kernel launch");
            last_member = next.index;
            ++cardinality;
            if (members != nullptr) {
                // Preserve the original greedy insertion order, not merely
                // the set of selected points.
                members->push_back(next.index);
            }
        }

        if (members != nullptr) {
            cudaCheck(cudaStreamSynchronize(stream_), "candidate-membership synchronization");
        } else {
            // Ensure the final one-thread select kernel has completed before
            // this workspace is reused for a different seed.
            cudaCheck(cudaStreamSynchronize(stream_), "candidate completion synchronization");
        }
        return cardinality;
    }

private:
    int point_count_;
    int block_count_;
    cudaStream_t stream_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    unsigned char* device_state_ = nullptr;
    double* device_max_distances_ = nullptr;
    CandidateChoice* device_block_choices_ = nullptr;
    CandidateChoice* host_block_choices_ = nullptr;
};

std::vector<Cluster> qtClusteringHybrid(const std::vector<Point>& points,
                                        const double threshold,
                                        const int rank,
                                        const int world_size) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, AVAILABLE);
    std::vector<Cluster> clusters;
    clusters.reserve(point_count);

    Point* device_points = nullptr;
    cudaCheck(cudaMalloc(&device_points, point_count * sizeof(*device_points)),
              "point-buffer allocation");
    cudaCheck(cudaMemcpy(device_points, points.data(), point_count * sizeof(*device_points),
                         cudaMemcpyHostToDevice),
              "point transfer");

    // Keep workspaces alive for the full run.  CUDA allocation/stream creation
    // is expensive relative to a short candidate, and the same per-worker
    // buffers are safe to reuse after every globally committed cluster.
    const int stream_limit = point_count < 8192 ? 8 : 2;
    const int worker_count = std::max(1, std::min(omp_get_max_threads(), stream_limit));
    int selected_device = 0;
    cudaCheck(cudaGetDevice(&selected_device), "selected-device query");
    std::vector<std::unique_ptr<CandidateWorkspace>> worker_workspaces;
    worker_workspaces.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        worker_workspaces.emplace_back(std::make_unique<CandidateWorkspace>(point_count));
    }
    CandidateWorkspace materialization_workspace(point_count);

    int remaining = point_count;
    while (remaining > 0) {
        CandidateResult rank_best{-1, INT_MAX};

        // CUDA's selected device is thread-local.  Query it on the MPI main
        // thread, then select that same GPU explicitly in every OpenMP worker.
#pragma omp parallel num_threads(worker_count)
        {
            cudaCheck(cudaSetDevice(selected_device), "OpenMP worker device selection");
            CandidateWorkspace& workspace = *worker_workspaces[omp_get_thread_num()];
            workspace.setClustered(clustered);
            CandidateResult thread_best{-1, INT_MAX};

#pragma omp for schedule(dynamic, 1) nowait
            for (int seed = rank; seed < point_count; seed += world_size) {
                if (clustered[seed] != AVAILABLE) {
                    continue;
                }
                const int cardinality = workspace.buildCandidate(seed, device_points, threshold);
                const CandidateResult result{cardinality, seed};
                if (betterResult(result, thread_best)) {
                    thread_best = result;
                }
            }

#pragma omp critical
            {
                if (betterResult(thread_best, rank_best)) {
                    rank_best = thread_best;
                }
            }
        }

        int local_choice[2] = {rank_best.cardinality, rank_best.seed};
        std::vector<int> all_choices(2 * world_size);
        MPI_Allgather(local_choice, 2, MPI_INT, all_choices.data(), 2, MPI_INT,
                      MPI_COMM_WORLD);

        CandidateResult best{-1, INT_MAX};
        for (int process = 0; process < world_size; ++process) {
            const CandidateResult candidate{all_choices[2 * process],
                                            all_choices[2 * process + 1]};
            if (betterResult(candidate, best)) {
                best = candidate;
            }
        }
        if (best.seed < 0 || best.cardinality <= 0) {
            cudaCheck(cudaFree(device_points), "point-buffer release");
            return clusters;
        }

        const int owner = best.seed % world_size;
        std::vector<int> best_members;
        if (rank == owner) {
            materialization_workspace.setClustered(clustered);
            const int regenerated_size = materialization_workspace.buildCandidate(
                best.seed, device_points, threshold, &best_members);
            if (regenerated_size != best.cardinality ||
                static_cast<int>(best_members.size()) != best.cardinality) {
                std::fprintf(stderr, "MPI rank %d: inconsistent candidate regeneration\n", rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        int member_count = best.cardinality;
        if (rank != owner) {
            best_members.resize(member_count);
        }
        MPI_Bcast(best_members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best.seed;
        cluster.members = std::move(best_members);
        for (const int member : cluster.members) {
            clustered[member] = ALREADY_CLUSTERED;
        }
        remaining -= member_count;
        clusters.push_back(std::move(cluster));
    }

    cudaCheck(cudaFree(device_points), "point-buffer release");
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
        const int member_count = static_cast<int>(cluster.members.size());

#pragma omp parallel for reduction(max : max_diameter) schedule(static)
        for (int i = 0; i < member_count; ++i) {
            double row_max = 0.0;
            for (int j = i + 1; j < member_count; ++j) {
                row_max = std::max(row_max,
                                   distance(points[cluster.members[i]],
                                            points[cluster.members[j]]));
            }
            max_diameter = std::max(max_diameter, row_max);
        }

        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c,
                        cluster.members.size(), cluster.seed_point, max_diameter);
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
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
#pragma omp parallel for reduction(+ : clustered_count) schedule(static)
    for (int i = 0; i < static_cast<int>(membership.size()); ++i) {
        if (membership[i] >= 0) {
            ++clustered_count;
        }
    }
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
                clustered_count, points.size() - clustered_count);
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
    int mpi_size = 1;
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int local_rank = 0;
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "device discovery");
    if (device_count <= 0) {
        if (mpi_rank == 0) {
            std::fprintf(stderr, "No CUDA device is available; this hybrid build requires CUDA.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int selected_device = local_rank % device_count;
    cudaCheck(cudaSetDevice(selected_device), "MPI rank device selection");
    MPI_Comm_free(&local_comm);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_enabled = false;
    int argument_error = 0;

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            argument_error = 1;
        }
    }
    if (num_points <= 0 || threshold <= 0.0 ||
        num_points > INT_MAX / static_cast<int>(sizeof(Point))) {
        if (mpi_rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        argument_error = 1;
    }
    if (argument_error != 0) {
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        std::printf("QT Clustering Benchmark\n");
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
    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClusteringHybrid(points, threshold, mpi_rank, mpi_size);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    const long long local_cluster_time = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            cluster_end - cluster_start).count());
    long long cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::printf("Clustering time: %lld ms\n", cluster_time);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
#pragma omp parallel for reduction(+ : total_clustered) reduction(max : max_cluster_size) schedule(static)
        for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
                    100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", avg_cluster_size);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = cluster_time / 1000.0;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / time_sec, num_points / time_sec);

        if (print_results_enabled) {
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
            argument_error = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&argument_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return argument_error;
}
