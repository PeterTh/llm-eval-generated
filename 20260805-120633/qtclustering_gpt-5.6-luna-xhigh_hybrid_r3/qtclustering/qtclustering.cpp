// QT Clustering Benchmark - MPI/OpenMP/CUDA implementation
//
// The QT selection loop is inherently sequential between clusters, but all
// candidate clusters for one selection round are independent.  MPI distributes
// those seeds between ranks.  Each rank evaluates all of its seeds on its
// assigned CUDA device, with one CUDA block per seed.  OpenMP is used for the
// rank-local preparation and winner reduction around the GPU work.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cfloat>
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
static constexpr int CUDA_BLOCK_SIZE = 256;

// Structure to represent a point in 2D space.  It is trivially copyable and
// therefore can be transferred directly between host and device.
struct Point {
    double x, y;
};

// Structure to represent a cluster.
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

static void checkCuda(const cudaError_t status, const char* operation,
                      const MPI_Comm comm) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, static_cast<int>(status));
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, comm)

// Generate synthetic 2D point data in clusters.  This retains the original
// rand_r sequence so the generated benchmark data is unchanged for normal
// benchmark sizes.  The minimum group size also makes the original generator
// well-defined for N < 30.
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
        int group_cnt = std::max(1, static_cast<int>(frand() * (N / 30.0)));

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

// Calculate Euclidean distance between two points.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Return true when candidate_point is a better result than current_point.
// The lower index tie-break reproduces findClosestPoint's ascending candidate
// traversal, which is important for deterministic cluster membership.
__device__ inline bool betterCandidate(const double candidate_diameter,
                                       const int candidate_point,
                                       const double current_diameter,
                                       const int current_point) {
    return candidate_diameter < current_diameter ||
           (candidate_diameter == current_diameter && candidate_point >= 0 &&
            (current_point < 0 || candidate_point < current_point));
}

// One block evaluates one seed.  Threads divide the candidate points, while
// the block reduction selects the same candidate as the original serial scan.
// The membership bitset makes the per-candidate in-cluster test O(1), and the
// member list remains in device memory across all growth iterations.
__global__ void growCandidateClusters(
    const Point* __restrict__ points,
    const unsigned char* __restrict__ clustered,
    int* __restrict__ candidate_members,
    int* __restrict__ candidate_sizes,
    std::uint32_t* __restrict__ candidate_bits,
    const int point_count,
    const int bit_words,
    const double threshold,
    int* __restrict__ progress) {
    const int seed_slot = blockIdx.x;
    const int lane = threadIdx.x;
    const std::size_t member_base =
        static_cast<std::size_t>(seed_slot) * point_count;
    const std::size_t bit_base =
        static_cast<std::size_t>(seed_slot) * bit_words;
    const int member_count = candidate_sizes[seed_slot];

    double best_diameter = DBL_MAX;
    int best_point = -1;

    for (int candidate = lane; candidate < point_count;
         candidate += blockDim.x) {
        if (clustered[candidate] ||
            (candidate_bits[bit_base + (candidate >> 5)] &
             (std::uint32_t(1) << (candidate & 31)))) {
            continue;
        }

        const Point candidate_point = points[candidate];
        double max_distance = 0.0;
        bool admissible = true;
        for (int member_index = 0; member_index < member_count;
             ++member_index) {
            const Point member_point =
                points[candidate_members[member_base + member_index]];
            const double candidate_distance =
                distance(candidate_point, member_point);
            max_distance = max(max_distance, candidate_distance);

            // The diameter can only grow.  This is equivalent to completing
            // the original max-distance scan, but avoids work for rejected
            // candidates in dense instances.
            if (max_distance >= threshold) {
                admissible = false;
                break;
            }
        }

        if (admissible && max_distance < threshold &&
            betterCandidate(max_distance, candidate, best_diameter,
                            best_point)) {
            best_diameter = max_distance;
            best_point = candidate;
        }
    }

    __shared__ double shared_diameters[CUDA_BLOCK_SIZE];
    __shared__ int shared_points[CUDA_BLOCK_SIZE];
    shared_diameters[lane] = best_diameter;
    shared_points[lane] = best_point;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset &&
            betterCandidate(shared_diameters[lane + offset],
                            shared_points[lane + offset],
                            shared_diameters[lane], shared_points[lane])) {
            shared_diameters[lane] = shared_diameters[lane + offset];
            shared_points[lane] = shared_points[lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0 && shared_points[0] >= 0) {
        const int selected_point = shared_points[0];
        const int selected_offset = candidate_sizes[seed_slot];
        if (selected_offset < point_count) {
            candidate_members[member_base + selected_offset] = selected_point;
            candidate_bits[bit_base + (selected_point >> 5)] |=
                (std::uint32_t(1) << (selected_point & 31));
            candidate_sizes[seed_slot] = selected_offset + 1;
            atomicExch(progress, 1);
        }
    }
}

__global__ void initializeCandidateClusters(
    int* __restrict__ candidate_members,
    int* __restrict__ candidate_sizes,
    std::uint32_t* __restrict__ candidate_bits,
    const int* __restrict__ seeds,
    const int seed_count,
    const int point_count) {
    const int seed_slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed_slot >= seed_count) return;

    const int seed = seeds[seed_slot];
    const std::size_t member_base =
        static_cast<std::size_t>(seed_slot) * point_count;
    const std::size_t bit_base =
        static_cast<std::size_t>(seed_slot) * ((point_count + 31) / 32);
    candidate_sizes[seed_slot] = 1;
    candidate_members[member_base] = seed;
    candidate_bits[bit_base + (seed >> 5)] |=
        (std::uint32_t(1) << (seed & 31));
}

static int localSeedCount(const std::size_t unclustered_count, const int rank,
                          const int rank_count) {
    if (static_cast<std::size_t>(rank) >= unclustered_count) return 0;
    return static_cast<int>((unclustered_count - 1 - rank) /
                                static_cast<std::size_t>(rank_count) +
                            1);
}

// Main QT clustering algorithm.  The cluster-selection round remains ordered
// globally to preserve the original semantics.  Candidate construction for
// every seed in that round is fully independent and is therefore distributed
// across MPI ranks and CUDA blocks.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const MPI_Comm comm,
                                  const int rank,
                                  const int rank_count) {
    const int N = static_cast<int>(points.size());
    const int bit_words = (N + 31) / 32;
    const int max_local_seeds =
        static_cast<int>((static_cast<std::size_t>(N) + rank_count - 1) /
                         rank_count);

    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "No CUDA device is available for MPI rank %d\n",
                     rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device));
    if (rank == 0) {
        std::printf("Hybrid execution: MPI ranks=%d, OpenMP threads/rank=%d, "
                    "CUDA device=%s\n",
                    rank_count, omp_get_max_threads(),
                    device_properties.name);
    }

    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_candidate_members = nullptr;
    int* d_candidate_sizes = nullptr;
    std::uint32_t* d_candidate_bits = nullptr;
    int* d_seeds = nullptr;
    int* d_progress = nullptr;

    const std::size_t member_capacity =
        static_cast<std::size_t>(std::max(1, max_local_seeds)) * N;
    const std::size_t bit_capacity =
        static_cast<std::size_t>(std::max(1, max_local_seeds)) * bit_words;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points),
                          static_cast<std::size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_clustered), N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidate_members),
                          member_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidate_sizes),
                          static_cast<std::size_t>(std::max(1, max_local_seeds)) *
                              sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidate_bits),
                          bit_capacity * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_seeds),
                          static_cast<std::size_t>(std::max(1, max_local_seeds)) *
                              sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_progress), sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(),
                          static_cast<std::size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<int> local_seeds(std::max(1, max_local_seeds));
    std::vector<int> local_sizes(std::max(1, max_local_seeds));
    std::vector<int> selected_members(N);
    std::vector<Cluster> clusters;
    clusters.reserve(N);

    while (!unclustered_indices.empty()) {
        const int local_count = localSeedCount(
            unclustered_indices.size(), rank, rank_count);

        // The stride distribution balances the work even after points are
        // removed from the front of the unclustered list.
        #pragma omp parallel for schedule(static)
        for (int slot = 0; slot < local_count; ++slot) {
            local_seeds[slot] =
                unclustered_indices[static_cast<std::size_t>(rank) +
                                    static_cast<std::size_t>(slot) *
                                        rank_count];
        }

        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), N,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_candidate_bits, 0,
                              static_cast<std::size_t>(local_count) *
                                  bit_words * sizeof(std::uint32_t)));

        if (local_count > 0) {
            CUDA_CHECK(cudaMemcpy(d_seeds, local_seeds.data(),
                                  static_cast<std::size_t>(local_count) *
                                      sizeof(int), cudaMemcpyHostToDevice));
            const int init_blocks =
                (local_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            initializeCandidateClusters<<<init_blocks, CUDA_BLOCK_SIZE>>>(
                d_candidate_members, d_candidate_sizes, d_candidate_bits,
                d_seeds, local_count, N);
            CUDA_CHECK(cudaGetLastError());

            int any_growth = 1;
            while (any_growth != 0) {
                CUDA_CHECK(cudaMemset(d_progress, 0, sizeof(int)));
                growCandidateClusters<<<local_count, CUDA_BLOCK_SIZE>>>(
                    d_points, d_clustered, d_candidate_members,
                    d_candidate_sizes, d_candidate_bits, N, bit_words,
                    threshold, d_progress);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(&any_growth, d_progress, sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }

            CUDA_CHECK(cudaMemcpy(local_sizes.data(), d_candidate_sizes,
                                  static_cast<std::size_t>(local_count) *
                                      sizeof(int), cudaMemcpyDeviceToHost));
        }

        int local_max_cardinality = 0;
        #pragma omp parallel for reduction(max : local_max_cardinality) \
            schedule(static)
        for (int slot = 0; slot < local_count; ++slot) {
            local_max_cardinality =
                std::max(local_max_cardinality, local_sizes[slot]);
        }

        int global_max_cardinality = 0;
        MPI_Allreduce(&local_max_cardinality, &global_max_cardinality, 1,
                      MPI_INT, MPI_MAX, comm);

        int local_best_seed = INT_MAX;
        int local_best_slot = -1;
        if (local_max_cardinality == global_max_cardinality) {
            for (int slot = 0; slot < local_count; ++slot) {
                if (local_sizes[slot] == local_max_cardinality &&
                    local_seeds[slot] < local_best_seed) {
                    local_best_seed = local_seeds[slot];
                    local_best_slot = slot;
                }
            }
        }

        int best_seed = INT_MAX;
        MPI_Allreduce(&local_best_seed, &best_seed, 1, MPI_INT, MPI_MIN,
                      comm);

        int owner = (local_best_seed == best_seed) ? rank : INT_MAX;
        int best_owner = INT_MAX;
        MPI_Allreduce(&owner, &best_owner, 1, MPI_INT, MPI_MIN, comm);

        if (best_seed == INT_MAX || global_max_cardinality <= 0 ||
            best_owner == INT_MAX) {
            break;
        }

        int selected_size = 0;
        if (rank == best_owner) {
            selected_size = local_sizes[local_best_slot];
            CUDA_CHECK(cudaMemcpy(selected_members.data(),
                                  d_candidate_members +
                                      static_cast<std::size_t>(local_best_slot) *
                                          N,
                                  static_cast<std::size_t>(selected_size) *
                                      sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&selected_size, 1, MPI_INT, best_owner, comm);
        MPI_Bcast(selected_members.data(), selected_size, MPI_INT, best_owner,
                  comm);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(selected_members.begin(),
                               selected_members.begin() + selected_size);
        clusters.push_back(std::move(cluster));

        for (int i = 0; i < selected_size; ++i) {
            clustered[selected_members[i]] = 1;
        }
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(),
                           unclustered_indices.end(),
                           [&clustered](const int index) {
                               return clustered[index] != 0;
                           }),
            unclustered_indices.end());
    }

    CUDA_CHECK(cudaFree(d_progress));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_candidate_bits));
    CUDA_CHECK(cudaFree(d_candidate_sizes));
    CUDA_CHECK(cudaFree(d_candidate_members));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));
    MPI_Comm_free(&local_comm);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties.
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;

    std::printf("Validating clusters:\n");

    for (std::size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (std::size_t i = 0; i < cluster.members.size(); ++i) {
            for (std::size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist =
                    distance(points[cluster.members[i]],
                             points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c,
                        cluster.members.size(), cluster.seed_point,
                        max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        c, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (std::size_t c = 0; c < clusters.size(); ++c) {
        for (const int member : clusters[c].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and "
                            "%zu)\n",
                            member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
    for (const int member : membership) {
        if (member >= 0) ++clustered_count;
    }

    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count,
                points.size() - static_cast<std::size_t>(clustered_count));

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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int parse_error = 0;

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            parse_error = 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                        num_points, threshold);
        }
        parse_error = 1;
    }
    if (parse_error != 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(),
              static_cast<int>(static_cast<std::size_t>(num_points) *
                               sizeof(Point)),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, MPI_COMM_WORLD, rank, rank_count);
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time_seconds = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        const long cluster_milliseconds =
            static_cast<long>(cluster_time_seconds * 1000.0);
        std::printf("Clustering time: %ld ms\n", cluster_milliseconds);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const auto& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double avg_cluster_size =
            clusters.empty()
                ? 0.0
                : static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
                    num_points, 100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", avg_cluster_size);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = std::max(cluster_time_seconds,
                                         std::numeric_limits<double>::min());
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (std::size_t c = 0; c < clusters.size(); ++c) {
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
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
