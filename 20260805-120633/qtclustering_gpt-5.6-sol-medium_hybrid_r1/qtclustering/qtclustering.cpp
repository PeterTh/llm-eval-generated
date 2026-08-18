// Hybrid MPI + OpenMP + CUDA Quality Threshold clustering benchmark.
// Each MPI rank evaluates a deterministic subset of seeds.  A CUDA block
// builds one candidate cluster, and MPI selects the globally best candidate.

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void generateSyntheticData(std::vector<Point>& points, int n,
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
        group_count = std::min(group_count, n - count);
        while (group_count > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = center_x + dx;
            const double y = center_y + dy;
            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT)
                continue;
            points[count++] = {x, y};
            --group_count;
        }
    }
}

__global__ void buildCandidateClusters(const Point* __restrict__ points,
                                       const unsigned char* __restrict__ clustered,
                                       const int* __restrict__ seeds,
                                       int seed_count, int n, double threshold,
                                       int* __restrict__ all_members,
                                       unsigned char* __restrict__ all_in_cluster,
                                       int* __restrict__ cardinalities) {
    const int candidate_cluster = blockIdx.x;
    if (candidate_cluster >= seed_count) return;

    int* members = all_members + static_cast<size_t>(candidate_cluster) * n;
    unsigned char* in_cluster =
        all_in_cluster + static_cast<size_t>(candidate_cluster) * n;

    __shared__ double best_distance[CUDA_BLOCK_SIZE];
    __shared__ int best_point[CUDA_BLOCK_SIZE];
    __shared__ int member_count;
    __shared__ int selected;

    for (int i = threadIdx.x; i < n; i += blockDim.x) in_cluster[i] = 0;
    if (threadIdx.x == 0) {
        const int seed = seeds[candidate_cluster];
        members[0] = seed;
        in_cluster[seed] = 1;
        member_count = 1;
    }
    __syncthreads();

    while (true) {
        double thread_best_distance = DBL_MAX;
        int thread_best_point = -1;
        const int count = member_count;

        for (int candidate = threadIdx.x; candidate < n;
             candidate += blockDim.x) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            double max_distance = 0.0;
            const Point p = points[candidate];
            for (int j = 0; j < count; ++j) {
                const Point q = points[members[j]];
                const double dx = p.x - q.x;
                const double dy = p.y - q.y;
                const double d = sqrt(dx * dx + dy * dy);
                if (d > max_distance) max_distance = d;
            }
            if (max_distance < threshold &&
                (max_distance < thread_best_distance ||
                 (max_distance == thread_best_distance &&
                  candidate < thread_best_point))) {
                thread_best_distance = max_distance;
                thread_best_point = candidate;
            }
        }

        best_distance[threadIdx.x] = thread_best_distance;
        best_point[threadIdx.x] = thread_best_point;
        __syncthreads();

        // Reduce a total ordering of (diameter, point index).  The index is the
        // deterministic tie break used by the sequential reference.
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                const int other_point = best_point[threadIdx.x + offset];
                const double other_distance = best_distance[threadIdx.x + offset];
                const int this_point = best_point[threadIdx.x];
                const double this_distance = best_distance[threadIdx.x];
                if (other_point >= 0 &&
                    (this_point < 0 || other_distance < this_distance ||
                     (other_distance == this_distance && other_point < this_point))) {
                    best_point[threadIdx.x] = other_point;
                    best_distance[threadIdx.x] = other_distance;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            selected = best_point[0];
            if (selected >= 0) {
                in_cluster[selected] = 1;
                members[member_count++] = selected;
            }
        }
        __syncthreads();
        if (selected < 0) break;
    }

    if (threadIdx.x == 0) cardinalities[candidate_cluster] = member_count;
}

class CudaClusterEngine {
  public:
    CudaClusterEngine(const std::vector<Point>& points, int maximum_local_seeds)
        : n_(static_cast<int>(points.size())) {
        cudaCheck(cudaMalloc(&d_points_, points.size() * sizeof(Point)),
                  "allocating device points");
        cudaCheck(cudaMalloc(&d_clustered_, points.size() * sizeof(unsigned char)),
                  "allocating device state");
        cudaCheck(cudaMemcpy(d_points_, points.data(), points.size() * sizeof(Point),
                             cudaMemcpyHostToDevice),
                  "copying points to device");

        size_t free_bytes = 0, total_bytes = 0;
        cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "querying GPU memory");
        (void)total_bytes;
        const size_t bytes_per_seed =
            static_cast<size_t>(n_) * (sizeof(int) + sizeof(unsigned char)) +
            sizeof(int) * 2;
        // Keep ample room for the CUDA runtime and other ranks sharing a GPU.
        const size_t memory_capacity = (free_bytes / 2) / std::max<size_t>(1, bytes_per_seed);
        capacity_ = std::max<size_t>(1, std::min<size_t>(
            static_cast<size_t>(std::max(1, maximum_local_seeds)), memory_capacity));

        cudaCheck(cudaMalloc(&d_seeds_, capacity_ * sizeof(int)),
                  "allocating device seeds");
        cudaCheck(cudaMalloc(&d_cardinalities_, capacity_ * sizeof(int)),
                  "allocating device cardinalities");
        cudaCheck(cudaMalloc(&d_members_, capacity_ * static_cast<size_t>(n_) * sizeof(int)),
                  "allocating candidate members");
        cudaCheck(cudaMalloc(&d_in_cluster_,
                             capacity_ * static_cast<size_t>(n_) * sizeof(unsigned char)),
                  "allocating candidate masks");
    }

    ~CudaClusterEngine() {
        cudaFree(d_in_cluster_);
        cudaFree(d_members_);
        cudaFree(d_cardinalities_);
        cudaFree(d_seeds_);
        cudaFree(d_clustered_);
        cudaFree(d_points_);
    }

    CudaClusterEngine(const CudaClusterEngine&) = delete;
    CudaClusterEngine& operator=(const CudaClusterEngine&) = delete;

    std::pair<int, int> bestLocalCandidate(
        const std::vector<int>& seeds,
        const std::vector<unsigned char>& clustered,
        double threshold) {
        cudaCheck(cudaMemcpy(d_clustered_, clustered.data(),
                             clustered.size() * sizeof(unsigned char),
                             cudaMemcpyHostToDevice),
                  "copying clustered mask");
        int best_cardinality = -1;
        int best_seed = -1;
        std::vector<int> cardinalities(capacity_);

        for (size_t begin = 0; begin < seeds.size(); begin += capacity_) {
            const int count = static_cast<int>(
                std::min(capacity_, seeds.size() - begin));
            cudaCheck(cudaMemcpy(d_seeds_, seeds.data() + begin,
                                 static_cast<size_t>(count) * sizeof(int),
                                 cudaMemcpyHostToDevice),
                      "copying seed batch");
            buildCandidateClusters<<<count, CUDA_BLOCK_SIZE>>>(
                d_points_, d_clustered_, d_seeds_, count, n_, threshold,
                d_members_, d_in_cluster_, d_cardinalities_);
            cudaCheck(cudaGetLastError(), "launching candidate kernel");
            cudaCheck(cudaMemcpy(cardinalities.data(), d_cardinalities_,
                                 static_cast<size_t>(count) * sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      "reading candidate cardinalities");
            for (int i = 0; i < count; ++i) {
                const int seed = seeds[begin + static_cast<size_t>(i)];
                if (cardinalities[i] > best_cardinality ||
                    (cardinalities[i] == best_cardinality && seed < best_seed)) {
                    best_cardinality = cardinalities[i];
                    best_seed = seed;
                }
            }
        }
        return {best_cardinality, best_seed};
    }

    std::vector<int> materialize(int seed,
                                 const std::vector<unsigned char>& clustered,
                                 double threshold) {
        cudaCheck(cudaMemcpy(d_clustered_, clustered.data(),
                             clustered.size() * sizeof(unsigned char),
                             cudaMemcpyHostToDevice),
                  "copying clustered mask");
        cudaCheck(cudaMemcpy(d_seeds_, &seed, sizeof(int), cudaMemcpyHostToDevice),
                  "copying winning seed");
        buildCandidateClusters<<<1, CUDA_BLOCK_SIZE>>>(
            d_points_, d_clustered_, d_seeds_, 1, n_, threshold,
            d_members_, d_in_cluster_, d_cardinalities_);
        cudaCheck(cudaGetLastError(), "launching winning candidate kernel");
        int count = 0;
        cudaCheck(cudaMemcpy(&count, d_cardinalities_, sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "reading winning cardinality");
        std::vector<int> members(count);
        cudaCheck(cudaMemcpy(members.data(), d_members_,
                             members.size() * sizeof(int), cudaMemcpyDeviceToHost),
                  "reading winning members");
        return members;
    }

  private:
    int n_;
    size_t capacity_ = 0;
    Point* d_points_ = nullptr;
    unsigned char* d_clustered_ = nullptr;
    int* d_seeds_ = nullptr;
    int* d_cardinalities_ = nullptr;
    int* d_members_ = nullptr;
    unsigned char* d_in_cluster_ = nullptr;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold, int rank, int world_size) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<unsigned char> eligible(n, 0);
    std::vector<Cluster> clusters;
    CudaClusterEngine engine(points, (n + world_size - 1) / world_size);
    int remaining = n;

    while (remaining > 0) {
        // OpenMP constructs the rank-local work mask; the compacting pass is
        // ordered so runs remain bit-for-bit deterministic.
#pragma omp parallel for schedule(static)
        for (int seed = 0; seed < n; ++seed) {
            eligible[seed] =
                (!clustered[seed] && seed % world_size == rank) ? 1 : 0;
        }
        std::vector<int> local_seeds;
        local_seeds.reserve((remaining + world_size - 1) / world_size);
        for (int seed = 0; seed < n; ++seed)
            if (eligible[seed]) local_seeds.push_back(seed);

        const auto local = engine.bestLocalCandidate(local_seeds, clustered, threshold);
        const long long base = static_cast<long long>(n) + 1;
        const long long local_key = local.first > 0
            ? static_cast<long long>(local.first) * base + (n - local.second)
            : 0;
        long long global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_LONG_LONG, MPI_MAX,
                      MPI_COMM_WORLD);
        if (global_key == 0) break;

        const int best_seed = n - static_cast<int>(global_key % base);
        const int owner = best_seed % world_size;
        std::vector<int> members;
        if (rank == owner)
            members = engine.materialize(best_seed, clustered, threshold);
        int member_count = rank == owner ? static_cast<int>(members.size()) : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner) members.resize(member_count);
        MPI_Bcast(members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);

        clusters.push_back({members, best_seed});
#pragma omp parallel for schedule(static)
        for (int i = 0; i < member_count; ++i) clustered[members[i]] = 1;
        remaining -= member_count;
    }
    return clusters;
}

static inline double pointDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points, double threshold) {
    bool valid = true;
    printf("Validating clusters:\n");
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        const Cluster& cluster = clusters[c];
        double max_diameter = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(max : max_diameter)
        for (int i = 0; i < static_cast<int>(cluster.members.size()); ++i) {
            for (size_t j = static_cast<size_t>(i) + 1;
                 j < cluster.members.size(); ++j) {
                max_diameter = std::max(
                    max_diameter,
                    pointDistance(points[cluster.members[i]], points[cluster.members[j]]));
            }
        }
        if (c < 10)
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c,
                   cluster.members.size(), cluster.seed_point, max_diameter);
        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
        for (int member : cluster.members) {
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    const int clustered_count = static_cast<int>(
        std::count_if(membership.begin(), membership.end(), [](int x) { return x >= 0; }));
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
           clustered_count, points.size() - clustered_count);
    return valid && clustered_count == static_cast<int>(points.size());
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        if (rank == 0)
            fprintf(stderr, "CUDA accelerator required: %s\n",
                    cudaGetErrorString(device_status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count), "selecting CUDA device");

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false, print_results_requested = false;
    bool arguments_ok = true, show_help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            num_points = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (strcmp(argv[i], "-r") == 0)
            print_results_requested = true;
        else if (strcmp(argv[i], "-h") == 0)
            show_help = true;
        else
            arguments_ok = false;
    }
    if (show_help || !arguments_ok || num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            if (!arguments_ok || num_points <= 0 || threshold <= 0.0)
                fprintf(stderr, "Invalid command-line parameters\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (show_help && arguments_ok) ? 0 : 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallel configuration: %d MPI ranks, %d OpenMP threads/rank, CUDA\n",
               world_size, omp_get_max_threads());
    }

    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    std::vector<Cluster> clusters;
    try {
        clusters = qtClustering(points, threshold, rank, world_size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const double local_seconds = MPI_Wtime() - start;
    double elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        printf("Clustering time: %.3f ms\n", elapsed_seconds * 1000.0);
        printf("Clusters found: %zu\n", clusters.size());
        int total_clustered = 0, maximum_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            maximum_cluster_size = std::max(maximum_cluster_size, size);
        }
        const double average = clusters.empty()
            ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
               num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", average);
        printf("Maximum cluster size: %d\n", maximum_cluster_size);
        if (elapsed_seconds > 0.0)
            printf("Performance: %.1f clusters/s, %.1f points/s\n",
                   clusters.size() / elapsed_seconds, num_points / elapsed_seconds);

        if (print_results_requested) {
            std::vector<double> membership_data(num_points, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (int member : clusters[c].members)
                    membership_data[member] = static_cast<double>(c);
            print_results(membership_data, "ClusterMembership");
        }
        if (validate) {
            result = validateClusters(clusters, points, threshold) ? 0 : 1;
            printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
