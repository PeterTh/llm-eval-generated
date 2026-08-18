// QT clustering benchmark -- hybrid MPI/OpenMP/CUDA implementation.
//
// MPI distributes candidate seeds, CUDA constructs the candidate clusters,
// and OpenMP is used for host-side state updates and validation.  Candidate
// selection and all global tie breaks deliberately retain the ordering of the
// original sequential implementation.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
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

static int mpi_rank = 0;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", mpi_rank,
                 operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: MPI failure in %s: %.*s\n", mpi_rank,
                 operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

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
        int group_count = static_cast<int>(frand() * (N / 30.0));
        group_count = std::min(group_count, N - count);

        while (group_count > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT)
                continue;
            points[count++] = {x, y};
            --group_count;
        }
    }
}

inline double distance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct DeviceChoice {
    double diameter;
    int point;
};

__device__ __forceinline__ DeviceChoice betterChoice(DeviceChoice a,
                                                       DeviceChoice b) {
    if (b.diameter < a.diameter ||
        (b.diameter == a.diameter && b.point < a.point))
        return b;
    return a;
}

// One block constructs one candidate cluster.  Maintaining each point's
// current maximum distance makes construction O(N^2), instead of repeatedly
// rescanning all cluster members (O(N^3)) as in the simple reference code.
__global__ void buildCandidateClusters(const double2* __restrict__ points,
                                       const unsigned char* __restrict__ clustered,
                                       const int* __restrict__ seeds,
                                       int seed_count, int point_count,
                                       double threshold,
                                       double* __restrict__ scores,
                                       int* __restrict__ cardinalities,
                                       int* __restrict__ output_members) {
    const int job = blockIdx.x;
    if (job >= seed_count) return;

    __shared__ DeviceChoice choices[CUDA_BLOCK_SIZE];
    __shared__ int newest;
    __shared__ int count;
    const int tid = threadIdx.x;
    const int seed = seeds[job];
    double* const job_scores = scores + static_cast<size_t>(job) * point_count;

    if (tid == 0) {
        count = 1;
        newest = seed;
        if (output_members) output_members[0] = seed;
    }
    __syncthreads();

    const double2 seed_point = points[seed];
    for (int p = tid; p < point_count; p += blockDim.x) {
        if (clustered[p] || p == seed) {
            job_scores[p] = DBL_MAX;
        } else {
            const double dx = points[p].x - seed_point.x;
            const double dy = points[p].y - seed_point.y;
            job_scores[p] = sqrt(dx * dx + dy * dy);
        }
    }
    __syncthreads();

    while (true) {
        DeviceChoice local{DBL_MAX, INT_MAX};
        for (int p = tid; p < point_count; p += blockDim.x) {
            const double value = job_scores[p];
            if (value < threshold) {
                const DeviceChoice candidate{value, p};
                local = betterChoice(local, candidate);
            }
        }
        choices[tid] = local;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset)
                choices[tid] = betterChoice(choices[tid], choices[tid + offset]);
            __syncthreads();
        }

        if (tid == 0) {
            newest = choices[0].point;
            if (newest != INT_MAX) {
                if (output_members) output_members[count] = newest;
                ++count;
                job_scores[newest] = DBL_MAX;
            }
        }
        __syncthreads();
        if (newest == INT_MAX) break;

        const double2 added = points[newest];
        for (int p = tid; p < point_count; p += blockDim.x) {
            double current = job_scores[p];
            if (current != DBL_MAX) {
                const double dx = points[p].x - added.x;
                const double dy = points[p].y - added.y;
                const double d = sqrt(dx * dx + dy * dy);
                if (d > current) current = d;
                job_scores[p] = current;
            }
        }
        __syncthreads();
    }

    if (tid == 0) cardinalities[job] = count;
}

class CudaWorkspace {
public:
    CudaWorkspace(const std::vector<Point>& points, int maximum_local_seeds)
        : point_count_(static_cast<int>(points.size())) {
        static_assert(sizeof(Point) == sizeof(double2));
        cudaCheck(cudaMalloc(&device_points_, points.size() * sizeof(Point)),
                  "allocating points");
        cudaCheck(cudaMalloc(&device_clustered_, points.size()),
                  "allocating clustered flags");

        size_t free_bytes = 0, total_bytes = 0;
        cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "querying GPU memory");
        const size_t bytes_per_seed = points.size() * sizeof(double);
        const size_t budget = free_bytes * 3 / 4;
        size_t affordable = bytes_per_seed ? budget / bytes_per_seed : 1;
        affordable = std::max<size_t>(1, affordable);
        batch_capacity_ = std::max(1, std::min(maximum_local_seeds,
                                               static_cast<int>(affordable)));

        cudaCheck(cudaMalloc(&device_scores_, static_cast<size_t>(batch_capacity_) *
                                                point_count_ * sizeof(double)),
                  "allocating candidate scores");
        cudaCheck(cudaMalloc(&device_seeds_, batch_capacity_ * sizeof(int)),
                  "allocating candidate seeds");
        cudaCheck(cudaMalloc(&device_cardinalities_, batch_capacity_ * sizeof(int)),
                  "allocating candidate cardinalities");
        cudaCheck(cudaMalloc(&device_members_, points.size() * sizeof(int)),
                  "allocating winning members");
        cudaCheck(cudaMemcpy(device_points_, points.data(), points.size() * sizeof(Point),
                             cudaMemcpyHostToDevice), "uploading points");
    }

    ~CudaWorkspace() {
        cudaFree(device_members_);
        cudaFree(device_cardinalities_);
        cudaFree(device_seeds_);
        cudaFree(device_scores_);
        cudaFree(device_clustered_);
        cudaFree(device_points_);
    }

    void uploadClustered(const std::vector<unsigned char>& clustered) {
        cudaCheck(cudaMemcpy(device_clustered_, clustered.data(), clustered.size(),
                             cudaMemcpyHostToDevice), "uploading clustered flags");
    }

    void evaluate(const std::vector<int>& seeds, int& best_cardinality,
                  int& best_seed) {
        std::vector<int> cardinalities(batch_capacity_);
        for (size_t begin = 0; begin < seeds.size(); begin += batch_capacity_) {
            const int count = static_cast<int>(
                std::min<size_t>(batch_capacity_, seeds.size() - begin));
            cudaCheck(cudaMemcpy(device_seeds_, seeds.data() + begin,
                                 count * sizeof(int), cudaMemcpyHostToDevice),
                      "uploading seed batch");
            buildCandidateClusters<<<count, CUDA_BLOCK_SIZE>>>(
                device_points_, device_clustered_, device_seeds_, count,
                point_count_, threshold_, device_scores_, device_cardinalities_, nullptr);
            cudaCheck(cudaGetLastError(), "launching candidate kernel");
            cudaCheck(cudaMemcpy(cardinalities.data(), device_cardinalities_,
                                 count * sizeof(int), cudaMemcpyDeviceToHost),
                      "downloading cardinalities");
            for (int i = 0; i < count; ++i) {
                const int seed = seeds[begin + i];
                if (cardinalities[i] > best_cardinality ||
                    (cardinalities[i] == best_cardinality && seed < best_seed)) {
                    best_cardinality = cardinalities[i];
                    best_seed = seed;
                }
            }
        }
    }

    std::vector<int> materialize(int seed, int expected_cardinality) {
        cudaCheck(cudaMemcpy(device_seeds_, &seed, sizeof(int), cudaMemcpyHostToDevice),
                  "uploading winning seed");
        buildCandidateClusters<<<1, CUDA_BLOCK_SIZE>>>(
            device_points_, device_clustered_, device_seeds_, 1, point_count_,
            threshold_, device_scores_, device_cardinalities_, device_members_);
        cudaCheck(cudaGetLastError(), "launching winner kernel");
        int cardinality = 0;
        cudaCheck(cudaMemcpy(&cardinality, device_cardinalities_, sizeof(int),
                             cudaMemcpyDeviceToHost), "downloading winner size");
        if (cardinality != expected_cardinality) {
            std::fprintf(stderr, "Rank %d: inconsistent winning cardinality (%d != %d)\n",
                         mpi_rank, cardinality, expected_cardinality);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        std::vector<int> members(cardinality);
        cudaCheck(cudaMemcpy(members.data(), device_members_,
                             cardinality * sizeof(int), cudaMemcpyDeviceToHost),
                  "downloading winning cluster");
        return members;
    }

    void setThreshold(double threshold) { threshold_ = threshold; }
    int batchCapacity() const { return batch_capacity_; }

private:
    int point_count_ = 0;
    int batch_capacity_ = 1;
    double threshold_ = 0.0;
    double2* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    double* device_scores_ = nullptr;
    int* device_seeds_ = nullptr;
    int* device_cardinalities_ = nullptr;
    int* device_members_ = nullptr;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold, int world_size) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;

    CudaWorkspace gpu(points, (n + world_size - 1) / world_size);
    gpu.setThreshold(threshold);

    while (remaining > 0) {
        gpu.uploadClustered(clustered);
        std::vector<int> local_seeds;
        local_seeds.reserve((remaining + world_size - 1) / world_size);
        for (int seed = mpi_rank; seed < n; seed += world_size)
            if (!clustered[seed]) local_seeds.push_back(seed);

        int local_cardinality = -1;
        int local_seed = INT_MAX;
        gpu.evaluate(local_seeds, local_cardinality, local_seed);

        struct { int cardinality; int seed; } local_best{local_cardinality, local_seed},
                                               global_best{-1, INT_MAX};
        mpiCheck(MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC,
                               MPI_COMM_WORLD), "reducing winning seed");
        if (global_best.cardinality <= 0 || global_best.seed == INT_MAX) break;

        const int owner = global_best.seed % world_size;
        std::vector<int> members;
        if (mpi_rank == owner)
            members = gpu.materialize(global_best.seed, global_best.cardinality);
        else
            members.resize(global_best.cardinality);
        mpiCheck(MPI_Bcast(members.data(), global_best.cardinality, MPI_INT, owner,
                           MPI_COMM_WORLD), "broadcasting winning cluster");

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < global_best.cardinality; ++i)
            clustered[members[i]] = 1;
        remaining -= global_best.cardinality;

        if (mpi_rank == 0)
            clusters.push_back(Cluster{std::move(members), global_best.seed});
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points, double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:max_diameter)
        for (long long i = 0; i < static_cast<long long>(cluster.members.size()); ++i) {
            for (size_t j = static_cast<size_t>(i) + 1; j < cluster.members.size(); ++j)
                max_diameter = std::max(max_diameter,
                    distance(points[cluster.members[i]], points[cluster.members[j]]));
        }
        if (c < 10)
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c,
                        cluster.members.size(), cluster.seed_point, max_diameter);
        if (max_diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        c, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (int member : clusters[c].members) {
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
    for (long long i = 0; i < static_cast<long long>(membership.size()); ++i)
        clustered_count += membership[i] >= 0;
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count, points.size() - clustered_count);
    return valid;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided),
             "initializing MPI");
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) std::fprintf(stderr, "MPI lacks required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int local_rank = 0;
    MPI_Comm local_comm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank,
                                 MPI_INFO_NULL, &local_comm), "creating node communicator");
    MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "enumerating CUDA devices");
    if (device_count <= 0) {
        if (mpi_rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count), "selecting CUDA device");
    MPI_Comm_free(&local_comm);

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
            if (mpi_rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            parse_status = 1;
        }
    }
    if (parse_status == 2) {
        if (mpi_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (parse_status || num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            if (!parse_status)
                std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                            num_points, threshold);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI ranks, %d OpenMP threads/rank, CUDA\n",
                    world_size, omp_get_max_threads());
    }

    std::vector<Point> points(num_points);
    if (mpi_rank == 0) generateSyntheticData(points, num_points);
    static_assert(std::is_standard_layout_v<Point> && sizeof(Point) == 2 * sizeof(double));
    mpiCheck(MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "broadcasting input points");

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "starting timed region");
    const double start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, world_size);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    mpiCheck(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD), "reducing execution time");

    int result = 0;
    if (mpi_rank == 0) {
        const long long elapsed_ms = static_cast<long long>(elapsed * 1000.0);
        std::printf("Clustering time: %lld ms\n", elapsed_ms);
        std::printf("Clusters found: %zu\n", clusters.size());
        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const auto& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
        const double average = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
                    num_points, 100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);
        const double safe_elapsed = std::max(elapsed, std::numeric_limits<double>::min());
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / safe_elapsed, num_points / safe_elapsed);

        if (print_results_requested) {
            std::vector<double> membership_data(num_points, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (int member : clusters[c].members)
                    membership_data[member] = static_cast<double>(c);
            print_results(membership_data, "ClusterMembership");
        }
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "broadcasting exit status");
    MPI_Finalize();
    return result;
}
