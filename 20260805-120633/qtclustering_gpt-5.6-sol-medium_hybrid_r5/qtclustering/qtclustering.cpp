// Hybrid MPI + OpenMP + CUDA Quality Threshold clustering benchmark.
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call, comm) cudaCheck((call), #call, (comm))

void generateSyntheticData(std::vector<Point>& points, int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (n / 30.0));
        group_count = std::min(group_count, n - count);
        while (group_count > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y};
            --group_count;
        }
    }
}

inline double pointDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances, int n) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const int member = blockIdx.y * blockDim.y + threadIdx.y;
    if (candidate < n && member < n) {
        const double dx = points[candidate].x - points[member].x;
        const double dy = points[candidate].y - points[member].y;
        // Member-major storage makes all candidate loads in a warp contiguous.
        distances[static_cast<size_t>(member) * n + candidate] = sqrt(dx * dx + dy * dy);
    }
}

__device__ __forceinline__ bool betterCandidate(double distance_a, int index_a,
                                                 double distance_b, int index_b) {
    return distance_a < distance_b || (distance_a == distance_b && index_a < index_b);
}

// One block constructs one candidate cluster. Candidate points are evaluated in
// parallel; the ordered (diameter,index) reduction exactly implements the serial tie rule.
__global__ void constructCandidateClusters(const int* __restrict__ seeds, int seed_count,
                                           const unsigned char* __restrict__ clustered,
                                           const double* __restrict__ distances,
                                           double threshold, int n,
                                           unsigned char* __restrict__ states,
                                           int* __restrict__ members,
                                           int* __restrict__ cardinalities) {
    const int cluster_index = blockIdx.x;
    if (cluster_index >= seed_count) return;

    const int tid = threadIdx.x;
    unsigned char* state = states + static_cast<size_t>(cluster_index) * n;
    int* cluster_members = members + static_cast<size_t>(cluster_index) * n;
    for (int p = tid; p < n; p += blockDim.x) state[p] = clustered[p];
    __syncthreads();

    const int seed = seeds[cluster_index];
    if (tid == 0) {
        state[seed] = 1;
        cluster_members[0] = seed;
    }
    __syncthreads();

    __shared__ double reduce_distance[CUDA_THREADS];
    __shared__ int reduce_index[CUDA_THREADS];
    __shared__ int member_count;
    if (tid == 0) member_count = 1;
    __syncthreads();

    while (member_count < n) {
        double thread_best_distance = DBL_MAX;
        int thread_best_index = INT_MAX;
        for (int candidate = tid; candidate < n; candidate += blockDim.x) {
            if (state[candidate]) continue;
            double diameter = 0.0;
            for (int m = 0; m < member_count; ++m) {
                diameter = fmax(diameter,
                    distances[static_cast<size_t>(cluster_members[m]) * n + candidate]);
            }
            if (diameter < threshold &&
                betterCandidate(diameter, candidate, thread_best_distance, thread_best_index)) {
                thread_best_distance = diameter;
                thread_best_index = candidate;
            }
        }

        reduce_distance[tid] = thread_best_distance;
        reduce_index[tid] = thread_best_index;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset && betterCandidate(reduce_distance[tid + offset],
                                                reduce_index[tid + offset],
                                                reduce_distance[tid], reduce_index[tid])) {
                reduce_distance[tid] = reduce_distance[tid + offset];
                reduce_index[tid] = reduce_index[tid + offset];
            }
            __syncthreads();
        }

        if (reduce_index[0] == INT_MAX) break;
        if (tid == 0) {
            const int selected = reduce_index[0];
            state[selected] = 1;
            cluster_members[member_count++] = selected;
        }
        __syncthreads();
    }
    if (tid == 0) cardinalities[cluster_index] = member_count;
}

class GpuWorkspace {
public:
    GpuWorkspace(const std::vector<Point>& points, int maximum_local_seeds, MPI_Comm comm)
        : comm_(comm), n_(static_cast<int>(points.size())) {
        int world_rank = 0;
        MPI_Comm_rank(comm_, &world_rank);
        MPI_Comm local_comm;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);
        int local_rank = 0, device_count = 0;
        MPI_Comm_rank(local_comm, &local_rank);
        CUDA_CHECK(cudaGetDeviceCount(&device_count), comm_);
        if (device_count == 0) {
            if (world_rank == 0) std::fprintf(stderr, "CUDA-capable GPU required.\n");
            MPI_Abort(comm_, EXIT_FAILURE);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % device_count), comm_);
        MPI_Comm_free(&local_comm);

        CUDA_CHECK(cudaMalloc(&d_points_, static_cast<size_t>(n_) * sizeof(Point)), comm_);
        CUDA_CHECK(cudaMalloc(&d_clustered_, static_cast<size_t>(n_)), comm_);
        CUDA_CHECK(cudaMemcpy(d_points_, points.data(), static_cast<size_t>(n_) * sizeof(Point),
                              cudaMemcpyHostToDevice), comm_);

        size_t matrix_bytes = static_cast<size_t>(n_) * n_ * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_distances_, matrix_bytes), comm_);
        const dim3 threads(16, 16);
        const dim3 blocks((n_ + threads.x - 1) / threads.x,
                          (n_ + threads.y - 1) / threads.y);
        buildDistanceMatrix<<<blocks, threads>>>(d_points_, d_distances_, n_);
        CUDA_CHECK(cudaGetLastError(), comm_);
        CUDA_CHECK(cudaDeviceSynchronize(), comm_);

        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes), comm_);
        const size_t bytes_per_seed = static_cast<size_t>(n_) * (sizeof(int) + sizeof(unsigned char));
        const size_t affordable = bytes_per_seed == 0 ? 1 : (free_bytes * 3 / 4) / bytes_per_seed;
        batch_capacity_ = std::max(1, std::min(maximum_local_seeds,
                                               static_cast<int>(std::min<size_t>(affordable, INT_MAX))));
        CUDA_CHECK(cudaMalloc(&d_seeds_, static_cast<size_t>(batch_capacity_) * sizeof(int)), comm_);
        CUDA_CHECK(cudaMalloc(&d_states_, static_cast<size_t>(batch_capacity_) * n_), comm_);
        CUDA_CHECK(cudaMalloc(&d_members_, static_cast<size_t>(batch_capacity_) * n_ * sizeof(int)), comm_);
        CUDA_CHECK(cudaMalloc(&d_cardinalities_, static_cast<size_t>(batch_capacity_) * sizeof(int)), comm_);
        host_cardinalities_.resize(batch_capacity_);
    }

    ~GpuWorkspace() {
        cudaFree(d_cardinalities_); cudaFree(d_members_); cudaFree(d_states_);
        cudaFree(d_seeds_); cudaFree(d_distances_); cudaFree(d_clustered_); cudaFree(d_points_);
    }

    void bestLocalCandidate(const std::vector<int>& seeds,
                            const std::vector<unsigned char>& clustered, double threshold,
                            int& best_cardinality, int& best_seed,
                            std::vector<int>& best_members) {
        CUDA_CHECK(cudaMemcpy(d_clustered_, clustered.data(), static_cast<size_t>(n_),
                              cudaMemcpyHostToDevice), comm_);
        best_cardinality = -1;
        best_seed = INT_MAX;
        best_members.clear();
        for (size_t begin = 0; begin < seeds.size(); begin += batch_capacity_) {
            const int count = static_cast<int>(std::min<size_t>(batch_capacity_, seeds.size() - begin));
            CUDA_CHECK(cudaMemcpy(d_seeds_, seeds.data() + begin, static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyHostToDevice), comm_);
            constructCandidateClusters<<<count, CUDA_THREADS>>>(
                d_seeds_, count, d_clustered_, d_distances_, threshold, n_,
                d_states_, d_members_, d_cardinalities_);
            CUDA_CHECK(cudaGetLastError(), comm_);
            CUDA_CHECK(cudaMemcpy(host_cardinalities_.data(), d_cardinalities_,
                                  static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost), comm_);
            for (int i = 0; i < count; ++i) {
                const int seed = seeds[begin + i];
                const int cardinality = host_cardinalities_[i];
                if (cardinality > best_cardinality ||
                    (cardinality == best_cardinality && seed < best_seed)) {
                    best_cardinality = cardinality;
                    best_seed = seed;
                    best_members.resize(cardinality);
                    CUDA_CHECK(cudaMemcpy(best_members.data(),
                                          d_members_ + static_cast<size_t>(i) * n_,
                                          static_cast<size_t>(cardinality) * sizeof(int),
                                          cudaMemcpyDeviceToHost), comm_);
                }
            }
        }
    }

private:
    MPI_Comm comm_;
    int n_, batch_capacity_;
    Point* d_points_ = nullptr;
    unsigned char* d_clustered_ = nullptr;
    double* d_distances_ = nullptr;
    int* d_seeds_ = nullptr;
    unsigned char* d_states_ = nullptr;
    int* d_members_ = nullptr;
    int* d_cardinalities_ = nullptr;
    std::vector<int> host_cardinalities_;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  MPI_Comm comm) {
    int rank = 0, ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) unclustered[i] = i;

    const int maximum_local_seeds = (n + ranks - 1) / ranks;
    GpuWorkspace gpu(points, std::max(1, maximum_local_seeds), comm);
    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        std::vector<int> local_seeds;
        local_seeds.reserve((unclustered.size() + ranks - 1) / ranks);
        for (size_t i = rank; i < unclustered.size(); i += ranks)
            local_seeds.push_back(unclustered[i]);

        int local_cardinality, local_seed;
        std::vector<int> local_members;
        gpu.bestLocalCandidate(local_seeds, clustered, threshold,
                               local_cardinality, local_seed, local_members);

        struct { int cardinality; int seed; } local_pair{local_cardinality, local_seed}, global_pair{};
        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_2INT, MPI_MAXLOC, comm);
        if (global_pair.cardinality <= 0 || global_pair.seed == INT_MAX) break;

        int owner = (local_seed == global_pair.seed) ? rank : -1;
        MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MAX, comm);
        std::vector<int> selected_members(global_pair.cardinality);
        if (rank == owner) selected_members = local_members;
        MPI_Bcast(selected_members.data(), global_pair.cardinality, MPI_INT, owner, comm);
        clusters.push_back({selected_members, global_pair.seed});

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < global_pair.cardinality; ++i)
            clustered[selected_members[i]] = 1;

        // Static chunks plus ordered concatenation preserve the ascending seed order.
        const int thread_count = omp_get_max_threads();
        std::vector<std::vector<int>> kept(thread_count);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const size_t begin = unclustered.size() * tid / omp_get_num_threads();
            const size_t end = unclustered.size() * (tid + 1) / omp_get_num_threads();
            kept[tid].reserve(end - begin);
            for (size_t i = begin; i < end; ++i)
                if (!clustered[unclustered[i]]) kept[tid].push_back(unclustered[i]);
        }
        unclustered.clear();
        for (auto& chunk : kept)
            unclustered.insert(unclustered.end(), chunk.begin(), chunk.end());
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points, double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const Cluster& cluster = clusters[c];
        double max_diameter = 0.0;
        #pragma omp parallel for reduction(max:max_diameter) schedule(dynamic)
        for (long long i = 0; i < static_cast<long long>(cluster.members.size()); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j)
                max_diameter = std::max(max_diameter,
                    pointDistance(points[cluster.members[i]], points[cluster.members[j]]));
        }
        if (c < 10)
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        c, cluster.members.size(), cluster.seed_point, max_diameter);
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
    std::printf("Options:\n  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false, print_results_requested = false;
    int exit_code = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) num_points = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) threshold = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
               MPI_Finalize(); return 1; }
    }
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                                   num_points, threshold);
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        std::printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\n",
                    num_points, threshold);
        std::printf("Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                    validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
    }

    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize(), MPI_COMM_WORLD);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long elapsed_ms = static_cast<long>(elapsed * 1000.0);
        std::printf("Clustering time: %ld ms\nClusters found: %zu\n", elapsed_ms, clusters.size());
        int total_clustered = 0, max_cluster_size = 0;
        #pragma omp parallel for reduction(+:total_clustered) reduction(max:max_cluster_size)
        for (long long i = 0; i < static_cast<long long>(clusters.size()); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
        const double average = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n",
                    total_clustered, num_points, 100.0 * total_clustered / num_points,
                    average, max_cluster_size);
        const double safe_elapsed = std::max(elapsed, std::numeric_limits<double>::min());
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters.size() / safe_elapsed, num_points / safe_elapsed);

        if (print_results_requested) {
            std::vector<double> membership_data(num_points, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (int member : clusters[c].members) membership_data[member] = static_cast<double>(c);
            print_results(membership_data, "ClusterMembership");
        }
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
