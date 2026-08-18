// Hybrid MPI/OpenMP/CUDA QT clustering benchmark.
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(e_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
} while (0)

static void generateSyntheticData(std::vector<Point>& points, int n,
                                  unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH;
        const double cy = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (n / 30.0));
        group_count = std::min(group_count, n - count);
        while (group_count > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cx + dx, y = cy + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y};
            --group_count;
        }
    }
}

__device__ __forceinline__ double pointDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// One block owns one seed.  The greedy dependency is kept inside that block;
// all admissible-point distance scans and their deterministic reduction are parallel.
__global__ void buildCandidates(const Point* __restrict__ points,
                                const unsigned char* __restrict__ clustered,
                                const int* __restrict__ seeds, int seed_count,
                                int n, double threshold,
                                double* __restrict__ diameter_workspace,
                                int* __restrict__ member_workspace,
                                int* __restrict__ cardinalities) {
    const int job = blockIdx.x;
    if (job >= seed_count) return;
    const int tid = threadIdx.x;
    double* diameters = diameter_workspace + static_cast<size_t>(job) * n;
    int* members = member_workspace + static_cast<size_t>(job) * n;
    extern __shared__ unsigned char shared_raw[];
    double* best_distance = reinterpret_cast<double*>(shared_raw);
    int* best_point = reinterpret_cast<int*>(best_distance + blockDim.x);
    int* done = best_point + blockDim.x;

    if (tid == 0) {
        members[0] = seeds[job];
        cardinalities[job] = 1;
    }
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x)
        diameters[i] = (clustered[i] || i == seeds[job])
                           ? DBL_MAX : pointDistance(points[i], points[seeds[job]]);
    __syncthreads();

    while (true) {
        const int count = cardinalities[job];
        double local_distance = DBL_MAX;
        int local_point = INT_MAX;
        for (int candidate = tid; candidate < n; candidate += blockDim.x) {
            const double maximum = diameters[candidate];
            if (maximum < threshold &&
                (maximum < local_distance ||
                 (maximum == local_distance && candidate < local_point))) {
                local_distance = maximum;
                local_point = candidate;
            }
        }
        best_distance[tid] = local_distance;
        best_point[tid] = local_point;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride) {
                const double other_d = best_distance[tid + stride];
                const int other_p = best_point[tid + stride];
                if (other_d < best_distance[tid] ||
                    (other_d == best_distance[tid] && other_p < best_point[tid])) {
                    best_distance[tid] = other_d;
                    best_point[tid] = other_p;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            if (best_point[0] != INT_MAX) {
                members[count] = best_point[0];
                cardinalities[job] = count + 1;
            }
            *done = (best_point[0] == INT_MAX);
        }
        __syncthreads();
        if (*done) break;
        const int added = best_point[0];
        for (int candidate = tid; candidate < n; candidate += blockDim.x) {
            if (candidate == added) diameters[candidate] = DBL_MAX;
            else if (diameters[candidate] != DBL_MAX) {
                const double d = pointDistance(points[candidate], points[added]);
                if (d > diameters[candidate]) diameters[candidate] = d;
            }
        }
        __syncthreads();
    }
}

static double hostDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int *d_seeds = nullptr, *d_members = nullptr, *d_cardinalities = nullptr;
    double* d_diameters = nullptr;
    const int capacity = (n + ranks - 1) / ranks;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_seeds, static_cast<size_t>(capacity) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, static_cast<size_t>(capacity) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(capacity) * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_diameters, static_cast<size_t>(capacity) * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point),
                          cudaMemcpyHostToDevice));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    std::vector<int> local_seeds(capacity), local_cards(capacity);
    const size_t shared_bytes = CUDA_THREADS * (sizeof(double) + sizeof(int)) + sizeof(int);

    int remaining = n;
    while (remaining > 0) {
        int local_count = 0;
        // Static cyclic ownership balances ranks as the unclustered set shrinks.
        #pragma omp parallel for schedule(static)
        for (int i = rank; i < n; i += ranks) {
            if (!clustered[i]) {
                int slot;
                #pragma omp atomic capture
                slot = local_count++;
                local_seeds[slot] = i;
            }
        }
        // Atomic insertion does not preserve order, but final MAXLOC tie-breaking does.
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(n),
                              cudaMemcpyHostToDevice));
        if (local_count) {
            CUDA_CHECK(cudaMemcpy(d_seeds, local_seeds.data(), local_count * sizeof(int),
                                  cudaMemcpyHostToDevice));
            buildCandidates<<<local_count, CUDA_THREADS, shared_bytes>>>(
                d_points, d_clustered, d_seeds, local_count, n, threshold,
                d_diameters, d_members, d_cardinalities);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(local_cards.data(), d_cardinalities,
                                  local_count * sizeof(int), cudaMemcpyDeviceToHost));
        }

        struct { int value; int index; } local_best{-1, INT_MAX}, global_best{-1, INT_MAX};
        #pragma omp parallel
        {
            int thread_value = -1, thread_seed = INT_MAX;
            #pragma omp for nowait
            for (int i = 0; i < local_count; ++i) {
                if (local_cards[i] > thread_value ||
                    (local_cards[i] == thread_value && local_seeds[i] < thread_seed)) {
                    thread_value = local_cards[i];
                    thread_seed = local_seeds[i];
                }
            }
            #pragma omp critical
            if (thread_value > local_best.value ||
                (thread_value == local_best.value && thread_seed < local_best.index)) {
                local_best.value = thread_value;
                local_best.index = thread_seed;
            }
        }
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global_best.value <= 0) break;

        std::vector<int> chosen(global_best.value);
        const int owner = global_best.index % ranks;
        if (rank == owner) {
            int slot = -1;
            for (int i = 0; i < local_count; ++i)
                if (local_seeds[i] == global_best.index) { slot = i; break; }
            CUDA_CHECK(cudaMemcpy(chosen.data(), d_members + static_cast<size_t>(slot) * n,
                                  chosen.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(chosen.data(), global_best.value, MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back({chosen, global_best.index});
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < global_best.value; ++i) clustered[chosen[i]] = 1;
        remaining -= global_best.value;
    }

    CUDA_CHECK(cudaFree(d_diameters));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));
    return clusters;
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points, double threshold) {
    int valid = 1;
    std::printf("Validating clusters:\n");
    #pragma omp parallel for schedule(dynamic) reduction(&:valid)
    for (int c = 0; c < static_cast<int>(clusters.size()); ++c) {
        double diameter = 0.0;
        for (size_t i = 0; i < clusters[c].members.size(); ++i)
            for (size_t j = i + 1; j < clusters[c].members.size(); ++j)
                diameter = std::max(diameter, hostDistance(points[clusters[c].members[i]],
                                                           points[clusters[c].members[j]]));
        if (c < 10) {
            #pragma omp critical
            std::printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n", c,
                        clusters[c].members.size(), clusters[c].seed_point, diameter);
        }
        if (diameter > threshold * 1.001) valid = 0;
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (int member : clusters[c].members) {
            if (membership[member] >= 0) valid = 0;
            membership[member] = static_cast<int>(c);
        }
    const int count = static_cast<int>(std::count_if(membership.begin(), membership.end(),
                                                     [](int c) { return c >= 0; }));
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), count, points.size() - count);
    return valid != 0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>    Number of points (default: 1000)\n"
                "  -t <float>  Distance threshold (default: 2.0)\n"
                "  -v          Enable validation\n  -r          Print result hash\n"
                "  -h          Show help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 1000;
    double threshold = 2.0;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threshold = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || threshold <= 0.0) {
        if (rank == 0) std::printf("Error: Invalid parameters\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { if (rank == 0) std::fprintf(stderr, "No CUDA device found\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    std::vector<Point> points(n);
    if (rank == 0) generateSyntheticData(points, n);
    MPI_Bcast(points.data(), static_cast<int>(n * sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\n"
                    "Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                    n, threshold, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        int total = 0, largest = 0;
        for (const auto& c : clusters) { total += static_cast<int>(c.members.size()); largest = std::max(largest, static_cast<int>(c.members.size())); }
        const double average = clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size();
        std::printf("Clustering time: %ld ms\nClusters found: %zu\nPoints clustered: %d / %d (%.1f%%)\n"
                    "Average cluster size: %.2f\nMaximum cluster size: %d\n"
                    "Performance: %.1f clusters/s, %.1f points/s\n",
                    static_cast<long>(max_elapsed * 1000.0), clusters.size(), total, n,
                    100.0 * total / n, average, largest,
                    max_elapsed > 0 ? clusters.size() / max_elapsed : 0.0,
                    max_elapsed > 0 ? n / max_elapsed : 0.0);
        if (print_results_flag) {
            std::vector<double> result(n, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (int member : clusters[c].members) result[member] = static_cast<double>(c);
            print_results(result, "ClusterMembership");
        }
        if (validate) {
            const bool ok = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            exit_code = ok ? 0 : 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
