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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(e_));                                   \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

static void generateSyntheticData(std::vector<Point>& points, int n,
                                  unsigned int seed = 42) {
    auto frand = [&seed]() { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        group = std::min(group, n - count);
        while (group > 0) {
            const double sign = frand() < .5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cx + dx, y = cy + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y}; --group;
        }
    }
}

// One block constructs one candidate cluster. Candidate evaluation is spread
// over the block; its reduction uses (diameter,index) ordering to exactly retain
// the sequential implementation's first-index tie break.
__global__ static void candidateKernel(const Point* __restrict__ points,
                                       const unsigned char* __restrict__ clustered,
                                       const int* __restrict__ seeds, int seed_count,
                                       int n, double threshold,
                                       int* __restrict__ cardinalities,
                                       int* __restrict__ all_members,
                                       unsigned char* __restrict__ all_in_cluster) {
    const int job = blockIdx.x;
    if (job >= seed_count) return;
    const int tid = threadIdx.x;
    int* members = all_members + static_cast<size_t>(job) * n;
    unsigned char* in_cluster = all_in_cluster + static_cast<size_t>(job) * n;
    extern __shared__ unsigned char smem[];
    double* best_dist = reinterpret_cast<double*>(smem);
    int* best_index = reinterpret_cast<int*>(best_dist + blockDim.x);

    for (int i = tid; i < n; i += blockDim.x) in_cluster[i] = 0;
    __syncthreads();
    const int seed = seeds[job];
    if (tid == 0) { in_cluster[seed] = 1; members[0] = seed; }
    __syncthreads();

    int count = 1;
    while (count < n) {
        double local_dist = DBL_MAX;
        int local_index = INT_MAX;
        for (int candidate = tid; candidate < n; candidate += blockDim.x) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double maximum = 0.0;
            const Point p = points[candidate];
            for (int j = 0; j < count; ++j) {
                const Point q = points[members[j]];
                const double dx = p.x - q.x, dy = p.y - q.y;
                maximum = fmax(maximum, sqrt(dx * dx + dy * dy));
            }
            if (maximum < threshold &&
                (maximum < local_dist || (maximum == local_dist && candidate < local_index))) {
                local_dist = maximum; local_index = candidate;
            }
        }
        best_dist[tid] = local_dist; best_index[tid] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride) {
                const double d = best_dist[tid + stride];
                const int idx = best_index[tid + stride];
                if (d < best_dist[tid] || (d == best_dist[tid] && idx < best_index[tid])) {
                    best_dist[tid] = d; best_index[tid] = idx;
                }
            }
            __syncthreads();
        }
        const int chosen = best_index[0];
        if (chosen == INT_MAX) break;
        if (tid == 0) { in_cluster[chosen] = 1; members[count] = chosen; }
        ++count;
        __syncthreads();
    }
    if (tid == 0) cardinalities[job] = count;
}

class GpuCandidates {
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int n;
public:
    GpuCandidates(const std::vector<Point>& points) : n(static_cast<int>(points.size())) {
        CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n)));
        CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point),
                              cudaMemcpyHostToDevice));
    }
    ~GpuCandidates() { cudaFree(d_clustered); cudaFree(d_points); }

    void update(const std::vector<unsigned char>& clustered) {
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(n),
                              cudaMemcpyHostToDevice));
    }

    void evaluate(const std::vector<int>& seeds, double threshold,
                  int& best_card, int& best_seed, std::vector<int>& best_members) {
        if (seeds.empty()) return;
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t bytes_per_seed = static_cast<size_t>(n) * (sizeof(int) + 1);
        // Keep ample space for the runtime and other ranks sharing a device.
        size_t batch_limit = std::max<size_t>(1, std::min<size_t>(4096,
            (free_bytes / 2) / std::max<size_t>(bytes_per_seed, 1)));
        for (size_t first = 0; first < seeds.size(); first += batch_limit) {
            const int count = static_cast<int>(std::min(batch_limit, seeds.size() - first));
            int *d_seeds = nullptr, *d_cards = nullptr, *d_members = nullptr;
            unsigned char* d_in = nullptr;
            CUDA_CHECK(cudaMalloc(&d_seeds, static_cast<size_t>(count) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cards, static_cast<size_t>(count) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(count) * n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(count) * n));
            CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data() + first,
                                  static_cast<size_t>(count) * sizeof(int), cudaMemcpyHostToDevice));
            candidateKernel<<<count, CUDA_THREADS,
                CUDA_THREADS * (sizeof(double) + sizeof(int))>>>(
                d_points, d_clustered, d_seeds, count, n, threshold, d_cards, d_members, d_in);
            CUDA_CHECK(cudaGetLastError());
            std::vector<int> cards(count);
            CUDA_CHECK(cudaMemcpy(cards.data(), d_cards, static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            for (int j = 0; j < count; ++j) {
                const int seed = seeds[first + j];
                if (cards[j] > best_card || (cards[j] == best_card && seed < best_seed)) {
                    best_card = cards[j]; best_seed = seed; best_members.resize(cards[j]);
                    CUDA_CHECK(cudaMemcpy(best_members.data(),
                        d_members + static_cast<size_t>(j) * n,
                        static_cast<size_t>(cards[j]) * sizeof(int), cudaMemcpyDeviceToHost));
                }
            }
            cudaFree(d_in); cudaFree(d_members); cudaFree(d_cards); cudaFree(d_seeds);
        }
    }
};

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    GpuCandidates gpu(points);
    while (true) {
        std::vector<int> local_seeds;
        local_seeds.reserve((n + ranks - 1) / ranks);
        for (int i = rank; i < n; i += ranks) if (!clustered[i]) local_seeds.push_back(i);
        gpu.update(clustered);
        int local_card = -1, local_seed = INT_MAX;
        std::vector<int> local_members;
        gpu.evaluate(local_seeds, threshold, local_card, local_seed, local_members);

        struct { int value; int index; } local{local_card, local_seed}, global{-1, INT_MAX};
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global.value <= 0 || global.index == INT_MAX) break;
        const int owner = global.index % ranks;
        std::vector<int> members(global.value);
        if (rank == owner) members = std::move(local_members);
        MPI_Bcast(members.data(), global.value, MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < global.value; ++i) clustered[members[i]] = 1;
        clusters.push_back({std::move(members), global.index});
    }
    return clusters;
}

static inline double distance(const Point& a, const Point& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points, double threshold) {
    std::printf("Validating clusters:\n");
    int valid = 1;
    #pragma omp parallel for schedule(dynamic) reduction(&:valid)
    for (int c = 0; c < static_cast<int>(clusters.size()); ++c) {
        double diameter = 0.0;
        const auto& m = clusters[c].members;
        for (size_t i = 0; i < m.size(); ++i)
            for (size_t j = i + 1; j < m.size(); ++j)
                diameter = std::max(diameter, distance(points[m[i]], points[m[j]]));
        if (diameter > threshold * 1.001) valid = 0;
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (int p : clusters[c].members) {
            if (membership[p] >= 0) valid = 0;
            membership[p] = static_cast<int>(c);
        }
    const int count = static_cast<int>(std::count_if(membership.begin(), membership.end(),
                                                      [](int x) { return x >= 0; }));
    for (size_t c = 0; c < std::min<size_t>(10, clusters.size()); ++c)
        std::printf("  Cluster %zu: size=%zu, seed=%d\n", c, clusters[c].members.size(),
                    clusters[c].seed_point);
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), count, points.size() - count);
    return valid != 0;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [-n points] [-t threshold] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0; MPI_Comm_rank(local_comm, &local_rank); MPI_Comm_free(&local_comm);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (!rank) std::fprintf(stderr, "A CUDA GPU is required.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));

    int n = 1000; double threshold = 2.0; bool validate = false, results = false;
    bool help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threshold = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad || n <= 0 || threshold <= 0) {
        if (!rank) printUsage(argv[0]); MPI_Finalize(); return bad || n <= 0 || threshold <= 0;
    }
    std::vector<Point> points(n);
    if (!rank) generateSyntheticData(points, n);
    MPI_Bcast(points.data(), 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) {
        std::printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\n"
                    "Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                    n, threshold, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const auto clusters = qtClustering(points, threshold, rank, ranks);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0; MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int rc = 0;
    if (!rank) {
        int total = 0, maximum = 0;
        for (const auto& c : clusters) { total += c.members.size(); maximum = std::max(maximum, static_cast<int>(c.members.size())); }
        const double average = clusters.empty() ? 0 : static_cast<double>(total) / clusters.size();
        std::printf("Clustering time: %.3f ms\nClusters found: %zu\nPoints clustered: %d / %d (%.1f%%)\n"
                    "Average cluster size: %.2f\nMaximum cluster size: %d\nPerformance: %.1f clusters/s, %.1f points/s\n",
                    elapsed * 1000, clusters.size(), total, n, 100.0 * total / n, average, maximum,
                    elapsed ? clusters.size() / elapsed : 0, elapsed ? n / elapsed : 0);
        if (results) {
            std::vector<int> membership(n, -1); std::vector<double> data(n);
            for (size_t c = 0; c < clusters.size(); ++c) for (int p : clusters[c].members) membership[p] = c;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) data[i] = membership[i];
            print_results(data, "ClusterMembership");
        }
        if (validate) { rc = validateClusters(clusters, points, threshold) ? 0 : 1; std::printf("Validation: %s\n", rc ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return rc;
}
