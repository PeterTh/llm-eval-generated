// Hybrid MPI + OpenMP + CUDA QT clustering benchmark.
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

static void cudaCheck(cudaError_t e, const char *where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), #x)

void generateSyntheticData(std::vector<Point>& points, int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
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

__device__ __forceinline__ bool better(double d, int i, double best_d, int best_i) {
    return d < best_d || (d == best_d && i < best_i);
}

// One block constructs one candidate cluster. Its sequential greedy steps are
// retained, while all candidate points in every step are evaluated in parallel.
__global__ void candidateKernel(const Point *__restrict__ points,
                                const unsigned char *__restrict__ clustered,
                                const int *__restrict__ seeds, int seed_count, int n,
                                double threshold, unsigned char *in_workspace,
                                int *member_workspace, int *cardinalities) {
    const int b = blockIdx.x;
    if (b >= seed_count) return;
    unsigned char *in = in_workspace + static_cast<size_t>(b) * n;
    int *members = member_workspace + static_cast<size_t>(b) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) in[i] = 0;
    __shared__ int count;
    __shared__ double best_dist[CUDA_THREADS];
    __shared__ int best_index[CUDA_THREADS];
    if (threadIdx.x == 0) { members[0] = seeds[b]; count = 1; }
    __syncthreads();
    if (threadIdx.x == 0) in[seeds[b]] = 1;
    __syncthreads();

    while (count < n) {
        double local_d = DBL_MAX;
        int local_i = INT_MAX;
        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            if (clustered[candidate] || in[candidate]) continue;
            double max_d = 0.0;
            const Point p = points[candidate];
            for (int j = 0; j < count; ++j) {
                const Point q = points[members[j]];
                const double dx = p.x - q.x, dy = p.y - q.y;
                max_d = fmax(max_d, sqrt(dx * dx + dy * dy));
            }
            if (max_d < threshold && better(max_d, candidate, local_d, local_i)) {
                local_d = max_d; local_i = candidate;
            }
        }
        best_dist[threadIdx.x] = local_d;
        best_index[threadIdx.x] = local_i;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride && better(best_dist[threadIdx.x + stride],
                                               best_index[threadIdx.x + stride],
                                               best_dist[threadIdx.x], best_index[threadIdx.x])) {
                best_dist[threadIdx.x] = best_dist[threadIdx.x + stride];
                best_index[threadIdx.x] = best_index[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (best_index[0] == INT_MAX) break;
        if (threadIdx.x == 0) {
            const int chosen = best_index[0];
            in[chosen] = 1; members[count++] = chosen;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) cardinalities[b] = count;
}

class GpuWorkspace {
public:
    Point *points = nullptr;
    unsigned char *clustered = nullptr, *in = nullptr;
    int *seeds = nullptr, *members = nullptr, *cards = nullptr;
    int n, capacity;

    GpuWorkspace(const std::vector<Point>& host, int requested) : n(static_cast<int>(host.size())) {
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const size_t fixed = sizeof(Point) * static_cast<size_t>(n) + n;
        const size_t per = static_cast<size_t>(n) * (sizeof(int) + 1) + 2 * sizeof(int);
        const size_t budget = free_b * 3 / 4;
        capacity = std::max(1, std::min(requested, static_cast<int>((budget > fixed ? budget - fixed : per) / per)));
        CUDA_CHECK(cudaMalloc(&points, sizeof(Point) * n));
        CUDA_CHECK(cudaMalloc(&clustered, n));
        CUDA_CHECK(cudaMalloc(&seeds, sizeof(int) * capacity));
        CUDA_CHECK(cudaMalloc(&in, static_cast<size_t>(n) * capacity));
        CUDA_CHECK(cudaMalloc(&members, sizeof(int) * static_cast<size_t>(n) * capacity));
        CUDA_CHECK(cudaMalloc(&cards, sizeof(int) * capacity));
        CUDA_CHECK(cudaMemcpy(points, host.data(), sizeof(Point) * n, cudaMemcpyHostToDevice));
    }
    ~GpuWorkspace() { cudaFree(cards); cudaFree(members); cudaFree(in); cudaFree(seeds); cudaFree(clustered); cudaFree(points); }
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    const int maximum_local = (n + ranks - 1) / ranks;
    GpuWorkspace gpu(points, std::max(1, maximum_local));

    int remaining = n;
    while (remaining) {
        CUDA_CHECK(cudaMemcpy(gpu.clustered, clustered.data(), n, cudaMemcpyHostToDevice));
        std::vector<int> local_seeds;
        local_seeds.reserve(maximum_local);
        for (int seed = rank; seed < n; seed += ranks)
            if (!clustered[seed]) local_seeds.push_back(seed);

        int local_card = -1, local_seed = INT_MAX;
        for (size_t begin = 0; begin < local_seeds.size(); begin += gpu.capacity) {
            const int batch = std::min<int>(gpu.capacity, local_seeds.size() - begin);
            CUDA_CHECK(cudaMemcpy(gpu.seeds, local_seeds.data() + begin, sizeof(int) * batch, cudaMemcpyHostToDevice));
            candidateKernel<<<batch, CUDA_THREADS>>>(gpu.points, gpu.clustered, gpu.seeds, batch,
                n, threshold, gpu.in, gpu.members, gpu.cards);
            CUDA_CHECK(cudaGetLastError());
            std::vector<int> cards(batch);
            CUDA_CHECK(cudaMemcpy(cards.data(), gpu.cards, sizeof(int) * batch, cudaMemcpyDeviceToHost));
            for (int b = 0; b < batch; ++b)
                if (cards[b] > local_card || (cards[b] == local_card && local_seeds[begin + b] < local_seed)) {
                    local_card = cards[b]; local_seed = local_seeds[begin + b];
                }
        }

        struct { int value, location; } local{local_card, local_seed}, global{-1, INT_MAX};
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global.value <= 0) break;

        std::vector<int> winning(global.value);
        const int owner = global.location % ranks;
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(gpu.seeds, &global.location, sizeof(int), cudaMemcpyHostToDevice));
            candidateKernel<<<1, CUDA_THREADS>>>(gpu.points, gpu.clustered, gpu.seeds, 1, n,
                threshold, gpu.in, gpu.members, gpu.cards);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(winning.data(), gpu.members, sizeof(int) * global.value, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(winning.data(), global.value, MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back({winning, global.location});
        for (int p : winning) clustered[p] = 1;
        remaining -= global.value;
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points,
                      double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cl = clusters[c];
        double max_d = 0.0;
#pragma omp parallel for schedule(static) reduction(max:max_d)
        for (long long i = 0; i < static_cast<long long>(cl.members.size()); ++i)
            for (size_t j = i + 1; j < cl.members.size(); ++j) {
                const Point a = points[cl.members[i]], b = points[cl.members[j]];
                const double dx = a.x-b.x, dy = a.y-b.y;
                max_d = std::max(max_d, std::sqrt(dx*dx + dy*dy));
            }
        if (c < 10) std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                                c, cl.members.size(), cl.seed_point, max_d);
        if (max_d > threshold * 1.001) { std::printf("ERROR: Cluster %zu exceeds threshold\n", c); valid = false; }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) for (int p : clusters[c].members) {
        if (membership[p] >= 0) { std::printf("ERROR: Point %d appears multiple times\n", p); valid = false; }
        membership[p] = static_cast<int>(c);
    }
    const int count = static_cast<int>(std::count_if(membership.begin(), membership.end(), [](int x){return x >= 0;}));
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), count, points.size()-count);
    return valid;
}

void printUsage(const char *p) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of points (default: 1000)\n"
                "  -t <float>   Distance threshold (default: 2.0)\n  -v           Enable validation\n"
                "  -r           Print results for external validation\n  -h           Show help\n", p);
}

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local_comm; int local_rank = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank); MPI_Comm_free(&local_comm);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "A CUDA device is required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));

    int n = 1000; double threshold = 2.0; bool validate = false, results = false;
    bool args_ok = true, help = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1<argc) n=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i+1<argc) threshold=std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate=true;
        else if (!std::strcmp(argv[i], "-r")) results=true;
        else if (!std::strcmp(argv[i], "-h")) help=true;
        else args_ok=false;
    }
    if (help) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
    if (!args_ok || n<=0 || threshold<=0) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    if (!rank) std::printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\n"
                           "Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                           n, threshold, validate?"enabled":"disabled", ranks, omp_get_max_threads());
    std::vector<Point> points(n);
    if (!rank) generateSyntheticData(points, n);
    MPI_Bcast(points.data(), static_cast<int>(sizeof(Point)*n), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const auto clusters = qtClustering(points, threshold, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();

    int total=0, maximum=0;
#pragma omp parallel for reduction(+:total) reduction(max:maximum) schedule(static)
    for (long long i=0; i<static_cast<long long>(clusters.size()); ++i) {
        const int s=static_cast<int>(clusters[i].members.size()); total += s; maximum=std::max(maximum,s);
    }
    int exit_code=0;
    if (!rank) {
        const double avg=clusters.empty()?0.0:static_cast<double>(total)/clusters.size();
        std::printf("Clustering time: %.0f ms\nClusters found: %zu\nPoints clustered: %d / %d (%.1f%%)\n"
                    "Average cluster size: %.2f\nMaximum cluster size: %d\nPerformance: %.1f clusters/s, %.1f points/s\n",
                    elapsed*1000, clusters.size(), total,n,100.0*total/n,avg,maximum,
                    clusters.size()/elapsed,n/elapsed);
        if (results) {
            std::vector<int> membership(n,-1); std::vector<double> data(n);
            for (size_t c=0;c<clusters.size();++c) for(int p:clusters[c].members) membership[p]=static_cast<int>(c);
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i) data[i]=static_cast<double>(membership[i]);
            print_results(data,"ClusterMembership");
        }
        if (validate) { const bool ok=validateClusters(clusters,points,threshold); std::printf("Validation: %s\n",ok?"PASSED":"FAILED"); exit_code=ok?0:1; }
    }
    MPI_Bcast(&exit_code,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize(); return exit_code;
}
