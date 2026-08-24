// QT clustering benchmark: mandatory MPI + OpenMP + CUDA implementation.
#include <algorithm>
#include <chrono>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA failure during %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void generateSyntheticData(std::vector<Point>& points, const int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (n / 30.0));
        if (group_count > n - count) group_count = n - count;
        while (group_count > 0) {
            const double sign = frand() < .5 ? -1.0 : 1.0;
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

inline double distance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// One block builds one candidate cluster.  Candidate points are partitioned among
// the threads; the deterministic reduction preserves the sequential lowest-index tie.
__global__ void buildCandidates(const Point* points, const unsigned char* clustered,
                                int* cardinalities, unsigned char* memberships,
                                int n, int rank, int ranks, double threshold) {
    const int local_seed = blockIdx.x;
    const int seed = rank + local_seed * ranks;
    if (seed >= n) return;
    __shared__ double scores[CUDA_THREADS];
    __shared__ int ids[CUDA_THREADS];
    __shared__ int member_count;
    __shared__ int keep_going;
    unsigned char* in_cluster = memberships + static_cast<size_t>(local_seed) * n;

    for (int i = threadIdx.x; i < n; i += blockDim.x) in_cluster[i] = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        member_count = clustered[seed] ? 0 : 1;
        keep_going = member_count;
        if (member_count) in_cluster[seed] = 1;
    }
    __syncthreads();

    for (int iteration = 1; iteration < n; ++iteration) {
        double best_score = 1.7976931348623157e308;
        int best_id = n;
        if (keep_going) {
            for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
                if (!clustered[candidate] && !in_cluster[candidate]) {
                    double maximum = 0.0;
                    for (int member = 0; member < n; ++member) {
                        if (in_cluster[member]) {
                            const double dx = points[candidate].x - points[member].x;
                            const double dy = points[candidate].y - points[member].y;
                            maximum = fmax(maximum, sqrt(dx * dx + dy * dy));
                        }
                    }
                    if (maximum < threshold) { best_score = maximum; best_id = candidate; }
                }
            }
        }
        scores[threadIdx.x] = best_score;
        ids[threadIdx.x] = best_id;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                const int other = threadIdx.x + stride;
                if (scores[other] < scores[threadIdx.x] ||
                    (scores[other] == scores[threadIdx.x] && ids[other] < ids[threadIdx.x])) {
                    scores[threadIdx.x] = scores[other]; ids[threadIdx.x] = ids[other];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (ids[0] == n) keep_going = 0;
            else { in_cluster[ids[0]] = 1; ++member_count; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) cardinalities[local_seed] = member_count;
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                         int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    const int local_count = rank < n ? (n - 1 - rank) / ranks + 1 : 0;
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_cardinalities = nullptr;
    unsigned char* d_memberships = nullptr;
    cudaCheck(cudaMalloc(&d_points, n * sizeof(Point)), "point allocation");
    cudaCheck(cudaMalloc(&d_clustered, n * sizeof(unsigned char)), "state allocation");
    cudaCheck(cudaMalloc(&d_cardinalities, std::max(1, local_count) * static_cast<int>(sizeof(int))), "cardinality allocation");
    cudaCheck(cudaMalloc(&d_memberships, std::max<size_t>(1, static_cast<size_t>(local_count) * n)), "membership allocation");
    cudaCheck(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice), "point upload");

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> local_cards(local_count);
    std::vector<Cluster> clusters;
    while (true) {
        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), n, cudaMemcpyHostToDevice), "state upload");
        if (local_count) {
            buildCandidates<<<local_count, CUDA_THREADS>>>(d_points, d_clustered, d_cardinalities,
                d_memberships, n, rank, ranks, threshold);
            cudaCheck(cudaGetLastError(), "candidate kernel launch");
            cudaCheck(cudaDeviceSynchronize(), "candidate kernel");
            cudaCheck(cudaMemcpy(local_cards.data(), d_cardinalities, local_count * sizeof(int), cudaMemcpyDeviceToHost), "cardinality download");
        }

        int local_best_card = 0, local_best_seed = n;
        #pragma omp parallel
        {
            int thread_card = 0, thread_seed = n;
            #pragma omp for nowait
            for (int i = 0; i < local_count; ++i) {
                const int seed = rank + i * ranks;
                if (local_cards[i] > thread_card || (local_cards[i] == thread_card && seed < thread_seed)) {
                    thread_card = local_cards[i]; thread_seed = seed;
                }
            }
            #pragma omp critical
            if (thread_card > local_best_card || (thread_card == local_best_card && thread_seed < local_best_seed)) {
                local_best_card = thread_card; local_best_seed = thread_seed;
            }
        }
        std::vector<int> gathered(2 * ranks);
        const int local_pair[2] = {local_best_card, local_best_seed};
        MPI_Allgather(local_pair, 2, MPI_INT, gathered.data(), 2, MPI_INT, MPI_COMM_WORLD);
        int best_card = 0, best_seed = n, winner = 0;
        for (int r = 0; r < ranks; ++r) {
            const int card = gathered[2 * r], seed = gathered[2 * r + 1];
            if (card > best_card || (card == best_card && seed < best_seed)) {
                best_card = card; best_seed = seed; winner = r;
            }
        }
        if (!best_card) break;
        std::vector<unsigned char> chosen(n);
        if (rank == winner) {
            const int local_seed = (best_seed - rank) / ranks;
            cudaCheck(cudaMemcpy(chosen.data(), d_memberships + static_cast<size_t>(local_seed) * n,
                                 n, cudaMemcpyDeviceToHost), "winning membership download");
        }
        MPI_Bcast(chosen.data(), n, MPI_UNSIGNED_CHAR, winner, MPI_COMM_WORLD);
        for (int i = 0; i < n; ++i) if (chosen[i]) clustered[i] = 1;
        if (rank == 0) {
            Cluster cluster; cluster.seed_point = best_seed;
            cluster.members.reserve(best_card);
            for (int i = 0; i < n; ++i) if (chosen[i]) cluster.members.push_back(i);
            clusters.push_back(std::move(cluster));
        }
    }
    cudaFree(d_memberships); cudaFree(d_cardinalities); cudaFree(d_clustered); cudaFree(d_points);
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true; printf("Validating clusters:\n");
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        double diameter = 0.0;
        for (size_t i = 0; i < clusters[c].members.size(); ++i) for (size_t j = i + 1; j < clusters[c].members.size(); ++j)
            diameter = std::max(diameter, distance(points[clusters[c].members[i]], points[clusters[c].members[j]]));
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, clusters[c].members.size(), clusters[c].seed_point, diameter);
        if (diameter > threshold * 1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, diameter, threshold); valid = false; }
        for (int member : clusters[c].members) {
            if (membership[member] >= 0) { printf("ERROR: Point %d appears in multiple clusters\n", member); valid = false; }
            membership[member] = static_cast<int>(c);
        }
    }
    int count = 0; for (int x : membership) if (x >= 0) ++count;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), count, points.size() - count);
    return valid;
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n  -n <num>     Number of points (default: 1000)\n  -t <float>   Distance threshold (default: 2.0)\n  -v           Enable validation\n  -r           Print results\n  -h           Show help\n", program);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank; MPI_Comm_rank(node_comm, &local_rank);
    int device_count = 0; cudaCheck(cudaGetDeviceCount(&device_count), "device discovery");
    if (!device_count) { if (!rank) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    // Use a node-local rank so every node maps ranks evenly over its visible GPUs.
    cudaCheck(cudaSetDevice(local_rank % device_count), "device selection");
    int n = 1000; double threshold = 2.0; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Comm_free(&node_comm); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Comm_free(&node_comm); MPI_Finalize(); return 1; }
    }
    if (n <= 0 || threshold <= 0) { if (!rank) fprintf(stderr, "Error: invalid parameters\n"); MPI_Comm_free(&node_comm); MPI_Finalize(); return 1; }
    std::vector<Point> points(n); generateSyntheticData(points, n);
    if (!rank) { printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\nMPI ranks: %d\n", n, threshold, validate ? "enabled" : "disabled", ranks); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    const long local_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    long elapsed = 0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    int exit_code = 0;
    if (!rank) {
        printf("Clustering time: %ld ms\nClusters found: %zu\n", elapsed, clusters.size());
        int total = 0, maximum = 0; for (const auto& c : clusters) { total += c.members.size(); maximum = std::max(maximum, static_cast<int>(c.members.size())); }
        const double seconds = elapsed / 1000.0;
        printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n", total, n, 100.0 * total / n, clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size(), maximum);
        printf("Performance: %.1f clusters/s, %.1f points/s\n", seconds ? clusters.size() / seconds : 0.0, seconds ? n / seconds : 0.0);
        if (results) { std::vector<double> out(n, -1); for (size_t c = 0; c < clusters.size(); ++c) for (int p : clusters[c].members) out[p] = c; print_results(out, "ClusterMembership"); }
        if (validate && !validateClusters(clusters, points, threshold)) exit_code = 1;
        else if (validate) printf("Validation: PASSED\n");
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return exit_code;
}
