// QT Clustering Benchmark -- MPI + OpenMP + CUDA implementation.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0, MAX_HEIGHT = 20.0;
struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void cudaCheck(cudaError_t status, const char *where) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void distanceMatrixKernel(const Point *points, double *distances, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const double dx = points[row].x - points[col].x;
        const double dy = points[row].y - points[col].y;
        distances[static_cast<size_t>(row) * n + col] = sqrt(dx * dx + dy * dy);
    }
}

void generateSyntheticData(std::vector<Point>& points, const int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double rmax = frand() * std::min(MAX_WIDTH, MAX_HEIGHT) / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        if (group > n - count) group = n - count;
        while (group > 0) {
            const double r = frand() * rmax;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = sqrt(r * r - dx * dx) * (frand() < .5 ? -1.0 : 1.0);
            const double x = cx + dx, y = cy + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y}; --group;
        }
    }
}

inline double distance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// This is intentionally local-state only: OpenMP can evaluate independent seeds safely.
static int generateCandidateCluster(int seed, const std::vector<unsigned char>& clustered,
                                    const std::vector<double>& distances, double threshold,
                                    int n, std::vector<int>* result = nullptr) {
    std::vector<unsigned char> in_cluster(n, 0);
    std::vector<int> members;
    members.reserve(n);
    in_cluster[seed] = 1; members.push_back(seed);
    while (static_cast<int>(members.size()) < n) {
        int closest = -1;
        double minimum = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double maximum = 0.0;
            const double *row = &distances[static_cast<size_t>(candidate) * n];
            for (int member : members) maximum = std::max(maximum, row[member]);
            if (maximum < threshold && maximum < minimum) {
                minimum = maximum; closest = candidate;
            }
        }
        if (closest < 0) break;
        in_cluster[closest] = 1; members.push_back(closest);
    }
    if (result) *result = std::move(members);
    return result ? static_cast<int>(result->size()) : static_cast<int>(members.size());
}

struct Candidate { int cardinality; int seed; };

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  const std::vector<double>& distances, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    while (!unclustered.empty()) {
        Candidate local{-1, std::numeric_limits<int>::max()};
        // Static ownership makes each rank's work disjoint; threads process its seeds concurrently.
        #pragma omp parallel
        {
            Candidate thread_best{-1, std::numeric_limits<int>::max()};
            #pragma omp for nowait schedule(dynamic)
            for (int pos = rank; pos < static_cast<int>(unclustered.size()); pos += ranks) {
                const int seed = unclustered[pos];
                const int count = generateCandidateCluster(seed, clustered, distances, threshold, n);
                if (count > thread_best.cardinality ||
                    (count == thread_best.cardinality && seed < thread_best.seed)) thread_best = {count, seed};
            }
            #pragma omp critical
            if (thread_best.cardinality > local.cardinality ||
                (thread_best.cardinality == local.cardinality && thread_best.seed < local.seed)) local = thread_best;
        }
        // MAXLOC selects the lowest global seed for equal cardinality, matching sequential scan order.
        struct { int value, index; } in{local.cardinality, -local.seed}, out{};
        MPI_Allreduce(&in, &out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_seed = -out.index;
        std::vector<int> best_members;
        const int owner = [&] { for (int p = 0; p < static_cast<int>(unclustered.size()); ++p)
            if (unclustered[p] == best_seed) return p % ranks; return 0; }();
        if (out.value <= 0 || best_seed < 0) break;
        if (rank == owner) generateCandidateCluster(best_seed, clustered, distances, threshold, n, &best_members);
        int member_count = rank == owner ? static_cast<int>(best_members.size()) : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner) best_members.resize(member_count);
        MPI_Bcast(best_members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back({best_members, best_seed});
        for (int p : best_members) clustered[p] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                          [&clustered](int p) { return clustered[p]; }), unclustered.end());
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true; printf("Validating clusters:\n");
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        double diameter = 0;
        for (size_t i = 0; i < clusters[c].members.size(); ++i) for (size_t j = i + 1; j < clusters[c].members.size(); ++j)
            diameter = std::max(diameter, distance(points[clusters[c].members[i]], points[clusters[c].members[j]]));
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, clusters[c].members.size(), clusters[c].seed_point, diameter);
        if (diameter > threshold * 1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, diameter, threshold); valid = false; }
        for (int member : clusters[c].members) { if (membership[member] >= 0) valid = false; membership[member] = static_cast<int>(c); }
    }
    int count = 0; for (int value : membership) count += value >= 0;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), count, points.size() - count);
    return valid;
}

void printUsage(const char* p) { printf("Usage: %s [options]\n  -n <num>     Number of points (default: 1000)\n  -t <float>   Distance threshold (default: 2.0)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    // Map ranks round-robin across accelerators on each node.  MPI_COMM_TYPE_SHARED
    // keeps this placement local even when different nodes have different GPU counts.
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                             MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0; cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) { if (!rank) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(local_rank % device_count), "cudaSetDevice");
    MPI_Comm_free(&local_comm);
    int n = 1000; double threshold = 2.0; bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) { if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]); else if (!strcmp(argv[i], "-t") && i + 1 < argc) threshold = atof(argv[++i]); else if (!strcmp(argv[i], "-v")) validate = true; else if (!strcmp(argv[i], "-r")) print_results_flag = true; else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; } else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; } }
    if (n <= 0 || threshold <= 0) { if (!rank) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", n, threshold); MPI_Finalize(); return 1; }
    std::vector<Point> points(n); if (!rank) generateSyntheticData(points, n); MPI_Bcast(points.data(), n * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (!rank) { printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\n", n, threshold, validate ? "enabled" : "disabled"); }
    const size_t matrix_size = static_cast<size_t>(n) * n;
    Point *device_points; double *device_distances; std::vector<double> distances(matrix_size);
    cudaCheck(cudaMalloc(&device_points, n * sizeof(Point)), "cudaMalloc(points)"); cudaCheck(cudaMalloc(&device_distances, matrix_size * sizeof(double)), "cudaMalloc(distances)");
    cudaCheck(cudaMemcpy(device_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice), "copy points");
    dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16); distanceMatrixKernel<<<grid, block>>>(device_points, device_distances, n);
    cudaCheck(cudaGetLastError(), "distance matrix kernel"); cudaCheck(cudaMemcpy(distances.data(), device_distances, matrix_size * sizeof(double), cudaMemcpyDeviceToHost), "copy distances"); cudaFree(device_points); cudaFree(device_distances);
    MPI_Barrier(MPI_COMM_WORLD); const auto start = std::chrono::high_resolution_clock::now();
    const auto clusters = qtClustering(points, threshold, distances, rank, ranks);
    double local_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count(), elapsed; MPI_Reduce(&local_seconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int status = 0;
    if (!rank) { const long ms = static_cast<long>(elapsed * 1000); printf("Clustering time: %ld ms\nClusters found: %zu\n", ms, clusters.size()); int total = 0, maximum = 0; for (const auto& c : clusters) { total += c.members.size(); maximum = std::max(maximum, static_cast<int>(c.members.size())); } printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n", total, n, 100.0 * total / n, clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size(), maximum); const double safe_seconds = std::max(elapsed, 1e-9); printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters.size() / safe_seconds, n / safe_seconds); if (print_results_flag) { std::vector<double> m(n, -1); for (size_t c = 0; c < clusters.size(); ++c) for (int p : clusters[c].members) m[p] = c; print_results(m, "ClusterMembership"); } if (validate && !validateClusters(clusters, points, threshold)) status = 1; if (validate && !status) printf("Validation: PASSED\n"); }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Finalize(); return status;
}
