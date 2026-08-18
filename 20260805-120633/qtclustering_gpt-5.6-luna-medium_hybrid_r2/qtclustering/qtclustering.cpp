// QT clustering benchmark: MPI distributes seeds between ranks, OpenMP
// evaluates local seeds, and CUDA builds the distance matrix once per rank.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

void generateSyntheticData(std::vector<Point>& points, int n, unsigned int seed = 42) {
    auto frand = [&seed]() { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH;
        const double cy = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        if (group > n - count) group = n - count;
        while (group > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cx + dx, y = cy + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y};
            --group;
        }
    }
}

inline double distance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

__global__ void buildDistanceMatrix(const Point* points, double* distances, int n) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    const double dx = points[i].x - points[j].x;
    const double dy = points[i].y - points[j].y;
    distances[static_cast<size_t>(i) * n + j] = sqrt(dx * dx + dy * dy);
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// The matrix is read-only after construction, so all OpenMP workers can use it.
static std::vector<double> makeDistanceMatrix(const std::vector<Point>& points, int rank) {
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) throw std::runtime_error("No CUDA accelerator is available");
    int local_rank = 0;
    MPI_Comm local = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    cudaCheck(cudaSetDevice(local_rank % device_count), "cudaSetDevice");

    const int n = static_cast<int>(points.size());
    const size_t bytes_points = points.size() * sizeof(Point);
    const size_t matrix_elements = points.size() * points.size();
    if (matrix_elements > std::numeric_limits<size_t>::max() / sizeof(double))
        throw std::runtime_error("Distance matrix is too large");
    const size_t bytes_matrix = matrix_elements * sizeof(double);
    std::vector<double> matrix(matrix_elements);
    Point* device_points = nullptr;
    double* device_matrix = nullptr;
    cudaCheck(cudaMalloc(&device_points, bytes_points), "cudaMalloc(points)");
    cudaCheck(cudaMalloc(&device_matrix, bytes_matrix), "cudaMalloc(distance matrix)");
    try {
        cudaCheck(cudaMemcpy(device_points, points.data(), bytes_points, cudaMemcpyHostToDevice),
                  "cudaMemcpy(points)");
        const dim3 block(16, 16);
        const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        buildDistanceMatrix<<<grid, block>>>(device_points, device_matrix, n);
        cudaCheck(cudaGetLastError(), "buildDistanceMatrix launch");
        cudaCheck(cudaDeviceSynchronize(), "buildDistanceMatrix synchronize");
        cudaCheck(cudaMemcpy(matrix.data(), device_matrix, bytes_matrix, cudaMemcpyDeviceToHost),
                  "cudaMemcpy(distance matrix)");
    } catch (...) {
        cudaFree(device_matrix);
        cudaFree(device_points);
        throw;
    }
    cudaCheck(cudaFree(device_matrix), "cudaFree(distance matrix)");
    cudaCheck(cudaFree(device_points), "cudaFree(points)");
    (void)rank;
    return matrix;
}

int generateCandidateCluster(int seed, const std::vector<uint8_t>& clustered,
                             const std::vector<double>& distances, double threshold,
                             int n, std::vector<int>* result) {
    std::vector<uint8_t> in_cluster(n, 0);
    std::vector<int> members;
    members.reserve(n);
    in_cluster[seed] = 1;
    members.push_back(seed);
    while (static_cast<int>(members.size()) < n) {
        int closest = -1;
        double min_distance = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double maximum = 0.0;
            for (int member : members)
                maximum = std::max(maximum, distances[static_cast<size_t>(candidate) * n + member]);
            // Strict comparisons intentionally retain the original tie-breaking behavior.
            if (maximum < threshold && maximum < min_distance) {
                min_distance = maximum;
                closest = candidate;
            }
        }
        if (closest < 0) break;
        in_cluster[closest] = 1;
        members.push_back(closest);
    }
    if (result) *result = std::move(members);
    return result ? static_cast<int>(result->size()) : static_cast<int>(members.size());
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  const std::vector<double>& distances, int rank, int size) {
    const int n = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(n, 0);
    std::vector<int> remaining(n);
    for (int i = 0; i < n; ++i) remaining[i] = i;
    std::vector<Cluster> clusters;

    while (!remaining.empty()) {
        int local_cardinality = -1, local_seed = std::numeric_limits<int>::max();
        std::vector<int> local_members;
        #pragma omp parallel
        {
            int thread_cardinality = -1, thread_seed = std::numeric_limits<int>::max();
            std::vector<int> thread_members;
            #pragma omp for schedule(dynamic, 1) nowait
            for (int position = 0; position < static_cast<int>(remaining.size()); ++position) {
                const int seed = remaining[position];
                std::vector<int> candidate;
                const int cardinality = generateCandidateCluster(seed, clustered, distances,
                                                                  threshold, n, &candidate);
                if (cardinality > thread_cardinality ||
                    (cardinality == thread_cardinality && seed < thread_seed)) {
                    thread_cardinality = cardinality;
                    thread_seed = seed;
                    thread_members = std::move(candidate);
                }
            }
            #pragma omp critical
            {
                if (thread_cardinality > local_cardinality ||
                    (thread_cardinality == local_cardinality && thread_seed < local_seed)) {
                    local_cardinality = thread_cardinality;
                    local_seed = thread_seed;
                    local_members = std::move(thread_members);
                }
            }
        }

        // Each rank reports its best local seed. Rank zero selects the global
        // maximum, with the original lowest-seed tie break, then broadcasts it.
        std::vector<int> reports(static_cast<size_t>(size) * 2);
        const int report[2] = {local_cardinality, local_seed};
        MPI_Allgather(report, 2, MPI_INT, reports.data(), 2, MPI_INT, MPI_COMM_WORLD);
        int winner_rank = 0, best_cardinality = reports[0], best_seed = reports[1];
        for (int r = 1; r < size; ++r) {
            const int card = reports[2 * r], seed = reports[2 * r + 1];
            if (card > best_cardinality || (card == best_cardinality && seed < best_seed)) {
                winner_rank = r;
                best_cardinality = card;
                best_seed = seed;
            }
        }
        MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&best_cardinality, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&best_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int member_count = winner_rank == rank ? static_cast<int>(local_members.size()) : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, winner_rank, MPI_COMM_WORLD);
        std::vector<int> best_members(member_count);
        if (winner_rank == rank) best_members = std::move(local_members);
        if (member_count) MPI_Bcast(best_members.data(), member_count, MPI_INT, winner_rank, MPI_COMM_WORLD);

        if (best_seed >= 0 && best_cardinality > 0) {
            clusters.push_back({std::move(best_members), best_seed});
            for (int member : clusters.back().members) clustered[member] = 1;
            remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                              [&clustered](int index) { return clustered[index] != 0; }), remaining.end());
        } else break;
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i)
            for (size_t j = i + 1; j < cluster.members.size(); ++j)
                diameter = std::max(diameter, distance(points[cluster.members[i]], points[cluster.members[j]]));
        if (c < 10) std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, diameter);
        if (diameter > threshold * 1.001) { std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, diameter, threshold); valid = false; }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (int member : clusters[c].members) {
            if (membership[member] >= 0) { std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c); valid = false; }
            membership[member] = static_cast<int>(c);
        }
    int clustered_count = 0;
    for (int member : membership) if (member >= 0) ++clustered_count;
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count, points.size() - clustered_count);
    return valid;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of points (default: 1000)\n  -t <float> Distance threshold (default: 2.0)\n  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 1000; double threshold = 2.0; bool validate = false, emit_results = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) threshold = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) emit_results = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || threshold <= 0.0) { if (rank == 0) std::printf("Error: Invalid parameters\n"); MPI_Finalize(); return 1; }
    std::vector<Point> points(n);
    if (rank == 0) generateSyntheticData(points, n);
    MPI_Bcast(points.data(), n * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    try {
        const std::vector<double> distances = makeDistanceMatrix(points, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        const std::vector<Cluster> clusters = qtClustering(points, threshold, distances, rank, size);
        const double local_time = MPI_Wtime() - start, cluster_time = [&] { double t = 0; MPI_Reduce(&local_time, &t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); return t; }();
        if (rank == 0) {
            std::printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\n", n, threshold, validate ? "enabled" : "disabled");
            std::printf("Clustering time: %.3f ms\nClusters found: %zu\n", cluster_time * 1000.0, clusters.size());
            int total = 0, maximum = 0;
            for (const auto& cluster : clusters) { total += static_cast<int>(cluster.members.size()); maximum = std::max(maximum, static_cast<int>(cluster.members.size())); }
            const double average = clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size();
            std::printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n", total, n, 100.0 * total / n, average, maximum);
            const double seconds = cluster_time;
            std::printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters.size() / seconds, n / seconds);
            if (emit_results) {
                std::vector<double> membership_data(n); std::vector<int> membership(n, -1);
                for (size_t c = 0; c < clusters.size(); ++c) for (int member : clusters[c].members) membership[member] = static_cast<int>(c);
                for (int i = 0; i < n; ++i) membership_data[i] = static_cast<double>(membership[i]);
                print_results(membership_data, "ClusterMembership");
            }
            if (validate) { const bool valid = validateClusters(clusters, points, threshold); MPI_Finalize(); return valid ? 0 : 1; }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
