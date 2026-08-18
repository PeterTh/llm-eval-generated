// QT clustering benchmark: MPI ranks, OpenMP seed evaluation, CUDA distances.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void squaredDistanceKernel(const Point* points, double* distances, int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n) {
        const double dx = points[row].x - points[column].x;
        const double dy = points[row].y - points[column].y;
        distances[static_cast<size_t>(row) * n + column] = dx * dx + dy * dy;
    }
}

static void generateSyntheticData(std::vector<Point>& points, const int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH;
        const double cy = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (n / 30.0));
        if (group_count > n - count) group_count = n - count;
        while (group_count > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double radius_at_point = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * radius_at_point;
            const double dy = std::sqrt(radius_at_point * radius_at_point - dx * dx) * sign;
            const double x = cx + dx, y = cy + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y};
            --group_count;
        }
    }
}

// The maintained maximum squared distance is exactly the candidate diameter
// used by the sequential implementation, but avoids rescanning all members.
static int generateCandidateCluster(int seed, const std::vector<unsigned char>& clustered,
                                    const std::vector<double>& distances, double threshold_squared,
                                    std::vector<int>& members, std::vector<double>& maximums) {
    const int n = static_cast<int>(clustered.size());
    members.clear();
    members.push_back(seed);
    // With just the seed present, every candidate's diameter is its distance
    // to that seed.  Later additions update this running maximum.
    std::copy_n(distances.data() + static_cast<size_t>(seed) * n, n, maximums.begin());
    maximums[seed] = -1.0;

    for (;;) {
        int closest = -1;
        double minimum_diameter = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || maximums[candidate] < 0.0) continue;
            const double diameter = maximums[candidate];
            // Strict tests and increasing scan order retain original tie semantics.
            if (diameter < threshold_squared && diameter < minimum_diameter) {
                minimum_diameter = diameter;
                closest = candidate;
            }
        }
        if (closest < 0) break;
        maximums[closest] = -1.0;
        members.push_back(closest);
        const double* row = distances.data() + static_cast<size_t>(closest) * n;
        #pragma omp simd
        for (int candidate = 0; candidate < n; ++candidate) {
            if (maximums[candidate] >= 0.0 && row[candidate] > maximums[candidate])
                maximums[candidate] = row[candidate];
        }
    }
    return static_cast<int>(members.size());
}

static std::vector<Cluster> qtClustering(const std::vector<double>& distances, int n,
                                          double threshold, int rank, int ranks) {
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    const double threshold_squared = threshold * threshold;

    while (!unclustered.empty()) {
        int local_size = -1;
        int local_seed = std::numeric_limits<int>::max();
        std::vector<int> local_members;

        #pragma omp parallel
        {
            std::vector<int> members;
            std::vector<double> maximums(n);
            int thread_size = -1;
            int thread_seed = std::numeric_limits<int>::max();
            std::vector<int> thread_members;
            #pragma omp for schedule(dynamic, 1) nowait
            for (int index = rank; index < static_cast<int>(unclustered.size()); index += ranks) {
                const int seed = unclustered[index];
                const int size = generateCandidateCluster(seed, clustered, distances,
                                                          threshold_squared, members, maximums);
                if (size > thread_size || (size == thread_size && seed < thread_seed)) {
                    thread_size = size;
                    thread_seed = seed;
                    thread_members = members;
                }
            }
            #pragma omp critical
            {
                if (thread_size > local_size || (thread_size == local_size && thread_seed < local_seed)) {
                    local_size = thread_size;
                    local_seed = thread_seed;
                    local_members.swap(thread_members);
                }
            }
        }

        struct Choice { int size; int seed; int rank; } local{local_size, local_seed, rank};
        std::vector<Choice> choices(ranks);
        MPI_Allgather(&local, sizeof(Choice), MPI_BYTE, choices.data(), sizeof(Choice), MPI_BYTE, MPI_COMM_WORLD);
        Choice winner{-1, std::numeric_limits<int>::max(), -1};
        for (const Choice& choice : choices)
            if (choice.size > winner.size || (choice.size == winner.size && choice.seed < winner.seed)) winner = choice;
        if (winner.size <= 0) break;

        std::vector<int> winning_members(static_cast<size_t>(winner.size));
        if (rank == winner.rank) winning_members = local_members;
        MPI_Bcast(winning_members.data(), winner.size, MPI_INT, winner.rank, MPI_COMM_WORLD);
        clusters.push_back({std::move(winning_members), winner.seed});
        for (int member : clusters.back().members) clustered[member] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                          [&clustered](int point) { return clustered[point] != 0; }), unclustered.end());
    }
    return clusters;
}

static bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true;
    printf("Validating clusters:\n");
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        double diameter = 0.0;
        for (size_t i = 0; i < clusters[c].members.size(); ++i) for (size_t j = i + 1; j < clusters[c].members.size(); ++j) {
            const Point& a = points[clusters[c].members[i]]; const Point& b = points[clusters[c].members[j]];
            diameter = std::max(diameter, std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y)));
        }
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, clusters[c].members.size(), clusters[c].seed_point, diameter);
        if (diameter > threshold * 1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, diameter, threshold); valid = false; }
        for (int member : clusters[c].members) {
            if (membership[member] >= 0) { printf("ERROR: Point %d appears in multiple clusters\n", member); valid = false; }
            membership[member] = static_cast<int>(c);
        }
    }
    int clustered_count = 0; for (int item : membership) clustered_count += item >= 0;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count, points.size() - clustered_count);
    return valid;
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n  -n <num>     Number of points (default: 1000)\n  -t <float>   Distance threshold (default: 2.0)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", program);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int num_points = 1000; double threshold = 2.0; bool validate = false, print_results_requested = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) num_points = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (num_points <= 0 || threshold <= 0.0) { if (!rank) printf("Error: Invalid parameters\n"); MPI_Finalize(); return 1; }
    if (static_cast<size_t>(num_points) > std::numeric_limits<size_t>::max() / static_cast<size_t>(num_points) / sizeof(double)) {
        if (!rank) fprintf(stderr, "Error: distance matrix is too large\n"); MPI_Finalize(); return 1;
    }
    int device_count = 0; checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (!device_count) { if (!rank) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    checkCuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    std::vector<Point> points(num_points);
    if (!rank) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (!rank) { printf("QT Clustering Benchmark (MPI ranks: %d, OpenMP threads/rank: %d)\n", ranks, omp_get_max_threads()); printf("Number of points: %d\nDistance threshold: %.2f\nValidation: %s\n", num_points, threshold, validate ? "enabled" : "disabled"); }

    // Include accelerator preprocessing in the benchmark interval and align ranks
    // before their independently assigned GPUs begin work.
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    Point* device_points = nullptr; double* device_distances = nullptr;
    const size_t matrix_entries = static_cast<size_t>(num_points) * num_points;
    checkCuda(cudaMalloc(&device_points, static_cast<size_t>(num_points) * sizeof(Point)), "cudaMalloc points");
    checkCuda(cudaMalloc(&device_distances, matrix_entries * sizeof(double)), "cudaMalloc distances");
    checkCuda(cudaMemcpy(device_points, points.data(), static_cast<size_t>(num_points) * sizeof(Point), cudaMemcpyHostToDevice), "cudaMemcpy points");
    dim3 block(32, 8), grid((num_points + block.x - 1) / block.x, (num_points + block.y - 1) / block.y);
    squaredDistanceKernel<<<grid, block>>>(device_points, device_distances, num_points);
    checkCuda(cudaGetLastError(), "distance kernel launch");
    std::vector<double> distances(matrix_entries);
    checkCuda(cudaMemcpy(distances.data(), device_distances, matrix_entries * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy distances");
    checkCuda(cudaFree(device_points), "cudaFree points"); checkCuda(cudaFree(device_distances), "cudaFree distances");

    const std::vector<Cluster> clusters = qtClustering(distances, num_points, threshold, rank, ranks);
    const auto end = std::chrono::high_resolution_clock::now();
    if (!rank) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        int total = 0, maximum = 0; for (const auto& cluster : clusters) { total += cluster.members.size(); maximum = std::max(maximum, static_cast<int>(cluster.members.size())); }
        printf("Clustering time: %ld ms\nClusters found: %zu\n", elapsed, clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n", total, num_points, 100.0 * total / num_points, clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size(), maximum);
        const double seconds = elapsed / 1000.0; printf("Performance: %.1f clusters/s, %.1f points/s\n", seconds ? clusters.size()/seconds : 0.0, seconds ? num_points/seconds : 0.0);
        if (print_results_requested) { std::vector<double> membership; membership.reserve(num_points); std::vector<int> labels(num_points, -1); for (size_t c = 0; c < clusters.size(); ++c) for (int point : clusters[c].members) labels[point] = c; for (int label : labels) membership.push_back(label); print_results(membership, "ClusterMembership"); }
        if (validate && !validateClusters(clusters, points, threshold)) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Finalize();
    return 0;
}
