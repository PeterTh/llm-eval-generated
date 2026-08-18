// QT clustering benchmark: MPI distributes seeds, OpenMP evaluates local
// candidates, and CUDA materializes the pairwise-distance lookup table.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void checkCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

// Every rank has the same input.  Its local GPU constructs the matrix, avoiding
// a large inter-node broadcast and making the CUDA stage scale with the ranks.
__global__ void distanceMatrixKernel(const Point* points, double* distances, int n) {
    const unsigned long long idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(n) * n;
    if (idx >= count) return;
    const int i = static_cast<int>(idx / n);
    const int j = static_cast<int>(idx - static_cast<unsigned long long>(i) * n);
    const double dx = points[i].x - points[j].x;
    const double dy = points[i].y - points[j].y;
    distances[idx] = sqrt(dx * dx + dy * dy);
}

static std::vector<double> buildDistanceMatrix(const std::vector<Point>& points) {
    const int n = static_cast<int>(points.size());
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(double);
    Point* devicePoints = nullptr;
    double* deviceDistances = nullptr;
    checkCuda(cudaMalloc(&devicePoints, static_cast<size_t>(n) * sizeof(Point)), "allocating CUDA points");
    try {
        checkCuda(cudaMalloc(&deviceDistances, bytes), "allocating CUDA distance matrix");
        checkCuda(cudaMemcpy(devicePoints, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice), "copying points to CUDA");
        const unsigned long long count = static_cast<unsigned long long>(n) * n;
        const int blockSize = 256;
        const unsigned long long grid = (count + blockSize - 1) / blockSize;
        if (grid > std::numeric_limits<unsigned int>::max()) throw std::runtime_error("point count exceeds CUDA launch limit");
        distanceMatrixKernel<<<static_cast<unsigned int>(grid), blockSize>>>(devicePoints, deviceDistances, n);
        checkCuda(cudaGetLastError(), "launching CUDA distance kernel");
        checkCuda(cudaDeviceSynchronize(), "synchronizing CUDA distance kernel");
        std::vector<double> distances(static_cast<size_t>(n) * n);
        checkCuda(cudaMemcpy(distances.data(), deviceDistances, bytes, cudaMemcpyDeviceToHost), "copying CUDA distance matrix");
        cudaFree(deviceDistances);
        cudaFree(devicePoints);
        return distances;
    } catch (...) {
        if (deviceDistances) cudaFree(deviceDistances);
        if (devicePoints) cudaFree(devicePoints);
        throw;
    }
}

void generateSyntheticData(std::vector<Point>& points, const int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double rmax = frand() * std::min(MAX_WIDTH, MAX_HEIGHT) / 2.0;
        // Retain the original generator sequence for normal benchmark sizes.
        // The small-N guard only prevents its zero-sized-group non-termination.
        int group = static_cast<int>(frand() * (n / 30.0));
        if (n < 30) group = std::max(1, group);
        group = std::min(group, n - count);
        while (group > 0) {
            const double r = frand() * rmax;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * (frand() < .5 ? -1.0 : 1.0);
            const double x = cx + dx, y = cy + dy;
            if (x >= 0 && x <= MAX_WIDTH && y >= 0 && y <= MAX_HEIGHT) {
                points[count++] = {x, y};
                --group;
            }
        }
    }
}

inline double distance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static int generateCandidateCluster(int seed, const std::vector<unsigned char>& clustered,
                                    const std::vector<double>& distances, double threshold, int n,
                                    std::vector<int>* output = nullptr) {
    std::vector<unsigned char> inCluster(n, 0);
    std::vector<int> members;
    members.reserve(n);
    inCluster[seed] = 1;
    members.push_back(seed);
    while (static_cast<int>(members.size()) < n) {
        int closest = -1;
        double minimumDiameter = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || inCluster[candidate]) continue;
            double maximumDistance = 0.0;
            const size_t row = static_cast<size_t>(candidate) * n;
            for (int member : members) maximumDistance = std::max(maximumDistance, distances[row + member]);
            // Strict comparisons and ascending candidate traversal retain the original tie behavior.
            if (maximumDistance < threshold && maximumDistance < minimumDiameter) {
                minimumDiameter = maximumDistance;
                closest = candidate;
            }
        }
        if (closest < 0) break;
        inCluster[closest] = 1;
        members.push_back(closest);
    }
    if (output) *output = std::move(members);
    return output ? static_cast<int>(output->size()) : static_cast<int>(members.size());
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                          const std::vector<double>& distances, MPI_Comm comm) {
    int rank = 0, ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        int localSize = -1, localSeed = n;
        #pragma omp parallel
        {
            int threadSize = -1, threadSeed = n;
            #pragma omp for schedule(dynamic, 1) nowait
            for (int pos = 0; pos < static_cast<int>(unclustered.size()); ++pos) {
                const int seed = unclustered[pos];
                if (pos % ranks != rank || clustered[seed]) continue;
                const int size = generateCandidateCluster(seed, clustered, distances, threshold, n);
                if (size > threadSize || (size == threadSize && seed < threadSeed)) {
                    threadSize = size;
                    threadSeed = seed;
                }
            }
            #pragma omp critical
            if (threadSize > localSize || (threadSize == localSize && threadSeed < localSeed)) {
                localSize = threadSize;
                localSeed = threadSeed;
            }
        }

        std::vector<int> choices(static_cast<size_t>(ranks) * 2);
        const int choice[2] = {localSize, localSeed};
        MPI_Allgather(choice, 2, MPI_INT, choices.data(), 2, MPI_INT, comm);
        int bestRank = 0, bestSize = -1, bestSeed = n;
        for (int r = 0; r < ranks; ++r) {
            const int size = choices[2 * r], seed = choices[2 * r + 1];
            if (size > bestSize || (size == bestSize && seed < bestSeed)) {
                bestSize = size; bestSeed = seed; bestRank = r;
            }
        }
        if (bestSize <= 0) break;
        std::vector<int> members;
        if (rank == bestRank) generateCandidateCluster(bestSeed, clustered, distances, threshold, n, &members);
        int memberCount = bestSize;
        MPI_Bcast(&memberCount, 1, MPI_INT, bestRank, comm);
        if (rank != bestRank) members.resize(memberCount);
        MPI_Bcast(members.data(), memberCount, MPI_INT, bestRank, comm);
        clusters.push_back({std::move(members), bestSeed});
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
        for (size_t i = 0; i < clusters[c].members.size(); ++i) for (size_t j = i + 1; j < clusters[c].members.size(); ++j)
            diameter = std::max(diameter, distance(points[clusters[c].members[i]], points[clusters[c].members[j]]));
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, clusters[c].members.size(), clusters[c].seed_point, diameter);
        if (diameter > threshold * 1.001) valid = false;
        for (int member : clusters[c].members) { if (membership[member] >= 0) valid = false; membership[member] = static_cast<int>(c); }
    }
    printf("Total points: %zu, Clustered: %zu, Unclustered: %zu\n", points.size(),
           std::count_if(membership.begin(), membership.end(), [](int c) { return c >= 0; }),
           std::count(membership.begin(), membership.end(), -1));
    return valid;
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n  -n <num>     Number of points (default: 1000)\n  -t <float>   Distance threshold (default: 2.0)\n  -v           Enable validation\n  -r           Print results\n  -h           Show help\n", program);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        int deviceCount = 0;
        checkCuda(cudaGetDeviceCount(&deviceCount), "discovering CUDA devices");
        if (deviceCount == 0) throw std::runtime_error("no CUDA device is available");
        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);
        checkCuda(cudaSetDevice(localRank % deviceCount), "selecting rank-local CUDA device");
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int n = 1000; double threshold = 2.0; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || threshold <= 0.0) { if (!rank) printf("Error: invalid parameters\n"); MPI_Finalize(); return 1; }
    try {
        std::vector<Point> points(n);
        generateSyntheticData(points, n);
        MPI_Barrier(MPI_COMM_WORLD);
        const auto start = std::chrono::high_resolution_clock::now();
        const std::vector<double> distances = buildDistanceMatrix(points);
        const std::vector<Cluster> clusters = qtClustering(points, threshold, distances, MPI_COMM_WORLD);
        const auto end = std::chrono::high_resolution_clock::now();
        if (!rank) {
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            int total = 0, maximum = 0;
            for (const auto& cluster : clusters) { total += static_cast<int>(cluster.members.size()); maximum = std::max(maximum, static_cast<int>(cluster.members.size())); }
            printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\n", n, threshold, validate ? "enabled" : "disabled");
            printf("Clustering time: %ld ms\nClusters found: %zu\nPoints clustered: %d / %d (%.1f%%)\n", ms, clusters.size(), total, n, 100.0 * total / n);
            printf("Average cluster size: %.2f\nMaximum cluster size: %d\n", clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size(), maximum);
            const double seconds = std::max(0.001, ms / 1000.0);
            printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters.size() / seconds, n / seconds);
            if (printResults) { std::vector<double> data(n, -1); for (size_t c = 0; c < clusters.size(); ++c) for (int p : clusters[c].members) data[p] = c; print_results(data, "ClusterMembership"); }
            if (validate && !validateClusters(clusters, points, threshold)) { MPI_Finalize(); return 1; }
            if (validate) printf("Validation: PASSED\n");
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
