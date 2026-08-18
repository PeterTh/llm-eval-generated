// QT Clustering Benchmark - MPI/OpenMP/CUDA implementation.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

// One GPU thread computes one entry of the symmetric distance matrix.  Keeping
// this matrix resident on the host after construction makes the highly
// irregular QT growth phase cheap and permits OpenMP seed parallelism.
__global__ void pairwiseDistances(const Point* points, double* distances, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        const double dx = points[row].x - points[col].x;
        const double dy = points[row].y - points[col].y;
        distances[static_cast<size_t>(row) * n + col] = sqrt(dx * dx + dy * dy);
    }
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    }
}

static void selectLocalGpu() {
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        std::fprintf(stderr, "CUDA error: no accelerator device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const char* localRankText = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!localRankText) localRankText = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (!localRankText) localRankText = std::getenv("SLURM_LOCALID");
    const int localRank = localRankText ? std::atoi(localRankText) : 0;
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
}

static void buildDistanceMatrix(const std::vector<Point>& points,
                                std::vector<double>& distances) {
    const int n = static_cast<int>(points.size());
    const size_t elements = static_cast<size_t>(n) * n;
    distances.resize(elements);
    Point* devicePoints = nullptr;
    double* deviceDistances = nullptr;
    cudaCheck(cudaMalloc(&devicePoints, sizeof(Point) * n), "cudaMalloc(points)");
    cudaCheck(cudaMalloc(&deviceDistances, sizeof(double) * elements),
              "cudaMalloc(distances)");
    cudaCheck(cudaMemcpy(devicePoints, points.data(), sizeof(Point) * n,
                         cudaMemcpyHostToDevice), "cudaMemcpy(points)");
    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    pairwiseDistances<<<grid, block>>>(devicePoints, deviceDistances, n);
    cudaCheck(cudaGetLastError(), "pairwiseDistances launch");
    cudaCheck(cudaDeviceSynchronize(), "pairwiseDistances synchronize");
    cudaCheck(cudaMemcpy(distances.data(), deviceDistances,
                         sizeof(double) * elements, cudaMemcpyDeviceToHost),
              "cudaMemcpy(distances)");
    cudaFree(deviceDistances);
    cudaFree(devicePoints);
}

static void generateSyntheticData(std::vector<Point>& points, int n,
                                  unsigned int seed = 42) {
    auto frand = [&seed]() { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double minDim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH;
        const double cy = frand() * MAX_HEIGHT;
        const double radius = frand() * minDim / 2.0;
        // The original expression is zero for N < 30 and never terminates.
        // A singleton group is its intended limiting behavior.
        int group = std::max(1, static_cast<int>(frand() * (n / 30.0)));
        group = std::min(group, n - count);
        while (group > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(std::max(0.0, r * r - dx * dx)) * sign;
            const double x = cx + dx;
            const double y = cy + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y};
            --group;
        }
    }
}

static inline double distance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static int findClosestPoint(const std::vector<int>& members,
                            const std::vector<unsigned char>& clustered,
                            const std::vector<unsigned char>& inCluster,
                            const std::vector<double>& distances,
                            double threshold, int n) {
    int closest = -1;
    double minimum = std::numeric_limits<double>::max();
    // Each candidate is independent.  The sequential reduction inside each
    // thread preserves the original member-order maximum exactly.
    #pragma omp parallel
    {
        int localPoint = -1;
        double localMinimum = std::numeric_limits<double>::max();
        #pragma omp for nowait schedule(static)
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || inCluster[candidate]) continue;
            double maxDistance = 0.0;
            for (int member : members)
                maxDistance = std::max(maxDistance,
                    distances[static_cast<size_t>(candidate) * n + member]);
            if (maxDistance < threshold && maxDistance < localMinimum) {
                localMinimum = maxDistance;
                localPoint = candidate;
            }
        }
        #pragma omp critical
        {
            // Strict comparison retains the lowest candidate on ties.
            if (localMinimum < minimum) {
                minimum = localMinimum;
                closest = localPoint;
            }
        }
    }
    return closest;
}

static std::vector<int> generateCandidateCluster(
    int seed, const std::vector<unsigned char>& clustered,
    const std::vector<double>& distances, double threshold, int n) {
    std::vector<unsigned char> inCluster(n, 0);
    std::vector<int> members;
    members.reserve(n);
    inCluster[seed] = 1;
    members.push_back(seed);
    while (static_cast<int>(members.size()) < n) {
        const int closest = findClosestPoint(members, clustered, inCluster,
                                             distances, threshold, n);
        if (closest < 0) break;
        inCluster[closest] = 1;
        members.push_back(closest);
    }
    return members;
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         double threshold, MPI_Comm communicator) {
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &worldSize);
    const int n = static_cast<int>(points.size());
    std::vector<double> distances;
    buildDistanceMatrix(points, distances);
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        int localCardinality = -1;
        int localSeed = -1;
        std::vector<int> localMembers;
        #pragma omp parallel
        {
            int threadCardinality = -1, threadSeed = -1;
            std::vector<int> threadMembers;
            #pragma omp for schedule(static)
            for (int position = rank; position < static_cast<int>(unclustered.size());
                 position += worldSize) {
                const int seed = unclustered[position];
                if (clustered[seed]) continue;
                std::vector<int> candidate = generateCandidateCluster(
                    seed, clustered, distances, threshold, n);
                if (static_cast<int>(candidate.size()) > threadCardinality ||
                    (static_cast<int>(candidate.size()) == threadCardinality &&
                     (threadSeed < 0 || seed < threadSeed))) {
                    threadCardinality = static_cast<int>(candidate.size());
                    threadSeed = seed;
                    threadMembers = std::move(candidate);
                }
            }
            #pragma omp critical
            {
                if (threadCardinality > localCardinality ||
                    (threadCardinality == localCardinality &&
                     (localSeed < 0 || threadSeed < localSeed))) {
                    localCardinality = threadCardinality;
                    localSeed = threadSeed;
                    localMembers = std::move(threadMembers);
                }
            }
        }
        int localPair[2] = {localCardinality, localSeed < 0 ? 0 : -localSeed};
        int globalPair[2] = {-1, 0};
        MPI_Allreduce(localPair, globalPair, 1, MPI_2INT, MPI_MAXLOC, communicator);
        const int bestCardinality = globalPair[0];
        const int bestSeed = globalPair[1] == 0 ? -1 : -globalPair[1];
        if (bestSeed < 0 || bestCardinality <= 0) break;

        int winner = (localSeed == bestSeed) ? rank : -1;
        MPI_Allreduce(MPI_IN_PLACE, &winner, 1, MPI_INT, MPI_MAX, communicator);
        int memberCount = winner == rank ? static_cast<int>(localMembers.size()) : 0;
        MPI_Bcast(&memberCount, 1, MPI_INT, winner, communicator);
        std::vector<int> bestMembers(memberCount);
        if (winner == rank) bestMembers = std::move(localMembers);
        MPI_Bcast(bestMembers.data(), memberCount, MPI_INT, winner, communicator);
        clusters.push_back({bestMembers, bestSeed});
        for (int member : bestMembers) clustered[member] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
            [&clustered](int index) { return clustered[index] != 0; }), unclustered.end());
    }
    return clusters;
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points, double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double diameter = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:diameter)
        for (int i = 0; i < static_cast<int>(cluster.members.size()); ++i)
            for (int j = i + 1; j < static_cast<int>(cluster.members.size()); ++j)
                diameter = std::max(diameter, distance(points[cluster.members[i]],
                                                        points[cluster.members[j]]));
        if (c < 10) std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                                c, cluster.members.size(), cluster.seed_point, diameter);
        if (diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        c, diameter, threshold);
            valid = false;
        }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (int member : clusters[c].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    int clustered = 0;
    for (int value : membership) clustered += value >= 0;
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered, points.size() - clustered);
    return valid;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>  Number of points (default: 1000)\n"
                "  -t <float> Distance threshold (default: 2.0)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", name);
}

int main(int argc, char** argv) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    selectLocalGpu();
    int n = 1000;
    double threshold = 2.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threshold = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || threshold <= 0.0) { if (rank == 0) std::printf("Error: Invalid parameters\n"); MPI_Finalize(); return 1; }
    std::vector<Point> points(n);
    generateSyntheticData(points, n);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double slowest = 0.0;
    MPI_Reduce(&seconds, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("QT Clustering Benchmark (MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled)\n", worldSize, omp_get_max_threads());
        std::printf("Number of points: %d\nDistance threshold: %.2f\nValidation: %s\n",
                    n, threshold, validate ? "enabled" : "disabled");
        std::printf("Clustering time: %.0f ms\nClusters found: %zu\n", slowest * 1000.0, clusters.size());
        int total = 0, maximum = 0;
        for (const auto& cluster : clusters) { total += static_cast<int>(cluster.members.size()); maximum = std::max(maximum, static_cast<int>(cluster.members.size())); }
        const double average = clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n",
                    total, n, 100.0 * total / n, average, maximum);
        const double rate = slowest > 0.0 ? 1.0 / slowest : 0.0;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters.size() * rate, n * rate);
        if (printResults) {
            std::vector<double> membershipData(n, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (int member : clusters[c].members) membershipData[member] = static_cast<double>(c);
            print_results(membershipData, "ClusterMembership");
        }
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) std::printf("Validation: PASSED\n"); else std::printf("Validation: FAILED\n");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
