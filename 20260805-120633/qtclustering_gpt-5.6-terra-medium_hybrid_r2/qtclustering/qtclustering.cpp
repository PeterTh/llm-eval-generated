// QT clustering benchmark -- distributed GPU implementation.
// MPI partitions candidate seeds, CUDA evaluates candidate distances, and
// OpenMP reduces the CUDA results without changing greedy tie-breaking.

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

static void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void generateSyntheticData(std::vector<Point>& points, const int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double radius = frand() * std::min(MAX_WIDTH, MAX_HEIGHT) / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        // The original generator has no terminating group for n < 30.
        // Preserve its stream for normal inputs while making this edge case usable.
        if (n < 30) group = std::max(1, group);
        group = std::min(group, n - count);
        while (group > 0) {
            const double sign = frand() < .5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double x = cx + dx, y = cy + std::sqrt(r * r - dx * dx) * sign;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y}; --group;
        }
    }
}

inline double distance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// One thread evaluates one possible next member.  The host reduction retains
// the original lowest-index tie rule exactly.
__global__ void candidateDiameters(const Point* points, const unsigned char* clustered,
                                   const unsigned char* inCluster, const int* members,
                                   int memberCount, int n, double* diameters) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= n) return;
    if (clustered[candidate] || inCluster[candidate]) {
        diameters[candidate] = HUGE_VAL;
        return;
    }
    double maximum = 0.0;
    for (int i = 0; i < memberCount; ++i) {
        const Point a = points[candidate], b = points[members[i]];
        const double dx = a.x - b.x, dy = a.y - b.y;
        maximum = fmax(maximum, sqrt(dx * dx + dy * dy));
    }
    diameters[candidate] = maximum;
}

class GpuEvaluator {
public:
    explicit GpuEvaluator(const std::vector<Point>& points) : n_(static_cast<int>(points.size())), hostDiameters_(n_) {
        cudaCheck(cudaMalloc(&d_points_, n_ * sizeof(Point)), "cudaMalloc(points)");
        cudaCheck(cudaMalloc(&d_clustered_, n_ * sizeof(unsigned char)), "cudaMalloc(clustered)");
        cudaCheck(cudaMalloc(&d_inCluster_, n_ * sizeof(unsigned char)), "cudaMalloc(inCluster)");
        cudaCheck(cudaMalloc(&d_members_, n_ * sizeof(int)), "cudaMalloc(members)");
        cudaCheck(cudaMalloc(&d_diameters_, n_ * sizeof(double)), "cudaMalloc(diameters)");
        cudaCheck(cudaMemcpy(d_points_, points.data(), n_ * sizeof(Point), cudaMemcpyHostToDevice), "copy(points)");
    }
    ~GpuEvaluator() {
        cudaFree(d_diameters_); cudaFree(d_members_); cudaFree(d_inCluster_);
        cudaFree(d_clustered_); cudaFree(d_points_);
    }

    std::vector<int> makeCluster(int seed, const std::vector<unsigned char>& clustered, double threshold) {
        cudaCheck(cudaMemcpy(d_clustered_, clustered.data(), n_ * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy(clustered)");
        cudaCheck(cudaMemset(d_inCluster_, 0, n_ * sizeof(unsigned char)), "clear(inCluster)");
        const unsigned char one = 1;
        cudaCheck(cudaMemcpy(d_inCluster_ + seed, &one, sizeof(one), cudaMemcpyHostToDevice), "set(seed)");
        std::vector<int> members(1, seed);
        while (static_cast<int>(members.size()) < n_) {
            cudaCheck(cudaMemcpy(d_members_, members.data(), members.size() * sizeof(int), cudaMemcpyHostToDevice), "copy(members)");
            constexpr int threads = 256;
            candidateDiameters<<<(n_ + threads - 1) / threads, threads>>>(d_points_, d_clustered_, d_inCluster_, d_members_, static_cast<int>(members.size()), n_, d_diameters_);
            cudaCheck(cudaGetLastError(), "candidateDiameters launch");
            cudaCheck(cudaMemcpy(hostDiameters_.data(), d_diameters_, n_ * sizeof(double), cudaMemcpyDeviceToHost), "copy(diameters)");

            int best = -1;
            double bestDiameter = std::numeric_limits<double>::max();
            // Static scheduling makes each thread's scan reproducible; the final
            // sequential merge is what enforces the original index tie break.
            #pragma omp parallel
            {
                int localBest = -1;
                double localDiameter = std::numeric_limits<double>::max();
                #pragma omp for nowait schedule(static)
                for (int i = 0; i < n_; ++i) {
                    const double d = hostDiameters_[i];
                    if (d < threshold && (d < localDiameter || (d == localDiameter && (localBest < 0 || i < localBest)))) {
                        localDiameter = d; localBest = i;
                    }
                }
                #pragma omp critical
                if (localBest >= 0 && (localDiameter < bestDiameter ||
                    (localDiameter == bestDiameter && (best < 0 || localBest < best)))) {
                    bestDiameter = localDiameter; best = localBest;
                }
            }
            if (best < 0) break;
            cudaCheck(cudaMemcpy(d_inCluster_ + best, &one, sizeof(one), cudaMemcpyHostToDevice), "set(member)");
            members.push_back(best);
        }
        return members;
    }
private:
    int n_;
    Point *d_points_ = nullptr;
    unsigned char *d_clustered_ = nullptr, *d_inCluster_ = nullptr;
    int* d_members_ = nullptr;
    double* d_diameters_ = nullptr;
    std::vector<double> hostDiameters_;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    GpuEvaluator gpu(points);

    while (!unclustered.empty()) {
        int localSize = -1, localSeed = std::numeric_limits<int>::max();
        std::vector<int> localMembers;
        // A rank owns a contiguous deterministic portion of the seed list.
        const size_t begin = unclustered.size() * rank / ranks;
        const size_t end = unclustered.size() * (rank + 1) / ranks;
        for (size_t p = begin; p < end; ++p) {
            const int seed = unclustered[p];
            std::vector<int> members = gpu.makeCluster(seed, clustered, threshold);
            if (static_cast<int>(members.size()) > localSize ||
                (static_cast<int>(members.size()) == localSize && seed < localSeed)) {
                localSize = static_cast<int>(members.size()); localSeed = seed; localMembers.swap(members);
            }
        }
        int pair[2] = {localSize, localSeed}, winner[2];
        MPI_Allreduce(pair, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int winnerSeed = winner[1];
        // Seed indices are not positions after the first round; locate the
        // owner from its current position in the contiguous rank partition.
        int owner = 0;
        for (size_t p = 0; p < unclustered.size(); ++p) {
            if (unclustered[p] != winnerSeed) continue;
            // This is the inverse of [size*r/ranks, size*(r+1)/ranks),
            // including the non-divisible boundary cases.
            while (owner + 1 < ranks && p >= unclustered.size() * static_cast<size_t>(owner + 1) / ranks) ++owner;
            break;
        }
        int count = rank == owner ? static_cast<int>(localMembers.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        std::vector<int> members(count);
        if (rank == owner) members = localMembers;
        MPI_Bcast(members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back({members, winnerSeed});
        for (int member : members) clustered[member] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(), [&clustered](int i) { return clustered[i]; }), unclustered.end());
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true; std::vector<int> membership(points.size(), -1);
    printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        double diameter = 0;
        for (size_t i = 0; i < clusters[c].members.size(); ++i) for (size_t j = i + 1; j < clusters[c].members.size(); ++j)
            diameter = std::max(diameter, distance(points[clusters[c].members[i]], points[clusters[c].members[j]]));
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, clusters[c].members.size(), clusters[c].seed_point, diameter);
        if (diameter > threshold * 1.001) valid = false;
        for (int m : clusters[c].members) { if (membership[m] >= 0) valid = false; membership[m] = static_cast<int>(c); }
    }
    int count = 0; for (int m : membership) count += m >= 0;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), count, points.size() - count);
    return valid;
}

void printUsage(const char* prog) { printf("Usage: %s [-n <num>] [-t <float>] [-v] [-r] [-h]\n", prog); }

int main(int argc, char** argv) {
    int mpiProvided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiProvided);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) {
        if (!rank) fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    cudaCheck(cudaSetDevice(rank % devices), "cudaSetDevice");
    int n = 1000; double threshold = 2.0; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n <= 0 || threshold <= 0) { if (!rank) printf("Error: Invalid parameters\n"); MPI_Finalize(); return 1; }
    std::vector<Point> points(n);
    if (!rank) generateSyntheticData(points, n);
    MPI_Bcast(points.data(), n * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (!rank) { printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\n", n, threshold, validate ? "enabled" : "disabled"); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    long ms = 0;
    MPI_Reduce(&local_ms, &ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        int total = 0, maximum = 0; for (const auto& c : clusters) { total += c.members.size(); maximum = std::max(maximum, static_cast<int>(c.members.size())); }
        printf("Clustering time: %ld ms\nClusters found: %zu\nPoints clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n", ms, clusters.size(), total, n, 100.0 * total / n, clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size(), maximum);
        const double seconds = std::max(0.001, ms / 1000.0); printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters.size() / seconds, n / seconds);
        if (printResults) { std::vector<double> result(n, -1); for (size_t c = 0; c < clusters.size(); ++c) for (int m : clusters[c].members) result[m] = c; print_results(result, "ClusterMembership"); }
        if (validate && !validateClusters(clusters, points, threshold)) { MPI_Finalize(); return 1; }
        if (validate) printf("Validation: PASSED\n");
    }
    MPI_Finalize(); return 0;
}
