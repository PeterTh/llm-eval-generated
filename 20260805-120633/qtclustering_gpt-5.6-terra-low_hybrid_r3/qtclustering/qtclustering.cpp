// QT clustering benchmark: MPI ranks evaluate seeds, CUDA evaluates distances,
// and OpenMP reduces the resulting candidate scores.
#include <algorithm>
#include <chrono>
#include <cfloat>
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

static constexpr double MAX_WIDTH = 20.0, MAX_HEIGHT = 20.0;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void cudaCheck(cudaError_t status, const char *where) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA failure at %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void generateSyntheticData(std::vector<Point>& points, int n, unsigned int seed = 42) {
    auto frand = [&seed]() { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double radius = frand() * std::min(MAX_WIDTH, MAX_HEIGHT) / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        group = std::min(group, n - count);
        while (group > 0) {
            const double sign = frand() < .5 ? -1.0 : 1.0;
            const double r = frand() * radius, dx = (2.0 * frand() - 1.0) * r;
            const double x = cx + dx, y = cy + std::sqrt(r * r - dx * dx) * sign;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y}; --group;
        }
    }
}

inline double distance(const Point& a, const Point& b) {
    const double x = a.x - b.x, y = a.y - b.y;
    return std::sqrt(x * x + y * y);
}

// One thread scores one candidate.  The strict comparison is intentional: it
// is the original QT diameter rule and also makes rank-to-rank results stable.
__global__ void candidateDiameterKernel(const Point *points, const int *members,
                                        int memberCount, const int *clustered,
                                        const int *inCluster, int n, double threshold,
                                        double *scores) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= n) return;
    if (clustered[candidate] || inCluster[candidate]) { scores[candidate] = DBL_MAX; return; }
    double maximum = 0.0;
    const Point p = points[candidate];
    for (int i = 0; i < memberCount; ++i) {
        const Point q = points[members[i]];
        const double dx = p.x - q.x, dy = p.y - q.y;
        maximum = fmax(maximum, sqrt(dx * dx + dy * dy));
        if (maximum >= threshold) break;
    }
    scores[candidate] = maximum < threshold ? maximum : DBL_MAX;
}

class GpuScorer {
    Point *points_ = nullptr; int *members_ = nullptr, *clustered_ = nullptr, *inCluster_ = nullptr;
    double *scores_ = nullptr; int n_;
public:
    GpuScorer(const std::vector<Point>& points) : n_(static_cast<int>(points.size())) {
        cudaCheck(cudaMalloc(&points_, n_ * sizeof(Point)), "allocate points");
        cudaCheck(cudaMalloc(&members_, n_ * sizeof(int)), "allocate members");
        cudaCheck(cudaMalloc(&clustered_, n_ * sizeof(int)), "allocate clustered");
        cudaCheck(cudaMalloc(&inCluster_, n_ * sizeof(int)), "allocate in-cluster");
        cudaCheck(cudaMalloc(&scores_, n_ * sizeof(double)), "allocate scores");
        cudaCheck(cudaMemcpy(points_, points.data(), n_ * sizeof(Point), cudaMemcpyHostToDevice), "upload points");
    }
    ~GpuScorer() { cudaFree(points_); cudaFree(members_); cudaFree(clustered_); cudaFree(inCluster_); cudaFree(scores_); }
    int closest(const std::vector<int>& members, const std::vector<int>& clustered,
                const std::vector<int>& inCluster, double threshold, std::vector<double>& hostScores) {
        cudaCheck(cudaMemcpy(members_, members.data(), members.size() * sizeof(int), cudaMemcpyHostToDevice), "upload members");
        cudaCheck(cudaMemcpy(clustered_, clustered.data(), n_ * sizeof(int), cudaMemcpyHostToDevice), "upload clustered");
        cudaCheck(cudaMemcpy(inCluster_, inCluster.data(), n_ * sizeof(int), cudaMemcpyHostToDevice), "upload in-cluster");
        candidateDiameterKernel<<<(n_ + 255) / 256, 256>>>(points_, members_, static_cast<int>(members.size()), clustered_, inCluster_, n_, threshold, scores_);
        cudaCheck(cudaGetLastError(), "launch candidate kernel");
        cudaCheck(cudaMemcpy(hostScores.data(), scores_, n_ * sizeof(double), cudaMemcpyDeviceToHost), "download scores");
        int result = -1;
        // Each OpenMP worker produces a deterministic local minimum; the final
        // serial merge resolves score ties by point index like the original loop.
        const int threads = omp_get_max_threads();
        std::vector<int> local(threads, -1);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            int best = -1;
            #pragma omp for nowait
            for (int i = 0; i < n_; ++i)
                if (hostScores[i] != DBL_MAX && (best < 0 || hostScores[i] < hostScores[best] || (hostScores[i] == hostScores[best] && i < best))) best = i;
            local[tid] = best;
        }
        for (int i : local) if (i >= 0 && (result < 0 || hostScores[i] < hostScores[result] || (hostScores[i] == hostScores[result] && i < result))) result = i;
        return result;
    }
};

static int generateCandidateCluster(int seed, const std::vector<int>& clustered,
                                    double threshold, GpuScorer& gpu, std::vector<int>& out) {
    const int n = static_cast<int>(clustered.size());
    std::vector<int> inCluster(n, 0); std::vector<double> scores(n);
    out.clear(); out.push_back(seed); inCluster[seed] = 1;
    while (static_cast<int>(out.size()) < n) {
        const int next = gpu.closest(out, clustered, inCluster, threshold, scores);
        if (next < 0) break;
        inCluster[next] = 1; out.push_back(next);
    }
    return static_cast<int>(out.size());
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    std::vector<int> clustered(n, 0), unclustered(n); for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters; GpuScorer gpu(points);
    while (!unclustered.empty()) {
        int localSize = -1, localSeed = std::numeric_limits<int>::max(); std::vector<int> candidate;
        for (size_t pos = rank; pos < unclustered.size(); pos += ranks) {
            const int seed = unclustered[pos]; const int size = generateCandidateCluster(seed, clustered, threshold, gpu, candidate);
            if (size > localSize || (size == localSize && seed < localSeed)) { localSize = size; localSeed = seed; }
        }
        // MPI_MAXLOC chooses the smallest location for equal values, which is
        // precisely the original first-seed tie break.
        int pair[2] = {localSize, localSeed}, winner[2];
        MPI_Allreduce(pair, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int bestSize = winner[0], bestSeed = winner[1];
        // Ownership of a seed changes as the compact unclustered list changes,
        // so use a fixed root for the reconstruction and broadcast.
        std::vector<int> members(bestSize);
        if (rank == 0) generateCandidateCluster(bestSeed, clustered, threshold, gpu, members);
        MPI_Bcast(members.data(), bestSize, MPI_INT, 0, MPI_COMM_WORLD);
        clusters.push_back({members, bestSeed});
        for (int m : members) clustered[m] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(), [&clustered](int i) { return clustered[i]; }), unclustered.end());
    }
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true; printf("Validating clusters:\n"); std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        double diameter = 0.; const auto& cluster = clusters[c];
        for (size_t i = 0; i < cluster.members.size(); ++i) for (size_t j = i + 1; j < cluster.members.size(); ++j) diameter = std::max(diameter, distance(points[cluster.members[i]], points[cluster.members[j]]));
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, diameter);
        if (diameter > threshold * 1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, diameter, threshold); valid = false; }
        for (int m : cluster.members) { if (membership[m] >= 0) { printf("ERROR: Point %d appears in multiple clusters\n", m); valid = false; } membership[m] = static_cast<int>(c); }
    }
    int count = 0; for (int m : membership) count += m >= 0;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), count, points.size() - count); return valid;
}

void printUsage(const char* p) { printf("Usage: %s [-n num] [-t threshold] [-v] [-r] [-h]\n", p); }
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    // Assign ranks round-robin over GPUs visible on this node.  This also
    // respects scheduler-provided CUDA_VISIBLE_DEVICES bindings.
    MPI_Comm nodeComm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank; MPI_Comm_rank(nodeComm, &localRank); MPI_Comm_free(&nodeComm);
    int deviceCount = 0; cudaCheck(cudaGetDeviceCount(&deviceCount), "query CUDA devices");
    if (deviceCount == 0) { if (!rank) fprintf(stderr, "No CUDA device is visible\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "select CUDA device");
    int n = 1000; double threshold = 2.; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) { if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]); else if (!strcmp(argv[i], "-t") && i + 1 < argc) threshold = atof(argv[++i]); else if (!strcmp(argv[i], "-v")) validate = true; else if (!strcmp(argv[i], "-r")) results = true; else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; } else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; } }
    if (n <= 0 || threshold <= 0.) { if (!rank) printf("Error: invalid parameters\n"); MPI_Finalize(); return 1; }
    std::vector<Point> points(n); generateSyntheticData(points, n);
    MPI_Barrier(MPI_COMM_WORLD); const auto start = std::chrono::high_resolution_clock::now();
    const auto clusters = qtClustering(points, threshold, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD); const auto end = std::chrono::high_resolution_clock::now();
    const long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long ms = 0;
    MPI_Reduce(&local_ms, &ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        int total = 0, maximum = 0;
        for (const auto& c : clusters) { total += c.members.size(); maximum = std::max(maximum, static_cast<int>(c.members.size())); }
        printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nMPI ranks: %d, OpenMP threads/rank: %d\nClustering time: %ld ms\nClusters found: %zu\n", n, threshold, ranks, omp_get_max_threads(), ms, clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n", total, n, 100. * total / n, clusters.empty() ? 0. : double(total) / clusters.size(), maximum);
        const double seconds = std::max(0.001, ms / 1000.); printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters.size() / seconds, n / seconds);
        if (results) { std::vector<double> membership(n, -1.); for (size_t c = 0; c < clusters.size(); ++c) for (int m : clusters[c].members) membership[m] = c; print_results(membership, "ClusterMembership"); }
        if (validate && !validateClusters(clusters, points, threshold)) { MPI_Abort(MPI_COMM_WORLD, 1); }
        if (validate) printf("Validation: PASSED\n");
    }
    MPI_Finalize(); return 0;
}
