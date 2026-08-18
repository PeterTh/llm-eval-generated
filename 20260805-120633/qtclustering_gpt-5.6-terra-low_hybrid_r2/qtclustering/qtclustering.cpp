// Distributed hybrid MPI/OpenMP/CUDA QT clustering benchmark.
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

static constexpr double MAX_WIDTH = 20.0, MAX_HEIGHT = 20.0;
struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void checkCuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void distanceMatrixKernel(const Point* points, double* distances, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const double dx = points[row].x - points[col].x;
        const double dy = points[row].y - points[col].y;
        distances[static_cast<size_t>(row) * n + col] = dx * dx + dy * dy;
    }
}

static void generateSyntheticData(std::vector<Point>& points, int n, unsigned seed = 42) {
    auto frand = [&seed]() { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double radius = frand() * std::min(MAX_WIDTH, MAX_HEIGHT) / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        // Preserve the original sequence for normal benchmark sizes, while making
        // the documented small inputs (n < 30) terminate.
        if (n < 30) group = std::max(1, group);
        group = std::min(group, n - count);
        while (group > 0) {
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * (frand() < .5 ? -1.0 : 1.0);
            const double x = cx + dx, y = cy + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y}; --group;
        }
    }
}

static std::vector<double> makeDistanceMatrix(const std::vector<Point>& points, int rank) {
    const size_t n = points.size();
    if (n && n > std::numeric_limits<size_t>::max() / n / sizeof(double)) {
        if (rank == 0) std::fprintf(stderr, "Distance matrix is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) { std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    checkCuda(cudaSetDevice(rank % devices), "cudaSetDevice");
    Point* d_points = nullptr; double* d_distances = nullptr;
    const size_t bytes = n * n * sizeof(double);
    checkCuda(cudaMalloc(&d_points, n * sizeof(Point)), "cudaMalloc(points)");
    checkCuda(cudaMalloc(&d_distances, bytes), "cudaMalloc(distances)");
    checkCuda(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice), "copy points");
    const dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
    distanceMatrixKernel<<<grid, block>>>(d_points, d_distances, static_cast<int>(n));
    checkCuda(cudaGetLastError(), "distanceMatrixKernel launch");
    std::vector<double> distances(n * n);
    checkCuda(cudaMemcpy(distances.data(), d_distances, bytes, cudaMemcpyDeviceToHost), "copy distances");
    checkCuda(cudaFree(d_distances), "cudaFree(distances)");
    checkCuda(cudaFree(d_points), "cudaFree(points)");
    return distances;
}

static int buildCandidate(int seed, const std::vector<unsigned char>& clustered,
                          const std::vector<double>& distances, double threshold2,
                          int n, std::vector<int>* result = nullptr) {
    std::vector<unsigned char> in_cluster(n, 0);
    std::vector<int> members; members.reserve(n);
    in_cluster[seed] = 1; members.push_back(seed);
    while (static_cast<int>(members.size()) < n) {
        int closest = -1; double best = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double maximum = 0.0;
            const size_t base = static_cast<size_t>(candidate) * n;
            for (int member : members) maximum = std::max(maximum, distances[base + member]);
            if (maximum < threshold2 && maximum < best) { best = maximum; closest = candidate; }
        }
        if (closest < 0) break;
        in_cluster[closest] = 1; members.push_back(closest);
    }
    if (result) *result = std::move(members);
    return result ? static_cast<int>(result->size()) : static_cast<int>(members.size());
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                         int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    const std::vector<double> distances = makeDistanceMatrix(points, rank);
    const double threshold2 = threshold * threshold;
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> remaining(n); for (int i = 0; i < n; ++i) remaining[i] = i;
    std::vector<Cluster> clusters;
    while (!remaining.empty()) {
        int local_count = -1, local_seed = std::numeric_limits<int>::max();
        #pragma omp parallel
        {
            int thread_count = -1, thread_seed = std::numeric_limits<int>::max();
            #pragma omp for nowait schedule(dynamic, 1)
            for (int pos = 0; pos < static_cast<int>(remaining.size()); ++pos) {
                const int seed = remaining[pos];
                if (seed % ranks != rank) continue;
                const int count = buildCandidate(seed, clustered, distances, threshold2, n);
                if (count > thread_count || (count == thread_count && seed < thread_seed))
                    thread_count = count, thread_seed = seed;
            }
            #pragma omp critical
            if (thread_count > local_count || (thread_count == local_count && thread_seed < local_seed))
                local_count = thread_count, local_seed = thread_seed;
        }
        int choice[2] = {local_count, local_seed};
        int global_choice[2] = {-1, std::numeric_limits<int>::max()};
        MPI_Allreduce(choice, global_choice, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global_choice[0] <= 0) break;
        std::vector<int> members;
        const int owner = global_choice[1] % ranks;
        if (rank == owner) buildCandidate(global_choice[1], clustered, distances, threshold2, n, &members);
        int member_count = static_cast<int>(members.size());
        MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner) members.resize(member_count);
        MPI_Bcast(members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back({std::move(members), global_choice[1]});
        for (int member : clusters.back().members) clustered[member] = 1;
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
            [&clustered](int p) { return clustered[p]; }), remaining.end());
    }
    return clusters;
}

static inline double distance(const Point& a, const Point& b) { return std::hypot(a.x-b.x, a.y-b.y); }
static bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true; std::vector<int> seen(points.size(), -1); std::printf("Validating clusters:\n");
    for (size_t c=0;c<clusters.size();++c) { double diameter=0;
        for (size_t i=0;i<clusters[c].members.size();++i) for (size_t j=i+1;j<clusters[c].members.size();++j)
            diameter=std::max(diameter,distance(points[clusters[c].members[i]],points[clusters[c].members[j]]));
        if(c<10) std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",c,clusters[c].members.size(),clusters[c].seed_point,diameter);
        if(diameter > threshold*1.001) valid=false;
        for(int p:clusters[c].members) { if(seen[p]>=0) valid=false; seen[p]=static_cast<int>(c); }
    }
    int used=0; for(int x:seen) used += x>=0;
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",points.size(),used,points.size()-used);
    return valid;
}

static void usage(const char* p) { std::printf("Usage: %s [-n num] [-t threshold] [-v] [-r] [-h]\n",p); }
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int n=1000; double threshold=2.; bool validation=false, results=false;
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"-n") && i+1<argc) n=std::atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"-t") && i+1<argc) threshold=std::atof(argv[++i]);
        else if(!std::strcmp(argv[i],"-v")) validation=true; else if(!std::strcmp(argv[i],"-r")) results=true;
        else if(!std::strcmp(argv[i],"-h")) { if(!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if(!rank) { std::printf("Unknown option: %s\n",argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if(n<=0 || threshold<=0) { if(!rank) std::printf("Error: invalid parameters\n"); MPI_Finalize(); return 1; }
    std::vector<Point> points(n); generateSyntheticData(points,n);
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    const auto clusters=qtClustering(points,threshold,rank,ranks);
    MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    int exit_code = 0;
    if(!rank) { const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\n",n,threshold,validation?"enabled":"disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\nClustering time: %ld ms\nClusters found: %zu\n",ranks,omp_get_max_threads(),ms,clusters.size());
        int total=0,maximum=0; for(const auto& c:clusters) total+=c.members.size(),maximum=std::max(maximum,(int)c.members.size());
        std::printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n",total,n,100.*total/n,clusters.empty()?0.:double(total)/clusters.size(),maximum);
        const double seconds=std::max(0.001,ms/1000.); std::printf("Performance: %.1f clusters/s, %.1f points/s\n",clusters.size()/seconds,n/seconds);
        if(results) { std::vector<double> data(n,-1.); for(size_t c=0;c<clusters.size();++c) for(int p:clusters[c].members) data[p]=c; print_results(data,"ClusterMembership"); }
        if(validation) {
            const bool valid = validateClusters(clusters,points,threshold);
            std::printf("Validation: %s\n",valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return exit_code;
}
