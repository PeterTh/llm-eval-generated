// QT clustering benchmark: MPI ranks coordinate greedy rounds, OpenMP evaluates
// independent seeds on each rank, and CUDA builds the distance matrix.
#include <algorithm>
#include <chrono>
#include <climits>
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

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void distancesKernel(const Point* points, double* distances, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const double dx = points[row].x - points[col].x;
        const double dy = points[row].y - points[col].y;
        distances[static_cast<size_t>(row) * n + col] = dx * dx + dy * dy;
    }
}

static std::vector<double> buildDistanceMatrix(const std::vector<Point>& points, int rank) {
    const int n = static_cast<int>(points.size());
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(double);
    Point* device_points = nullptr;
    double* device_distances = nullptr;
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % device_count), "cudaSetDevice");
    cudaCheck(cudaMalloc(&device_points, static_cast<size_t>(n) * sizeof(Point)), "cudaMalloc(points)");
    cudaCheck(cudaMalloc(&device_distances, bytes), "cudaMalloc(distances)");
    cudaCheck(cudaMemcpy(device_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice), "cudaMemcpy(points)");
    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    distancesKernel<<<grid, block>>>(device_points, device_distances, n);
    cudaCheck(cudaGetLastError(), "distancesKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "distancesKernel");
    std::vector<double> distances(static_cast<size_t>(n) * n);
    cudaCheck(cudaMemcpy(distances.data(), device_distances, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(distances)");
    cudaCheck(cudaFree(device_distances), "cudaFree(distances)");
    cudaCheck(cudaFree(device_points), "cudaFree(points)");
    return distances;
}

void generateSyntheticData(std::vector<Point>& points, int n, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double radius = frand() * std::min(MAX_WIDTH, MAX_HEIGHT) / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        if (group > n - count) group = n - count;
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

static std::vector<int> generateCandidateCluster(int seed, const std::vector<unsigned char>& clustered,
                                                   const std::vector<double>& distances, double threshold2, int n) {
    std::vector<unsigned char> in_cluster(n, 0);
    std::vector<int> members; members.reserve(n);
    in_cluster[seed] = 1; members.push_back(seed);
    while (static_cast<int>(members.size()) < n) {
        int closest = -1;
        double minimum = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double maximum = 0.0;
            const size_t base = static_cast<size_t>(candidate) * n;
            for (int member : members) maximum = std::max(maximum, distances[base + member]);
            // Strict comparison intentionally matches the original QT definition.
            if (maximum < threshold2 && maximum < minimum) { minimum = maximum; closest = candidate; }
        }
        if (closest < 0) break;
        in_cluster[closest] = 1; members.push_back(closest);
    }
    return members;
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    const std::vector<double> distances = buildDistanceMatrix(points, rank);
    const double threshold2 = threshold * threshold;
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    while (!unclustered.empty()) {
        int local_cardinality = -1, local_seed = INT_MAX;
#pragma omp parallel
        {
            int thread_cardinality = -1, thread_seed = INT_MAX;
#pragma omp for schedule(dynamic, 1) nowait
            for (int pos = rank; pos < static_cast<int>(unclustered.size()); pos += ranks) {
                const int seed = unclustered[pos];
                const int cardinality = static_cast<int>(generateCandidateCluster(seed, clustered, distances, threshold2, n).size());
                if (cardinality > thread_cardinality || (cardinality == thread_cardinality && seed < thread_seed)) {
                    thread_cardinality = cardinality; thread_seed = seed;
                }
            }
#pragma omp critical
            if (thread_cardinality > local_cardinality || (thread_cardinality == local_cardinality && thread_seed < local_seed)) {
                local_cardinality = thread_cardinality; local_seed = thread_seed;
            }
        }
        int global_cardinality = -1;
        MPI_Allreduce(&local_cardinality, &global_cardinality, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int eligible_seed = local_cardinality == global_cardinality ? local_seed : INT_MAX;
        int best_seed = INT_MAX;
        MPI_Allreduce(&eligible_seed, &best_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (global_cardinality <= 0 || best_seed == INT_MAX) break;
        const int owner = std::find(unclustered.begin(), unclustered.end(), best_seed) - unclustered.begin();
        const int owner_rank = owner % ranks;
        std::vector<int> members;
        if (rank == owner_rank) members = generateCandidateCluster(best_seed, clustered, distances, threshold2, n);
        int member_count = rank == owner_rank ? static_cast<int>(members.size()) : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, owner_rank, MPI_COMM_WORLD);
        if (rank != owner_rank) members.resize(member_count);
        MPI_Bcast(members.data(), member_count, MPI_INT, owner_rank, MPI_COMM_WORLD);
        clusters.push_back({members, best_seed});
        for (int member : members) clustered[member] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(), [&clustered](int i) { return clustered[i]; }), unclustered.end());
    }
    return clusters;
}

static inline double distance(const Point& a, const Point& b) { const double x = a.x-b.x, y = a.y-b.y; return std::sqrt(x*x+y*y); }
static bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true; std::vector<int> membership(points.size(), -1); printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        double diameter = 0;
        for (size_t i=0;i<clusters[c].members.size();++i) for (size_t j=i+1;j<clusters[c].members.size();++j) diameter=std::max(diameter,distance(points[clusters[c].members[i]],points[clusters[c].members[j]]));
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, clusters[c].members.size(), clusters[c].seed_point, diameter);
        if (diameter > threshold * 1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, diameter, threshold); valid=false; }
        for (int member : clusters[c].members) { if (membership[member] >= 0) valid=false; membership[member]=static_cast<int>(c); }
    }
    int count=0; for (int x:membership) if(x>=0) ++count;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), count, points.size()-count); return valid;
}
static void printUsage(const char* p) { printf("Usage: %s [-n num] [-t threshold] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0; MPI_Comm_rank(local_comm, &local_rank);
    int n=1000; double threshold=2.0; bool validate=false, results=false;
    for(int i=1;i<argc;++i) { if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]); else if(!strcmp(argv[i],"-t")&&i+1<argc)threshold=atof(argv[++i]); else if(!strcmp(argv[i],"-v"))validate=true; else if(!strcmp(argv[i],"-r"))results=true; else if(!strcmp(argv[i],"-h")){if(!rank)printUsage(argv[0]);MPI_Finalize();return 0;} else {if(!rank)printUsage(argv[0]);MPI_Finalize();return 1;} }
    if(n<=0||threshold<=0) { if(!rank) fprintf(stderr,"Error: invalid parameters\n"); MPI_Finalize(); return 1; }
    std::vector<Point> points(n); generateSyntheticData(points,n);
    if(!rank) { printf("QT Clustering Benchmark (MPI ranks=%d, OpenMP threads=%d, CUDA enabled)\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\n",ranks,omp_get_max_threads(),n,threshold,validate?"enabled":"disabled"); }
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::high_resolution_clock::now();
    // local_rank maps one rank per node to each visible accelerator.
    const std::vector<Cluster> clusters=qtClustering(points,threshold,local_rank,ranks);
    MPI_Barrier(MPI_COMM_WORLD); const double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-start).count(); double maximum=0; MPI_Reduce(&elapsed,&maximum,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    int exit_code=0;
    if(!rank) { int total=0,max_size=0; for(const auto& c:clusters){total+=c.members.size();max_size=std::max(max_size,static_cast<int>(c.members.size()));} printf("Clustering time: %.3f ms\nClusters found: %zu\nPoints clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\nPerformance: %.1f clusters/s, %.1f points/s\n",maximum*1000,clusters.size(),total,n,100.0*total/n,clusters.empty()?0.0:double(total)/clusters.size(),max_size,clusters.size()/maximum,n/maximum); if(results){std::vector<double> membership(n,-1);for(size_t c=0;c<clusters.size();++c)for(int x:clusters[c].members)membership[x]=c;print_results(membership,"ClusterMembership");} if(validate&&!validateClusters(clusters,points,threshold)){printf("Validation: FAILED\n");exit_code=1;} else if(validate) printf("Validation: PASSED\n"); }
    MPI_Bcast(&exit_code,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Comm_free(&local_comm); MPI_Finalize(); return exit_code;
}
