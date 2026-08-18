// QT clustering benchmark: MPI ranks distribute seeds, OpenMP evaluates them,
// and CUDA constructs the immutable pairwise Euclidean-distance matrix.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0, MAX_HEIGHT = 20.0;
struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void distanceMatrixKernel(const Point* points, double* distances, int n) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * n;
    if (k >= total) return;
    const int i = static_cast<int>(k / n), j = static_cast<int>(k - static_cast<size_t>(i) * n);
    const double dx = points[i].x - points[j].x;
    const double dy = points[i].y - points[j].y;
    distances[k] = sqrt(dx * dx + dy * dy);
}

static std::vector<double> buildDistanceMatrix(const std::vector<Point>& points, int rank) {
    const size_t n = points.size();
    if (n && n > std::numeric_limits<size_t>::max() / n) {
        fprintf(stderr, "Distance matrix size overflow\n"); MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { fprintf(stderr, "No CUDA accelerator available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    Point* devicePoints = nullptr; double* deviceDistances = nullptr;
    const size_t pointBytes = n * sizeof(Point), matrixBytes = n * n * sizeof(double);
    cudaCheck(cudaMalloc(&devicePoints, pointBytes), "cudaMalloc(points)");
    cudaCheck(cudaMalloc(&deviceDistances, matrixBytes), "cudaMalloc(distances)");
    cudaCheck(cudaMemcpy(devicePoints, points.data(), pointBytes, cudaMemcpyHostToDevice), "copy points");
    constexpr int threads = 256;
    const size_t blocks = (n * n + threads - 1) / threads;
    distanceMatrixKernel<<<static_cast<unsigned int>(blocks), threads>>>(devicePoints, deviceDistances, static_cast<int>(n));
    cudaCheck(cudaGetLastError(), "distance matrix kernel launch");
    std::vector<double> distances(n * n);
    cudaCheck(cudaMemcpy(distances.data(), deviceDistances, matrixBytes, cudaMemcpyDeviceToHost), "copy distances");
    cudaCheck(cudaFree(deviceDistances), "cudaFree(distances)");
    cudaCheck(cudaFree(devicePoints), "cudaFree(points)");
    return distances;
}

static void generateSyntheticData(std::vector<Point>& points, int n, unsigned int seed = 42) {
    auto frand = [&seed]() { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH, cy = frand() * MAX_HEIGHT;
        const double radius = frand() * std::min(MAX_WIDTH, MAX_HEIGHT) / 2.0;
        int group = static_cast<int>(frand() * (n / 30.0));
        if (group > n - count) group = n - count;
        while (group > 0) {
            const double sign = frand() < .5 ? -1.0 : 1.0, r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r, dy = std::sqrt(r * r - dx * dx) * sign;
            if (cx + dx < 0 || cx + dx > MAX_WIDTH || cy + dy < 0 || cy + dy > MAX_HEIGHT) continue;
            points[count++] = {cx + dx, cy + dy}; --group;
        }
    }
}

static std::vector<int> candidateCluster(int seed, const std::vector<unsigned char>& clustered,
                                         const std::vector<double>& distances, double threshold, int n) {
    std::vector<unsigned char> inCluster(n, 0); std::vector<int> members;
    inCluster[seed] = 1; members.push_back(seed);
    while (static_cast<int>(members.size()) < n) {
        double bestDiameter = std::numeric_limits<double>::max(); int best = -1;
        // This loop remains ordered intentionally: strict comparison implements original tie semantics.
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || inCluster[candidate]) continue;
            double diameter = 0.0;
            for (int member : members) diameter = std::max(diameter, distances[static_cast<size_t>(candidate) * n + member]);
            if (diameter < threshold && diameter < bestDiameter) { bestDiameter = diameter; best = candidate; }
        }
        if (best < 0) break;
        inCluster[best] = 1; members.push_back(best);
    }
    return members;
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    const std::vector<double> distances = buildDistanceMatrix(points, rank);
    std::vector<unsigned char> clustered(n, 0); std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    while (!unclustered.empty()) {
        int localSize = -1, localSeed = std::numeric_limits<int>::max(); std::vector<int> localMembers;
        #pragma omp parallel
        {
            int threadSize = -1, threadSeed = std::numeric_limits<int>::max(); std::vector<int> threadMembers;
            #pragma omp for schedule(dynamic, 1) nowait
            for (int pos = 0; pos < static_cast<int>(unclustered.size()); ++pos) {
                const int seed = unclustered[pos];
                if (seed % ranks != rank || clustered[seed]) continue;
                std::vector<int> candidate = candidateCluster(seed, clustered, distances, threshold, n);
                const int size = static_cast<int>(candidate.size());
                if (size > threadSize || (size == threadSize && seed < threadSeed)) {
                    threadSize = size; threadSeed = seed; threadMembers.swap(candidate);
                }
            }
            #pragma omp critical
            if (threadSize > localSize || (threadSize == localSize && threadSeed < localSeed)) {
                localSize = threadSize; localSeed = threadSeed; localMembers.swap(threadMembers);
            }
        }
        int local[2] = {localSize, localSeed}, *all = new int[2 * ranks];
        MPI_Allgather(local, 2, MPI_INT, all, 2, MPI_INT, MPI_COMM_WORLD);
        int winner = 0;
        for (int r = 1; r < ranks; ++r)
            if (all[2*r] > all[2*winner] || (all[2*r] == all[2*winner] && all[2*r+1] < all[2*winner+1])) winner = r;
        const int count = all[2*winner], seed = all[2*winner+1]; delete[] all;
        if (count <= 0) break;
        std::vector<int> members(count);
        if (rank == winner) members = localMembers;
        MPI_Bcast(members.data(), count, MPI_INT, winner, MPI_COMM_WORLD);
        clusters.push_back({members, seed});
        for (int member : members) clustered[member] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(), [&clustered](int i) { return clustered[i]; }), unclustered.end());
    }
    return clusters;
}

static bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, double threshold) {
    bool valid = true; std::vector<int> membership(points.size(), -1); printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        double diameter = 0;
        for (size_t i=0;i<clusters[c].members.size();++i) for (size_t j=i+1;j<clusters[c].members.size();++j) {
            const Point &a=points[clusters[c].members[i]], &b=points[clusters[c].members[j]];
            diameter=std::max(diameter,std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)));
        }
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, clusters[c].members.size(), clusters[c].seed_point, diameter);
        if (diameter > threshold * 1.001) { printf("ERROR: Cluster %zu exceeds threshold\n", c); valid=false; }
        for (int member : clusters[c].members) { if (membership[member]>=0) valid=false; membership[member]=static_cast<int>(c); }
    }
    int count=0; for(int m:membership) count += m>=0; printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),count,points.size()-count); return valid;
}

static void printUsage(const char* p) { printf("Usage: %s [-n num] [-t threshold] [-v] [-r] [-h]\n", p); }
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int n=1000; double threshold=2.0; bool validate=false, results=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-n") && i+1<argc) n=atoi(argv[++i]); else if(!strcmp(argv[i],"-t")&&i+1<argc) threshold=atof(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true; else if(!strcmp(argv[i],"-r")) results=true; else if(!strcmp(argv[i],"-h")) { if(!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(!rank) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if(n<=0 || threshold<=0) { if(!rank) printf("Error: invalid parameters\n"); MPI_Finalize(); return 1; }
    std::vector<Point> points(n); if(!rank) generateSyntheticData(points,n); MPI_Bcast(points.data(), n*sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
    if(!rank) printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nMPI ranks: %d, OpenMP threads/rank: %d\n",n,threshold,ranks,omp_get_max_threads());
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::steady_clock::now(); auto clusters=qtClustering(points,threshold,rank,ranks); MPI_Barrier(MPI_COMM_WORLD);
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(), maxElapsed=0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank) { printf("Clustering time: %.3f ms\nClusters found: %zu\n",maxElapsed*1000,clusters.size()); int total=0,maxSize=0; for(const auto& c:clusters){total+=c.members.size();maxSize=std::max(maxSize,(int)c.members.size());} printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\nPerformance: %.1f clusters/s, %.1f points/s\n",total,n,100.0*total/n,clusters.empty()?0.0:(double)total/clusters.size(),maxSize,clusters.size()/maxElapsed,n/maxElapsed); if(results){std::vector<double> m(n,-1);for(size_t c=0;c<clusters.size();++c)for(int p:clusters[c].members)m[p]=c;print_results(m,"ClusterMembership");} bool valid=!validate || validateClusters(clusters,points,threshold); if(validate) printf("Validation: %s\n",valid?"PASSED":"FAILED"); MPI_Finalize(); return valid?0:1; }
    MPI_Finalize(); return 0;
}
