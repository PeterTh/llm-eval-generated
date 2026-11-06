// QT Clustering Benchmark - MPI, OpenMP, and CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <climits>
#include <cstdint>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (N <= 30 && group_cnt == 0) group_cnt = 1;
        
        // Make sure we don't make more points than we need
        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }
            
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Every CUDA block grows one seed's candidate cluster.  A candidate's
// distance to the cluster is the maximum of its distances to the members;
// updating that maximum only for the newly added member preserves the greedy
// search while avoiding a repeated scan of all earlier members.
static constexpr int BLOCK_SIZE = 256;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__device__ double pointDistance(const Point* points, int a, int b) {
    const double dx = points[a].x - points[b].x;
    const double dy = points[a].y - points[b].y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void calculateDistances(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int a = static_cast<int>(i / n);
        const int b = static_cast<int>(i % n);
        distances[i] = pointDistance(points, a, b);
    }
}

__global__ void growClusters(const Point* points, const double* distances,
                             const int* active, const int* seeds,
                             int n, int count, double threshold,
                             double* scores, int* members, int* sizes) {
    const int slot = blockIdx.x;
    const int tid = threadIdx.x;
    const int seed = seeds[slot];
    double* score = scores + static_cast<size_t>(slot) * count;
    int* member = members + static_cast<size_t>(slot) * count;
    __shared__ double bestDistance[BLOCK_SIZE];
    __shared__ int bestIndex[BLOCK_SIZE];
    __shared__ int chosen;

    for (int i = tid; i < count; i += BLOCK_SIZE) {
        int point = active[i];
        double d = distances ? distances[static_cast<size_t>(seed) * n + point]
                             : pointDistance(points, seed, point);
        score[i] = point == seed ? CUDART_INF : (d > 0.0 ? d : 0.0);
    }
    if (tid == 0) member[0] = seed;
    __syncthreads();

    int size = 1;
    while (size < count) {
        double localBest = threshold;
        int localIndex = INT_MAX;
        for (int i = tid; i < count; i += BLOCK_SIZE) {
            const double d = score[i];
            if (d < localBest || (d == localBest && d < threshold && i < localIndex)) {
                localBest = d;
                localIndex = i;
            }
        }
        bestDistance[tid] = localBest;
        bestIndex[tid] = localIndex;
        __syncthreads();
        for (int step = BLOCK_SIZE / 2; step > 0; step /= 2) {
            if (tid < step) {
                const double d = bestDistance[tid + step];
                const int idx = bestIndex[tid + step];
                if (d < bestDistance[tid] ||
                    (d == bestDistance[tid] && idx < bestIndex[tid])) {
                    bestDistance[tid] = d;
                    bestIndex[tid] = idx;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            chosen = bestIndex[0];
            if (chosen != INT_MAX) member[size] = active[chosen];
        }
        __syncthreads();
        if (chosen == INT_MAX) break;
        const int newMember = active[chosen];
        for (int i = tid; i < count; i += BLOCK_SIZE) {
            if (i == chosen) {
                score[i] = CUDART_INF;
            } else if (score[i] < threshold) {
                const int point = active[i];
                const double d = distances ? distances[static_cast<size_t>(newMember) * n + point]
                                           : pointDistance(points, newMember, point);
                if (d > score[i]) score[i] = d;
            }
        }
        ++size;
        __syncthreads();
    }
    if (tid == 0) sizes[slot] = size;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    Point* devicePoints = nullptr;
    int *deviceActive = nullptr, *deviceSeeds = nullptr, *deviceSizes = nullptr;
    double *deviceDistances = nullptr, *deviceScores = nullptr;
    int* deviceMembers = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePoints, sizeof(Point) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&deviceActive, sizeof(int) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMemcpy(devicePoints, points.data(), sizeof(Point) * static_cast<size_t>(n), cudaMemcpyHostToDevice));

    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t matrixBytes = static_cast<size_t>(n) * n * sizeof(double);
    // The matrix is optional: a large input may instead recompute distances.
    // Limit its size so that batches still fit and multiple ranks can share a GPU.
    if (matrixBytes <= freeBytes / 4 && matrixBytes <= (size_t(1) << 30)) {
        cudaError_t status = cudaMalloc(&deviceDistances, matrixBytes);
        if (status == cudaSuccess) {
            size_t blocks = (static_cast<size_t>(n) * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            calculateDistances<<<static_cast<int>(std::min<size_t>(blocks, 65535)), BLOCK_SIZE>>>(devicePoints, deviceDistances, n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            deviceDistances = nullptr;
            cudaGetLastError();
        }
    }
    const size_t perSeed = static_cast<size_t>(n) * (sizeof(double) + sizeof(int));
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    size_t batchCapacity = std::max<size_t>(1, std::min<size_t>(n, std::min<size_t>(freeBytes / 4, size_t(256) << 20) / perSeed));
    CUDA_CHECK(cudaMalloc(&deviceSeeds, batchCapacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceSizes, batchCapacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceScores, batchCapacity * static_cast<size_t>(n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceMembers, batchCapacity * static_cast<size_t>(n) * sizeof(int)));

    std::vector<int> active(n);
    for (int i = 0; i < n; ++i) active[i] = i;
    std::vector<Cluster> clusters;
    std::vector<int> seedBatch(batchCapacity), sizes(batchCapacity);
    while (!active.empty()) {
        const int count = static_cast<int>(active.size());
        CUDA_CHECK(cudaMemcpy(deviceActive, active.data(), sizeof(int) * active.size(), cudaMemcpyHostToDevice));
        int localSize = -1, localSeed = INT_MAX;
        std::vector<int> localMembers;
        // Round-robin distribution balances seeds when candidate sizes vary.
        const int localCount = rank < count ? (count - 1 - rank) / ranks + 1 : 0;
        for (int begin = 0; begin < localCount; begin += static_cast<int>(batchCapacity)) {
            const int batch = std::min<int>(static_cast<int>(batchCapacity), localCount - begin);
            #pragma omp parallel for if(batch > 4096)
            for (int j = 0; j < batch; ++j) seedBatch[j] = active[rank + (begin + j) * ranks];
            CUDA_CHECK(cudaMemcpy(deviceSeeds, seedBatch.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
            growClusters<<<batch, BLOCK_SIZE>>>(devicePoints, deviceDistances, deviceActive, deviceSeeds,
                                                n, count, threshold, deviceScores, deviceMembers, deviceSizes);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), deviceSizes, batch * sizeof(int), cudaMemcpyDeviceToHost));
            int batchBest = 0;
            for (int j = 1; j < batch; ++j)
                if (sizes[j] > sizes[batchBest] ||
                    (sizes[j] == sizes[batchBest] && seedBatch[j] < seedBatch[batchBest])) batchBest = j;
            if (sizes[batchBest] > localSize ||
                (sizes[batchBest] == localSize && seedBatch[batchBest] < localSeed)) {
                localSize = sizes[batchBest];
                localSeed = seedBatch[batchBest];
                localMembers.resize(localSize);
                CUDA_CHECK(cudaMemcpy(localMembers.data(), deviceMembers + static_cast<size_t>(batchBest) * count,
                                      localSize * sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
        int localPair[2] = {localSize, localSeed}, globalPair[2];
        MPI_Allreduce(localPair, globalPair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // Seeds are assigned round-robin by position in the sorted active
        // list, so every rank can derive the winner's owner without a second
        // collective.
        const auto winner = std::lower_bound(active.begin(), active.end(), globalPair[1]);
        const int owner = static_cast<int>(winner - active.begin()) % ranks;
        Cluster cluster;
        cluster.seed_point = globalPair[1];
        cluster.members.resize(globalPair[0]);
        if (rank == owner) cluster.members.swap(localMembers);
        MPI_Bcast(cluster.members.data(), globalPair[0], MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back(std::move(cluster));

        std::vector<unsigned char> removed(n, 0);
        const std::vector<int>& members = clusters.back().members;
        // Filter contiguous portions in parallel, then compact them in order.
        // Keeping the list sorted preserves the original tie-breaking rule.
        const int hostThreads = std::min(omp_get_max_threads(), std::max(2, count / 512 + (count % 512 != 0)));
        std::vector<int> next(count), lengths(hostThreads, 0), starts(hostThreads, 0);
        #pragma omp parallel num_threads(hostThreads)
        {
            const int tid = omp_get_thread_num();
            const int teamSize = omp_get_num_threads();
            #pragma omp for
            for (size_t i = 0; i < members.size(); ++i) removed[members[i]] = 1;
            const int begin = static_cast<int>(static_cast<long long>(count) * tid / teamSize);
            const int end = static_cast<int>(static_cast<long long>(count) * (tid + 1) / teamSize);
            int length = 0;
            for (int i = begin; i < end; ++i)
                if (!removed[active[i]]) next[begin + length++] = active[i];
            starts[tid] = begin;
            lengths[tid] = length;
        }
        int remaining = 0;
        for (int t = 0; t < hostThreads; ++t) {
            std::memmove(next.data() + remaining, next.data() + starts[t], lengths[t] * sizeof(int));
            remaining += lengths[t];
        }
        next.resize(remaining);
        active.swap(next);
    }
    CUDA_CHECK(cudaFree(deviceMembers));
    CUDA_CHECK(cudaFree(deviceScores));
    CUDA_CHECK(cudaFree(deviceSizes));
    CUDA_CHECK(cudaFree(deviceSeeds));
    CUDA_CHECK(cudaFree(deviceDistances));
    CUDA_CHECK(cudaFree(deviceActive));
    CUDA_CHECK(cudaFree(devicePoints));
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Calculate independent cluster diameters concurrently while retaining
    // the original output order below.
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
                diameter = std::max(diameter, dist);
            }
        }
        diameters[c] = diameter;
    }
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
        
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    
    // Count clustered points
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int runBenchmark(int argc, char** argv, int rank, int ranks) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                              num_points, threshold);
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype mpiPoint;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpiPoint);
    MPI_Type_commit(&mpiPoint);
    MPI_Bcast(points.data(), num_points, mpiPoint, 0, MPI_COMM_WORLD);
    MPI_Type_free(&mpiPoint);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    const std::chrono::milliseconds cluster_time(max_cluster_time);
    
    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }
    
    const double avg_cluster_size = clusters.empty() ? 0.0 : 
        static_cast<double>(total_clustered) / clusters.size();
    
    printf("Points clustered: %d / %d (%.1f%%)\n", 
           total_clustered, num_points, 
           100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);
    
    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", 
           clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults) {
        // Serialize cluster membership for hashing
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                membership[clusters[c].members[i]] = static_cast<int>(c);
            }
        }
        for (int m : membership) {
            membershipData.push_back(static_cast<double>(m));
        }
        print_results(membershipData, "ClusterMembership");
    }
    
    // Validation
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
