// QT Clustering Benchmark - MPI/OpenMP/CUDA
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
#include <vector>
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

// All MPI calls are made by the main thread (MPI_THREAD_FUNNELED).
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK_SIZE = 128;

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    // Compile without FMA contraction to match the sequential distance exactly.
    return sqrt(dx * dx + dy * dy);
}

// One cooperative block per seed. Only seed neighbors can ever join its
// cluster. Cache their running maximum distances, avoiding rescans of all
// previously selected members. Scratch is bounded by the host batch size.
__global__ void growCandidates(const Point* points, const unsigned char* clustered,
                               const int* seeds, int seedOffset, int n,
                               double threshold, int* neighbors, double* maxima,
                               int* sizes, int* members) {
    const int seed = seeds[seedOffset + blockIdx.x];
    int* ids = neighbors + size_t(blockIdx.x) * n;
    double* maxDist = maxima + size_t(blockIdx.x) * n;
    __shared__ int neighborCount, selected, cardinality;
    __shared__ double bestDistance[BLOCK_SIZE];
    __shared__ int bestId[BLOCK_SIZE], bestSlot[BLOCK_SIZE];
    if (threadIdx.x == 0) {
        neighborCount = 0;
        cardinality = 1;
        if (members) members[0] = seed;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == seed || clustered[i]) continue;
        const double d = pointDistance(points[i], points[seed]);
        if (d < threshold) {
            const int slot = atomicAdd(&neighborCount, 1);
            ids[slot] = i;
            maxDist[slot] = d;
        }
    }
    __syncthreads();
    while (true) {
        double best = DBL_MAX;
        int id = INT_MAX, slot = -1;
        for (int i = threadIdx.x; i < neighborCount; i += blockDim.x) {
            const double d = maxDist[i];
            if (d < best || (d == best && d < DBL_MAX && ids[i] < id)) {
                best = d;
                id = ids[i];
                slot = i;
            }
        }
        bestDistance[threadIdx.x] = best;
        bestId[threadIdx.x] = id;
        bestSlot[threadIdx.x] = slot;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                int other = threadIdx.x + stride;
                if (bestDistance[other] < bestDistance[threadIdx.x] ||
                    (bestDistance[other] == bestDistance[threadIdx.x] &&
                     bestId[other] < bestId[threadIdx.x])) {
                    bestDistance[threadIdx.x] = bestDistance[other];
                    bestId[threadIdx.x] = bestId[other];
                    bestSlot[threadIdx.x] = bestSlot[other];
                }
            }
            __syncthreads();
        }
        if (bestSlot[0] < 0) break;
        if (threadIdx.x == 0) {
            selected = bestId[0];
            maxDist[bestSlot[0]] = DBL_MAX;
            if (members) members[cardinality] = selected;
            ++cardinality;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < neighborCount; i += blockDim.x) {
            if (maxDist[i] == DBL_MAX) continue;
            const double d = pointDistance(points[ids[i]], points[selected]);
            const double updated = fmax(maxDist[i], d);
            maxDist[i] = updated < threshold ? updated : DBL_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) sizes[seedOffset + blockIdx.x] = cardinality;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                 const double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    // Host work is linear and small compared with GPU candidate growth.
    // Avoid starting a large OpenMP team for tiny bookkeeping loops.
    const int hostThreads = std::min(omp_get_max_threads(), std::max(2, n / 2048));
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> active(n), seeds(n), sizes(n);
    #pragma omp parallel for schedule(static) num_threads(hostThreads)
    for (int i = 0; i < n; ++i) active[i] = i;
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<unsigned char> deviceClustered(n);
    DeviceBuffer<int> deviceSeeds(n), deviceSizes(n), deviceMembers(n);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    size_t freeBytes, totalBytes;
    cudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Leave space for the runtime and other ranks sharing this device. Never
    // allocate an N-by-N distance matrix, even on large problem instances.
    const size_t budget = std::min<size_t>(freeBytes / 4, 256ULL * 1024 * 1024);
    const size_t bytesPerSeed = size_t(n) * (sizeof(int) + sizeof(double));
    if (budget < bytesPerSeed) {
        fprintf(stderr, "Insufficient GPU memory for one candidate\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int batchSize = static_cast<int>(std::min<size_t>(
        std::min<size_t>(budget / bytesPerSeed, 4096), (size_t(n) + ranks - 1) / ranks));
    DeviceBuffer<int> neighbors(size_t(batchSize) * n);
    DeviceBuffer<double> maxima(size_t(batchSize) * n);
    std::vector<Cluster> clusters;
    while (!active.empty()) {
        const int count = static_cast<int>(active.size());
        const int localCount = count > rank ? (count - 1 - rank) / ranks + 1 : 0;
        #pragma omp parallel for schedule(static) num_threads(hostThreads)
        for (int i = 0; i < localCount; ++i) seeds[i] = active[rank + i * ranks];
        cudaCheck(cudaMemcpy(deviceClustered.data, clustered.data(), n, cudaMemcpyHostToDevice));
        if (localCount) {
            cudaCheck(cudaMemcpy(deviceSeeds.data, seeds.data(), localCount * sizeof(int), cudaMemcpyHostToDevice));
            for (int offset = 0; offset < localCount; offset += batchSize) {
                const int batch = std::min(batchSize, localCount - offset);
                growCandidates<<<batch, BLOCK_SIZE>>>(devicePoints.data, deviceClustered.data,
                    deviceSeeds.data, offset, n, threshold, neighbors.data, maxima.data,
                    deviceSizes.data, nullptr);
                cudaCheck(cudaGetLastError());
            }
            cudaCheck(cudaMemcpy(sizes.data(), deviceSizes.data, localCount * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // MAXLOC chooses the lowest seed index among equal cardinalities,
        // exactly the winner selected by the original ordered seed traversal.
        int localBest[2] = {-1, INT_MAX}, globalBest[2];
        #pragma omp parallel num_threads(hostThreads)
        {
            int bestSize = -1, bestSeed = INT_MAX;
            #pragma omp for nowait schedule(static)
            for (int i = 0; i < localCount; ++i) {
                if (sizes[i] > bestSize || (sizes[i] == bestSize && seeds[i] < bestSeed)) {
                    bestSize = sizes[i];
                    bestSeed = seeds[i];
                }
            }
            #pragma omp critical
            {
                if (bestSize > localBest[0] || (bestSize == localBest[0] && bestSeed < localBest[1])) {
                    localBest[0] = bestSize;
                    localBest[1] = bestSeed;
                }
            }
        }
        MPI_Allreduce(localBest, globalBest, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (globalBest[0] == 1) {
            // No remaining pair can join a cluster. All subsequent winners
            // are singletons in ascending seed order, without more launches.
            if (rank == 0)
                for (int seed : active) clusters.push_back({{seed}, seed});
            break;
        }
        const int seed = globalBest[1];
        const int position = static_cast<int>(std::lower_bound(active.begin(), active.end(), seed) - active.begin());
        const int owner = position % ranks;
        std::vector<int> members(globalBest[0]);
        if (rank == owner) {
            cudaCheck(cudaMemcpy(deviceSeeds.data, &seed, sizeof(int), cudaMemcpyHostToDevice));
            growCandidates<<<1, BLOCK_SIZE>>>(devicePoints.data, deviceClustered.data,
                deviceSeeds.data, 0, n, threshold, neighbors.data, maxima.data,
                deviceSizes.data, deviceMembers.data);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(members.data(), deviceMembers.data, members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), globalBest[0], MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static) num_threads(hostThreads)
        for (int i = 0; i < globalBest[0]; ++i) clustered[members[i]] = 1;
        active.erase(std::remove_if(active.begin(), active.end(),
            [&clustered](int i) { return clustered[i] != 0; }), active.end());
        if (rank == 0) clusters.push_back({std::move(members), seed});
    }
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        // Check diameter (max distance between any two points)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    struct MpiLifetime { ~MpiLifetime() { MPI_Finalize(); } } mpiLifetime;
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount));
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    // Chunk the byte broadcast to avoid MPI's int count limit.
    const size_t pointBytes = points.size() * sizeof(Point);
    for (size_t offset = 0; offset < pointBytes; ) {
        const int bytes = static_cast<int>(std::min<size_t>(pointBytes - offset, INT_MAX));
        MPI_Bcast(reinterpret_cast<char*>(points.data()) + offset, bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
        offset += bytes;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (rank != 0) return 0;

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
    const double time_sec = std::chrono::duration<double>(cluster_end - cluster_start).count();
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
