// QT Clustering Benchmark - MPI / OpenMP / CUDA
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
#include <vector>
#include <climits>
#include <utility>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>

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
        // The original generator cannot make progress for N <= 30.
        if (N <= 30) group_cnt = 1;
        
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

// All MPI calls are on the main thread (MPI_THREAD_FUNNELED).
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Owning buffers are allocated once and reused throughout clustering.
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    // No FMA contraction: match the original double-precision distance.
    return sqrt(dx * dx + dy * dy);
}

constexpr int BLOCK_SIZE = 128;

// Each rank stores only the threshold neighborhoods of its own seeds.
// A point outside the seed's neighborhood can never enter that cluster.
__global__ void countNeighbors(const Point* points, int n, int rank, int ranks,
                              double threshold, int* counts) {
    const int row = blockIdx.x;
    const int seed = rank + row * ranks;
    int count = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (i != seed && pointDistance(points[seed], points[i]) < threshold) ++count;
    __shared__ int partial[BLOCK_SIZE];
    partial[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) counts[row] = partial[0];
}

__global__ void fillNeighbors(const Point* points, int n, int rank, int ranks,
                             double threshold, const size_t* offsets,
                             int* neighbors, double* seedDistances) {
    const int row = blockIdx.x;
    const int seed = rank + row * ranks;
    __shared__ int next;
    if (!threadIdx.x) next = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == seed) continue;
        double d = pointDistance(points[seed], points[i]);
        if (d < threshold) {
            size_t slot = offsets[row] + atomicAdd(&next, 1);
            neighbors[slot] = i;
            seedDistances[slot] = d;
        }
    }
}

// Min distance, then min global point index. Neighborhood insertion order
// is immaterial because every reduction explicitly applies this tie rule.
__device__ void choose(double otherDistance, int otherPoint, double& d, int& p) {
    if (otherDistance < d || (otherDistance == d && otherPoint < p)) {
        d = otherDistance;
        p = otherPoint;
    }
}

// One cooperative block grows each candidate. Maximum distances are updated
// only against the last added member, replacing repeated all-member scans.
__global__ void growCandidates(const Point* points, const unsigned char* clustered,
                               const int* rows, int rank, int ranks,
                               double threshold, const size_t* offsets,
                               const int* neighbors, const double* seedDistances,
                               double* maxima, int* sizes, int* memberLists) {
    const int row = rows[blockIdx.x];
    const int seed = rank + row * ranks;
    if (clustered[seed]) {
        if (!threadIdx.x) sizes[row] = 0;
        return;
    }
    const size_t begin = offsets[row], end = offsets[row + 1];
    // A previous candidate remains identical if none of its selected members
    // were removed. Removing other points cannot alter its greedy choices.
    __shared__ int dirty;
    if (!threadIdx.x) dirty = sizes[row] == 0;
    __syncthreads();
    const int previousSize = sizes[row];
    for (int i = threadIdx.x; i < previousSize - 1; i += blockDim.x)
        if (clustered[memberLists[begin + i]]) atomicExch(&dirty, 1);
    __syncthreads();
    if (!dirty) return;
    for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x)
        maxima[i] = clustered[neighbors[i]] ? CUDART_INF : seedDistances[i];
    __shared__ double warpDistances[BLOCK_SIZE / 32];
    __shared__ int warpPoints[BLOCK_SIZE / 32];
    __shared__ int selected;
    int last = seed, cardinality = 1;
    __syncthreads();
    for (;;) {
        double bestDistance = CUDART_INF;
        int bestPoint = INT_MAX;
        for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
            double d = maxima[i];
            if (d >= threshold) continue;
            const int candidate = neighbors[i];
            if (candidate == last) {
                maxima[i] = CUDART_INF;
                continue;
            }
            if (last != seed) {
                d = fmax(d, pointDistance(points[candidate], points[last]));
                maxima[i] = d;
            }
            if (d < threshold) choose(d, candidate, bestDistance, bestPoint);
        }
        for (int delta = 16; delta; delta /= 2) {
            double d = __shfl_down_sync(0xffffffff, bestDistance, delta);
            int p = __shfl_down_sync(0xffffffff, bestPoint, delta);
            choose(d, p, bestDistance, bestPoint);
        }
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        if (!lane) {
            warpDistances[warp] = bestDistance;
            warpPoints[warp] = bestPoint;
        }
        __syncthreads();
        if (!warp) {
            bestDistance = lane < BLOCK_SIZE / 32 ? warpDistances[lane] : CUDART_INF;
            bestPoint = lane < BLOCK_SIZE / 32 ? warpPoints[lane] : INT_MAX;
            for (int delta = 16; delta; delta /= 2) {
                double d = __shfl_down_sync(0xffffffff, bestDistance, delta);
                int p = __shfl_down_sync(0xffffffff, bestPoint, delta);
                choose(d, p, bestDistance, bestPoint);
            }
            if (!lane) selected = bestPoint;
        }
        __syncthreads();
        if (selected == INT_MAX) break;
        last = selected;
        if (!threadIdx.x) memberLists[begin + cardinality - 1] = last;
        ++cardinality;
    }
    if (!threadIdx.x) sizes[row] = cardinality;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int localCount = n / ranks + (rank < n % ranks ? 1 : 0);
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<unsigned char> deviceClustered(n);
    DeviceBuffer<int> deviceCounts(localCount), deviceRows(localCount);
    DeviceBuffer<size_t> deviceOffsets(static_cast<size_t>(localCount) + 1);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(deviceClustered.data, 0, n));
    std::vector<int> counts(localCount), rows(localCount);
    std::vector<size_t> offsets(static_cast<size_t>(localCount) + 1, 0);
    if (localCount) {
        countNeighbors<<<localCount, BLOCK_SIZE>>>(devicePoints.data, n, rank, ranks,
                                                  threshold, deviceCounts.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(counts.data(), deviceCounts.data, localCount * sizeof(int),
                             cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < localCount; ++i) offsets[i + 1] = offsets[i] + counts[i];
    cudaCheck(cudaMemcpy(deviceOffsets.data, offsets.data(), offsets.size() * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    DeviceBuffer<int> neighbors(offsets.back()), memberLists(offsets.back());
    DeviceBuffer<double> seedDistances(offsets.back()), maxima(offsets.back());
    if (localCount) {
        fillNeighbors<<<localCount, BLOCK_SIZE>>>(devicePoints.data, n, rank, ranks, threshold,
            deviceOffsets.data, neighbors.data, seedDistances.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemset(deviceCounts.data, 0, localCount * sizeof(int)));
    }
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        int active = 0;
        for (int row = 0; row < localCount; ++row)
            if (!clustered[rank + row * ranks]) rows[active++] = row;
        if (active) {
            cudaCheck(cudaMemcpy(deviceRows.data, rows.data(), active * sizeof(int), cudaMemcpyHostToDevice));
            growCandidates<<<active, BLOCK_SIZE>>>(devicePoints.data, deviceClustered.data,
                deviceRows.data, rank, ranks, threshold, deviceOffsets.data, neighbors.data,
                seedDistances.data, maxima.data, deviceCounts.data, memberLists.data);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(counts.data(), deviceCounts.data, localCount * sizeof(int),
                                 cudaMemcpyDeviceToHost));
        }
        // Packed key makes the OpenMP maximum deterministic for both fields.
        unsigned long long best = 0;
        #pragma omp parallel for reduction(max:best) schedule(static)
        for (int i = 0; i < active; ++i) {
            const int row = rows[i], seed = rank + row * ranks;
            unsigned long long key = (static_cast<unsigned long long>(counts[row]) << 32)
                                   | static_cast<unsigned int>(INT_MAX - seed);
            best = std::max(best, key);
        }
        int localBest[2] = {static_cast<int>(best >> 32),
                           best ? INT_MAX - static_cast<int>(best & 0xffffffffULL) : INT_MAX};
        int globalBest[2];
        MPI_Allreduce(localBest, globalBest, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int cardinality = globalBest[0], seed = globalBest[1];
        // If all candidates are singletons, removing a point cannot create
        // an admissible pair. Emit the remaining seeds in original order.
        if (cardinality == 1) {
            if (!rank)
                for (int i = 0; i < n; ++i)
                    if (!clustered[i]) clusters.push_back({{i}, i});
            break;
        }
        const int owner = seed % ranks;
        std::vector<int> members(cardinality);
        if (rank == owner) {
            const int row = seed / ranks;
            members[0] = seed;
            if (cardinality > 1)
                cudaCheck(cudaMemcpy(members.data() + 1, memberLists.data + offsets[row],
                                     (cardinality - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < cardinality; ++i) clustered[members[i]] = 1;
        cudaCheck(cudaMemcpy(deviceClustered.data, clustered.data(), n, cudaMemcpyHostToDevice));
        remaining -= cardinality;
        if (!rank) clusters.push_back({std::move(members), seed});
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
        #pragma omp parallel for reduction(max:max_diameter) schedule(dynamic, 8)
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

struct MpiSession {
    MpiSession(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ~MpiSession() { MPI_Finalize(); }
};

int main(int argc, char** argv) {
    MpiSession mpi(argc, argv);
    int rank;
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
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (!rank) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    // Map node-local MPI ranks to accelerators. A launcher can alternatively
    // expose one device per rank with CUDA_VISIBLE_DEVICES.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    // Host work is small compared with GPU work. Avoid creating hundreds
    // of threads per rank by default; honor explicit OpenMP settings.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / localRanks)));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        if (!rank) fprintf(stderr, "QT clustering requires a CUDA accelerator on every rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    if (!rank) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (!rank) generateSyntheticData(points, num_points);
    // Broadcast the original point bytes in chunks to avoid MPI count overflow.
    unsigned char* coordinates = reinterpret_cast<unsigned char*>(points.data());
    const size_t coordinateCount = static_cast<size_t>(num_points) * sizeof(Point);
    for (size_t first = 0; first < coordinateCount;) {
        int count = static_cast<int>(std::min(coordinateCount - first, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(coordinates + first, count, MPI_BYTE, 0, MPI_COMM_WORLD);
        first += count;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    int exitStatus = 0;
    if (rank) {
        MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return exitStatus;
    }

    printf("Clustering time: %ld ms\n", cluster_time);
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
    const double time_sec = static_cast<double>(cluster_time) / 1000.0;
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
        } else {
            exitStatus = 1;
            printf("Validation: FAILED\n");
        }
    }
    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exitStatus;
}
