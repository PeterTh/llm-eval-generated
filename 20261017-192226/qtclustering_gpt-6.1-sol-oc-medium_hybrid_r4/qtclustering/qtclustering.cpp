// QT Clustering Benchmark - distributed GPU candidate growth
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
#include <utility>
#include <climits>
#include <cfloat>
#include <cstdint>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) MPI_Abort(MPI_COMM_WORLD, 1);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

constexpr int BLOCK_SIZE = 128;

// Threshold-neighbor CSR rows are distributed cyclically by seed. Points are
// replicated, but adjacency and all candidate workspace scale with 1 / ranks.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int localCount, int* counts) {
    for (int row = blockIdx.x; row < localCount; row += gridDim.x) {
        const int seed = rank + row * ranks;
        int count = 0;
        for (int j = threadIdx.x; j < n; j += blockDim.x)
            count += j == seed || deviceDistance(points[seed], points[j]) < threshold;
        __shared__ int sums[BLOCK_SIZE];
        sums[threadIdx.x] = count;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) counts[row] = sums[0];
        __syncthreads();
    }
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, int localCount,
                              const size_t* offsets, int* neighbors) {
    for (int row = blockIdx.x; row < localCount; row += gridDim.x) {
        const int seed = rank + row * ranks;
        __shared__ int next;
        if (threadIdx.x == 0) next = 0;
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            if (j == seed || deviceDistance(points[seed], points[j]) < threshold) {
                const int slot = atomicAdd(&next, 1);
                neighbors[offsets[row] + slot] = j;
            }
        }
        __syncthreads();
    }
}

// One block grows one candidate. Maintain each point's maximum distance
// incrementally, reducing growth from O(N*k*k) to O(degree*k). Explicit index
// tie-breaking makes the unsorted CSR layout immaterial to the result.
__global__ void growCandidates(const Point* points, double threshold,
                               int rank, int ranks, int localCount,
                               const size_t* offsets, const int* neighbors,
                               const unsigned char* active, double* maxima,
                               int* members, int* sizes) {
    __shared__ double bestDistance[BLOCK_SIZE];
    __shared__ int bestIndex[BLOCK_SIZE];
    __shared__ int changed, last, cardinality;
    for (int row = blockIdx.x; row < localCount; row += gridDim.x) {
        const int seed = rank + row * ranks;
        const size_t begin = offsets[row], end = offsets[row + 1];
        if (!active[seed]) {
            if (threadIdx.x == 0) sizes[row] = 0;
            continue;
        }
        // Removing points not in this candidate cannot change its greedy path.
        if (threadIdx.x == 0) changed = sizes[row] == 0;
        __syncthreads();
        for (int j = threadIdx.x; j < sizes[row]; j += blockDim.x)
            if (!active[members[begin + j]]) atomicExch(&changed, 1);
        __syncthreads();
        if (!changed) continue;
        for (size_t j = begin + threadIdx.x; j < end; j += blockDim.x)
            maxima[j] = active[neighbors[j]] && neighbors[j] != seed ? 0.0 : CUDART_INF;
        if (threadIdx.x == 0) {
            last = seed;
            cardinality = 1;
            members[begin] = seed;
        }
        __syncthreads();
        while (true) {
            double best = DBL_MAX;
            int index = INT_MAX;
            for (size_t j = begin + threadIdx.x; j < end; j += blockDim.x) {
                double value = maxima[j];
                if (value < threshold) {
                    const int candidate = neighbors[j];
                    if (candidate == last) value = CUDART_INF;
                    else value = fmax(value, deviceDistance(points[candidate], points[last]));
                    maxima[j] = value;
                    if (value < threshold && (value < best || (value == best && candidate < index))) {
                        best = value;
                        index = candidate;
                    }
                }
            }
            bestDistance[threadIdx.x] = best;
            bestIndex[threadIdx.x] = index;
            __syncthreads();
            for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
                if (threadIdx.x < stride) {
                    const int other = threadIdx.x + stride;
                    if (bestDistance[other] < bestDistance[threadIdx.x] ||
                        (bestDistance[other] == bestDistance[threadIdx.x] &&
                         bestIndex[other] < bestIndex[threadIdx.x])) {
                        bestDistance[threadIdx.x] = bestDistance[other];
                        bestIndex[threadIdx.x] = bestIndex[other];
                    }
                }
                __syncthreads();
            }
            if (bestIndex[0] == INT_MAX) break;
            if (threadIdx.x == 0) {
                last = bestIndex[0];
                members[begin + cardinality++] = last;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) sizes[row] = cardinality;
        __syncthreads();
    }
}

__global__ void removeMembers(unsigned char* active, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) active[members[i]] = 0;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int localCount = n / ranks + (rank < n % ranks);
    const int blocks = std::max(1, std::min(localCount, 65535));
    DeviceBuffer<Point> dPoints(n);
    DeviceBuffer<unsigned char> active(n);
    DeviceBuffer<int> counts(localCount), sizes(localCount), removed(n);
    DeviceBuffer<size_t> dOffsets(size_t(localCount) + 1);
    CUDA_CHECK(cudaMemcpy(dPoints.data, points.data(), size_t(n) * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(active.data, 1, n));
    CUDA_CHECK(cudaMemset(sizes.data, 0, size_t(localCount) * sizeof(int)));
    countNeighbors<<<blocks, BLOCK_SIZE>>>(dPoints.data, n, threshold, rank, ranks, localCount, counts.data);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> hostSizes(localCount);
    CUDA_CHECK(cudaMemcpy(hostSizes.data(), counts.data, size_t(localCount) * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> offsets(size_t(localCount) + 1, 0);
    // Parallel prefix sum, with only O(threads) serial work.
    const int hostThreads = std::min(omp_get_max_threads(), localCount ? (localCount - 1) / 256 + 1 : 1);
    std::vector<size_t> partial(size_t(hostThreads) + 1, 0);
    #pragma omp parallel num_threads(hostThreads)
    {
        const int tid = omp_get_thread_num(), threads = omp_get_num_threads();
        const int first = static_cast<long long>(localCount) * tid / threads;
        const int finish = static_cast<long long>(localCount) * (tid + 1) / threads;
        size_t sum = 0;
        for (int i = first; i < finish; ++i) sum += hostSizes[i];
        partial[tid + 1] = sum;
        #pragma omp barrier
        #pragma omp single
        for (int t = 1; t <= threads; ++t) partial[t] += partial[t - 1];
        sum = partial[tid];
        for (int i = first; i < finish; ++i) {
            offsets[i] = sum;
            sum += hostSizes[i];
        }
    }
    if (localCount) offsets[localCount] = offsets[localCount - 1] + hostSizes.back();
    DeviceBuffer<int> neighbors(offsets.back()), members(offsets.back());
    DeviceBuffer<double> maxima(offsets.back());
    CUDA_CHECK(cudaMemcpy(dOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    fillNeighbors<<<blocks, BLOCK_SIZE>>>(dPoints.data, n, threshold, rank, ranks, localCount, dOffsets.data, neighbors.data);
    CUDA_CHECK(cudaGetLastError());
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        growCandidates<<<blocks, BLOCK_SIZE>>>(dPoints.data, threshold, rank, ranks, localCount,
            dOffsets.data, neighbors.data, active.data, maxima.data, members.data, sizes.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hostSizes.data(), sizes.data, size_t(localCount) * sizeof(int), cudaMemcpyDeviceToHost));
        unsigned long long localBest = 0, globalBest = 0;
        #pragma omp parallel for reduction(max:localBest) schedule(static) num_threads(hostThreads)
        for (int row = 0; row < localCount; ++row) {
            if (hostSizes[row]) {
                const unsigned int seed = rank + row * ranks;
                const unsigned long long key = (static_cast<unsigned long long>(hostSizes[row]) << 32) |
                    (UINT_MAX - seed);
                localBest = std::max(localBest, key);
            }
        }
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int count = static_cast<int>(globalBest >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(globalBest));
        if (!count) MPI_Abort(MPI_COMM_WORLD, 1);
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(count);
        const int owner = seed % ranks;
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(cluster.members.data(), members.data + offsets[seed / ranks],
                size_t(count) * sizeof(int), cudaMemcpyDeviceToHost));
        MPI_Bcast(cluster.members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(removed.data, cluster.members.data(), size_t(count) * sizeof(int), cudaMemcpyHostToDevice));
        removeMembers<<<std::min(65535, (count - 1) / BLOCK_SIZE + 1), BLOCK_SIZE>>>(active.data, removed.data, count);
        CUDA_CHECK(cudaGetLastError());
        remaining -= count;
        clusters.push_back(std::move(cluster));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& members = clusters[c].members;
        double diameter = 0.0;
        for (size_t i = 0; i < members.size(); ++i)
            for (size_t j = i + 1; j < members.size(); ++j)
                diameter = std::max(diameter, distance(points[members[i]], points[members[j]]));
        diameters[c] = diameter;
    }
    
    // Check each cluster
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
    if (clustered_count != static_cast<int>(points.size())) valid = false;
    
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
    
    // Local rank, rather than global rank, selects the GPU on each node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype pointType;
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must be packed");
    MPI_Type_contiguous(2, MPI_DOUBLE, &pointType);
    MPI_Type_commit(&pointType);
    MPI_Bcast(points.data(), num_points, pointType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pointType);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
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
