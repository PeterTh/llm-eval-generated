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
#include <limits>
#include <vector>
#include <cstdint>
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

// MPI calls are confined to the main thread (MPI_THREAD_FUNNELED).
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    // Compile without contraction/fast math to preserve strict threshold tests.
    return sqrt(dx * dx + dy * dy);
}

constexpr int THREADS = 128;

// A seed's threshold neighbors are a superset of every cluster it can grow.
// Only local seeds' rows are stored, distributing both memory and computation.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int rows, int* counts) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int seed = rank + row * ranks;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (j != seed && deviceDistance(points[seed], points[j]) < threshold) ++count;
    __shared__ int sums[THREADS];
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[row] = sums[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, const size_t* offsets, int* neighbors) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    __shared__ int cursor;
    if (threadIdx.x == 0) cursor = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j != seed && deviceDistance(points[seed], points[j]) < threshold) {
            int slot = atomicAdd(&cursor, 1);
            neighbors[offsets[row] + slot] = j;
        }
    }
}

// One block grows one cluster. Each candidate stores its running maximum
// distance; adding a member therefore needs only one new distance per candidate.
// Rejected points cannot become eligible again, since maxima only increase.
__global__ void growClusters(const Point* points, const unsigned char* clustered,
                             const int* activeRows, int rank, int ranks,
                             const size_t* offsets, const int* neighbors,
                             double* maxima, double threshold,
                             unsigned long long* best, int* cachedSizes, int* cachedMembers) {
    const int row = activeRows[blockIdx.x], seed = rank + row * ranks;
    const size_t begin = offsets[row], end = offsets[row + 1];
    __shared__ double values[THREADS / 32];
    __shared__ int indices[THREADS / 32];
    const size_t memberBegin = begin + row;
    __shared__ int last, cardinality, invalid;
    if (threadIdx.x == 0) invalid = cachedSizes[row] == 0;
    __syncthreads();
    // Removing points outside this candidate leaves every greedy choice intact.
    // Reuse it until at least one of its members is selected by another seed.
    for (int i = threadIdx.x; i < cachedSizes[row]; i += THREADS)
        if (clustered[cachedMembers[memberBegin + i]]) atomicExch(&invalid, 1);
    __syncthreads();
    if (!invalid) {
        if (threadIdx.x == 0) {
            const unsigned long long key = (static_cast<unsigned long long>(cachedSizes[row]) << 32)
                                          | (UINT_MAX - static_cast<unsigned int>(seed));
            atomicMax(best, key);
        }
        return;
    }
    if (threadIdx.x == 0) {
        last = seed;
        cardinality = 1;
        cachedMembers[memberBegin] = seed;
    }
    for (size_t e = begin + threadIdx.x; e < end; e += THREADS)
        maxima[e] = clustered[neighbors[e]] ? DBL_MAX : 0.0;
    __syncthreads();
    while (true) {
        double value = DBL_MAX;
        int index = INT_MAX;
        for (size_t e = begin + threadIdx.x; e < end; e += THREADS) {
            const int candidate = neighbors[e];
            double m = maxima[e];
            if (m == DBL_MAX) continue;
            if (candidate == last) {
                maxima[e] = DBL_MAX;
                continue;
            }
            m = fmax(m, deviceDistance(points[candidate], points[last]));
            maxima[e] = m < threshold ? m : DBL_MAX;
            if (m < threshold && (m < value || (m == value && candidate < index))) {
                value = m;
                index = candidate;
            }
        }
        // Reduce within warps first, limiting block-wide barriers to two.
        for (int stride = 16; stride; stride /= 2) {
            const double other = __shfl_down_sync(0xffffffff, value, stride);
            const int otherIndex = __shfl_down_sync(0xffffffff, index, stride);
            if (other < value || (other == value && otherIndex < index)) {
                value = other;
                index = otherIndex;
            }
        }
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        if (lane == 0) { values[warp] = value; indices[warp] = index; }
        __syncthreads();
        if (warp == 0) {
            value = lane < THREADS / 32 ? values[lane] : DBL_MAX;
            index = lane < THREADS / 32 ? indices[lane] : INT_MAX;
            for (int stride = 16; stride; stride /= 2) {
                const double other = __shfl_down_sync(0xffffffff, value, stride);
                const int otherIndex = __shfl_down_sync(0xffffffff, index, stride);
                if (other < value || (other == value && otherIndex < index)) {
                    value = other;
                    index = otherIndex;
                }
            }
            if (lane == 0) indices[0] = index;
        }
        __syncthreads();
        if (indices[0] == INT_MAX) break;
        if (threadIdx.x == 0) {
            last = indices[0];
            cachedMembers[memberBegin + cardinality] = last;
            ++cardinality;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        cachedSizes[row] = cardinality;
        // Largest cluster, then lowest original seed index (sequential tie rule).
        const unsigned long long key = (static_cast<unsigned long long>(cardinality) << 32)
                                      | (UINT_MAX - static_cast<unsigned int>(seed));
        atomicMax(best, key);
    }
}

__global__ void markCluster(unsigned char* clustered, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = rank < n ? 1 + (n - 1 - rank) / ranks : 0;
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<unsigned char> deviceClustered(n);
    DeviceBuffer<int> counts(rows), deviceRows(rows ? rows : 1), deviceMembers(n);
    DeviceBuffer<size_t> deviceOffsets(static_cast<size_t>(rows) + 1);
    DeviceBuffer<unsigned long long> deviceBest(1);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(deviceClustered.data, 0, n));
    if (rows) {
        countNeighbors<<<rows, THREADS>>>(devicePoints.data, n, threshold, rank, ranks, rows, counts.data);
        cudaCheck(cudaGetLastError());
    }
    std::vector<int> hostCounts(rows), activeRows(rows);
    std::vector<size_t> offsets(static_cast<size_t>(rows) + 1, 0);
    if (rows) cudaCheck(cudaMemcpy(hostCounts.data(), counts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
    // Parallel exclusive prefix sum, with one short prefix over thread totals.
    // GPU work dominates; a small host team avoids oversubscribing CPUs when
    // multiple GPU ranks share a node, especially for short prefix scans.
    const int hostThreads = std::min(omp_get_max_threads(), std::min(8, 1 + rows / 256));
    std::vector<size_t> totals(static_cast<size_t>(hostThreads) + 1, 0);
    #pragma omp parallel num_threads(hostThreads)
    {
        const int t = omp_get_thread_num(), threads = omp_get_num_threads();
        const int first = static_cast<int>(static_cast<int64_t>(rows) * t / threads);
        const int limit = static_cast<int>(static_cast<int64_t>(rows) * (t + 1) / threads);
        size_t sum = 0;
        for (int i = first; i < limit; ++i) { offsets[i] = sum; sum += hostCounts[i]; }
        totals[t + 1] = sum;
        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= threads; ++i) totals[i] += totals[i - 1];
            offsets[rows] = totals[threads];
        }
        for (int i = first; i < limit; ++i) offsets[i] += totals[t];
    }
    cudaCheck(cudaMemcpy(deviceOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    DeviceBuffer<int> neighbors(offsets.back());
    DeviceBuffer<double> maxima(offsets.back());
    DeviceBuffer<int> cachedSizes(rows), cachedMembers(offsets.back() + rows);
    if (rows) cudaCheck(cudaMemset(cachedSizes.data, 0, rows * sizeof(int)));
    if (rows) {
        fillNeighbors<<<rows, THREADS>>>(devicePoints.data, n, threshold, rank, ranks, deviceOffsets.data, neighbors.data);
        cudaCheck(cudaGetLastError());
    }
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        int activeCount = 0;
        for (int row = 0; row < rows; ++row)
            if (!clustered[rank + row * ranks]) activeRows[activeCount++] = row;
        cudaCheck(cudaMemset(deviceBest.data, 0, sizeof(unsigned long long)));
        if (activeCount) {
            cudaCheck(cudaMemcpy(deviceRows.data, activeRows.data(), activeCount * sizeof(int), cudaMemcpyHostToDevice));
            growClusters<<<activeCount, THREADS>>>(devicePoints.data, deviceClustered.data,
                deviceRows.data, rank, ranks, deviceOffsets.data, neighbors.data, maxima.data,
                threshold, deviceBest.data, cachedSizes.data, cachedMembers.data);
            cudaCheck(cudaGetLastError());
        }
        unsigned long long localBest = 0, globalBest = 0;
        cudaCheck(cudaMemcpy(&localBest, deviceBest.data, sizeof(localBest), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int cardinality = static_cast<int>(globalBest >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(globalBest));
        if (cardinality == 1) {
            // All remaining candidates are singletons; sequential seed order
            // determines the rest without additional global synchronization.
            if (rank == 0)
                for (int i = 0; i < n; ++i)
                    if (!clustered[i]) clusters.push_back({{i}, i});
            break;
        }
        const int owner = seed % ranks;
        std::vector<int> members(cardinality);
        if (rank == owner) {
            const int row = seed / ranks;
            cudaCheck(cudaMemcpy(members.data(), cachedMembers.data + offsets[row] + row,
                                 cardinality * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(deviceMembers.data, members.data(), cardinality * sizeof(int), cudaMemcpyHostToDevice));
        markCluster<<<(cardinality + THREADS - 1) / THREADS, THREADS>>>(deviceClustered.data, deviceMembers.data, cardinality);
        cudaCheck(cudaGetLastError());
        #pragma omp parallel for schedule(static) num_threads(hostThreads) if(cardinality > 1024)
        for (int i = 0; i < cardinality; ++i) clustered[members[i]] = 1;
        remaining -= cardinality;
        if (rank == 0) clusters.push_back({std::move(members), seed});
    }
    cudaCheck(cudaDeviceSynchronize());
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic) num_threads(std::min(omp_get_max_threads(), 8))
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
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    ~MpiSession() { MPI_Finalize(); }
};

int main(int argc, char** argv) {
    MpiSession session(argc, argv);
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
    
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    int devices;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "A CUDA device is required on every MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    // Chunk broadcasts to avoid MPI int-count overflow.
    for (size_t offset = 0; offset < points.size() * sizeof(Point);) {
        const int bytes = static_cast<int>(std::min<size_t>(INT_MAX, points.size() * sizeof(Point) - offset));
        MPI_Bcast(reinterpret_cast<char*>(points.data()) + offset, bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
        offset += bytes;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_time_ms = 0;
    MPI_Reduce(&local_time_ms, &max_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;

    auto cluster_time = std::chrono::milliseconds(max_time_ms);
    
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
