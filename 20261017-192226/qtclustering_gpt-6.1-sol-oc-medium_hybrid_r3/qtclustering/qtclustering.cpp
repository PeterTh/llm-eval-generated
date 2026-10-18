// QT Clustering Benchmark - MPI/OpenMP/CUDA implementation
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),
                              std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct MpiSession {
    int rank = 0;
    MpiSession(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    ~MpiSession() { MPI_Finalize(); }
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

constexpr int BLOCK_SIZE = 128;

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Only points within the seed's threshold can ever enter its candidate.
// Each rank stores only its cyclic subset of seed rows, in sparse CSR form.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int* counts) {
    const int row = blockIdx.x;
    const int seed = rank + row * ranks;
    int count = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x)
        count += (j == seed || pointDistance(points[seed], points[j]) < threshold);
    __shared__ int sums[BLOCK_SIZE];
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[row] = sums[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, const size_t* offsets, int* neighbors) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    __shared__ int next;
    if (threadIdx.x == 0) next = 0;
    __syncthreads();
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == seed || pointDistance(points[seed], points[j]) < threshold) {
            const int slot = atomicAdd(&next, 1);
            neighbors[offsets[row] + slot] = static_cast<int>(j);
        }
    }
}

// One cooperative block per seed. Maintain max distance incrementally rather
// than rescanning all prior members. Reductions explicitly break ties by point
// index, independent of thread order and of the sparse row's storage order.
__device__ void warpMinimum(double& best, int& index) {
    #pragma unroll
    for (int offset = 16; offset; offset /= 2) {
        const double other = __shfl_down_sync(0xffffffff, best, offset);
        const int p = __shfl_down_sync(0xffffffff, index, offset);
        if (other < best || (other == best && p < index)) {
            best = other;
            index = p;
        }
    }
}

__global__ void candidateClusters(const Point* points, double threshold,
                                  int rank, int ranks, const size_t* offsets,
                                  const int* neighbors, const unsigned char* clustered,
                                  double* maxima, int* members, int* counts) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    const int tid = threadIdx.x;
    if (clustered[seed]) {
        if (tid == 0) counts[row] = 0;
        return;
    }
    const size_t begin = offsets[row], end = offsets[row + 1];
    const int old_count = counts[row];
    int dirty = (old_count == 0);
    for (int64_t i = tid; i < old_count; i += BLOCK_SIZE)
        dirty |= clustered[members[begin + i]];
    // If no old member was removed, every greedy choice is still available,
    // and removing other points cannot change this candidate's construction.
    if (!__syncthreads_or(dirty)) return;

    for (size_t i = begin + tid; i < end; i += BLOCK_SIZE) {
        const int p = neighbors[i];
        maxima[i] = (clustered[p] || p == seed) ? CUDART_INF :
                    pointDistance(points[seed], points[p]);
    }
    if (tid == 0) members[begin] = seed;
    __syncthreads();

    __shared__ double distances[BLOCK_SIZE / 32];
    __shared__ int indices[BLOCK_SIZE / 32];
    int count = 1;
    while (true) {
        double best = CUDART_INF;
        int index = INT_MAX;
        for (size_t i = begin + tid; i < end; i += BLOCK_SIZE) {
            const double d = maxima[i];
            const int p = neighbors[i];
            if (d < threshold && (d < best || (d == best && p < index))) {
                best = d;
                index = p;
            }
        }
        warpMinimum(best, index);
        if ((tid % 32) == 0) {
            distances[tid / 32] = best;
            indices[tid / 32] = index;
        }
        __syncthreads();
        if (tid < 32) {
            best = tid < BLOCK_SIZE / 32 ? distances[tid] : CUDART_INF;
            index = tid < BLOCK_SIZE / 32 ? indices[tid] : INT_MAX;
            warpMinimum(best, index);
            if (tid == 0) indices[0] = index;
        }
        __syncthreads();
        const int chosen = indices[0];
        if (chosen == INT_MAX) break;
        if (tid == 0) members[begin + count] = chosen;
        ++count;
        for (size_t i = begin + tid; i < end; i += BLOCK_SIZE) {
            const int p = neighbors[i];
            if (p == chosen) maxima[i] = CUDART_INF;
            else if (maxima[i] < threshold)
                maxima[i] = fmax(maxima[i], pointDistance(points[p], points[chosen]));
        }
        __syncthreads();
    }
    if (tid == 0) counts[row] = count;
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        clustered[members[i]] = 1;
}

// Initialize the accelerator before the timed clustering region. Respect
// scheduler visibility masks and map node-local MPI ranks round-robin to GPUs.
void initializeAccelerator() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, devices;
    MPI_Comm_rank(local, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "QT clustering requires a CUDA device on every MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    CUDA_CHECK(cudaFree(nullptr));
    MPI_Comm_free(&local);
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = n / ranks + (rank < n % ranks);
    // These host operations are bandwidth-light; avoid oversubscribing nodes
    // with hundreds of OpenMP workers for a small local seed partition.
    const int host_threads = std::min(omp_get_max_threads(),
                                     std::min(8, std::max(1, (rows - 1) / 256 + 1)));
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<int> counts(rows), winner(n);
    DeviceBuffer<size_t> d_offsets(size_t(rows) + 1);
    CUDA_CHECK(cudaMemcpy(d_points.data, points.data(), size_t(n) * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(clustered.data, 0, n));
    std::vector<int> host_counts(rows);
    std::vector<size_t> offsets(size_t(rows) + 1, 0);
    if (rows) {
        countNeighbors<<<rows, BLOCK_SIZE>>>(d_points.data, n, threshold, rank, ranks, counts.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_counts.data(), counts.data, size_t(rows) * sizeof(int), cudaMemcpyDeviceToHost));
    }
    // Parallel prefix over chunks, avoiding an O(ranks * n^2) replicated graph.
    std::vector<size_t> chunk_sums(size_t(host_threads) + 1, 0);
    #pragma omp parallel num_threads(host_threads)
    {
        const int thread = omp_get_thread_num(), threads = omp_get_num_threads();
        const int first = static_cast<int>(int64_t(rows) * thread / threads);
        const int last = static_cast<int>(int64_t(rows) * (thread + 1) / threads);
        size_t sum = 0;
        for (int i = first; i < last; ++i) {
            sum += host_counts[i];
            offsets[i + 1] = sum;
        }
        chunk_sums[thread + 1] = sum;
        #pragma omp barrier
        #pragma omp single
        {
            for (int t = 1; t <= threads; ++t) chunk_sums[t] += chunk_sums[t - 1];
        }
        const size_t base = chunk_sums[thread];
        for (int i = first; i < last; ++i) offsets[i + 1] += base;
    }
    CUDA_CHECK(cudaMemcpy(d_offsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    DeviceBuffer<int> neighbors(offsets.back()), members(offsets.back());
    DeviceBuffer<double> maxima(offsets.back());
    if (rows) {
        fillNeighbors<<<rows, BLOCK_SIZE>>>(d_points.data, n, threshold, rank, ranks, d_offsets.data, neighbors.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemset(counts.data, 0, size_t(rows) * sizeof(int)));
    }

    std::vector<Cluster> clusters;
    std::vector<int> winning_members(n);
    int remaining = n;
    while (remaining) {
        if (rows) {
            candidateClusters<<<rows, BLOCK_SIZE>>>(d_points.data, threshold, rank, ranks,
                d_offsets.data, neighbors.data, clustered.data, maxima.data, members.data, counts.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host_counts.data(), counts.data, size_t(rows) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // Lexicographic maximum: largest cardinality, then smallest seed.
        unsigned long long best = 0, global_best = 0;
        #pragma omp parallel for num_threads(host_threads) reduction(max:best) schedule(static)
        for (int i = 0; i < rows; ++i) {
            if (host_counts[i]) {
                const unsigned int seed = rank + i * ranks;
                const unsigned long long key = (static_cast<unsigned long long>(host_counts[i]) << 32)
                                              | (UINT_MAX - seed);
                best = std::max(best, key);
            }
        }
        MPI_Allreduce(&best, &global_best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int count = static_cast<int>(global_best >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(global_best));
        if (count <= 0 || count > remaining) MPI_Abort(MPI_COMM_WORLD, 1);
        const int owner = seed % ranks;
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(winning_members.data(), members.data + offsets[seed / ranks],
                                  size_t(count) * sizeof(int), cudaMemcpyDeviceToHost));
        MPI_Bcast(winning_members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank == 0) clusters.push_back({std::vector<int>(winning_members.begin(), winning_members.begin() + count), seed});
        CUDA_CHECK(cudaMemcpy(winner.data, winning_members.data(), size_t(count) * sizeof(int), cudaMemcpyHostToDevice));
        markClustered<<<std::min(65535, (count - 1) / BLOCK_SIZE + 1), BLOCK_SIZE>>>(clustered.data, winner.data, count);
        CUDA_CHECK(cudaGetLastError());
        remaining -= count;
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
    MpiSession mpi(argc, argv);
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
            if (mpi.rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (mpi.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi.rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (mpi.rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (mpi.rank == 0) generateSyntheticData(points, num_points);
    // Chunk broadcasts to keep MPI's int count valid for large inputs.
    const size_t bytes = points.size() * sizeof(Point);
    for (size_t offset = 0; offset < bytes;) {
        const int count = static_cast<int>(std::min(size_t(INT_MAX), bytes - offset));
        MPI_Bcast(reinterpret_cast<unsigned char*>(points.data()) + offset,
                  count, MPI_BYTE, 0, MPI_COMM_WORLD);
        offset += count;
    }
    
    // Perform QT clustering
    initializeAccelerator();
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi.rank != 0) return 0;
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
