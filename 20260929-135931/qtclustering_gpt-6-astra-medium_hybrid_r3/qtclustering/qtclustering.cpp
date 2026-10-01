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
        
        // For these sizes the original truncated count is always zero.
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

// MPI is confined to the main thread. Every failure aborts the communicator,
// so another rank cannot hang in a collective after a device failure.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
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

constexpr int BLOCK = 128;

__device__ double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// A candidate can only contain points strictly within the seed's threshold.
// Store only these neighbors, distributed cyclically by seed over MPI ranks.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int* counts) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    __shared__ int sums[BLOCK];
    int count = 0;
    for (int j = threadIdx.x; j < n; j += BLOCK)
        count += j == seed || deviceDistance(points[seed], points[j]) < threshold;
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int step = BLOCK / 2; step; step /= 2) {
        if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x + step];
        __syncthreads();
    }
    if (!threadIdx.x) counts[row] = sums[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, const size_t* offsets, int* neighbors) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    __shared__ int cursor;
    if (!threadIdx.x) cursor = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += BLOCK) {
        if (j == seed || deviceDistance(points[seed], points[j]) < threshold)
            neighbors[offsets[row] + atomicAdd(&cursor, 1)] = j;
    }
}

// One cooperative block per candidate. Incrementally maintaining the maximum
// distance removes the original repeated scan over all previously added members.
// Index-aware reductions preserve the sequential lowest-index tie break even
// though the neighbor lists are stored in arbitrary order.
__global__ void growCandidates(const Point* points, double threshold,
                               int rank, int ranks, const unsigned char* clustered,
                               const size_t* offsets, const int* neighbors,
                               double* maxima, int* members, int* cardinalities) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    const int tid = threadIdx.x;
    const size_t begin = offsets[row], end = offsets[row + 1];
    __shared__ double values[BLOCK];
    __shared__ int indices[BLOCK];
    __shared__ int dirty, last, count;
    if (clustered[seed]) {
        if (!tid) cardinalities[row] = 0;
        return;
    }
    if (!tid) dirty = cardinalities[row] == 0;
    __syncthreads();
    // Removing points outside an existing candidate cannot change its greedy
    // choices. Such candidates need no recomputation.
    for (int j = tid; j < cardinalities[row]; j += BLOCK)
        if (clustered[members[begin + j]]) atomicExch(&dirty, 1);
    __syncthreads();
    if (!dirty) return;
    for (size_t j = begin + tid; j < end; j += BLOCK)
        maxima[j] = (clustered[neighbors[j]] || neighbors[j] == seed) ? CUDART_INF : 0.0;
    if (!tid) { last = seed; count = 1; members[begin] = seed; }
    __syncthreads();
    while (true) {
        double best = CUDART_INF;
        int index = INT_MAX;
        for (size_t j = begin + tid; j < end; j += BLOCK) {
            double value = maxima[j];
            if (value == CUDART_INF) continue;
            const int candidate = neighbors[j];
            if (candidate == last) { maxima[j] = CUDART_INF; continue; }
            value = fmax(value, deviceDistance(points[candidate], points[last]));
            // A rejected point can never become eligible as the cluster grows.
            maxima[j] = value < threshold ? value : CUDART_INF;
            if (value < threshold && (value < best || (value == best && candidate < index))) {
                best = value; index = candidate;
            }
        }
        values[tid] = best; indices[tid] = index;
        __syncthreads();
        for (int step = BLOCK / 2; step; step /= 2) {
            if (tid < step && (values[tid + step] < values[tid] ||
                (values[tid + step] == values[tid] && indices[tid + step] < indices[tid]))) {
                values[tid] = values[tid + step]; indices[tid] = indices[tid + step];
            }
            __syncthreads();
        }
        if (indices[0] == INT_MAX) break;
        if (!tid) { last = indices[0]; members[begin + count++] = last; }
        __syncthreads();
    }
    if (!tid) cardinalities[row] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = n / ranks + (rank < n % ranks);
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned char> d_clustered(n);
    DeviceBuffer<int> d_counts(rows);
    DeviceBuffer<size_t> d_offsets(static_cast<size_t>(rows) + 1);
    cudaCheck(cudaMemcpy(d_points.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_clustered.data, 0, n));
    std::vector<int> counts(rows);
    std::vector<size_t> offsets(static_cast<size_t>(rows) + 1, 0);
    if (rows) {
        countNeighbors<<<rows, BLOCK>>>(d_points.data, n, threshold, rank, ranks, d_counts.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(counts.data(), d_counts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < rows; ++i) offsets[i + 1] = offsets[i] + counts[i];
    DeviceBuffer<int> d_neighbors(offsets.back()), d_members(offsets.back());
    DeviceBuffer<double> d_maxima(offsets.back());
    cudaCheck(cudaMemcpy(d_offsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (rows) {
        fillNeighbors<<<rows, BLOCK>>>(d_points.data, n, threshold, rank, ranks, d_offsets.data, d_neighbors.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemset(d_counts.data, 0, rows * sizeof(int)));
    }
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        if (rows) {
            growCandidates<<<rows, BLOCK>>>(d_points.data, threshold, rank, ranks,
                d_clustered.data, d_offsets.data, d_neighbors.data, d_maxima.data,
                d_members.data, d_counts.data);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(counts.data(), d_counts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // Packed key: largest cardinality wins; equal sizes choose lowest seed.
        unsigned long long local_best = 0, global_best = 0;
        #pragma omp parallel for reduction(max:local_best) schedule(static) if(rows > 512) \
            num_threads(std::min(omp_get_max_threads(), 1 + rows / 512))
        for (int i = 0; i < rows; ++i) {
            if (counts[i]) {
                const unsigned int seed = rank + i * ranks;
                const unsigned long long key = (static_cast<unsigned long long>(counts[i]) << 32) |
                                               (UINT_MAX - seed);
                local_best = std::max(local_best, key);
            }
        }
        MPI_Allreduce(&local_best, &global_best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int count = static_cast<int>(global_best >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(global_best));
        const int owner = seed % ranks;
        std::vector<int> members(count);
        if (rank == owner)
            cudaCheck(cudaMemcpy(members.data(), d_members.data + offsets[seed / ranks],
                                 count * sizeof(int), cudaMemcpyDeviceToHost));
        MPI_Bcast(members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static) if(count > 512) \
            num_threads(std::min(omp_get_max_threads(), 1 + count / 512))
        for (int i = 0; i < count; ++i) clustered[members[i]] = 1;
        cudaCheck(cudaMemcpy(d_clustered.data, clustered.data(), n, cudaMemcpyHostToDevice));
        if (!rank) clusters.push_back({std::move(members), seed});
        remaining -= count;
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
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalize;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    // Respect scheduler GPU visibility; otherwise assign by node-local rank.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank = 0, devices = 0;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        if (!rank) fprintf(stderr, "QT clustering requires a CUDA GPU on each MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(local_rank % devices));
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
    
    if (!rank) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (!rank) generateSyntheticData(points, num_points);
    // Point contains exactly two doubles. Broadcast in chunks to avoid MPI's
    // signed-int count limit on large inputs.
    for (size_t first = 0; first < points.size();) {
        const int count = static_cast<int>(std::min(points.size() - first, size_t(INT_MAX / 2)));
        MPI_Bcast(&points[first].x, 2 * count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        first += count;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank) return 0;
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
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
