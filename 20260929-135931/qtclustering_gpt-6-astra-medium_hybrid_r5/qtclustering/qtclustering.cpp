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
#include <utility>
#include <vector>
#include <climits>
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mpi.h>
#include <omp.h>

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
        
        if (N <= 30) group_cnt = 1; // Ensure progress for small inputs.

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

// All ranks participate in the algorithm; only rank zero emits results.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) MPI_Abort(MPI_COMM_WORLD, 2);
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

static constexpr int BLOCK = 128;

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Build with --fmad=false: rounding and strict threshold match the CPU.
    return sqrt(dx * dx + dy * dy);
}

// A seed can only admit points within threshold of itself. Store just these
// neighborhoods, distributed cyclically across ranks for balanced seed work.
__global__ void neighborhoods(const Point* points, int n, double threshold,
                              int rank, int ranks, int rows, int* counts,
                              const size_t* offsets, int* neighbors) {
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int seed = rank + row * ranks;
        __shared__ int count;
        if (threadIdx.x == 0) count = 0;
        __syncthreads();
        for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
            if (j != seed && pointDistance(points[seed], points[j]) < threshold) {
                const int slot = atomicAdd(&count, 1);
                if (neighbors) neighbors[offsets[row] + slot] = static_cast<int>(j);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) counts[row] = count;
        __syncthreads();
    }
}

__device__ void warpNearest(double& value, int& point) {
    for (int offset = 16; offset; offset /= 2) {
        const double otherValue = __shfl_down_sync(0xffffffffu, value, offset);
        const int otherPoint = __shfl_down_sync(0xffffffffu, point, offset);
        if (otherValue < value || (otherValue == value && otherPoint < point)) {
            value = otherValue;
            point = otherPoint;
        }
    }
}

// Each block grows one cluster. Maintain each candidate's maximum distance
// incrementally, instead of rescanning all previous members at every step.
// Removing points outside the cached cluster cannot affect its greedy choices.
__global__ void candidates(const Point* points, double threshold,
                           int rank, int ranks, int rows, const int* counts,
                           const size_t* offsets, const int* neighbors,
                           const unsigned char* clustered, double* maxima,
                           int* members, int* sizes) {
    __shared__ double bestDistance[BLOCK / 32];
    __shared__ int bestPoint[BLOCK / 32];
    __shared__ int selected;
    __shared__ int dirty;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int seed = rank + row * ranks;
        if (clustered[seed]) {
            if (threadIdx.x == 0) sizes[row] = 0;
            continue;
        }
        const size_t start = offsets[row];
        const int length = counts[row];
        const int oldSize = sizes[row];
        if (threadIdx.x == 0) dirty = (oldSize == 0);
        __syncthreads();
        for (int k = threadIdx.x; k < oldSize; k += BLOCK) {
            if (clustered[members[start + k]]) atomicExch(&dirty, 1);
        }
        __syncthreads();
        const int rebuild = dirty;
        __syncthreads();
        if (!rebuild) continue;
        for (int k = threadIdx.x; k < length; k += BLOCK) {
            maxima[start + k] = clustered[neighbors[start + k]] ? CUDART_INF : 0.0;
        }
        if (threadIdx.x == 0) members[start] = seed;
        int last = seed;
        int size = 1;
        __syncthreads();
        while (true) {
            double nearest = CUDART_INF;
            int next = INT_MAX;
            for (int k = threadIdx.x; k < length; k += BLOCK) {
                const size_t pos = start + k;
                double value = maxima[pos];
                if (value == CUDART_INF) continue;
                const int point = neighbors[pos];
                if (point == last) value = CUDART_INF;
                else {
                    value = fmax(value, pointDistance(points[point], points[last]));
                    if (!(value < threshold)) value = CUDART_INF;
                }
                maxima[pos] = value;
                if (value < nearest || (value == nearest && value != CUDART_INF && point < next)) {
                    nearest = value;
                    next = point;
                }
            }
            warpNearest(nearest, next);
            const int lane = threadIdx.x % 32;
            const int warp = threadIdx.x / 32;
            if (lane == 0) {
                bestDistance[warp] = nearest;
                bestPoint[warp] = next;
            }
            __syncthreads();
            if (warp == 0) {
                nearest = lane < BLOCK / 32 ? bestDistance[lane] : CUDART_INF;
                next = lane < BLOCK / 32 ? bestPoint[lane] : INT_MAX;
                warpNearest(nearest, next);
                if (lane == 0) selected = next;
            }
            __syncthreads();
            last = selected;
            if (last == INT_MAX) break;
            if (threadIdx.x == 0) members[start + size] = last;
            ++size;
            // All threads must consume the reduction before reusing shared memory.
            __syncthreads();
        }
        if (threadIdx.x == 0) sizes[row] = size;
        __syncthreads();
    }
}

__global__ void removeMembers(unsigned char* clustered, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = n / ranks + (rank < n % ranks);
    const int blocks = std::min(rows, 65535);
    DeviceBuffer<Point> dPoints(n);
    DeviceBuffer<unsigned char> dClustered(n);
    DeviceBuffer<int> dCounts(rows), dSizes(rows), dWinner(n);
    DeviceBuffer<size_t> dOffsets(static_cast<size_t>(rows) + 1);
    CUDA_CHECK(cudaMemcpy(dPoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dClustered.data, 0, n));
    std::vector<int> counts(rows), sizes(rows);
    std::vector<size_t> offsets(static_cast<size_t>(rows) + 1, 0);
    if (rows) {
        CUDA_CHECK(cudaMemset(dSizes.data, 0, rows * sizeof(int)));
        neighborhoods<<<blocks, BLOCK>>>(dPoints.data, n, threshold, rank, ranks,
                                        rows, dCounts.data, nullptr, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(counts.data(), dCounts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
    }
    // One extra slot per row accommodates the seed in the cached member list.
    for (int row = 0; row < rows; ++row) offsets[row + 1] = offsets[row] + counts[row] + 1;
    DeviceBuffer<int> dNeighbors(offsets.back()), dMembers(offsets.back());
    DeviceBuffer<double> dMaxima(offsets.back());
    CUDA_CHECK(cudaMemcpy(dOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (rows) {
        neighborhoods<<<blocks, BLOCK>>>(dPoints.data, n, threshold, rank, ranks,
                                        rows, dCounts.data, dOffsets.data, dNeighbors.data);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        if (rows) {
            candidates<<<blocks, BLOCK>>>(dPoints.data, threshold, rank, ranks, rows,
                dCounts.data, dOffsets.data, dNeighbors.data, dClustered.data,
                dMaxima.data, dMembers.data, dSizes.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), dSizes.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // Lexicographic (largest size, smallest seed), exactly as the original
        // ascending seed loop. One scalar all-reduce selects the global winner.
        unsigned long long localBest = 0, globalBest = 0;
        #pragma omp parallel for reduction(max:localBest) schedule(static)
        for (int row = 0; row < rows; ++row) {
            if (sizes[row]) {
                const unsigned int seed = rank + row * ranks;
                const unsigned long long key = (static_cast<unsigned long long>(sizes[row]) << 32)
                                             | (UINT_MAX - seed);
                localBest = std::max(localBest, key);
            }
        }
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int size = static_cast<int>(globalBest >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(globalBest));
        if (!size) MPI_Abort(MPI_COMM_WORLD, 3);
        // Once the largest candidate is a singleton, every remaining point
        // must be emitted individually in ascending seed order. Avoid one
        // kernel launch and two collectives per point in this common tail.
        if (size == 1) {
            if (rank == 0) {
                std::vector<unsigned char> clustered(n);
                CUDA_CHECK(cudaMemcpy(clustered.data(), dClustered.data, n, cudaMemcpyDeviceToHost));
                for (int point = 0; point < n; ++point) {
                    if (!clustered[point]) clusters.push_back({{point}, point});
                }
            }
            break;
        }
        const int owner = seed % ranks;
        std::vector<int> winner(size);
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(winner.data(), dMembers.data + offsets[seed / ranks],
                                  size * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(winner.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dWinner.data, winner.data(), size * sizeof(int), cudaMemcpyHostToDevice));
        removeMembers<<<std::min(65535, (size - 1) / BLOCK + 1), BLOCK>>>(dClustered.data, dWinner.data, size);
        CUDA_CHECK(cudaGetLastError());
        if (rank == 0) clusters.push_back({std::move(winner), seed});
        remaining -= size;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    // Finalize on every normal return, including usage and parameter errors.
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalizeMPI;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_max_threads()));
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
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        if (rank == 0) fprintf(stderr, "QT clustering requires a CUDA device on every rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    // Chunk broadcasts to avoid MPI's signed-int count limit.
    for (size_t first = 0; first < points.size(); ) {
        const int count = static_cast<int>(std::min(points.size() - first,
                                                  static_cast<size_t>(INT_MAX / sizeof(Point))));
        MPI_Bcast(points.data() + first, count * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
        first += count;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);
    
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
