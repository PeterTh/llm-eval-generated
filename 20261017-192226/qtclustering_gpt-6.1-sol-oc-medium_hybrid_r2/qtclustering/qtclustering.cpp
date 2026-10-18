// QT Clustering Benchmark - MPI / OpenMP / CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <cfloat>
#include <cstdint>
#include <cuda_runtime.h>
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

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) MPI_Abort(MPI_COMM_WORLD, 1);
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// All MPI calls stay on the initial thread (MPI_THREAD_FUNNELED).
struct ParallelRuntime {
    int rank = 0, ranks = 1;
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int local_rank, devices;
        MPI_Comm_rank(local, &local_rank);
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Also supports launchers that expose only the assigned GPU to each rank.
        CUDA_CHECK(cudaSetDevice(local_rank % devices));
        CUDA_CHECK(cudaFree(nullptr));
        MPI_Comm_free(&local);
    }
    ~ParallelRuntime() { MPI_Finalize(); }
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

constexpr int BLOCK_SIZE = 256;

__device__ double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    // Do not replace with squared distances: rounding can change greedy ties.
    return sqrt(dx * dx + dy * dy);
}

// Store only points within the seed's threshold ball. Every possible member
// must lie in this ball, so the sparse representation is exact.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int* counts) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    __shared__ int sums[BLOCK_SIZE];
    int count = 0;
    for (size_t j = threadIdx.x; j < static_cast<size_t>(n); j += blockDim.x)
        count += (j == static_cast<size_t>(seed) ||
                  deviceDistance(points[seed], points[j]) < threshold);
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
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
    // One shared atomic per warp of neighbors rather than per point.
    for (size_t base = 0; base < static_cast<size_t>(n); base += blockDim.x) {
        const size_t j = base + threadIdx.x;
        const bool keep = j < static_cast<size_t>(n) &&
            (j == static_cast<size_t>(seed) || deviceDistance(points[seed], points[j]) < threshold);
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        const unsigned lane = threadIdx.x % 32;
        int start = 0;
        if (!lane) start = atomicAdd(&cursor, __popc(mask));
        start = __shfl_sync(0xffffffffu, start, 0);
        if (keep) neighbors[offsets[row] + start + __popc(mask & ((1u << lane) - 1))] = j;
    }
}

// One cooperative block per seed. Incremental maxima reduce a candidate's
// construction from O(n*k*k) distances to O(degree*k). The original serial
// order is reproduced with lexicographic (distance, point index) reductions.
__global__ void evaluateCandidates(const Point* points, double threshold,
                                   int rank, int ranks, const size_t* offsets,
                                   const int* neighbors, const unsigned char* clustered,
                                   double* maxima, int* members, int* cardinalities) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    const size_t begin = offsets[row], end = offsets[row + 1];
    if (clustered[seed]) {
        if (!threadIdx.x) cardinalities[row] = 0;
        return;
    }
    __shared__ int invalid, last, length;
    __shared__ double best_dist[BLOCK_SIZE];
    __shared__ int best_point[BLOCK_SIZE];
    if (!threadIdx.x) invalid = cardinalities[row] == 0;
    __syncthreads();
    for (int j = threadIdx.x; j < cardinalities[row]; j += blockDim.x)
        if (clustered[members[begin + j]]) atomicExch(&invalid, 1);
    __syncthreads();
    // Removing points outside this candidate cannot change its greedy path.
    if (!invalid) return;
    for (size_t j = begin + threadIdx.x; j < end; j += blockDim.x)
        maxima[j] = (clustered[neighbors[j]] || neighbors[j] == seed) ? DBL_MAX : 0.0;
    if (!threadIdx.x) {
        last = seed;
        length = 1;
        members[begin] = seed;
    }
    __syncthreads();
    while (true) {
        double local_dist = DBL_MAX;
        int local_point = INT_MAX;
        for (size_t j = begin + threadIdx.x; j < end; j += blockDim.x) {
            const int point = neighbors[j];
            double value = maxima[j];
            if (point == last) value = DBL_MAX;
            if (value < DBL_MAX && value < threshold) {
                value = fmax(value, deviceDistance(points[point], points[last]));
                if (value < threshold &&
                    (value < local_dist || (value == local_dist && point < local_point))) {
                    local_dist = value;
                    local_point = point;
                }
            }
            maxima[j] = value;
        }
        best_dist[threadIdx.x] = local_dist;
        best_point[threadIdx.x] = local_point;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                const int other = threadIdx.x + stride;
                if (best_dist[other] < best_dist[threadIdx.x] ||
                    (best_dist[other] == best_dist[threadIdx.x] &&
                     best_point[other] < best_point[threadIdx.x])) {
                    best_dist[threadIdx.x] = best_dist[other];
                    best_point[threadIdx.x] = best_point[other];
                }
            }
            __syncthreads();
        }
        if (best_point[0] == INT_MAX) break;
        if (!threadIdx.x) {
            last = best_point[0];
            members[begin + length++] = last;
        }
        __syncthreads();
    }
    if (!threadIdx.x) cardinalities[row] = length;
}

__global__ void markMembers(unsigned char* clustered, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = n / ranks + (rank < n % ranks);
    // Avoid launching a large host team for tiny local partitions.
    const int host_threads = std::min(omp_get_max_threads(), std::max(1, rows / 256));
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned char> d_clustered(n);
    DeviceBuffer<int> d_counts(rows), d_winner(n);
    DeviceBuffer<size_t> d_offsets(static_cast<size_t>(rows) + 1);
    CUDA_CHECK(cudaMemcpy(d_points.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered.data, 0, n));
    std::vector<int> counts(rows);
    std::vector<size_t> offsets(static_cast<size_t>(rows) + 1, 0);
    if (rows) {
        countNeighbors<<<rows, BLOCK_SIZE>>>(d_points.data, n, threshold, rank, ranks, d_counts.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(counts.data(), d_counts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < rows; ++i) offsets[i + 1] = offsets[i] + counts[i];
    CUDA_CHECK(cudaMemcpy(d_offsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    DeviceBuffer<int> d_neighbors(offsets.back()), d_members(offsets.back());
    DeviceBuffer<double> d_maxima(offsets.back());
    if (rows) {
        fillNeighbors<<<rows, BLOCK_SIZE>>>(d_points.data, n, threshold, rank, ranks,
                                          d_offsets.data, d_neighbors.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemset(d_counts.data, 0, rows * sizeof(int)));
    }
    std::vector<int> seeds(rows);
    #pragma omp parallel for schedule(static) num_threads(host_threads)
    for (int i = 0; i < rows; ++i) seeds[i] = rank + i * ranks;
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        if (rows) {
            evaluateCandidates<<<rows, BLOCK_SIZE>>>(d_points.data, threshold, rank, ranks,
                d_offsets.data, d_neighbors.data, d_clustered.data, d_maxima.data,
                d_members.data, d_counts.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(counts.data(), d_counts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // Cardinality dominates, then the smallest seed wins, exactly as in
        // the ascending sequential seed traversal. One scalar winner reduction.
        unsigned long long best = 0, global_best = 0;
        #pragma omp parallel for reduction(max:best) schedule(static) num_threads(host_threads)
        for (int i = 0; i < rows; ++i) {
            if (counts[i]) {
                const auto score = (static_cast<unsigned long long>(counts[i]) << 32) |
                                   (static_cast<unsigned>(UINT_MAX) - static_cast<unsigned>(seeds[i]));
                best = std::max(best, score);
            }
        }
        MPI_Allreduce(&best, &global_best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int count = global_best >> 32;
        const int seed = UINT_MAX - static_cast<unsigned>(global_best);
        if (!count) MPI_Abort(MPI_COMM_WORLD, 1);
        const int owner = seed % ranks;
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(count);
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(cluster.members.data(), d_members.data + offsets[seed / ranks],
                                  count * sizeof(int), cudaMemcpyDeviceToHost));
        MPI_Bcast(cluster.members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_winner.data, cluster.members.data(), count * sizeof(int), cudaMemcpyHostToDevice));
        markMembers<<<(count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_clustered.data, d_winner.data, count);
        CUDA_CHECK(cudaGetLastError());
        remaining -= count;
        if (!rank) clusters.push_back(std::move(cluster));
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
    ParallelRuntime runtime(argc, argv);
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
            if (!runtime.rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!runtime.rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (!runtime.rank) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (!runtime.rank) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (!runtime.rank) generateSyntheticData(points, num_points);
    // Chunk broadcasts to keep MPI's signed count valid for large inputs.
    for (size_t first = 0; first < points.size();) {
        const int count = static_cast<int>(std::min(points.size() - first,
                                                   static_cast<size_t>(INT_MAX / 2)));
        MPI_Bcast(&points[first].x, count * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        first += count;
    }
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (runtime.rank) return 0;
    
    printf("Clustering time: %.0f ms\n", time_sec * 1000.0);
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
