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
#include <utility>
#include <vector>
#include <climits>
#include <cfloat>
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
        // Otherwise the original generator cannot make progress for N <= 30.
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

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// All MPI calls are made by the main thread; early returns finalize MPI too.
struct MpiSession {
    int rank, size;
    MpiSession(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    ~MpiSession() { MPI_Finalize(); }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            MPI_Abort(MPI_COMM_WORLD, 1);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data),
                             std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK_SIZE = 128;

__device__ double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ void warpMinimum(double& value, int& index) {
    for (int step = 16; step; step /= 2) {
        const double other_value = __shfl_down_sync(0xffffffffu, value, step);
        const int other_index = __shfl_down_sync(0xffffffffu, index, step);
        if (other_value < value || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
}

// Only neighbors strictly within the seed's threshold can ever be selected.
// Each rank stores just its own rows of this sparse graph, not an N x N matrix.
template<bool Fill>
__global__ void buildNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int* counts,
                               const size_t* offsets, int* neighbors) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    for (size_t base = 0; base < static_cast<size_t>(n); base += blockDim.x) {
        const size_t j = base + threadIdx.x;
        const bool include = j < static_cast<size_t>(n) &&
            (j == static_cast<size_t>(seed) ||
             deviceDistance(points[seed], points[j]) < threshold);
        const unsigned mask = __ballot_sync(0xffffffffu, include);
        const int lane = threadIdx.x & 31;
        int start = 0;
        if (lane == 0) start = atomicAdd(&count, __popc(mask));
        start = __shfl_sync(0xffffffffu, start, 0);
        if (Fill && include) {
            const int position = start + __popc(mask & ((1u << lane) - 1u));
            neighbors[offsets[row] + position] = static_cast<int>(j);
        }
    }
    __syncthreads();
    if (!Fill && threadIdx.x == 0) counts[row] = count;
}

// A block cooperatively grows one seed's cluster. A candidate's maximum
// distance only needs updating against the most recently added member.
// Unchanged cached clusters remain exact: deleting points that were never
// selected cannot alter any of the greedy choices.
__global__ void candidateClusters(const Point* points, double threshold,
                                  int rank, int ranks, const size_t* offsets,
                                  const int* neighbors, const unsigned char* clustered,
                                  double* maxima, int* members, int* sizes,
                                  unsigned long long* best) {
    const int row = blockIdx.x, seed = rank + row * ranks;
    if (clustered[seed]) return;
    const size_t begin = offsets[row], end = offsets[row + 1];
    const int tid = threadIdx.x;
    __shared__ double distances[BLOCK_SIZE / 32];
    __shared__ int indices[BLOCK_SIZE / 32];
    __shared__ int dirty, selected, cardinality;
    if (tid == 0) dirty = sizes[row] == 0;
    __syncthreads();
    for (int i = tid; i < sizes[row]; i += blockDim.x)
        if (clustered[members[begin + i]]) atomicExch(&dirty, 1);
    __syncthreads();
    if (dirty) {
        for (size_t i = begin + tid; i < end; i += blockDim.x)
            maxima[i] = clustered[neighbors[i]] ? DBL_MAX : 0.0;
        if (tid == 0) {
            selected = seed;
            cardinality = 1;
            members[begin] = seed;
        }
        __syncthreads();
        while (true) {
            double minimum = DBL_MAX;
            int closest = INT_MAX;
            const Point last = points[selected];
            for (size_t i = begin + tid; i < end; i += blockDim.x) {
                const int candidate = neighbors[i];
                double maximum = maxima[i];
                if (candidate == selected) maximum = DBL_MAX;
                if (maximum != DBL_MAX) {
                    maximum = fmax(maximum, deviceDistance(points[candidate], last));
                    if (!(maximum < threshold)) maximum = DBL_MAX;
                }
                maxima[i] = maximum;
                if (maximum < minimum || (maximum == minimum && candidate < closest)) {
                    minimum = maximum;
                    closest = candidate;
                }
            }
            // Reduce within warps first, avoiding a block-wide barrier at
            // every level of the reduction for every greedy insertion.
            warpMinimum(minimum, closest);
            if ((tid & 31) == 0) {
                distances[tid / 32] = minimum;
                indices[tid / 32] = closest;
            }
            __syncthreads();
            if (tid < 32) {
                minimum = tid < BLOCK_SIZE / 32 ? distances[tid] : DBL_MAX;
                closest = tid < BLOCK_SIZE / 32 ? indices[tid] : INT_MAX;
                warpMinimum(minimum, closest);
                if (tid == 0) {
                    distances[0] = minimum;
                    indices[0] = closest;
                }
            }
            __syncthreads();
            if (distances[0] == DBL_MAX) break;
            if (tid == 0) {
                selected = indices[0];
                members[begin + cardinality++] = selected;
            }
            __syncthreads();
        }
        if (tid == 0) sizes[row] = cardinality;
    }
    if (tid == 0) {
        // Maximizing this key chooses largest size, then smallest seed.
        const unsigned long long key =
            (static_cast<unsigned long long>(sizes[row]) << 32) |
            (UINT_MAX - static_cast<unsigned>(seed));
        atomicMax(best, key);
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = rank < n ? 1 + (n - 1 - rank) / ranks : 0;
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned char> d_clustered(n);
    DeviceBuffer<int> d_counts(rows), d_sizes(rows), d_winner(n);
    DeviceBuffer<unsigned long long> d_best(1);
    cudaCheck(cudaMemcpy(d_points.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_clustered.data, 0, n));
    cudaCheck(cudaMemset(d_sizes.data, 0, rows * sizeof(int)));
    if (rows) {
        buildNeighbors<false><<<rows, BLOCK_SIZE>>>(d_points.data, n, threshold,
            rank, ranks, d_counts.data, nullptr, nullptr);
        cudaCheck(cudaGetLastError());
    }
    std::vector<int> counts(rows);
    std::vector<size_t> offsets(static_cast<size_t>(rows) + 1, 0);
    cudaCheck(cudaMemcpy(counts.data(), d_counts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> partial(static_cast<size_t>(omp_get_max_threads()) + 1, 0);
    // Parallel exclusive scan, with only the small per-thread scan serialized.
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num(), threads = omp_get_num_threads();
        const size_t first = static_cast<size_t>(rows) * tid / threads;
        const size_t last = static_cast<size_t>(rows) * (tid + 1) / threads;
        size_t sum = 0;
        for (size_t i = first; i < last; ++i) {
            offsets[i] = sum;
            sum += counts[i];
        }
        partial[tid + 1] = sum;
        #pragma omp barrier
        #pragma omp single
        {
            for (int t = 1; t <= threads; ++t) partial[t] += partial[t - 1];
            offsets[rows] = partial[threads];
        }
        for (size_t i = first; i < last; ++i) offsets[i] += partial[tid];
    }
    const size_t edges = offsets.back();
    DeviceBuffer<size_t> d_offsets(offsets.size());
    DeviceBuffer<int> d_neighbors(edges), d_members(edges);
    DeviceBuffer<double> d_maxima(edges);
    cudaCheck(cudaMemcpy(d_offsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (rows) {
        buildNeighbors<true><<<rows, BLOCK_SIZE>>>(d_points.data, n, threshold,
            rank, ranks, nullptr, d_offsets.data, d_neighbors.data);
        cudaCheck(cudaGetLastError());
    }
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        cudaCheck(cudaMemset(d_best.data, 0, sizeof(unsigned long long)));
        if (rows) {
            candidateClusters<<<rows, BLOCK_SIZE>>>(d_points.data, threshold, rank, ranks,
                d_offsets.data, d_neighbors.data, d_clustered.data, d_maxima.data,
                d_members.data, d_sizes.data, d_best.data);
            cudaCheck(cudaGetLastError());
        }
        unsigned long long local_best = 0, global_best = 0;
        cudaCheck(cudaMemcpy(&local_best, d_best.data, sizeof(local_best), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&local_best, &global_best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int count = static_cast<int>(global_best >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned>(global_best));
        if (count <= 0) MPI_Abort(MPI_COMM_WORLD, 1);
        const int owner = seed % ranks;
        Cluster winner;
        winner.seed_point = seed;
        winner.members.resize(count);
        if (rank == owner)
            cudaCheck(cudaMemcpy(winner.members.data(), d_members.data + offsets[seed / ranks],
                                 count * sizeof(int), cudaMemcpyDeviceToHost));
        MPI_Bcast(winner.members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(d_winner.data, winner.members.data(), count * sizeof(int), cudaMemcpyHostToDevice));
        markClustered<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_clustered.data, d_winner.data, count);
        cudaCheck(cudaGetLastError());
        remaining -= count;
        if (rank == 0) clusters.push_back(std::move(winner));
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
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& members = clusters[c].members;
        double maximum = 0.0;
        for (size_t i = 0; i < members.size(); ++i)
            for (size_t j = i + 1; j < members.size(); ++j)
                maximum = std::max(maximum, distance(points[members[i]], points[members[j]]));
        diameters[c] = maximum;
    }
    // Print in the original order, independently of OpenMP scheduling.
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
        if (mpi.rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        return 1;
    }
    
    // Bind processes by node-local rank, also respecting CUDA_VISIBLE_DEVICES.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi.rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank, device_count;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    cudaCheck(cudaGetDeviceCount(&device_count));
    if (!device_count) {
        fprintf(stderr, "A CUDA device is required on every MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count));
    cudaCheck(cudaFree(nullptr));

    if (mpi.rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (mpi.rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    if (mpi.rank != 0) return 0;
    
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
