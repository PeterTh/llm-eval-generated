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
#include <stdexcept>
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

// All MPI calls are made by the main thread (MPI_THREAD_FUNNELED).
static int world_rank = 0;
static int world_size = 1;

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", world_rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) throw std::runtime_error("GPU buffer size overflow");
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

static constexpr int BLOCK = 128;

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Store only neighbors within the strict threshold. Every member of a
// candidate must be a neighbor of its seed. Rows belong to this MPI rank.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int* counts) {
    const int seed = int(blockIdx.x) * ranks + rank;
    __shared__ int sums[BLOCK];
    int count = 0;
    for (int j = threadIdx.x; j < n; j += BLOCK)
        if (j != seed && pointDistance(points[seed], points[j]) < threshold) ++count;
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[blockIdx.x] = sums[0] + 1;
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, const size_t* offsets, int* neighbors) {
    const int seed = int(blockIdx.x) * ranks + rank;
    const size_t base = offsets[blockIdx.x];
    __shared__ int next;
    if (threadIdx.x == 0) {
        next = 1;
        neighbors[base] = seed;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += BLOCK) {
        if (j != seed && pointDistance(points[seed], points[j]) < threshold)
            neighbors[base + atomicAdd(&next, 1)] = j;
    }
}

// One block per seed; threads cooperate over that seed's sparse neighbor row.
// Maintain each candidate's maximum distance incrementally, instead of
// scanning the entire growing cluster again on every greedy insertion.
__global__ void evaluateCandidates(const Point* points, double threshold,
                                   const size_t* offsets, const int* neighbors,
                                   const unsigned char* clustered, double* maxima,
                                   int* members, int* sizes) {
    const int row = blockIdx.x;
    const size_t base = offsets[row], end = offsets[row + 1];
    const int seed = neighbors[base];
    const int tid = threadIdx.x;
    if (clustered[seed]) {
        if (tid == 0) sizes[row] = 0;
        return;
    }
    __shared__ int invalid, selected, cardinality;
    __shared__ double bestDistance[BLOCK / 32];
    __shared__ int bestIndex[BLOCK / 32];
    if (tid == 0) invalid = (sizes[row] == 0);
    __syncthreads();
    // Removing points outside a cached candidate cannot change its greedy
    // sequence. Recompute only candidates that lost at least one member.
    for (int i = tid; i < sizes[row]; i += BLOCK)
        if (clustered[members[base + i]]) atomicExch(&invalid, 1);
    __syncthreads();
    if (!invalid) return;

    for (size_t k = base + tid; k < end; k += BLOCK) {
        const int p = neighbors[k];
        maxima[k] = (p == seed || clustered[p]) ? CUDART_INF : 0.0;
    }
    if (tid == 0) {
        selected = seed;
        cardinality = 1;
        members[base] = seed;
    }
    __syncthreads();
    while (true) {
        double best = CUDART_INF;
        int index = INT_MAX;
        const Point last = points[selected];
        for (size_t k = base + tid; k < end; k += BLOCK) {
            double value = maxima[k];
            if (value == CUDART_INF) continue;
            const int p = neighbors[k];
            if (p == selected) {
                maxima[k] = CUDART_INF;
                continue;
            }
            value = fmax(value, pointDistance(points[p], last));
            // An infeasible point can never become feasible as a cluster grows.
            maxima[k] = value < threshold ? value : CUDART_INF;
            if (value < threshold && (value < best || (value == best && p < index))) {
                best = value;
                index = p;
            }
        }
        // Reduce within warps first, then reduce the four warp winners.
        // This needs only two block barriers per greedy insertion.
        for (int delta = 16; delta; delta /= 2) {
            const double other = __shfl_down_sync(0xffffffffu, best, delta);
            const int otherIndex = __shfl_down_sync(0xffffffffu, index, delta);
            if (other < best || (other == best && otherIndex < index)) {
                best = other;
                index = otherIndex;
            }
        }
        if ((tid & 31) == 0) {
            bestDistance[tid / 32] = best;
            bestIndex[tid / 32] = index;
        }
        __syncthreads();
        if (tid == 0) {
            best = bestDistance[0];
            selected = bestIndex[0];
            for (int warp = 1; warp < BLOCK / 32; ++warp) {
                if (bestDistance[warp] < best ||
                    (bestDistance[warp] == best && bestIndex[warp] < selected)) {
                    best = bestDistance[warp];
                    selected = bestIndex[warp];
                }
            }
            if (selected != INT_MAX) members[base + cardinality++] = selected;
        }
        __syncthreads();
        if (selected == INT_MAX) break;
    }
    if (tid == 0) sizes[row] = cardinality;
}

__global__ void markCluster(unsigned char* clustered, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    const int rows = n / world_size + (world_rank < n % world_size);
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned char> d_clustered(n);
    DeviceBuffer<int> d_sizes(rows), d_winner(n);
    DeviceBuffer<size_t> d_offsets(size_t(rows) + 1);
    cudaCheck(cudaMemcpy(d_points.data, points.data(), size_t(n) * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_clustered.data, 0, n));
    std::vector<int> sizes(rows);
    std::vector<size_t> offsets(size_t(rows) + 1, 0);
    if (rows) {
        countNeighbors<<<rows, BLOCK>>>(d_points.data, n, threshold, world_rank, world_size, d_sizes.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(sizes.data(), d_sizes.data, size_t(rows) * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < rows; ++i) offsets[i + 1] = offsets[i] + sizes[i];
    }
    DeviceBuffer<int> d_neighbors(offsets.back()), d_members(offsets.back());
    DeviceBuffer<double> d_maxima(offsets.back());
    cudaCheck(cudaMemcpy(d_offsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (rows) {
        fillNeighbors<<<rows, BLOCK>>>(d_points.data, n, threshold, world_rank, world_size,
                                      d_offsets.data, d_neighbors.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemset(d_sizes.data, 0, size_t(rows) * sizeof(int)));
    }
    std::vector<Cluster> clusters;
    int remaining = n;
    const int host_threads = std::min(omp_get_max_threads(), std::max(1, rows / 1024));
    while (remaining) {
        if (rows) {
            evaluateCandidates<<<rows, BLOCK>>>(d_points.data, threshold, d_offsets.data,
                d_neighbors.data, d_clustered.data, d_maxima.data, d_members.data, d_sizes.data);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(sizes.data(), d_sizes.data, size_t(rows) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // An unsigned key orders by largest size, then smallest global seed.
        unsigned long long local = 0, global = 0;
        #pragma omp parallel for reduction(max:local) schedule(static) num_threads(host_threads)
        for (int i = 0; i < rows; ++i) {
            if (sizes[i]) {
                const unsigned int seed = i * world_size + world_rank;
                const unsigned long long key = (static_cast<unsigned long long>(sizes[i]) << 32)
                                               | (UINT_MAX - seed);
                local = std::max(local, key);
            }
        }
        MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int count = static_cast<int>(global >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(global));
        if (count <= 0) throw std::runtime_error("No candidate for remaining points");
        const int owner = seed % world_size;
        Cluster winner;
        winner.seed_point = seed;
        winner.members.resize(count);
        if (world_rank == owner) {
            cudaCheck(cudaMemcpy(winner.members.data(), d_members.data + offsets[seed / world_size],
                                 size_t(count) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(winner.members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(d_winner.data, winner.members.data(), size_t(count) * sizeof(int), cudaMemcpyHostToDevice));
        markCluster<<<(count + BLOCK - 1) / BLOCK, BLOCK>>>(d_clustered.data, d_winner.data, count);
        cudaCheck(cudaGetLastError());
        remaining -= count;
        if (world_rank == 0) clusters.push_back(std::move(winner));
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

int benchmarkMain(int argc, char** argv) {
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
            if (world_rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (world_rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (world_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    cudaCheck(cudaGetDeviceCount(&device_count));
    if (!device_count) throw std::runtime_error("CUDA device required on every rank");
    cudaCheck(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (world_rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (world_rank != 0) return 0;
    
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
    const double time_sec = cluster_time / 1000.0;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try {
        result = benchmarkMain(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", world_rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Propagate root validation failures to every process exit status.
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
