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
#include <cfloat>
#include <cstdint>
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
        // The original generator otherwise never makes progress for N <= 30.
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), n) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK_SIZE = 128;

__device__ inline void reduceClosest(double& value, int& index) {
    for (int stride = 16; stride; stride /= 2) {
        const double other = __shfl_down_sync(0xffffffff, value, stride);
        const int other_index = __shfl_down_sync(0xffffffff, index, stride);
        if (other < value || (other == value && other_index < index)) {
            value = other;
            index = other_index;
        }
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        clustered[members[i]] = 1;
}

// Only points within the seed's threshold can ever enter its candidate.
// Partition these sparse neighbor rows cyclically across ranks.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int rows, int* counts) {
    __shared__ int sums[BLOCK_SIZE];
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        int seed = rank + row * ranks;
        int count = 0;
        for (int j = threadIdx.x; j < n; j += blockDim.x)
            if (j != seed && distance(points[seed], points[j]) < threshold) ++count;
        sums[threadIdx.x] = count;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }
        if (!threadIdx.x) counts[row] = sums[0];
        __syncthreads();
    }
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, int rows,
                              const size_t* offsets, int* neighbors) {
    __shared__ int next;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        if (!threadIdx.x) next = 0;
        __syncthreads();
        int seed = rank + row * ranks;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            if (j != seed && distance(points[seed], points[j]) < threshold) {
                int slot = atomicAdd(&next, 1);
                neighbors[offsets[row] + slot] = j;
            }
        }
        __syncthreads();
    }
}

// One block per candidate. Maintain each neighbor's maximum distance
// incrementally, rather than rescanning all previous members at every step.
// Cached candidates remain exact until one of their members is removed.
__global__ void growCandidates(const Point* points, double threshold,
                               int rank, int ranks, int rows,
                               const size_t* offsets, const int* neighbors,
                               const unsigned char* clustered, double* maxima,
                               int* members, int* cardinalities) {
    __shared__ double best_distance[BLOCK_SIZE / 32];
    __shared__ int best_index[BLOCK_SIZE / 32];
    __shared__ int invalid;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int seed = rank + row * ranks;
        if (clustered[seed]) {
            if (!threadIdx.x) cardinalities[row] = 0;
            continue;
        }
        const size_t begin = offsets[row], end = offsets[row + 1];
        const int old_size = cardinalities[row];
        if (!threadIdx.x) invalid = (old_size == 0);
        __syncthreads();
        for (int i = threadIdx.x; i < old_size - 1; i += blockDim.x)
            if (clustered[members[begin + i]]) atomicExch(&invalid, 1);
        __syncthreads();
        const bool rebuild = invalid != 0;
        // All warps must consume invalid before a grid-stride iteration
        // reuses that shared flag for another row.
        __syncthreads();
        if (!rebuild) continue;

        for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
            int candidate = neighbors[i];
            maxima[i] = clustered[candidate] ? DBL_MAX : distance(points[seed], points[candidate]);
        }
        __syncthreads();
        int count = 1;
        int last = seed;
        while (true) {
            double value = DBL_MAX;
            int index = INT_MAX;
            for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
                int candidate = neighbors[i];
                double d = maxima[i];
                if (d == DBL_MAX) continue;
                if (candidate == last) d = DBL_MAX;
                else if (count > 1) d = fmax(d, distance(points[last], points[candidate]));
                // An infeasible neighbor can never become feasible later.
                if (!(d < threshold)) d = DBL_MAX;
                maxima[i] = d;
                if (d < value || (d == value && candidate < index)) {
                    value = d;
                    index = candidate;
                }
            }
            reduceClosest(value, index);
            const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
            if (!lane) {
                best_distance[warp] = value;
                best_index[warp] = index;
            }
            __syncthreads();
            if (!warp) {
                value = lane < BLOCK_SIZE / 32 ? best_distance[lane] : DBL_MAX;
                index = lane < BLOCK_SIZE / 32 ? best_index[lane] : INT_MAX;
                reduceClosest(value, index);
                if (!lane) {
                    best_distance[0] = value;
                    best_index[0] = index;
                }
            }
            __syncthreads();
            if (best_distance[0] == DBL_MAX) break;
            last = best_index[0];
            if (!threadIdx.x) members[begin + count - 1] = last;
            ++count;
            // Prevent the next reduction from overwriting its shared result
            // before all threads have consumed it.
            __syncthreads();
        }
        if (!threadIdx.x) cardinalities[row] = count;
        __syncthreads();
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = n / ranks + (rank < n % ranks);
    const int blocks = std::min(rows, 65535);
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned char> d_clustered(n);
    DeviceBuffer<int> d_selected(n);
    DeviceBuffer<int> d_sizes(rows);
    DeviceBuffer<size_t> d_offsets(size_t(rows) + 1);
    CUDA_CHECK(cudaMemcpy(d_points.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered.data, 0, n));
    std::vector<int> sizes(rows);
    if (rows) {
        countNeighbors<<<blocks, BLOCK_SIZE>>>(d_points.data, n, threshold, rank, ranks, rows, d_sizes.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::vector<size_t> offsets(size_t(rows) + 1, 0);
    for (int i = 0; i < rows; ++i) offsets[i + 1] = offsets[i] + sizes[i];
    CUDA_CHECK(cudaMemcpy(d_offsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    DeviceBuffer<int> d_neighbors(offsets.back()), d_members(offsets.back());
    DeviceBuffer<double> d_maxima(offsets.back());
    CUDA_CHECK(cudaMemset(d_sizes.data, 0, rows * sizeof(int)));
    if (rows) {
        fillNeighbors<<<blocks, BLOCK_SIZE>>>(d_points.data, n, threshold, rank, ranks, rows,
                                             d_offsets.data, d_neighbors.data);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<Cluster> clusters;
    std::vector<unsigned char> clustered(n, 0);
    int remaining = n;
    while (remaining) {
        if (rows) {
            growCandidates<<<blocks, BLOCK_SIZE>>>(d_points.data, threshold, rank, ranks, rows,
                d_offsets.data, d_neighbors.data, d_clustered.data, d_maxima.data, d_members.data, d_sizes.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // Encode both tie breakers in one integer so OpenMP and MPI can use
        // their native max reductions (largest size, then smallest seed).
        unsigned long long local_best = 0, global_best = 0;
        #pragma omp parallel for reduction(max:local_best) schedule(static) if(rows > 1024)
        for (int i = 0; i < rows; ++i) {
            if (sizes[i]) {
                unsigned long long key = (static_cast<unsigned long long>(sizes[i]) << 32)
                    | static_cast<unsigned int>(INT_MAX - (rank + i * ranks));
                local_best = std::max(local_best, key);
            }
        }
        MPI_Allreduce(&local_best, &global_best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int cardinality = static_cast<int>(global_best >> 32);
        // If all remaining candidates are singletons, their order is known.
        // Finish the entire tail without one collective and launch per point.
        if (cardinality == 1) {
            if (!rank) {
                for (int seed = 0; seed < n; ++seed)
                    if (!clustered[seed]) clusters.push_back(Cluster{{seed}, seed});
            }
            break;
        }
        const int seed = INT_MAX - static_cast<int>(global_best & 0xffffffffULL);
        const int owner = seed % ranks;
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(cardinality);
        if (rank == owner) {
            cluster.members[0] = seed;
            if (cardinality > 1)
                CUDA_CHECK(cudaMemcpy(cluster.members.data() + 1,
                    d_members.data + offsets[seed / ranks], (cardinality - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(cluster.members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static) if(cardinality > 1024)
        for (int i = 0; i < cardinality; ++i) clustered[cluster.members[i]] = 1;
        CUDA_CHECK(cudaMemcpy(d_selected.data, cluster.members.data(), cardinality * sizeof(int), cudaMemcpyHostToDevice));
        markClustered<<<(cardinality + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_clustered.data, d_selected.data, cardinality);
        CUDA_CHECK(cudaGetLastError());
        remaining -= cardinality;
        if (!rank) clusters.push_back(std::move(cluster));
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
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalize_mpi;
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
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (!rank) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                          num_points, threshold);
        return 1;
    }
    
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, local_ranks;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_ranks);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "A CUDA device is required on every MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    CUDA_CHECK(cudaFree(nullptr)); // Initialize the context before timing.
    if (!getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / local_ranks)));
    MPI_Comm_free(&local_comm);

    if (!rank) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (!rank) generateSyntheticData(points, num_points);
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
    if (rank) return 0;
    
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
