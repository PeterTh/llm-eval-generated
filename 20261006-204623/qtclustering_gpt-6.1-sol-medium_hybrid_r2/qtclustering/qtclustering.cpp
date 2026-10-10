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
#include <string>

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

// MPI is called only by the main thread (MPI_THREAD_FUNNELED).
static int world_rank = 0;
static int world_size = 1;

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

// Persistent allocations: no allocation or point transfer inside a growth loop.
template<class T> class DeviceBuffer {
    T* ptr_ = nullptr;
public:
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) throw std::runtime_error("GPU allocation overflow");
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
};

// Do not use squared distances or fused operations: rounding and strict threshold
// comparisons must match the original double-precision Euclidean distance.
__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

static constexpr int BLOCK = 128;

// Only points within the seed's threshold can ever enter its candidate cluster.
// Store those neighbors in CSR form, distributed cyclically over MPI ranks.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int* counts) {
    const int seed = rank + static_cast<int>(blockIdx.x) * ranks;
    int count = 0;
    for (int p = threadIdx.x; p < n; p += blockDim.x)
        count += p != seed && gpuDistance(points[seed], points[p]) < threshold;
    for (int d = 16; d; d >>= 1)
        count += __shfl_down_sync(0xffffffff, count, d);
    __shared__ int sums[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < BLOCK / 32; ++w) total += sums[w];
        counts[blockIdx.x] = total;
    }
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, const size_t* offsets, int* neighbors) {
    const int seed = rank + static_cast<int>(blockIdx.x) * ranks;
    __shared__ int used;
    if (threadIdx.x == 0) used = 0;
    __syncthreads();
    const unsigned lane = threadIdx.x & 31;
    for (size_t base = 0; base < static_cast<size_t>(n); base += blockDim.x) {
        const size_t p = base + threadIdx.x;
        const bool keep = p < static_cast<size_t>(n) && static_cast<int>(p) != seed &&
                          gpuDistance(points[seed], points[p]) < threshold;
        const unsigned mask = __ballot_sync(0xffffffff, keep);
        int start = 0;
        if (lane == 0) start = atomicAdd(&used, __popc(mask));
        start = __shfl_sync(0xffffffff, start, 0);
        if (keep) {
            const int before = __popc(mask & ((1u << lane) - 1u));
            neighbors[offsets[blockIdx.x] + start + before] = static_cast<int>(p);
        }
    }
}

// Lexicographic minimum (distance, point index), including across warps.
// The original ascending scan picks the smallest index on equal distances.
__device__ int closestInBlock(double best, int index) {
    const int lane = threadIdx.x & 31;
    for (int d = 16; d; d >>= 1) {
        const double other = __shfl_down_sync(0xffffffff, best, d);
        const int oi = __shfl_down_sync(0xffffffff, index, d);
        if (lane + d < 32 && (other < best || (other == best && oi < index))) {
            best = other;
            index = oi;
        }
    }
    __shared__ double values[BLOCK / 32];
    __shared__ int indices[BLOCK / 32];
    __shared__ int winner;
    if (lane == 0) {
        values[threadIdx.x / 32] = best;
        indices[threadIdx.x / 32] = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double value = CUDART_INF;
        int point = INT_MAX;
        for (int w = 0; w < BLOCK / 32; ++w) {
            if (values[w] < value || (values[w] == value && indices[w] < point)) {
                value = values[w];
                point = indices[w];
            }
        }
        winner = point;
    }
    __syncthreads();
    return winner;
}

// One block grows one seed. Cache each neighbor's maximum distance and update
// it only against the newly added member, reducing growth from O(N*k*k) to
// O(degree*k). Rejected neighbors cannot become eligible again.
template<bool Record>
__global__ void growCandidates(const Point* points, double threshold,
                               int rank, int ranks, const int* rows,
                               const size_t* offsets, const int* neighbors, const size_t* work_offsets,
                               const unsigned char* clustered, double* maxima,
                               int* counts, int* members, unsigned long long* incumbent) {
    const int row = rows[blockIdx.x];
    const int seed = rank + row * ranks;
    if (clustered[seed]) {
        if (threadIdx.x == 0) counts[blockIdx.x] = 0;
        return;
    }
    const size_t begin = offsets[row], end = offsets[row + 1];
    const size_t work_begin = work_offsets[blockIdx.x];
    int eligible = 0;
    for (size_t e = begin + threadIdx.x; e < end; e += blockDim.x) {
        const int p = neighbors[e];
        eligible += !clustered[p];
        maxima[work_begin + e - begin] = clustered[p] ? CUDART_INF : gpuDistance(points[seed], points[p]);
    }
    if (!Record) {
        for (int d = 16; d; d >>= 1)
            eligible += __shfl_down_sync(0xffffffff, eligible, d);
        __shared__ int sums[BLOCK / 32];
        __shared__ bool prune;
        if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = eligible;
        __syncthreads();
        if (threadIdx.x == 0) {
            int upper = 1;
            for (int w = 0; w < BLOCK / 32; ++w) upper += sums[w];
            const unsigned long long bound = (static_cast<unsigned long long>(upper) << 32) |
                                             (static_cast<unsigned>(INT_MAX) - seed);
            // A stale incumbent only permits extra work; it cannot change the winner.
            prune = bound <= atomicAdd(incumbent, 0ULL);
        }
        __syncthreads();
        if (prune) {
            if (threadIdx.x == 0) counts[blockIdx.x] = 0;
            return;
        }
    }
    if (Record && threadIdx.x == 0) members[0] = seed;
    __syncthreads();
    int count = 1;
    while (true) {
        double best = CUDART_INF;
        int index = INT_MAX;
        for (size_t e = begin + threadIdx.x; e < end; e += blockDim.x) {
            const double v = maxima[work_begin + e - begin];
            const int p = neighbors[e];
            if (v < threshold && (v < best || (v == best && p < index))) {
                best = v;
                index = p;
            }
        }
        const int selected = closestInBlock(best, index);
        if (selected == INT_MAX) break;
        if (Record && threadIdx.x == 0) members[count] = selected;
        ++count;
        for (size_t e = begin + threadIdx.x; e < end; e += blockDim.x) {
            double value = maxima[work_begin + e - begin];
            if (value == CUDART_INF) continue;
            const int p = neighbors[e];
            if (p == selected) value = CUDART_INF;
            else {
                const double d = gpuDistance(points[p], points[selected]);
                value = value > d ? value : d;
                if (!(value < threshold)) value = CUDART_INF;
            }
            maxima[work_begin + e - begin] = value;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        counts[blockIdx.x] = count;
        if (!Record) {
            const unsigned long long key = (static_cast<unsigned long long>(count) << 32) |
                                           (static_cast<unsigned>(INT_MAX) - seed);
            atomicMax(incumbent, key);
        }
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(count)) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node);
    int local_rank, local_size;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_size(node, &local_size);
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / local_size)));
    MPI_Comm_free(&node);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("A CUDA device is required on every rank");
    // CUDA_VISIBLE_DEVICES is honored; one rank per visible GPU is recommended.
    cudaCheck(cudaSetDevice(local_rank % devices));

    const int n = static_cast<int>(points.size());
    const int rows = world_rank < n ? (n - 1 - world_rank) / world_size + 1 : 0;
    DeviceBuffer<Point> d_points(n);
    DeviceBuffer<unsigned char> d_clustered(n);
    DeviceBuffer<int> d_counts(rows), d_rows(rows), d_members(n);
    DeviceBuffer<unsigned long long> d_incumbent(1);
    DeviceBuffer<size_t> d_offsets(static_cast<size_t>(rows) + 1), d_work_offsets(rows);
    cudaCheck(cudaMemcpy(d_points.get(), points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_clustered.get(), 0, n));
    std::vector<int> degree(rows), counts(rows), active(rows);
    std::vector<size_t> offsets(static_cast<size_t>(rows) + 1, 0);
    if (rows) {
        countNeighbors<<<rows, BLOCK>>>(d_points.get(), n, threshold, world_rank,
                                       world_size, d_counts.get());
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(degree.data(), d_counts.get(), rows * sizeof(int), cudaMemcpyDeviceToHost));
    }
    for (int r = 0; r < rows; ++r) {
        if (offsets[r] > SIZE_MAX - static_cast<size_t>(degree[r]))
            throw std::runtime_error("Neighbor offset overflow");
        offsets[r + 1] = offsets[r] + degree[r];
    }
    DeviceBuffer<int> d_neighbors(offsets.back());
    // Bound scratch storage independently of the complete neighbor graph. Large
    // seed sets are processed in batches, retaining GPU occupancy without an
    // additional eight bytes of permanent storage for every graph edge.
    size_t free_bytes, total_bytes;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t largest_row = degree.empty() ? 0 : *std::max_element(degree.begin(), degree.end());
    const size_t desired = std::max(largest_row, (size_t(256) << 20) / sizeof(double));
    const size_t scratch_size = std::min(offsets.back(), std::min(desired, free_bytes / (2 * sizeof(double))));
    if (scratch_size < largest_row) throw std::runtime_error("Insufficient GPU memory for one seed's neighbors");
    DeviceBuffer<double> d_maxima(scratch_size);
    std::vector<size_t> work_offsets(rows);
    cudaCheck(cudaMemcpy(d_offsets.get(), offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (rows) {
        fillNeighbors<<<rows, BLOCK>>>(d_points.get(), n, threshold, world_rank,
                                      world_size, d_offsets.get(), d_neighbors.get());
        cudaCheck(cudaGetLastError());
    }
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        // Stable compaction keeps cyclic ownership, with no replicated seed work.
        int active_count = 0;
        for (int r = 0; r < rows; ++r)
            if (!clustered[world_rank + r * world_size]) active[active_count++] = r;
        unsigned long long best_key = 0;
        if (active_count) {
            cudaCheck(cudaMemset(d_incumbent.get(), 0, sizeof(unsigned long long)));
            for (int first = 0; first < active_count; ) {
                size_t used = 0;
                int last = first;
                while (last < active_count) {
                    const size_t length = static_cast<size_t>(degree[active[last]]);
                    if (length > scratch_size - used) break;
                    work_offsets[last - first] = used;
                    used += length;
                    ++last;
                }
                const int batch = last - first;
                cudaCheck(cudaMemcpy(d_rows.get(), active.data() + first, batch * sizeof(int), cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(d_work_offsets.get(), work_offsets.data(), batch * sizeof(size_t), cudaMemcpyHostToDevice));
                growCandidates<false><<<batch, BLOCK>>>(d_points.get(), threshold,
                    world_rank, world_size, d_rows.get(), d_offsets.get(), d_neighbors.get(), d_work_offsets.get(),
                    d_clustered.get(), d_maxima.get(), d_counts.get(), nullptr, d_incumbent.get());
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaMemcpy(counts.data() + first, d_counts.get(), batch * sizeof(int), cudaMemcpyDeviceToHost));
                first = last;
            }
            // Pack size and reversed seed index so a maximum resolves both ties.
            #pragma omp parallel for reduction(max:best_key) schedule(static)
            for (int i = 0; i < active_count; ++i) {
                const unsigned seed = world_rank + active[i] * world_size;
                const unsigned long long key = (static_cast<unsigned long long>(counts[i]) << 32) |
                                               (static_cast<unsigned>(INT_MAX) - seed);
                best_key = std::max(best_key, key);
            }
        }
        unsigned long long global_key;
        MPI_Allreduce(&best_key, &global_key, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int cardinality = static_cast<int>(global_key >> 32);
        const int seed = INT_MAX - static_cast<int>(global_key & 0xffffffffULL);
        if (cardinality <= 0 || seed < 0 || seed >= n)
            throw std::runtime_error("Invalid distributed cluster winner");
        const int owner = seed % world_size;
        std::vector<int> members(cardinality);
        if (world_rank == owner) {
            const int row = seed / world_size;
            cudaCheck(cudaMemcpy(d_rows.get(), &row, sizeof(int), cudaMemcpyHostToDevice));
            cudaCheck(cudaMemset(d_work_offsets.get(), 0, sizeof(size_t)));
            growCandidates<true><<<1, BLOCK>>>(d_points.get(), threshold,
                world_rank, world_size, d_rows.get(), d_offsets.get(), d_neighbors.get(), d_work_offsets.get(),
                d_clustered.get(), d_maxima.get(), d_counts.get(), d_members.get(), d_incumbent.get());
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(members.data(), d_members.get(), cardinality * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < cardinality; ++i) clustered[members[i]] = 1;
        cudaCheck(cudaMemcpy(d_members.get(), members.data(), cardinality * sizeof(int), cudaMemcpyHostToDevice));
        markClustered<<<(static_cast<size_t>(cardinality) + BLOCK - 1) / BLOCK, BLOCK>>>(
            d_clustered.get(), d_members.get(), cardinality);
        cudaCheck(cudaGetLastError());
        remaining -= cardinality;
        if (world_rank == 0) clusters.push_back({std::move(members), seed});
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

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (world_rank == 0) generateSyntheticData(points, num_points);
    // Broadcast in chunks to stay within MPI's int count limit.
    for (size_t first = 0; first < points.size(); ) {
        const int count = static_cast<int>(std::min(points.size() - first,
                                                   static_cast<size_t>(INT_MAX / 2)));
        MPI_Bcast(&points[first].x, 2 * count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        first += count;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (world_rank != 0) return 0;

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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int result = 1;
    try {
        result = benchmarkMain(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", world_rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    // Propagate rank zero's validation status to every rank.
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
