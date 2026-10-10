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
#include <stdexcept>

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

template<class T> class DeviceArray {
public:
    T* data = nullptr;
    explicit DeviceArray(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

// Disabling FMA in the build preserves the reference's double-precision
// distance comparisons, including ties and the strict threshold test.
__device__ double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

constexpr int BLOCK_SIZE = 256;

// Store only points within the threshold of each owned seed. Any point that
// could ever join this seed's cluster must be in this list. No dense global
// distance matrix or per-seed N-element membership array is needed.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                              int rank, int ranks, int rows, int* counts) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int seed = rank + row * ranks;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (j != seed && deviceDistance(points[seed], points[j]) < threshold) ++count;
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
                             int rank, int ranks, int rows,
                             const size_t* offsets, int* neighbors) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int seed = rank + row * ranks;
    __shared__ int next;
    if (threadIdx.x == 0) next = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j != seed && deviceDistance(points[seed], points[j]) < threshold) {
            const int slot = atomicAdd(&next, 1);
            neighbors[offsets[row] + slot] = j;
        }
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        clustered[members[i]] = 1;
}

// One cooperative block per seed. Updating a candidate's running maximum
// against only the newly added member is equivalent to rescanning every
// member, but removes a factor of cluster size from the work.
__global__ void evaluateSeeds(const Point* points, const unsigned char* clustered,
                             double threshold, int rank, int ranks,
                             const int* activeRows, const size_t* offsets,
                             const int* neighbors, double* maxima,
                             int* cardinalities, int* incumbent,
                             int* outputMembers) {
    const int row = activeRows[blockIdx.x];
    const int seed = rank + row * ranks;
    const size_t begin = offsets[row], end = offsets[row + 1];
    const int tid = threadIdx.x;
    __shared__ double bestDistance[BLOCK_SIZE];
    __shared__ int bestPoint[BLOCK_SIZE];
    __shared__ int selected, count, prune;
    int possible = 0;
    for (size_t k = begin + tid; k < end; k += BLOCK_SIZE) {
        const int candidate = neighbors[k];
        maxima[k] = clustered[candidate] ? DBL_MAX : deviceDistance(points[seed], points[candidate]);
        possible += !clustered[candidate];
    }
    bestPoint[tid] = possible;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (tid < stride) bestPoint[tid] += bestPoint[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        count = 1;
        // Strict inequality retains smaller-seed ties. This bound can only
        // discard a seed that cannot win, irrespective of block scheduling.
        prune = !outputMembers && bestPoint[0] + 1 < atomicAdd(incumbent, 0);
        if (outputMembers) outputMembers[0] = seed;
    }
    __syncthreads();
    if (prune) {
        if (tid == 0) cardinalities[blockIdx.x] = 0;
        return;
    }
    while (true) {
        double localDistance = DBL_MAX;
        int localPoint = INT_MAX;
        for (size_t k = begin + tid; k < end; k += BLOCK_SIZE) {
            const double d = maxima[k];
            const int candidate = neighbors[k];
            if (d < localDistance || (d == localDistance && candidate < localPoint)) {
                localDistance = d;
                localPoint = candidate;
            }
        }
        // Warp shuffles avoid a block-wide barrier at every reduction level.
        const int lane = tid & 31, warp = tid / 32;
        for (int delta = 16; delta; delta /= 2) {
            const double d = __shfl_down_sync(0xffffffff, localDistance, delta);
            const int id = __shfl_down_sync(0xffffffff, localPoint, delta);
            if (lane + delta < 32 && (d < localDistance || (d == localDistance && id < localPoint))) {
                localDistance = d;
                localPoint = id;
            }
        }
        if (lane == 0) {
            bestDistance[warp] = localDistance;
            bestPoint[warp] = localPoint;
        }
        __syncthreads();
        if (warp == 0) {
            localDistance = lane < BLOCK_SIZE / 32 ? bestDistance[lane] : DBL_MAX;
            localPoint = lane < BLOCK_SIZE / 32 ? bestPoint[lane] : INT_MAX;
            for (int delta = 16; delta; delta /= 2) {
                const double d = __shfl_down_sync(0xffffffff, localDistance, delta);
                const int id = __shfl_down_sync(0xffffffff, localPoint, delta);
                if (lane + delta < 32 && (d < localDistance || (d == localDistance && id < localPoint))) {
                    localDistance = d;
                    localPoint = id;
                }
            }
            if (lane == 0) {
                selected = localDistance < threshold && localDistance < DBL_MAX ? localPoint : -1;
                if (selected >= 0) {
                    if (outputMembers) outputMembers[count] = selected;
                    ++count;
                }
            }
        }
        __syncthreads();
        if (selected < 0) break;
        const Point member = points[selected];
        for (size_t k = begin + tid; k < end; k += BLOCK_SIZE) {
            const int candidate = neighbors[k];
            double d = maxima[k];
            if (candidate == selected) d = DBL_MAX;
            else if (d < threshold) {
                d = fmax(d, deviceDistance(points[candidate], member));
                if (!(d < threshold)) d = DBL_MAX;
            }
            maxima[k] = d;
        }
        __syncthreads();
    }
    if (tid == 0) {
        cardinalities[blockIdx.x] = count;
        atomicMax(incumbent, count);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int rows = n / ranks + (rank < n % ranks);
    DeviceArray<Point> devicePoints(n);
    DeviceArray<unsigned char> deviceClustered(n);
    DeviceArray<int> deviceCounts(rows), deviceRows(rows), deviceMembers(n), incumbent(1);
    DeviceArray<size_t> deviceOffsets(static_cast<size_t>(rows) + 1);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    std::vector<int> counts(rows);
    std::vector<size_t> offsets(static_cast<size_t>(rows) + 1, 0);
    if (rows) {
        countNeighbors<<<rows, BLOCK_SIZE>>>(devicePoints.data, n, threshold, rank, ranks, rows, deviceCounts.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(counts.data(), deviceCounts.data, rows * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < rows; ++i) {
            if (offsets[i] > SIZE_MAX - static_cast<size_t>(counts[i])) throw std::bad_alloc();
            offsets[i + 1] = offsets[i] + counts[i];
        }
    }
    DeviceArray<int> neighbors(offsets.back());
    DeviceArray<double> maxima(offsets.back());
    cudaCheck(cudaMemcpy(deviceOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (rows) {
        fillNeighbors<<<rows, BLOCK_SIZE>>>(devicePoints.data, n, threshold, rank, ranks, rows, deviceOffsets.data, neighbors.data);
        cudaCheck(cudaGetLastError());
    }
    std::vector<unsigned char> clustered(n, 0);
    cudaCheck(cudaMemset(deviceClustered.data, 0, n * sizeof(unsigned char)));
    std::vector<int> activeRows(rows), alive(rows), cardinalities(rows), seedOrder(rows);
    for (int i = 0; i < rows; ++i) seedOrder[i] = i;
    // Launch promising seeds first so completed blocks quickly tighten the
    // cardinality bound used to skip seeds that cannot win.
    std::sort(seedOrder.begin(), seedOrder.end(), [&counts](int a, int b) {
        return counts[a] != counts[b] ? counts[a] > counts[b] : a < b;
    });
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        // OpenMP filters local seeds and later reduces their cardinalities;
        // CUDA owns the expensive candidate construction.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) alive[i] = !clustered[rank + i * ranks];
        int active = 0;
        for (int row : seedOrder) if (alive[row]) activeRows[active++] = row;
        if (active) {
            cudaCheck(cudaMemcpy(deviceRows.data, activeRows.data(), active * sizeof(int), cudaMemcpyHostToDevice));
            cudaCheck(cudaMemset(incumbent.data, 0, sizeof(int)));
            evaluateSeeds<<<active, BLOCK_SIZE>>>(devicePoints.data, deviceClustered.data, threshold,
                rank, ranks, deviceRows.data, deviceOffsets.data, neighbors.data, maxima.data,
                deviceCounts.data, incumbent.data, nullptr);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(cardinalities.data(), deviceCounts.data, active * sizeof(int), cudaMemcpyDeviceToHost));
        }
        // Encode the reference's size-descending, seed-ascending comparison.
        long long best = -1;
        #pragma omp parallel for reduction(max:best) schedule(static)
        for (int i = 0; i < active; ++i) {
            const int row = activeRows[i], seed = rank + row * ranks;
            const long long key = static_cast<long long>(cardinalities[i]) * (static_cast<long long>(n) + 1) + n - seed;
            best = std::max(best, key);
        }
        int local[2] = {-1, INT_MAX}, winner[2];
        if (best >= 0) {
            local[0] = static_cast<int>(best / (static_cast<long long>(n) + 1));
            local[1] = n - static_cast<int>(best % (static_cast<long long>(n) + 1));
        }
        MPI_Allreduce(local, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int owner = winner[1] % ranks;
        Cluster cluster;
        cluster.seed_point = winner[1];
        cluster.members.resize(winner[0]);
        if (rank == owner) {
            const int row = winner[1] / ranks;
            cudaCheck(cudaMemcpy(deviceRows.data, &row, sizeof(int), cudaMemcpyHostToDevice));
            evaluateSeeds<<<1, BLOCK_SIZE>>>(devicePoints.data, deviceClustered.data, threshold,
                rank, ranks, deviceRows.data, deviceOffsets.data, neighbors.data, maxima.data,
                deviceCounts.data, incumbent.data, deviceMembers.data);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(cluster.members.data(), deviceMembers.data, cluster.members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(cluster.members.data(), winner[0], MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner)
            cudaCheck(cudaMemcpy(deviceMembers.data, cluster.members.data(), cluster.members.size() * sizeof(int), cudaMemcpyHostToDevice));
        markClustered<<<std::min(65535, (winner[0] - 1) / BLOCK_SIZE + 1), BLOCK_SIZE>>>(deviceClustered.data, deviceMembers.data, winner[0]);
        cudaCheck(cudaGetLastError());
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < winner[0]; ++i) clustered[cluster.members[i]] = 1;
        remaining -= winner[0];
        if (rank == 0) clusters.push_back(std::move(cluster));
    }
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

static int runBenchmark(int argc, char** argv, int rank) {
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
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    // Chunk broadcasts to avoid overflowing MPI's int count for large inputs.
    for (size_t offset = 0; offset < points.size(); ) {
        const size_t count = std::min(points.size() - offset, static_cast<size_t>(INT_MAX) / sizeof(Point));
        MPI_Bcast(points.data() + offset, static_cast<int>(count * sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
        offset += count;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    const long local_cluster_time_ms = cluster_time.count();
    long max_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &max_cluster_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    cluster_time = std::chrono::milliseconds(max_cluster_time_ms);

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
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int result = 0;
    try {
        // Ranks sharing a node use distinct visible GPUs when available.
        // Respect scheduler-set CUDA_VISIBLE_DEVICES (including one GPU/rank).
        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank, devices = 0;
        MPI_Comm_rank(localComm, &localRank);
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        cudaCheck(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&localComm);
        // Avoid launching more host workers than this rank's CPU affinity allows.
        if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_num_procs()));
        result = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
