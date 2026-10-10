// QT Clustering Benchmark - CUDA Version
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
#include <cfloat>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// All allocations and graph resources are released on both success and failure.
template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count = 0) { allocate(count); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(size_t count) {
        if (data) CUDA_CHECK(cudaFree(data));
        data = nullptr;
        if (count > SIZE_MAX / sizeof(T)) throw std::runtime_error("CUDA allocation size overflow");
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
};

struct Progress {
    int remaining;
    int cluster_count;
    int written;
};

class Execution {
public:
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Progress* progress = nullptr;
    Execution() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto status = cudaMallocHost(reinterpret_cast<void**>(&progress), sizeof(Progress));
        if (status != cudaSuccess) {
            cudaStreamDestroy(stream);
            checkCuda(status, "cudaMallocHost");
        }
    }
    ~Execution() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (progress) cudaFreeHost(progress);
        if (stream) cudaStreamDestroy(stream);
    }
    Execution(const Execution&) = delete;
    Execution& operator=(const Execution&) = delete;
};

constexpr int THREADS = 128;

__device__ __forceinline__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Match the native host compiler's double precision Euclidean distance.
    // Explicit rounding also prevents fast-math changes to the strict threshold.
    return __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
}

__device__ __forceinline__ double readDistance(const Point* points,
                                                const double* distances,
                                                int n, int a, int b) {
    return distances ? distances[static_cast<size_t>(a) * n + b]
                     : pointDistance(points[a], points[b]);
}

// Compute each pair once. The transpose writes also provide contiguous rows
// for all subsequent candidate updates.
__global__ void computeDistances(const Point* points, double* distances, int n) {
    const int a = blockIdx.y * 16 + threadIdx.y;
    const int b = blockIdx.x * 16 + threadIdx.x;
    if (blockIdx.x < blockIdx.y || a >= n || b >= n || a > b) return;
    const double d = pointDistance(points[a], points[b]);
    distances[static_cast<size_t>(a) * n + b] = d;
    distances[static_cast<size_t>(b) * n + a] = d;
}

__global__ void countNeighbors(const Point* points, const double* distances,
                               int n, double threshold, int* counts) {
    const int seed = blockIdx.x;
    int count = 0;
    for (int i = threadIdx.x; i < n; i += THREADS) {
        count += i == seed || readDistance(points, distances, n, seed, i) < threshold;
    }
    for (int delta = 16; delta; delta >>= 1)
        count += __shfl_down_sync(0xffffffffu, count, delta);
    __shared__ int warp_counts[THREADS / 32];
    if ((threadIdx.x & 31) == 0) warp_counts[threadIdx.x / 32] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        count = 0;
        for (int w = 0; w < THREADS / 32; ++w) count += warp_counts[w];
        counts[seed] = count;
    }
}

// CSR rows contain exactly the seed and points strictly inside its threshold.
// Warp ballots compact them in point-index order without global atomics.
__global__ void fillNeighbors(const Point* points, const double* distances,
                              int n, double threshold, const size_t* offsets,
                              int* neighbors) {
    const int seed = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    __shared__ int warp_offsets[THREADS / 32];
    __shared__ int total;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    for (int base = 0; base < n; base += THREADS) {
        const int i = base + threadIdx.x;
        const bool keep = i < n &&
            (i == seed || readDistance(points, distances, n, seed, i) < threshold);
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) warp_offsets[warp] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < THREADS / 32; ++w) {
                const int count = warp_offsets[w];
                warp_offsets[w] = total;
                total += count;
            }
        }
        __syncthreads();
        if (keep) {
            const int position = warp_offsets[warp] + __popc(mask & ((1u << lane) - 1u));
            neighbors[offsets[seed] + position] = i;
        }
        __syncthreads();
    }
}

struct Choice {
    double diameter;
    int point;
    int slot;
};

__device__ __forceinline__ unsigned long long winnerKey(int size, int seed) {
    return (static_cast<unsigned long long>(size) << 32) |
           (UINT_MAX - static_cast<unsigned>(seed));
}

__device__ __forceinline__ Choice betterChoice(Choice a, Choice b) {
    return b.diameter < a.diameter ||
           (b.diameter == a.diameter && b.point < a.point) ? b : a;
}

__device__ __forceinline__ Choice warpMinimum(Choice value) {
    for (int delta = 16; delta; delta >>= 1) {
        Choice other;
        other.diameter = __shfl_down_sync(0xffffffffu, value.diameter, delta);
        other.point = __shfl_down_sync(0xffffffffu, value.point, delta);
        other.slot = __shfl_down_sync(0xffffffffu, value.slot, delta);
        value = betterChoice(value, other);
    }
    return value;
}

// One block grows one seed's cluster, with all of its prospective members
// evaluated in parallel. Maxima are updated only against the newest member;
// max(max(d(a,b), d(a,c)), d(a,d)) is the original full diameter scan.
template<bool SHARED_MAXIMA>
__global__ void growCandidates(const Point* points, const double* distances,
                               int n, double threshold, const size_t* offsets,
                               const int* neighbors, const unsigned char* clustered,
                               int* sizes, int* members, double* global_maxima,
                               unsigned long long* winner) {
    const int seed = blockIdx.x;
    if (clustered[seed]) return;
    const size_t start = offsets[seed];
    const int degree = static_cast<int>(offsets[seed + 1] - start);
    // Every member must be a neighbor of the seed. Once a completed candidate
    // beats this upper bound, this seed cannot win the current round. Its cache
    // is left intact so it can still be reconsidered in subsequent rounds.
    __shared__ bool pruned;
    if (threadIdx.x == 0)
        pruned = winnerKey(degree, seed) <= atomicAdd(winner, 0ull);
    __syncthreads();
    if (pruned) return;
    int size = sizes[seed];
    bool affected = size == 0;
    for (int i = threadIdx.x; i < size; i += THREADS)
        affected |= clustered[members[start + i]] != 0;
    // Removing points outside a cached cluster cannot change its greedy choices.
    if (__syncthreads_or(affected)) {
        extern __shared__ double shared_maxima[];
        double* maxima = SHARED_MAXIMA ? shared_maxima : global_maxima + start;
        for (int i = threadIdx.x; i < degree; i += THREADS) {
            const int candidate = neighbors[start + i];
            maxima[i] = candidate == seed || clustered[candidate] ? DBL_MAX : 0.0;
        }
        if (threadIdx.x == 0) members[start] = seed;
        size = 1;
        int last = seed;
        __shared__ Choice warp_choices[THREADS / 32];
        __shared__ Choice selected;
        __syncthreads();
        while (true) {
            Choice best{DBL_MAX, INT_MAX, -1};
            for (int i = threadIdx.x; i < degree; i += THREADS) {
                double maximum = maxima[i];
                if (maximum == DBL_MAX) continue;
                const int candidate = neighbors[start + i];
                const double d = readDistance(points, distances, n, last, candidate);
                maximum = maximum > d ? maximum : d;
                if (maximum < threshold) {
                    maxima[i] = maximum;
                    best = betterChoice(best, Choice{maximum, candidate, i});
                } else {
                    // Maxima can only grow, so this point can never become valid.
                    maxima[i] = DBL_MAX;
                }
            }
            best = warpMinimum(best);
            if ((threadIdx.x & 31) == 0) warp_choices[threadIdx.x / 32] = best;
            __syncthreads();
            if (threadIdx.x < 32) {
                best = threadIdx.x < THREADS / 32 ? warp_choices[threadIdx.x]
                                                : Choice{DBL_MAX, INT_MAX, -1};
                best = warpMinimum(best);
                if (threadIdx.x == 0) selected = best;
            }
            __syncthreads();
            if (selected.slot < 0) break;
            last = selected.point;
            if (threadIdx.x == 0) {
                maxima[selected.slot] = DBL_MAX;
                members[start + size] = last;
            }
            ++size;
            __syncthreads();
        }
        if (threadIdx.x == 0) sizes[seed] = size;
    }
    if (threadIdx.x == 0) {
        // Largest size wins; equal sizes choose the smallest original seed.
        atomicMax(winner, winnerKey(size, seed));
    }
}

__global__ void commitCluster(unsigned long long* winner, const size_t* offsets,
                              const int* members, unsigned char* clustered,
                              int* output_members, int* output_offsets,
                              int* output_seeds, Progress* progress, int n) {
    const unsigned long long key = *winner;
    const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned>(key));
    const int size = static_cast<int>(key >> 32);
    const int start = progress->written;
    const int cluster = progress->cluster_count;
    // The metadata must remain unchanged until every thread has read it.
    __syncthreads();
    if (size == 1) {
        // If even the largest candidate is a singleton, all remaining rounds
        // produce singleton seeds in ascending order. Compact that exact tail
        // in one GPU pass instead of launching a graph for every point.
        __shared__ int warp_offsets[THREADS / 32];
        __shared__ int total;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x / 32;
        if (threadIdx.x == 0) total = 0;
        __syncthreads();
        for (int base = 0; base < n; base += THREADS) {
            const int point = base + threadIdx.x;
            const bool keep = point < n && !clustered[point];
            const unsigned mask = __ballot_sync(0xffffffffu, keep);
            if (lane == 0) warp_offsets[warp] = __popc(mask);
            __syncthreads();
            if (threadIdx.x == 0) {
                for (int w = 0; w < THREADS / 32; ++w) {
                    const int count = warp_offsets[w];
                    warp_offsets[w] = total;
                    total += count;
                }
            }
            __syncthreads();
            if (keep) {
                const int position = warp_offsets[warp] + __popc(mask & ((1u << lane) - 1u));
                output_members[start + position] = point;
                output_offsets[cluster + position] = start + position;
                output_seeds[cluster + position] = point;
                clustered[point] = 1;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            output_offsets[cluster + total] = start + total;
            progress->written = start + total;
            progress->remaining = 0;
            progress->cluster_count = cluster + total;
            *winner = 0;
        }
        return;
    }
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const int member = members[offsets[seed] + i];
        clustered[member] = 1;
        output_members[start + i] = member;
    }
    if (threadIdx.x == 0) {
        output_seeds[cluster] = seed;
        output_offsets[cluster] = start;
        output_offsets[cluster + 1] = start + size;
        progress->written = start + size;
        progress->remaining -= size;
        progress->cluster_count = cluster + 1;
        *winner = 0;
    }
}

} // namespace

// The greedy rounds remain ordered, while seed growth, distance evaluation,
// winner selection and membership updates all execute on the GPU.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    if (!n) return {};
    Execution execution;
    const auto stream = execution.stream;
    DeviceBuffer<Point> device_points(n);
    CUDA_CHECK(cudaMemcpyAsync(device_points.data, points.data(), n * sizeof(Point),
                               cudaMemcpyHostToDevice, stream));

    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    DeviceBuffer<double> distances;
    const size_t matrix_entries = static_cast<size_t>(n) * n;
    // A dense cache saves repeated double precision square roots when it fits.
    // Large inputs retain the same CUDA algorithm with distances computed on demand.
    if (matrix_entries <= free_memory / (4 * sizeof(double))) {
        distances.allocate(matrix_entries);
        const dim3 blocks((n + 15) / 16, (n + 15) / 16);
        computeDistances<<<blocks, dim3(16, 16), 0, stream>>>(device_points.data,
                                                            distances.data, n);
        CUDA_CHECK(cudaGetLastError());
    }
    DeviceBuffer<int> counts(n);
    countNeighbors<<<n, THREADS, 0, stream>>>(device_points.data, distances.data,
                                             n, threshold, counts.data);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> host_counts(n);
    CUDA_CHECK(cudaMemcpyAsync(host_counts.data(), counts.data, n * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<size_t> offsets(n + 1, 0);
    int max_degree = 0;
    for (int i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + host_counts[i];
        max_degree = std::max(max_degree, host_counts[i]);
    }
    counts.allocate(0);
    const size_t entries = offsets.back();
    const size_t shared_bytes = static_cast<size_t>(max_degree) * sizeof(double);
    const bool use_shared = shared_bytes <= 32 * 1024;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    const size_t workspace_bytes = entries * (2 * sizeof(int) +
                                              (use_shared ? 0 : sizeof(double)));
    if (distances.data && workspace_bytes > free_memory / 2) distances.allocate(0);

    DeviceBuffer<size_t> device_offsets(n + 1);
    DeviceBuffer<int> neighbors(entries), members(entries), sizes(n);
    DeviceBuffer<double> global_maxima(use_shared ? 0 : entries);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<unsigned long long> winner(1);
    DeviceBuffer<Progress> progress(1);
    DeviceBuffer<int> output_members(n), output_offsets(n + 1), output_seeds(n);
    CUDA_CHECK(cudaMemcpyAsync(device_offsets.data, offsets.data(), (n + 1) * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
    fillNeighbors<<<n, THREADS, 0, stream>>>(device_points.data, distances.data, n, threshold,
                                            device_offsets.data, neighbors.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(sizes.data, 0, n * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(clustered.data, 0, n, stream));
    CUDA_CHECK(cudaMemsetAsync(winner.data, 0, sizeof(unsigned long long), stream));
    *execution.progress = Progress{n, 0, 0};
    CUDA_CHECK(cudaMemcpyAsync(progress.data, execution.progress, sizeof(Progress),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Replaying this graph avoids repeated kernel setup and transfers only three
    // integers per round. Candidate lists and final memberships stay on device.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (use_shared) {
        growCandidates<true><<<n, THREADS, shared_bytes, stream>>>(
            device_points.data, distances.data, n, threshold, device_offsets.data,
            neighbors.data, clustered.data, sizes.data, members.data,
            global_maxima.data, winner.data);
    } else {
        growCandidates<false><<<n, THREADS, 0, stream>>>(
            device_points.data, distances.data, n, threshold, device_offsets.data,
            neighbors.data, clustered.data, sizes.data, members.data,
            global_maxima.data, winner.data);
    }
    CUDA_CHECK(cudaGetLastError());
    commitCluster<<<1, THREADS, 0, stream>>>(winner.data, device_offsets.data, members.data,
        clustered.data, output_members.data, output_offsets.data, output_seeds.data, progress.data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(execution.progress, progress.data, sizeof(Progress),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &execution.graph));
    CUDA_CHECK(cudaGraphInstantiate(&execution.executable, execution.graph, nullptr, nullptr, 0));
    do {
        const int previous_remaining = execution.progress->remaining;
        CUDA_CHECK(cudaGraphLaunch(execution.executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (execution.progress->remaining >= previous_remaining)
            throw std::runtime_error("CUDA clustering did not make progress");
    } while (execution.progress->remaining);

    const int cluster_count = execution.progress->cluster_count;
    std::vector<int> host_members(n), host_offsets(cluster_count + 1), host_seeds(cluster_count);
    CUDA_CHECK(cudaMemcpyAsync(host_members.data(), output_members.data, n * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_offsets.data(), output_offsets.data,
                               (cluster_count + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_seeds.data(), output_seeds.data, cluster_count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<Cluster> clusters;
    clusters.reserve(cluster_count);
    for (int i = 0; i < cluster_count; ++i) {
        Cluster cluster;
        cluster.seed_point = host_seeds[i];
        cluster.members.assign(host_members.begin() + host_offsets[i],
                               host_members.begin() + host_offsets[i + 1]);
        clusters.push_back(std::move(cluster));
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    std::vector<Cluster> clusters;
    try {
        clusters = qtClustering(points, threshold);
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA clustering failed: %s\n", error.what());
        return 1;
    }
    
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
