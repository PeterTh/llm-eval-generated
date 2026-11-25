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
#include <stdexcept>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct alignas(16) Point {
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

constexpr int blockSize = 128;
constexpr int sharedCapacity = 1024;
using Offset = unsigned long long;
using Winner = unsigned long long;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data = nullptr;
};

struct RunState {
    Winner winner;
    int pointCount;
    int clusterCount;
    int finished;
};

struct Choice {
    double squaredDiameter;
    int point;
};

struct EarlierChoice {
    __device__ Choice operator()(Choice a, Choice b) const {
        if (a.squaredDiameter == b.squaredDiameter) return a.point < b.point ? a : b;
        if (b.squaredDiameter < a.squaredDiameter) {
            const Choice swap = a;
            a = b;
            b = swap;
        }
        // Positive doubles whose square roots round to the same value are
        // within eight encodings of each other. Only check roots when the tie
        // could change which point wins.
        if (b.point < a.point &&
            __double_as_longlong(b.squaredDiameter) - __double_as_longlong(a.squaredDiameter) <= 8 &&
            __dsqrt_rn(a.squaredDiameter) == __dsqrt_rn(b.squaredDiameter)) {
            return b;
        }
        return a;
    }
};

__device__ double squaredDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
#if defined(__FMA__)
    const double squared = __fma_rn(dx, dx, __dmul_rn(dy, dy));
#else
    const double squared = __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
#endif
    // The reference's max operation ignores unordered distances.
    return isnan(squared) ? 0.0 : squared;
}

// Find the first squared distance whose rounded root fails the strict cutoff.
double squaredCutoff(double threshold) {
    Offset first = 0;
    Offset last = 0x7ff0000000000000ull; // Positive infinity.
    while (first < last) {
        const Offset middle = first + (last - first) / 2;
        double squared;
        std::memcpy(&squared, &middle, sizeof(squared));
        if (std::sqrt(squared) < threshold) first = middle + 1;
        else last = middle;
    }
    double cutoff;
    std::memcpy(&cutoff, &first, sizeof(cutoff));
    return cutoff;
}

// Each row reserves a slot for the seed, followed by its eligible neighbors.
__global__ void countNeighbors(const Point* points, int count, double cutoff,
                               Offset* degrees, int* maxDegree) {
    using Reduce = cub::BlockReduce<int, blockSize>;
    __shared__ Reduce::TempStorage storage;
    const int seed = blockIdx.x;
    const Point origin = points[seed];
    int degree = 0;
    for (Offset i = threadIdx.x; i < static_cast<Offset>(count); i += blockSize) {
        degree += i != static_cast<Offset>(seed) && squaredDistance(origin, points[i]) < cutoff;
    }
    degree = Reduce(storage).Sum(degree);
    if (threadIdx.x == 0) {
        degrees[seed + 1] = static_cast<Offset>(degree) + 1;
        atomicMax(maxDegree, degree);
    }
}

__global__ void storeNeighbors(const Point* points, int count, double cutoff,
                               const Offset* offsets, int* neighbors) {
    __shared__ int stored;
    if (threadIdx.x == 0) stored = 0;
    __syncthreads();

    const int seed = blockIdx.x;
    const Point origin = points[seed];
    int* row = neighbors + offsets[seed] + 1;
    const int lane = threadIdx.x % warpSize;
    for (Offset base = threadIdx.x; base - lane < static_cast<Offset>(count);
         base += blockSize) {
        const bool eligible = base < static_cast<Offset>(count) &&
            base != static_cast<Offset>(seed) && squaredDistance(origin, points[base]) < cutoff;
        const unsigned mask = __ballot_sync(0xffffffffu, eligible);
        if (mask != 0) {
            int first = 0;
            if (lane == 0) first = atomicAdd(&stored, __popc(mask));
            first = __shfl_sync(0xffffffffu, first, 0);
            if (eligible) {
                row[first + __popc(mask & ((1u << lane) - 1))] = static_cast<int>(base);
            }
        }
    }
}

__device__ double extendDiameter(double diameter, Point candidate, Point last,
                                 double cutoff) {
    const double next = fmax(diameter, squaredDistance(candidate, last));
    return next < cutoff ? next : CUDART_INF;
}

template <bool complete>
__global__ void evaluateCandidates(const Point* points, int count, double cutoff,
                                   const Offset* offsets, const int* neighbors,
                                   const unsigned char* assigned, int* members, int* sizes,
                                   double* largeScores, RunState* state) {
    using Reduce = cub::BlockReduce<Choice, blockSize>;
    __shared__ Reduce::TempStorage reduction;
    __shared__ int selected;
    extern __shared__ double sharedScores[];

    const int seed = blockIdx.x;
    if (state->finished || assigned[seed]) return;

    const Offset begin = complete ? 0 : offsets[seed];
    const int degree = complete ? count - 1 : static_cast<int>(offsets[seed + 1] - begin - 1);
    const int* row = complete || degree == 0 ? nullptr : neighbors + begin + 1;
    int* candidate = members + begin;
    const int lane = threadIdx.x;
    int size = sizes[seed];

    // A deletion can only change a greedy candidate if it removes a member.
    bool invalid = size == 0;
    for (int i = lane + 1; i < size; i += blockSize) {
        invalid |= assigned[candidate[i]] != 0;
    }
    if (__syncthreads_or(invalid)) {
        double* scores = degree <= sharedCapacity ? sharedScores : largeScores + begin;
        const int point = lane < degree ? (complete ? lane + 1 : row[lane]) : -1;
        const bool eligible = point >= 0 && !assigned[point];
        const Point position = eligible ? points[point] : Point{};
        Point last = points[seed];
        double diameter = eligible ? 0.0 : CUDART_INF;

        for (int i = lane + blockSize; i < degree; i += blockSize) {
            const int other = complete ? i + 1 : row[i];
            scores[i - blockSize] = assigned[other] ? CUDART_INF : 0.0;
        }
        if (lane == 0) candidate[0] = seed;
        size = 1;

        // Only the newest member can increase a candidate's diameter.
        while (size <= degree) {
            Choice best{CUDART_INF, -1};
            int bestSlot = lane;
            if (diameter < CUDART_INF) {
                diameter = extendDiameter(diameter, position, last, cutoff);
                if (diameter < CUDART_INF) best = {diameter, point};
            }
            for (int i = lane + blockSize; i < degree; i += blockSize) {
                double& score = scores[i - blockSize];
                if (score < CUDART_INF) {
                    const int other = complete ? i + 1 : row[i];
                    score = extendDiameter(score, points[other], last, cutoff);
                    best = EarlierChoice{}(best, {score, other});
                    if (best.point == other) bestSlot = i;
                }
            }

            const Choice winner = Reduce(reduction).Reduce(best, EarlierChoice{});
            if (lane == 0) {
                selected = winner.point;
                if (winner.point >= 0) candidate[size] = winner.point;
            }
            __syncthreads();
            const int next = selected;
            if (next < 0) break;

            if (best.point == next) {
                if (bestSlot == lane) diameter = CUDART_INF;
                else scores[bestSlot - blockSize] = CUDART_INF;
            }
            last = points[next];
            ++size;
            __syncthreads();
        }

        if (lane == 0) sizes[seed] = size;
    }

    if (lane == 0) {
        const Winner key = (static_cast<Winner>(size) << 32) | ~static_cast<unsigned>(seed);
        atomicMax(&state->winner, key);
    }
}

__global__ void commitWinner(const Offset* offsets, const int* members,
                             unsigned char* assigned, int* orderedMembers,
                             int* clusterSizes, int count, RunState* state) {
    if (state->finished) return;

    const Winner winner = state->winner;
    const int size = winner >> 32;
    const int seed = static_cast<int>(~static_cast<unsigned>(winner));
    const int start = state->pointCount;
    const int cluster = state->clusterCount;
    __syncthreads();

    // Every remaining seed is a singleton if the largest candidate has size one.
    if (size <= 1) {
        if (threadIdx.x == 0) state->finished = 1;
        return;
    }

    const int* candidate = members + offsets[seed];
    for (int i = threadIdx.x; i < size; i += blockSize) {
        const int point = candidate[i];
        assigned[point] = 1;
        orderedMembers[start + i] = point;
    }

    if (threadIdx.x == 0) {
        clusterSizes[cluster] = size;
        state->pointCount = start + size;
        state->clusterCount = cluster + 1;
        state->winner = 0;
        state->finished = start + size == count;
    }
}

struct RoundGraph {
    ~RoundGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (result) cudaFreeHost(result);
    }

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    RunState* result = nullptr;
};

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int count = static_cast<int>(points.size());
    if (count == 0) return {};

    const double cutoff = squaredCutoff(threshold);
    DeviceBuffer<Point> devicePoints(count);
    DeviceBuffer<Offset> offsets(static_cast<size_t>(count) + 1);
    DeviceBuffer<int> maxDegree(1);
    DeviceBuffer<unsigned char> assigned(count);
    DeviceBuffer<int> sizes(count);
    DeviceBuffer<int> orderedMembers(count);
    DeviceBuffer<int> clusterSizes(count);
    DeviceBuffer<RunState> state(1);

    checkCuda(cudaMemcpy(devicePoints.data, points.data(), count * sizeof(Point),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(offsets.data, 0, sizeof(Offset)));
    checkCuda(cudaMemset(maxDegree.data, 0, sizeof(int)));
    checkCuda(cudaMemset(assigned.data, 0, count * sizeof(unsigned char)));
    checkCuda(cudaMemset(sizes.data, 0, count * sizeof(int)));
    checkCuda(cudaMemset(state.data, 0, sizeof(RunState)));

    countNeighbors<<<count, blockSize>>>(
        devicePoints.data, count, cutoff, offsets.data, maxDegree.data);
    checkCuda(cudaGetLastError());
    const size_t offsetCount = static_cast<size_t>(count) + 1;
    size_t scanBytes = 0;
    checkCuda(cub::DeviceScan::InclusiveSum(nullptr, scanBytes, offsets.data,
                                          offsets.data, offsetCount));
    DeviceBuffer<unsigned char> scanStorage(scanBytes);
    checkCuda(cub::DeviceScan::InclusiveSum(scanStorage.data, scanBytes, offsets.data,
                                          offsets.data, offsetCount));

    Offset entryCount;
    int largestDegree;
    checkCuda(cudaMemcpy(&entryCount, offsets.data + count, sizeof(entryCount),
                         cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(&largestDegree, maxDegree.data, sizeof(largestDegree),
                         cudaMemcpyDeviceToHost));

    // In a complete graph every seed reaches every point, so only seed zero
    // needs to grow a candidate to determine the original insertion order.
    const bool complete = entryCount == static_cast<Offset>(count) * count;
    const Offset candidateEntries = complete ? count : entryCount;
    DeviceBuffer<int> neighbors(!complete && largestDegree > 0 ? entryCount : 0);
    DeviceBuffer<int> members(candidateEntries);
    DeviceBuffer<double> largeScores(largestDegree > sharedCapacity ? candidateEntries : 0);

    RoundGraph rounds;
    checkCuda(cudaStreamCreateWithFlags(&rounds.stream, cudaStreamNonBlocking));
    checkCuda(cudaMallocHost(&rounds.result, sizeof(RunState)));
    if (neighbors.data) {
        storeNeighbors<<<count, blockSize, 0, rounds.stream>>>(
            devicePoints.data, count, cutoff, offsets.data, neighbors.data);
        checkCuda(cudaGetLastError());
    }

    // Batch rounds to keep the winner selection and removal on the device.
    const int roundsPerLaunch = complete || largestDegree == 0 ? 1 : 8;
    const size_t sharedBytes =
        std::max(0, std::min(largestDegree, sharedCapacity) - blockSize) * sizeof(double);
    checkCuda(cudaStreamBeginCapture(rounds.stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < roundsPerLaunch; ++i) {
        if (complete) {
            evaluateCandidates<true><<<1, blockSize, sharedBytes, rounds.stream>>>(
                devicePoints.data, count, cutoff, offsets.data, nullptr, assigned.data,
                members.data, sizes.data, largeScores.data, state.data);
        } else {
            evaluateCandidates<false><<<count, blockSize, sharedBytes, rounds.stream>>>(
                devicePoints.data, count, cutoff, offsets.data, neighbors.data, assigned.data,
                members.data, sizes.data, largeScores.data, state.data);
        }
        checkCuda(cudaGetLastError());
        commitWinner<<<1, blockSize, 0, rounds.stream>>>(
            offsets.data, members.data, assigned.data, orderedMembers.data,
            clusterSizes.data, count, state.data);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaMemcpyAsync(rounds.result, state.data, sizeof(RunState),
                              cudaMemcpyDeviceToHost, rounds.stream));
    checkCuda(cudaStreamEndCapture(rounds.stream, &rounds.graph));
    checkCuda(cudaGraphInstantiate(&rounds.executable, rounds.graph, nullptr, nullptr, 0));
    do {
        checkCuda(cudaGraphLaunch(rounds.executable, rounds.stream));
        checkCuda(cudaStreamSynchronize(rounds.stream));
    } while (!rounds.result->finished);

    std::vector<int> output(rounds.result->pointCount);
    std::vector<int> counts(rounds.result->clusterCount);
    if (!counts.empty()) {
        checkCuda(cudaMemcpy(output.data(), orderedMembers.data, output.size() * sizeof(int),
                             cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(counts.data(), clusterSizes.data, counts.size() * sizeof(int),
                             cudaMemcpyDeviceToHost));
    }

    std::vector<Cluster> clusters;
    clusters.reserve(counts.size() + count - output.size());
    std::vector<unsigned char> used(count, 0);
    auto begin = output.begin();
    for (int size : counts) {
        const auto end = begin + size;
        clusters.push_back({std::vector<int>(begin, end), *begin});
        for (; begin != end; ++begin) used[*begin] = 1;
    }
    for (int point = 0; point < count; ++point) {
        if (!used[point]) clusters.push_back({{point}, point});
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
    } catch (const std::runtime_error& error) {
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
