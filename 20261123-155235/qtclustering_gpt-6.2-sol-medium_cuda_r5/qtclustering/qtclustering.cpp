// QT Clustering Benchmark - CUDA version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/device/device_scan.cuh>
#include <vector>

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
        // Below this size the original group-size draw cannot make progress.
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
    return std::sqrt(std::fma(dx, dx, dy * dy));
}

namespace {

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) {
            throw std::runtime_error("device allocation size overflow");
        }
        if (count != 0) {
            checkCuda(cudaMalloc(&data_, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

constexpr int warpSizeQT = 32;
constexpr int threadsPerBlock = 128;
constexpr int warpsPerBlock = threadsPerBlock / warpSizeQT;
constexpr unsigned fullWarp = 0xffffffffu;
constexpr int registerSlots = 4;
constexpr int registerCapacity = registerSlots * warpSizeQT;
using WinnerKey = unsigned long long;

struct IndexedPoint {
    Point point;
    int index;
};

struct RowOffset {
    size_t neighbors;
    size_t scores;
};

struct AddOffsets {
    __host__ __device__ RowOffset operator()(RowOffset a, RowOffset b) const {
        return {a.neighbors + b.neighbors, a.scores + b.scores};
    }
};

struct Progress {
    WinnerKey winner;
    int points;
    int clusters;
    int done;
};

// The original compares rounded Euclidean distances to a strict threshold.
// Find the greatest squared distance that passes that same comparison.
double squaredCutoff(double threshold) {
    if (!(threshold > 0.0)) return -1.0;

    uint64_t low = 0;
    uint64_t high = std::bit_cast<uint64_t>(std::numeric_limits<double>::infinity());
    while (low + 1 < high) {
        const uint64_t mid = low + (high - low) / 2;
        if (std::sqrt(std::bit_cast<double>(mid)) < threshold) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return std::bit_cast<double>(low);
}

__device__ double squaredDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __fma_rn(dx, dx, dy * dy);
}

__device__ int neighborhoodBound(const IndexedPoint* sorted, int count, double x,
                                 double cutoff, bool upper) {
    int first = 0;
    int last = count;
    while (first < last) {
        const int mid = first + (last - first) / 2;
        const double dx = sorted[mid].point.x - x;
        const bool before = upper ? (dx <= 0.0 || dx * dx <= cutoff)
                                  : (dx < 0.0 && dx * dx > cutoff);
        if (before) {
            first = mid + 1;
        } else {
            last = mid;
        }
    }
    return first;
}

// Two passes build compact rows of eligible neighbors. The seed itself is implicit.
template <bool WriteNeighbors>
__global__ void buildNeighborhoods(const Point* points, const IndexedPoint* sorted,
                                   int count, double cutoff, bool finitePoints,
                                   RowOffset* offsets, int* neighbors) {
    const int lane = threadIdx.x % warpSizeQT;
    const int seed = blockIdx.x * warpsPerBlock + threadIdx.x / warpSizeQT;
    if (seed >= count) return;

    const Point center = points[seed];
    int first = 0;
    int last = 0;
    if (lane == 0 && cutoff >= 0.0) {
        last = count;
        if (finitePoints) {
            first = neighborhoodBound(sorted, count, center.x, cutoff, false);
            last = neighborhoodBound(sorted, count, center.x, cutoff, true);
        }
    }
    first = __shfl_sync(fullWarp, first, 0);
    last = __shfl_sync(fullWarp, last, 0);

    size_t output = WriteNeighbors ? offsets[seed].neighbors : 0;
    int degree = 0;
    for (int base = first; base < last; base += warpSizeQT) {
        const int column = base + lane;
        int index = -1;
        if (column < last) {
            const IndexedPoint candidate = sorted[column];
            // A NaN distance was ignored by the original maximum-distance update.
            if (candidate.index != seed && !(squaredDistance(candidate.point, center) > cutoff)) {
                index = candidate.index;
            }
        }
        if constexpr (WriteNeighbors) {
            const unsigned mask = __ballot_sync(fullWarp, index >= 0);
            if (index >= 0) {
                neighbors[output + __popc(mask & ((1u << lane) - 1))] = index;
            }
            output += __popc(mask);
        } else {
            degree += index >= 0;
        }
    }

    if constexpr (!WriteNeighbors) {
        for (int delta = warpSizeQT / 2; delta > 0; delta /= 2) {
            degree += __shfl_down_sync(fullWarp, degree, delta);
        }
        if (lane == 0) {
            offsets[seed + 1] = {static_cast<size_t>(degree),
                                 degree > registerCapacity ? static_cast<size_t>(degree) : 0};
        }
    }
}

__device__ bool cacheIsCurrent(const int* members, int size, const int* assigned, int lane) {
    bool removed = false;
    for (int i = lane; i < size - 1; i += warpSizeQT) {
        removed |= assigned[members[i]] != 0;
    }
    return !__any_sync(fullWarp, removed);
}

__device__ double warpMinimum(double value) {
    for (int delta = warpSizeQT / 2; delta > 0; delta /= 2) {
        value = fmin(value, __shfl_down_sync(fullWarp, value, delta));
    }
    return __shfl_sync(fullWarp, value, 0);
}

__device__ int warpMinimum(int value) {
#if __CUDA_ARCH__ >= 800
    return __reduce_min_sync(fullWarp, value);
#else
    for (int delta = warpSizeQT / 2; delta > 0; delta /= 2) {
        value = min(value, __shfl_down_sync(fullWarp, value, delta));
    }
    return __shfl_sync(fullWarp, value, 0);
#endif
}

__device__ bool sameDistance(double score, double minimum) {
    // Rounded square roots can tie for distinct squares. Their rounding interval
    // spans at most four representable squares, so only nearby scores need sqrt.
    return score == minimum ||
           (__double_as_longlong(score) - __double_as_longlong(minimum) <= 4 &&
            __dsqrt_rn(score) == __dsqrt_rn(minimum));
}

template <int Slots>
__device__ int growInRegisters(int seed, int lane, int degree, double cutoff,
                               const Point* points, const int* assigned,
                               const int* neighbors, int* members) {
    Point positions[Slots];
    int indices[Slots];
    double scores[Slots];
    #pragma unroll
    for (int slot = 0; slot < Slots; ++slot) {
        const int offset = lane + slot * warpSizeQT;
        const int index = offset < degree ? neighbors[offset] : -1;
        indices[slot] = index;
        positions[slot] = {};
        scores[slot] = CUDART_INF;
        if (index >= 0 && !assigned[index]) {
            positions[slot] = points[index];
            scores[slot] = 0.0;
        }
    }

    int size = 1;
    Point latest = points[seed];
    while (true) {
        double localMinimum = CUDART_INF;
        #pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const double score = fmax(scores[slot], squaredDistance(positions[slot], latest));
            scores[slot] = score <= cutoff ? score : CUDART_INF;
            localMinimum = fmin(localMinimum, scores[slot]);
        }
        const double minimum = warpMinimum(localMinimum);
        if (minimum == CUDART_INF) return size;

        int closest = INT_MAX;
        #pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            if (sameDistance(scores[slot], minimum)) closest = min(closest, indices[slot]);
        }
        closest = warpMinimum(closest);

        if (lane == 0) members[size - 1] = closest;
        ++size;
        if (size > degree) return size;

        #pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            if (indices[slot] == closest) scores[slot] = CUDART_INF;
        }
        latest = points[closest];
    }
}

template <bool Complete = false>
__device__ int growWithScratch(int seed, int lane, int degree, double cutoff,
                               const Point* points, const int* assigned,
                               const int* neighbors, int* members, double* scores) {
    int size = 1;
    int latest = seed;
    int limit = degree;
    while (true) {
        const Point center = points[latest];
        double localMinimum = CUDART_INF;
        int nextLimit = 0;
        for (int i = lane; i < limit; i += warpSizeQT) {
            double score = size == 1 ? 0.0 : scores[i];
            if (score == CUDART_INF) continue;

            const int index = Complete ? i + 1 : neighbors[i];
            if (index == latest || (!Complete && size == 1 && assigned[index])) {
                score = CUDART_INF;
            } else {
                score = fmax(score, squaredDistance(points[index], center));
            }
            if (score <= cutoff) {
                localMinimum = fmin(localMinimum, score);
                nextLimit = i + 1;
            } else {
                score = CUDART_INF;
            }
            scores[i] = score;
        }
        limit = nextLimit;

        const double minimum = warpMinimum(localMinimum);
        if (minimum == CUDART_INF) return size;

        int closest = INT_MAX;
        for (int i = lane; i < limit; i += warpSizeQT) {
            if (sameDistance(scores[i], minimum)) {
                closest = min(closest, Complete ? i + 1 : neighbors[i]);
            }
        }
        latest = warpMinimum(closest);
        if (lane == 0) members[size - 1] = latest;
        ++size;
        if (size > degree) return size;
    }
}

__global__ void growCompleteCluster(const Point* points, int count, double cutoff,
                                    int* members, double* scores) {
    if (threadIdx.x == 0) members[0] = 0;
    growWithScratch<true>(0, threadIdx.x, count - 1, cutoff, points,
                          nullptr, nullptr, members + 1, scores);
}

std::vector<Cluster> completeCluster(const Point* points, int count, double cutoff) {
    DeviceBuffer<int> members(count);
    DeviceBuffer<double> scores(count - 1);
    growCompleteCluster<<<1, warpSizeQT>>>(points, count, cutoff, members.get(), scores.get());
    checkCuda(cudaGetLastError());

    std::vector<Cluster> clusters(1);
    clusters.front().seed_point = 0;
    clusters.front().members.resize(count);
    checkCuda(cudaMemcpy(clusters.front().members.data(), members.get(), count * sizeof(int),
                         cudaMemcpyDeviceToHost));
    return clusters;
}

__global__ void evaluateSeeds(const Point* points, const RowOffset* offsets,
                              const int* neighbors, int* members, double* scores,
                              int* sizes, const int* assigned, int count,
                              double cutoff, Progress* progress) {
    if (progress->done) return;

    __shared__ WinnerKey warpWinners[warpsPerBlock];
    const int warp = threadIdx.x / warpSizeQT;
    const int lane = threadIdx.x % warpSizeQT;
    const int seed = blockIdx.x * warpsPerBlock + warp;
    WinnerKey key = 0;

    if (seed < count && !assigned[seed]) {
        const RowOffset row = offsets[seed];
        const int degree = static_cast<int>(offsets[seed + 1].neighbors - row.neighbors);
        if (degree > 0) {
            int* candidate = members + row.neighbors;
            int size = sizes[seed];
            // Removing points outside a candidate cannot change any of its greedy choices.
            if (size == 0 || !cacheIsCurrent(candidate, size, assigned, lane)) {
                const int* eligible = neighbors + row.neighbors;
                if (degree <= warpSizeQT) {
                    size = growInRegisters<1>(seed, lane, degree, cutoff, points,
                                              assigned, eligible, candidate);
                } else if (degree <= registerCapacity) {
                    size = growInRegisters<registerSlots>(seed, lane, degree, cutoff, points,
                                                          assigned, eligible, candidate);
                } else {
                    size = growWithScratch(seed, lane, degree, cutoff, points, assigned,
                                           eligible, candidate, scores + row.scores);
                }
                if (lane == 0) sizes[seed] = size;
            }
            if (size > 1) {
                // Atomic max prefers the largest cluster, then the earliest seed.
                key = (static_cast<WinnerKey>(size) << 32) | (UINT_MAX - seed);
            }
        }
    }

    if (lane == 0) warpWinners[warp] = key;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < warpsPerBlock; ++i) {
            key = max(key, warpWinners[i]);
        }
        if (key != 0) atomicMax(&progress->winner, key);
    }
}

__global__ void commitWinner(const RowOffset* offsets, const int* members,
                              int* assigned, int count, int* output, int* outputSizes,
                              Progress* progress) {
    __shared__ WinnerKey key;
    __shared__ int start;
    if (threadIdx.x == 0) {
        key = progress->winner;
        start = progress->points;
    }
    __syncthreads();

    if (key == 0) {
        // Once all candidates are singletons, the remaining seeds finish in index order.
        if (threadIdx.x == 0) progress->done = 1;
        return;
    }

    const int size = static_cast<int>(key >> 32);
    const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned>(key));
    const int* candidate = members + offsets[seed].neighbors;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const int member = i == 0 ? seed : candidate[i - 1];
        assigned[member] = 1;
        output[start + i] = member;
    }
    if (threadIdx.x == 0) {
        outputSizes[progress->clusters++] = size;
        progress->points = start + size;
        progress->winner = 0;
        progress->done = start + size == count;
    }
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int count = static_cast<int>(points.size());
    if (count == 0) return {};

    const double cutoff = squaredCutoff(threshold);
    DeviceBuffer<Point> devicePoints(count);
    checkCuda(cudaMemcpy(devicePoints.get(), points.data(), count * sizeof(Point),
                         cudaMemcpyHostToDevice));

    Point lower = points.front();
    Point upper = points.front();
    bool finitePoints = true;
    for (const Point& point : points) {
        lower = {std::min(lower.x, point.x), std::min(lower.y, point.y)};
        upper = {std::max(upper.x, point.x), std::max(upper.y, point.y)};
        finitePoints &= std::isfinite(point.x) && std::isfinite(point.y);
    }
    const double dx = upper.x - lower.x;
    const double dy = upper.y - lower.y;
    // If every seed can reach every point, seed zero must win.
    if (finitePoints && std::fma(dx, dx, dy * dy) <= cutoff) {
        return completeCluster(devicePoints.get(), count, cutoff);
    }

    std::vector<IndexedPoint> sorted;
    sorted.reserve(count);
    for (int i = 0; i < count; ++i) sorted.push_back({points[i], i});
    if (finitePoints) {
        std::sort(sorted.begin(), sorted.end(), [](const IndexedPoint& a, const IndexedPoint& b) {
            return a.point.x < b.point.x;
        });
    }

    DeviceBuffer<IndexedPoint> deviceSorted(count);
    DeviceBuffer<RowOffset> offsets(static_cast<size_t>(count) + 1);
    checkCuda(cudaMemcpy(deviceSorted.get(), sorted.data(), count * sizeof(IndexedPoint),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(offsets.get(), 0, sizeof(RowOffset)));

    const int blocks = (count - 1) / warpsPerBlock + 1;
    buildNeighborhoods<false><<<blocks, threadsPerBlock>>>(
        devicePoints.get(), deviceSorted.get(), count, cutoff, finitePoints, offsets.get(), nullptr);
    checkCuda(cudaGetLastError());

    // Scan both the adjacency sizes and the workspace needed by high-degree seeds.
    RowOffset* rows = offsets.get() + 1;
    size_t scanBytes = 0;
    checkCuda(cub::DeviceScan::InclusiveScan(nullptr, scanBytes, rows, rows, AddOffsets{}, count));
    DeviceBuffer<unsigned char> scanStorage(scanBytes);
    checkCuda(cub::DeviceScan::InclusiveScan(scanStorage.get(), scanBytes, rows, rows,
                                             AddOffsets{}, count));
    RowOffset totals;
    checkCuda(cudaMemcpy(&totals, offsets.get() + count, sizeof(totals), cudaMemcpyDeviceToHost));

    if (totals.neighbors == static_cast<size_t>(count) * (count - 1)) {
        return completeCluster(devicePoints.get(), count, cutoff);
    }

    DeviceBuffer<int> neighbors(totals.neighbors);
    DeviceBuffer<int> members(totals.neighbors);
    DeviceBuffer<double> scores(totals.scores);
    DeviceBuffer<int> sizes(count);
    DeviceBuffer<int> assigned(count);
    DeviceBuffer<int> output(count);
    DeviceBuffer<int> outputSizes(count);
    DeviceBuffer<Progress> deviceProgress(1);
    checkCuda(cudaMemset(sizes.get(), 0, count * sizeof(int)));
    checkCuda(cudaMemset(assigned.get(), 0, count * sizeof(int)));
    checkCuda(cudaMemset(deviceProgress.get(), 0, sizeof(Progress)));
    buildNeighborhoods<true><<<blocks, threadsPerBlock>>>(
        devicePoints.get(), deviceSorted.get(), count, cutoff, finitePoints,
        offsets.get(), neighbors.get());
    checkCuda(cudaGetLastError());

    Progress progress{};
    for (int batch = 1; !progress.done; batch = 8) {
        for (int round = 0; round < batch; ++round) {
            evaluateSeeds<<<blocks, threadsPerBlock>>>(
                devicePoints.get(), offsets.get(), neighbors.get(), members.get(), scores.get(),
                sizes.get(), assigned.get(), count, cutoff, deviceProgress.get());
            commitWinner<<<1, threadsPerBlock>>>(
                offsets.get(), members.get(), assigned.get(), count, output.get(),
                outputSizes.get(), deviceProgress.get());
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(&progress, deviceProgress.get(), sizeof(progress), cudaMemcpyDeviceToHost));
    }

    std::vector<int> order(progress.points);
    std::vector<int> cardinalities(progress.clusters);
    if (!order.empty()) {
        checkCuda(cudaMemcpy(order.data(), output.get(), order.size() * sizeof(int),
                             cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(cardinalities.data(), outputSizes.get(),
                             cardinalities.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }

    std::vector<Cluster> clusters;
    clusters.reserve(progress.clusters + (count - progress.points));
    auto member = order.begin();
    for (int size : cardinalities) {
        clusters.push_back({std::vector<int>(member, member + size), *member});
        member += size;
    }
    if (progress.points < count) {
        std::vector<bool> clustered(count, false);
        for (int index : order) clustered[index] = true;
        for (int seed = 0; seed < count; ++seed) {
            if (!clustered[seed]) clusters.push_back({{seed}, seed});
        }
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
