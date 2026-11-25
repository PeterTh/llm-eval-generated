// QT Clustering Benchmark
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
#include <cstdint>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

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

constexpr int warpSizeCuda = 32;
constexpr int blockSize = 128;
constexpr int warpsPerBlock = blockSize / warpSizeCuda;
constexpr unsigned fullWarp = 0xffffffffu;
constexpr int warpNeighbors = 128;
constexpr int blockNeighbors = 4 * blockSize;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) {
        if (size > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA buffer size overflow");
        }
        if (size != 0) {
            checkCuda(cudaMalloc(&data, size * sizeof(T)), "cudaMalloc");
        }
    }

    ~DeviceBuffer() { if (data) cudaFree(data); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data; }

private:
    T* data = nullptr;
};

__device__ double squaredDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double yy = __dmul_rn(dy, dy);
#ifdef __FMA__
    // Match the original host build's contraction of dx * dx + dy * dy.
    return __fma_rn(dx, dx, yy);
#else
    return __dadd_rn(__dmul_rn(dx, dx), yy);
#endif
}

// Find the largest squared distance whose rounded square root passes the
// original strict comparison. This lets every edge test skip the square root.
double squaredCutoff(double threshold) {
    if (!(threshold > 0.0)) return -1.0;

    uint64_t low = 0;
    uint64_t high = UINT64_C(0x7ff0000000000000); // Positive infinity.
    while (high - low > 1) {
        const uint64_t mid = low + (high - low) / 2;
        double squared;
        std::memcpy(&squared, &mid, sizeof(squared));
        if (std::sqrt(squared) < threshold) {
            low = mid;
        } else {
            high = mid;
        }
    }
    double cutoff;
    std::memcpy(&cutoff, &low, sizeof(cutoff));
    return cutoff;
}

// Build rows in point-index order so a smaller neighbor slot wins ties.
template <bool writeNeighbors>
__global__ void buildNeighborhoods(const Point* points, int n, double cutoff,
                                   int seedCount, const size_t* offsets,
                                   size_t* degrees, int* neighbors, int* maxDegree) {
    const int seed = blockIdx.x * warpsPerBlock + threadIdx.x / warpSizeCuda;
    const int lane = threadIdx.x % warpSizeCuda;
    if (seed >= seedCount) return;

    const Point origin = points[seed];
    int degree = 0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += warpSizeCuda) {
        const size_t candidate = base + lane;
        const bool fits = candidate < static_cast<size_t>(n) &&
                          candidate != static_cast<size_t>(seed) &&
                          fmax(0.0, squaredDistance(points[candidate], origin)) <= cutoff;
        const unsigned mask = __ballot_sync(fullWarp, fits);
        if constexpr (writeNeighbors) {
            if (fits) {
                const int slot = degree + __popc(mask & ((1u << lane) - 1));
                neighbors[offsets[seed] + slot] = static_cast<int>(candidate);
            }
        }
        degree += __popc(mask);
    }

    if constexpr (!writeNeighbors) {
        if (lane == 0) {
            degrees[seed] = degree;
            if (degree > warpNeighbors) atomicMax(maxDegree, degree);
            if (seed == 0) degrees[n] = 0;
        }
    }
}

constexpr int noNeighbor = std::numeric_limits<int>::max();
constexpr double rejected = std::numeric_limits<double>::infinity();

struct NeighborChoice {
    double distance;
    int slot;
};

struct ClosestNeighbor {
    __device__ NeighborChoice operator()(NeighborChoice a, NeighborChoice b) const {
        return (b.distance < a.distance || (b.distance == a.distance && b.slot < a.slot))
            ? b : a;
    }
};

template <int width>
__device__ int reduceNeighbor(NeighborChoice choice, int rank) {
    const ClosestNeighbor closest;
    if constexpr (width == warpSizeCuda) {
        #pragma unroll
        for (int offset = warpSizeCuda / 2; offset > 0; offset /= 2) {
            const double distance = __shfl_down_sync(fullWarp, choice.distance, offset);
            const int slot = __shfl_down_sync(fullWarp, choice.slot, offset);
            choice = closest(choice, {distance, slot});
        }
        return __shfl_sync(fullWarp, choice.slot, 0);
    } else {
        using Reduce = cub::BlockReduce<NeighborChoice, width>;
        __shared__ typename Reduce::TempStorage reduce;
        __shared__ int winner;
        const auto best = Reduce(reduce).Reduce(choice, closest);
        if (rank == 0) winner = best.slot;
        __syncthreads();
        const int slot = winner;
        __syncthreads();
        return slot;
    }
}

template <int width, int slots>
__device__ int growCandidate(const Point* points, const int* neighbors,
                             const int* clustered, int* selected, double* scratch,
                             int degree, int rank, Point newest, double cutoff) {
    // Update each point's farthest distance against the member just added.
    // A point above the cutoff can never reenter the candidate set.
    double bounds[slots == 0 ? 1 : slots];
    Point positions[slots == 0 ? 1 : slots];
    const int owned = slots == 0 ? degree / width + (rank < degree % width) : slots;

    #pragma unroll
    for (int i = 0; i < owned; ++i) {
        const int slot = rank + i * width;
        const int point = slot < degree ? neighbors[slot] : -1;
        const bool active = point >= 0 && !clustered[point];
        const double initial = active ? 0.0 : rejected;
        if constexpr (slots == 0) {
            scratch[slot] = initial;
        } else {
            positions[i] = active ? points[point] : Point{};
            bounds[i] = initial;
        }
    }

    int count = 1;
    for (;;) {
        NeighborChoice best{rejected, noNeighbor};

        #pragma unroll
        for (int i = 0; i < owned; ++i) {
            const int slot = rank + i * width;
            const double oldBound = slots == 0 ? scratch[slot] : bounds[i];
            if (oldBound != rejected) {
                const Point position = slots == 0 ? points[neighbors[slot]] : positions[i];
                const double bound = fmax(oldBound, squaredDistance(position, newest));
                const double updated = bound <= cutoff ? bound : rejected;
                if constexpr (slots == 0) {
                    scratch[slot] = updated;
                } else {
                    bounds[i] = updated;
                }

                // Rounded square roots can tie even when squared distances differ.
                if (updated != rejected) {
                    const double distance = __dsqrt_rn(updated);
                    if (distance < best.distance) best = {distance, slot};
                }
            }
        }

        const int winner = reduceNeighbor<width>(best, rank);
        if (winner == noNeighbor) return count;

        const int member = neighbors[winner];
        if (rank == 0) selected[count - 1] = member;
        ++count;
        if (count > degree) return count;

        if constexpr (slots == 0) {
            if (rank == winner % width) scratch[winner] = rejected;
        } else {
            #pragma unroll
            for (int i = 0; i < slots; ++i) {
                if (rank + i * width == winner) bounds[i] = rejected;
            }
        }
        newest = points[member];
    }
}

using CandidateKey = unsigned long long;

struct ClusteringState {
    CandidateKey best;
    int remaining;
    int clusterCount;
};

template <int width>
__global__ void evaluateSeeds(const Point* points, int seedCount, double cutoff,
                              const size_t* offsets, const int* neighbors, int* selected,
                              double* scratch, int* sizes, const int* clustered,
                              ClusteringState* state) {
    if (state->remaining == 0) return;

    constexpr int groups = blockSize / width;
    const int group = threadIdx.x / width;
    const int rank = threadIdx.x % width;
    const int seed = blockIdx.x * groups + group;
    CandidateKey key = 0;
    if (seed < seedCount && !clustered[seed]) {
        const size_t begin = offsets[seed];
        const int degree = offsets[seed + 1] - begin;
        int count = 1;
        if (degree != 0) {
            if constexpr (width == warpSizeCuda) {
                if (rank == 0) count = sizes[seed];
                count = __shfl_sync(fullWarp, count, 0);
            } else {
                __shared__ int cachedSize;
                if (rank == 0) cachedSize = sizes[seed];
                __syncthreads();
                count = cachedSize;
            }
        }
        bool dirty = count == 0;
        // Removing an unselected point cannot change a greedy choice. Only a
        // cached candidate that lost a member needs to be generated again.
        if (!dirty) {
            if constexpr (width == warpSizeCuda) {
                for (int base = 0; !dirty && base < count - 1; base += width) {
                    const int slot = base + rank;
                    dirty = __any_sync(fullWarp, slot < count - 1 && clustered[selected[begin + slot]]);
                }
            } else {
                for (int slot = rank; slot < count - 1 && !dirty; slot += width) {
                    dirty = clustered[selected[begin + slot]];
                }
                dirty = __syncthreads_or(dirty);
            }
        }
        if (dirty) {
            const Point origin = points[seed];
            if (degree <= width) {
                count = growCandidate<width, 1>(points, neighbors + begin, clustered, selected + begin,
                                                nullptr, degree, rank, origin, cutoff);
            } else if (degree <= 2 * width) {
                count = growCandidate<width, 2>(points, neighbors + begin, clustered, selected + begin,
                                                nullptr, degree, rank, origin, cutoff);
            } else if (degree <= 4 * width) {
                count = growCandidate<width, 4>(points, neighbors + begin, clustered, selected + begin,
                                                nullptr, degree, rank, origin, cutoff);
            } else if constexpr (width == blockSize) {
                count = growCandidate<width, 0>(points, neighbors + begin, clustered, selected + begin,
                                                scratch + begin, degree, rank, origin, cutoff);
            }
            if (rank == 0) sizes[seed] = count;
        }
        // Maximize cardinality, favoring the lowest seed index on a tie.
        key = (static_cast<CandidateKey>(count) << 32) | (fullWarp - seed);
    }

    if constexpr (groups == 1) {
        if (rank == 0 && key != 0) atomicMax(&state->best, key);
    } else {
        __shared__ CandidateKey winners[groups];
        if (rank == 0) winners[group] = key;
        __syncthreads();
        if (threadIdx.x == 0) {
            CandidateKey best = 0;
            for (int i = 0; i < groups; ++i) best = max(best, winners[i]);
            if (best != 0) atomicMax(&state->best, best);
        }
    }
}

__global__ void commitCluster(int n, const size_t* offsets, const int* selected,
                              int* clustered, int* members, int* ends, ClusteringState* state) {
    __shared__ ClusteringState round;
    if (threadIdx.x == 0) round = *state;
    __syncthreads();
    if (round.remaining == 0) return;

    const int count = round.best >> 32;
    const int seed = fullWarp - static_cast<unsigned>(round.best);
    const int start = n - round.remaining;
    if (count == 1) {
        // Once the largest candidate is a singleton, all remaining points
        // form singleton clusters in ascending point order.
        using Scan = cub::BlockScan<int, blockSize>;
        __shared__ typename Scan::TempStorage scan;
        int written = 0;
        for (size_t base = 0; base < static_cast<size_t>(n); base += blockSize) {
            const size_t point = base + threadIdx.x;
            const bool active = point < static_cast<size_t>(n) && !clustered[point];
            int prefix, total;
            Scan(scan).ExclusiveSum(static_cast<int>(active), prefix, total);
            if (active) {
                members[start + written + prefix] = static_cast<int>(point);
                ends[round.clusterCount + written + prefix] = start + written + prefix + 1;
            }
            written += total;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            *state = {0, 0, round.clusterCount + round.remaining};
        }
        return;
    }

    const size_t begin = offsets[seed];
    for (int i = threadIdx.x; i < count; i += blockSize) {
        const int point = i == 0 ? seed : selected[begin + i - 1];
        members[start + i] = point;
        clustered[point] = 1;
    }
    if (threadIdx.x == 0) {
        ends[round.clusterCount] = start + count;
        *state = {0, round.remaining - count, round.clusterCount + 1};
    }
}

int seedBlocks(int count) {
    return (count - 1) / warpsPerBlock + 1;
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    if (points.empty()) return {};

    const int n = static_cast<int>(points.size());
    const double cutoff = squaredCutoff(threshold);
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<size_t> offsets(static_cast<size_t>(n) + 1);
    checkCuda(cudaMemcpy(devicePoints.get(), points.data(), n * sizeof(Point),
                         cudaMemcpyHostToDevice), "copy input points");

    size_t edgeCount;
    int maxDegree;
    {
        DeviceBuffer<size_t> degrees(static_cast<size_t>(n) + 1);
        DeviceBuffer<int> largest(1);
        checkCuda(cudaMemset(largest.get(), 0, sizeof(int)), "initialize maximum degree");
        buildNeighborhoods<false><<<seedBlocks(n), blockSize>>>(
            devicePoints.get(), n, cutoff, n, nullptr, degrees.get(), nullptr, largest.get());
        checkCuda(cudaGetLastError(), "count neighborhoods");

        size_t tempBytes = 0;
        checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, degrees.get(), offsets.get(),
                                                static_cast<size_t>(n) + 1), "size offset scan");
        DeviceBuffer<unsigned char> temp(tempBytes);
        checkCuda(cub::DeviceScan::ExclusiveSum(temp.get(), tempBytes, degrees.get(), offsets.get(),
                                                static_cast<size_t>(n) + 1), "scan neighborhood sizes");
        checkCuda(cudaMemcpy(&edgeCount, offsets.get() + n, sizeof(edgeCount),
                             cudaMemcpyDeviceToHost), "read total neighborhood size");
        checkCuda(cudaMemcpy(&maxDegree, largest.get(), sizeof(maxDegree),
                             cudaMemcpyDeviceToHost), "read maximum degree");
    }

    // If every pair fits, every seed grows to size n and seed zero wins.
    // Grow just that seed to retain its original member order.
    const bool complete = edgeCount == static_cast<size_t>(n) * (n - 1);
    const int seedCount = complete ? 1 : n;
    if (complete) edgeCount = n - 1;

    DeviceBuffer<int> neighbors(edgeCount);
    DeviceBuffer<int> selected(edgeCount);
    DeviceBuffer<double> scratch(maxDegree > blockNeighbors ? edgeCount : 0);
    DeviceBuffer<int> sizes(seedCount);
    DeviceBuffer<int> clustered(n);
    DeviceBuffer<int> members(n);
    DeviceBuffer<int> ends(n);
    DeviceBuffer<ClusteringState> deviceState(1);

    if (edgeCount != 0) {
        buildNeighborhoods<true><<<seedBlocks(seedCount), blockSize>>>(
            devicePoints.get(), n, cutoff, seedCount, offsets.get(), nullptr, neighbors.get(), nullptr);
        checkCuda(cudaGetLastError(), "fill neighborhoods");
    }
    checkCuda(cudaMemset(sizes.get(), 0, seedCount * sizeof(int)), "initialize candidate sizes");
    checkCuda(cudaMemset(clustered.get(), 0, n * sizeof(int)), "initialize point status");
    ClusteringState state{0, n, 0};
    checkCuda(cudaMemcpy(deviceState.get(), &state, sizeof(state), cudaMemcpyHostToDevice),
              "initialize clustering state");

    // Keep the greedy iterations ordered on the device and batch launches
    // so the host need not synchronize after each committed cluster.
    int rounds = 1;
    do {
        for (int i = 0; i < rounds; ++i) {
            if (maxDegree <= warpNeighbors) {
                evaluateSeeds<warpSizeCuda><<<seedBlocks(seedCount), blockSize>>>(
                    devicePoints.get(), seedCount, cutoff, offsets.get(), neighbors.get(), selected.get(),
                    nullptr, sizes.get(), clustered.get(), deviceState.get());
            } else {
                evaluateSeeds<blockSize><<<seedCount, blockSize>>>(
                    devicePoints.get(), seedCount, cutoff, offsets.get(), neighbors.get(), selected.get(),
                    scratch.get(), sizes.get(), clustered.get(), deviceState.get());
            }
            checkCuda(cudaGetLastError(), "evaluate candidate clusters");
            commitCluster<<<1, blockSize>>>(n, offsets.get(), selected.get(), clustered.get(),
                                            members.get(), ends.get(), deviceState.get());
            checkCuda(cudaGetLastError(), "commit winning cluster");
        }
        checkCuda(cudaMemcpy(&state, deviceState.get(), sizeof(state), cudaMemcpyDeviceToHost),
                  "read clustering progress");
        rounds = std::min(16, state.remaining);
    } while (state.remaining != 0);

    std::vector<int> orderedMembers(n);
    std::vector<int> clusterEnds(state.clusterCount);
    checkCuda(cudaMemcpy(orderedMembers.data(), members.get(), n * sizeof(int), cudaMemcpyDeviceToHost),
              "copy cluster members");
    checkCuda(cudaMemcpy(clusterEnds.data(), ends.get(), clusterEnds.size() * sizeof(int),
                         cudaMemcpyDeviceToHost), "copy cluster boundaries");

    std::vector<Cluster> clusters;
    clusters.reserve(clusterEnds.size());
    int start = 0;
    for (int end : clusterEnds) {
        clusters.push_back({{orderedMembers.begin() + start, orderedMembers.begin() + end},
                            orderedMembers[start]});
        start = end;
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
    
    std::vector<Cluster> clusters;
    std::chrono::high_resolution_clock::time_point cluster_start;
    try {
        // Initialize the CUDA context before timing the clustering work.
        checkCuda(cudaFree(nullptr), "initialize CUDA");
        cluster_start = std::chrono::high_resolution_clock::now();
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
