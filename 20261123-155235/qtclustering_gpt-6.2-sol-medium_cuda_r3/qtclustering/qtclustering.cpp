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

#include <cuda_runtime.h>
#include <math_constants.h>
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
        // The original group-size formula cannot reliably advance at these sizes.
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

namespace {

using Offset = unsigned long long;
using Winner = unsigned long long;
constexpr int blockSize = 128;
constexpr int roundsPerBatch = 16;
constexpr int maxRegisterSlots = 16;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) checkCuda(cudaMalloc(&data_, count * sizeof(T)));
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

struct Neighborhoods {
    const Point* points;
    double* distances;
    Offset* offsets;
    int* neighbors;
    int count;
    int seedCount;
    double threshold;
};

struct Progress {
    int clusters;
    int members;
};

struct ClusteringState {
    int* clustered;
    int* sizes;
    int* candidates;
    Winner* winners;
    int* resultOffsets;
    int* resultMembers;
    Progress* progress;
};

__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
#if defined(__FMA__)
    // Match the native host compiler's contraction of the Euclidean distance.
    const double squared = __fma_rn(dx, dx, __dmul_rn(dy, dy));
#else
    const double squared = __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
#endif
    // The original running maximum starts at zero and ignores NaN distances.
    return fmax(0.0, __dsqrt_rn(squared));
}

__device__ double pairDistance(Neighborhoods graph, int from, int to) {
    if (graph.distances) {
        return graph.distances[static_cast<size_t>(from) * graph.count + to];
    }
    return gpuDistance(graph.points[from], graph.points[to]);
}

// Build a compact row of neighbors for each seed. Every member of a candidate
// cluster must be within the threshold of its seed.
__global__ void countNeighbors(Neighborhoods graph, int* maxDegree) {
    using Reduce = cub::BlockReduce<int, blockSize>;
    __shared__ Reduce::TempStorage reduction;
    const int seed = blockIdx.x;
    const Point point = graph.points[seed];
    int count = 0;
    for (Offset i = threadIdx.x; i < graph.count; i += blockSize) {
        const double d = gpuDistance(point, graph.points[i]);
        if (graph.distances) graph.distances[seed * static_cast<Offset>(graph.count) + i] = d;
        count += i != seed && d < graph.threshold;
    }
    count = Reduce(reduction).Sum(count);
    if (threadIdx.x == 0) {
        graph.offsets[seed + 1] = count;
        atomicMax(maxDegree, count);
        if (seed == 0) graph.offsets[0] = 0;
    }
}

__global__ void writeNeighbors(Neighborhoods graph) {
    using Scan = cub::BlockScan<int, blockSize>;
    __shared__ Scan::TempStorage scan;
    const int seed = blockIdx.x;
    Offset out = graph.offsets[seed];
    const Offset end = graph.offsets[seed + 1];
    for (Offset base = 0; base < graph.count && out < end; base += blockSize) {
        const Offset i = base + threadIdx.x;
        const int keep = i < graph.count && i != seed && pairDistance(graph, seed, i) < graph.threshold;
        int index, total;
        Scan(scan).ExclusiveSum(keep, index, total);
        if (keep) graph.neighbors[out + index] = static_cast<int>(i);
        out += total;
        __syncthreads();
    }
}

struct Closest {
    double diameter;
    int point;
};

struct MinDiameter {
    __device__ Closest operator()(Closest a, Closest b) const {
        if (b.diameter < a.diameter || (b.diameter == a.diameter && b.point < a.point)) {
            return b;
        }
        return a;
    }
};

__device__ void updateCandidate(Neighborhoods graph, int added, int point,
                                double& radius, Closest& best) {
    if (radius < graph.threshold) {
        radius = point == added ? CUDART_INF : fmax(radius, pairDistance(graph, added, point));
        if (radius < graph.threshold) best = MinDiameter{}(best, {radius, point});
    }
}

// One block grows a seed's cluster. Small rows keep their running maxima in
// registers. Larger rows use reusable per-block workspace.
template <int Slots>
__global__ void evaluateSeeds(Neighborhoods graph, ClusteringState state,
                              double* workspace, int maxDegree) {
    using Reduce = cub::BlockReduce<Closest, blockSize>;
    __shared__ typename Reduce::TempStorage reduction;
    __shared__ int nextPoints[2];
    const int lane = threadIdx.x;
    if (state.progress->members == graph.count) return;

    for (Offset row = blockIdx.x; row < graph.seedCount; row += gridDim.x) {
        const int seed = static_cast<int>(row);
        if (state.clustered[seed]) continue;
        const Offset begin = graph.offsets[seed];
        const int degree = static_cast<int>(graph.offsets[seed + 1] - begin);
        const int* neighbors = degree ? graph.neighbors + begin : nullptr;
        int* candidate = degree ? state.candidates + begin : nullptr;
        int size = state.sizes[seed];

        // Removing points outside a cached cluster cannot change any of its
        // greedy choices, or enable a choice that was previously rejected.
        bool dirty = size == 0;
        for (int i = lane; i < size - 1 && !dirty; i += blockSize) {
            dirty = state.clustered[candidate[i]] != 0;
        }
        if (__syncthreads_or(dirty)) {
            int points[Slots > 0 ? Slots : 1];
            double radii[Slots > 0 ? Slots : 1];
            double* scratch = Slots == 0
                ? workspace + static_cast<size_t>(blockIdx.x) * maxDegree
                : nullptr;
            if constexpr (Slots > 0) {
                #pragma unroll
                for (int slot = 0; slot < Slots; ++slot) {
                    const int i = lane + slot * blockSize;
                    const int point = i < degree ? neighbors[i] : 0;
                    points[slot] = point;
                    radii[slot] = i < degree && !state.clustered[point] ? 0.0 : CUDART_INF;
                }
            } else {
                for (int i = lane; i < degree; i += blockSize) {
                    scratch[i] = state.clustered[neighbors[i]] ? CUDART_INF : 0.0;
                }
            }

            int added = seed;
            size = 1;
            while (size <= degree) {
                Closest best{CUDART_INF, -1};
                if constexpr (Slots > 0) {
                    #pragma unroll
                    for (int slot = 0; slot < Slots; ++slot) {
                        updateCandidate(graph, added, points[slot], radii[slot], best);
                    }
                } else {
                    for (int i = lane; i < degree; i += blockSize) {
                        double radius = scratch[i];
                        if (radius < graph.threshold) {
                            updateCandidate(graph, added, neighbors[i], radius, best);
                            scratch[i] = radius;
                        }
                    }
                }
                best = Reduce(reduction).Reduce(best, MinDiameter{});
                if (lane == 0) {
                    nextPoints[size % 2] = best.point;
                    if (best.point >= 0) candidate[size - 1] = best.point;
                }
                // Alternate slots so the next iteration can publish while other
                // warps are still reading this iteration's point.
                __syncthreads();
                added = nextPoints[size % 2];
                if (added < 0) break;
                ++size;
            }
            if (lane == 0) state.sizes[seed] = size;
        }

        if (lane == 0) {
            // Compare cardinality first and prefer the smaller seed on ties.
            const Winner key = (static_cast<Winner>(size) << 32) | (0xffffffffu - seed);
            atomicMax(state.winners + blockIdx.x % blockSize, key);
        }
    }
}

__global__ void commitCluster(Neighborhoods graph, ClusteringState state) {
    using Reduce = cub::BlockReduce<Winner, blockSize>;
    using Scan = cub::BlockScan<int, blockSize>;
    __shared__ Reduce::TempStorage reduction;
    __shared__ Scan::TempStorage scan;
    __shared__ Winner winner;
    const int lane = threadIdx.x;
    const Progress progress = *state.progress;
    if (progress.members == graph.count) return;

    const Winner best = Reduce(reduction).Reduce(state.winners[lane], cub::Max{});
    state.winners[lane] = 0;
    if (lane == 0) winner = best;
    __syncthreads();

    const int size = winner >> 32;
    const int seed = 0xffffffffu - static_cast<unsigned int>(winner);
    if (size == 1) {
        // Every remaining candidate is a singleton. Emit them in seed order.
        int out = progress.members;
        for (Offset base = 0; base < graph.count; base += blockSize) {
            const Offset point = base + lane;
            const int keep = point < graph.count && !state.clustered[point];
            int index, total;
            Scan(scan).ExclusiveSum(keep, index, total);
            if (keep) {
                state.resultMembers[out + index] = static_cast<int>(point);
                state.resultOffsets[progress.clusters + (out - progress.members) + index] = out + index;
            }
            out += total;
            __syncthreads();
        }
        if (lane == 0) {
            *state.progress = {progress.clusters + (graph.count - progress.members), graph.count};
        }
        return;
    }

    const int* candidate = state.candidates + graph.offsets[seed];
    for (int i = lane; i < size; i += blockSize) {
        const int point = i == 0 ? seed : candidate[i - 1];
        state.clustered[point] = 1;
        state.resultMembers[progress.members + i] = point;
    }
    if (lane == 0) {
        state.resultOffsets[progress.clusters] = progress.members;
        *state.progress = {progress.clusters + 1, progress.members + size};
    }
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int count = static_cast<int>(points.size());
    if (count == 0) return {};

    // Cap the optional distance cache. Larger problems use compact neighbor
    // rows and evaluate pair distances on demand.
    const size_t pairs = static_cast<size_t>(count) * count;
    constexpr size_t distanceCacheBytes = 128 * 1024 * 1024;
    DeviceBuffer<Point> devicePoints(count);
    DeviceBuffer<double> distances(pairs <= distanceCacheBytes / sizeof(double) ? pairs : 0);
    DeviceBuffer<Offset> offsets(static_cast<size_t>(count) + 1);
    DeviceBuffer<int> degree(1);
    Neighborhoods graph{devicePoints.get(), distances.get(), offsets.get(),
                        nullptr, count, count, threshold};

    const size_t pointBytes = static_cast<size_t>(count) * sizeof(int);
    checkCuda(cudaMemcpy(devicePoints.get(), points.data(), points.size() * sizeof(Point),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(degree.get(), 0, sizeof(int)));

    size_t scanBytes = 0;
    Offset* rowEnds = offsets.get() + 1;
    checkCuda(cub::DeviceScan::InclusiveSum(nullptr, scanBytes, rowEnds, rowEnds, count));
    DeviceBuffer<unsigned char> scanStorage(scanBytes);
    countNeighbors<<<count, blockSize>>>(graph, degree.get());
    checkCuda(cudaGetLastError());
    checkCuda(cub::DeviceScan::InclusiveSum(scanStorage.get(), scanBytes, rowEnds, rowEnds, count));

    Offset edges;
    int maxDegree;
    checkCuda(cudaMemcpy(&edges, offsets.get() + count, sizeof(edges), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(&maxDegree, degree.get(), sizeof(maxDegree), cudaMemcpyDeviceToHost));

    // In a complete graph every seed produces the full cluster. Only the
    // first seed's member order is needed.
    const bool completeGraph = edges == static_cast<Offset>(count) * (count - 1);
    if (completeGraph) graph.seedCount = 1;
    const Offset storedEdges = completeGraph ? maxDegree : edges;
    DeviceBuffer<int> neighbors(storedEdges);
    DeviceBuffer<int> candidates(storedEdges);
    DeviceBuffer<int> clustered(count);
    DeviceBuffer<int> sizes(count);
    DeviceBuffer<Winner> winners(blockSize);
    DeviceBuffer<int> resultOffsets(count);
    DeviceBuffer<int> resultMembers(count);
    DeviceBuffer<Progress> deviceProgress(1);
    graph.neighbors = neighbors.get();
    const ClusteringState state{clustered.get(), sizes.get(), candidates.get(), winners.get(),
                                resultOffsets.get(), resultMembers.get(), deviceProgress.get()};
    checkCuda(cudaMemset(state.clustered, 0, pointBytes));
    checkCuda(cudaMemset(state.sizes, 0, pointBytes));
    checkCuda(cudaMemset(state.winners, 0, blockSize * sizeof(Winner)));
    checkCuda(cudaMemset(state.progress, 0, sizeof(Progress)));
    if (storedEdges != 0) {
        writeNeighbors<<<graph.seedCount, blockSize>>>(graph);
        checkCuda(cudaGetLastError());
    }

    int blocks = graph.seedCount;
    const bool needsWorkspace = maxDegree > maxRegisterSlots * blockSize;
    if (needsWorkspace) {
        int device, multiprocessors;
        checkCuda(cudaGetDevice(&device));
        checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
        blocks = std::min(blocks, 4 * multiprocessors);
    }
    DeviceBuffer<double> workspace(needsWorkspace ? static_cast<size_t>(maxDegree) * blocks : 0);

    Progress progress{};
    do {
        const int rounds = completeGraph || edges == 0 ? 1 :
            std::min(roundsPerBatch, count - progress.members);
        for (int round = 0; round < rounds; ++round) {
            if (maxDegree <= blockSize) {
                evaluateSeeds<1><<<blocks, blockSize>>>(graph, state, nullptr, maxDegree);
            } else if (maxDegree <= 2 * blockSize) {
                evaluateSeeds<2><<<blocks, blockSize>>>(graph, state, nullptr, maxDegree);
            } else if (maxDegree <= 4 * blockSize) {
                evaluateSeeds<4><<<blocks, blockSize>>>(graph, state, nullptr, maxDegree);
            } else if (maxDegree <= 8 * blockSize) {
                evaluateSeeds<8><<<blocks, blockSize>>>(graph, state, nullptr, maxDegree);
            } else if (!needsWorkspace) {
                evaluateSeeds<maxRegisterSlots><<<blocks, blockSize>>>(graph, state, nullptr, maxDegree);
            } else {
                evaluateSeeds<0><<<blocks, blockSize>>>(graph, state, workspace.get(), maxDegree);
            }
            commitCluster<<<1, blockSize>>>(graph, state);
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(&progress, state.progress, sizeof(progress), cudaMemcpyDeviceToHost));
    } while (progress.members < count);

    std::vector<int> hostOffsets(static_cast<size_t>(progress.clusters) + 1);
    std::vector<int> hostMembers(count);
    checkCuda(cudaMemcpy(hostOffsets.data(), state.resultOffsets, progress.clusters * sizeof(int),
                         cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(hostMembers.data(), state.resultMembers, pointBytes, cudaMemcpyDeviceToHost));
    hostOffsets.back() = count;
    std::vector<Cluster> clusters;
    clusters.reserve(progress.clusters);
    for (int c = 0; c < progress.clusters; ++c) {
        const auto begin = hostMembers.begin() + hostOffsets[c];
        const auto end = hostMembers.begin() + hostOffsets[c + 1];
        clusters.push_back({std::vector<int>(begin, end), *begin});
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

int main(int argc, char** argv) try {
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
    
    // Initialize CUDA before timing the clustering work.
    checkCuda(cudaFree(nullptr));
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
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
} catch (const std::runtime_error& error) {
    fprintf(stderr, "CUDA error: %s\n", error.what());
    return 1;
}
