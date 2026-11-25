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
#include <climits>
#include <numeric>
#include <utility>
#include <cuda_runtime.h>
#include <math_constants.h>
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
        // For small inputs the original range rounds every group size to zero.
        double group_range = N <= 30 ? 2.0 : N / 30.0;
        int group_cnt = static_cast<int>(frand() * group_range);
        
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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double deviceDistance(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(fma(dx, dx, dy * dy));
}

constexpr int warpSizeQT = 32;
constexpr unsigned fullWarp = 0xffffffffu;

// Each warp counts or writes one seed's neighbors in a compact row. The
// host supplies an x-sorted window, so narrow thresholds skip distant points.
template<bool write>
__global__ void neighborhoods(const Point* points, const int* sorted, const int2* windows,
                              double threshold, const size_t* offsets, int* neighbors,
                              int* degrees, int n) {
    int seed = blockIdx.x * (blockDim.x / warpSizeQT) + threadIdx.x / warpSizeQT;
    if (seed >= n) return;
    int lane = threadIdx.x % warpSizeQT;
    int2 window = windows[seed];
    int count = 0;
    for (int base = window.x; base < window.y; base += warpSizeQT) {
        int id = base + lane < window.y ? sorted[base + lane] : seed;
        bool keep = id != seed && deviceDistance(points[seed], points[id]) < threshold;
        unsigned mask = __ballot_sync(fullWarp, keep);
        if (write && keep) {
            int rank = __popc(mask & ((1u << lane) - 1));
            neighbors[offsets[seed] + count + rank] = id;
        }
        count += __popc(mask);
    }
    if (!write && lane == 0) degrees[seed] = count;
}

struct Choice {
    double score;
    int id;
};

__device__ Choice better(Choice a, Choice b) {
    return b.score < a.score || (b.score == a.score && b.id < a.id) ? b : a;
}

__device__ Choice warpBest(Choice value) {
    for (int step = 16; step; step /= 2) {
        Choice other{__shfl_down_sync(fullWarp, value.score, step),
                     __shfl_down_sync(fullWarp, value.id, step)};
        value = better(value, other);
    }
    return value;
}

// A single unsigned maximum orders candidates by size, then by lowest seed.
using Winner = unsigned long long;
__device__ void updateWinner(Winner* winner, int count, int seed) {
    Winner key = (static_cast<Winner>(count) << 32) | static_cast<unsigned>(INT_MAX - seed);
    atomicMax(winner, key);
}

// One block per seed. Candidates can only come from the seed's neighbor
// row. Their scores increase monotonically as new members are selected.
constexpr int threads = 128;
__global__ void growCandidates(const Point* points, double threshold, const size_t* offsets,
                               const int* neighbors, const int* degrees, int* members,
                               int* sizes, const int* claimed, double* overflowScores,
                               const size_t* overflowOffsets, int sharedCapacity, Winner* winner) {
    const int seed = blockIdx.x;
    int tid = threadIdx.x;
    if (claimed[seed]) {
        if (!tid) sizes[seed] = 0;
        return;
    }
    size_t row = offsets[seed];
    int degree = degrees[seed];
    __shared__ int stale;
    __shared__ Choice winners[threads / warpSizeQT];
    __shared__ int next;
    int oldSize = sizes[seed];
    if (oldSize) {
        if (!tid) stale = 0;
        __syncthreads();
        for (int j = tid; j < oldSize - 1; j += threads) {
            if (claimed[members[row + j]]) atomicExch(&stale, 1);
        }
        __syncthreads();
        // Removing points outside the old greedy sequence cannot change it.
        if (!stale) {
            if (!tid) updateWinner(winner, oldSize, seed);
            return;
        }
    }

    extern __shared__ double localScores[];
    double* scores = degree <= sharedCapacity ? localScores : overflowScores + overflowOffsets[seed];
    int last = seed;
    int count = 1;
    while (count <= degree) {
        Point p = points[last];
        Choice best{CUDART_INF, INT_MAX};
        for (int j = tid; j < degree; j += threads) {
            int id = neighbors[row + j];
            double score = count == 1 ? (claimed[id] ? CUDART_INF : 0.0) : scores[j];
            if (score < threshold) {
                score = id == last ? CUDART_INF : fmax(score, deviceDistance(points[id], p));
                if (score < threshold) best = better(best, {score, id});
            }
            scores[j] = score;
        }
        best = warpBest(best);
        if (tid % warpSizeQT == 0) winners[tid / warpSizeQT] = best;
        __syncthreads();
        if (tid < warpSizeQT) {
            Choice finalist = tid < threads / warpSizeQT ? winners[tid] : Choice{CUDART_INF, INT_MAX};
            finalist = warpBest(finalist);
            if (!tid) {
                next = finalist.id;
                if (next != INT_MAX) members[row + count - 1] = next;
            }
        }
        __syncthreads();
        last = next;
        if (last == INT_MAX) break;
        ++count;
    }
    if (!tid) {
        sizes[seed] = count;
        updateWinner(winner, count, seed);
    }
}

__global__ void claim(int seed, int count, const int* members, int* claimed, Winner* winner) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) *winner = 0;
    if (i < count) claimed[i == 0 ? seed : members[i - 1]] = 1;
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int n = static_cast<int>(points.size());
    if (n == 0) return {};
    std::vector<int> sorted(n);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return points[a].x < points[b].x; });
    // If the entire bounding box fits, seed zero is guaranteed to win.
    // Grow just that row to retain the original member insertion order.
    auto yBounds = std::minmax_element(points.begin(), points.end(),
        [](Point a, Point b) { return a.y < b.y; });
    double width = points[sorted.back()].x - points[sorted.front()].x;
    double height = yBounds.second->y - yBounds.first->y;
    int seeds = std::sqrt(std::fma(width, width, height * height)) < threshold ? 1 : n;
    std::vector<int2> windows(n);
    int left = 0, right = 0;
    for (int seed : sorted) {
        while (left < n && points[seed].x - points[sorted[left]].x >= threshold) ++left;
        while (right < n && points[sorted[right]].x - points[seed].x < threshold) ++right;
        windows[seed] = make_int2(left, right);
    }

    DeviceBuffer<Point> gpuPoints(n);
    DeviceBuffer<int2> gpuWindows(n);
    DeviceBuffer<int> gpuSorted(n), degrees(seeds), sizes(seeds), claimed(n);
    DeviceBuffer<Winner> winner(1);
    checkCuda(cudaMemcpy(gpuPoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(gpuWindows.data, windows.data(), n * sizeof(int2), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(gpuSorted.data, sorted.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    int blocks = (seeds + threads / warpSizeQT - 1) / (threads / warpSizeQT);
    neighborhoods<false><<<blocks, threads>>>(gpuPoints.data, gpuSorted.data, gpuWindows.data,
        threshold, nullptr, nullptr, degrees.data, seeds);
    std::vector<int> counts(seeds);
    checkCuda(cudaMemcpy(counts.data(), degrees.data, seeds * sizeof(int), cudaMemcpyDeviceToHost));

    // Bound shared memory per block to allow multiple seeds per SM. Very
    // dense rows keep their scores in a separate global scratch allocation.
    int capacity = std::min(2048, *std::max_element(counts.begin(), counts.end()));
    std::vector<size_t> offsets(seeds + 1), overflow(seeds + 1);
    for (int i = 0; i < seeds; ++i) {
        offsets[i + 1] = offsets[i] + counts[i];
        overflow[i + 1] = overflow[i] + (counts[i] > capacity ? counts[i] : 0);
    }
    DeviceBuffer<size_t> rows(seeds), overflowRows(seeds);
    DeviceBuffer<int> neighbors(offsets.back()), members(offsets.back());
    DeviceBuffer<double> scratch(overflow.back());
    checkCuda(cudaMemcpy(rows.data, offsets.data(), seeds * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(overflowRows.data, overflow.data(), seeds * sizeof(size_t), cudaMemcpyHostToDevice));
    neighborhoods<true><<<blocks, threads>>>(gpuPoints.data, gpuSorted.data, gpuWindows.data,
        threshold, rows.data, neighbors.data, degrees.data, seeds);
    checkCuda(cudaMemset(winner.data, 0, sizeof(Winner)));
    checkCuda(cudaMemset(sizes.data, 0, seeds * sizeof(int)));
    checkCuda(cudaMemset(claimed.data, 0, n * sizeof(int)));

    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        growCandidates<<<seeds, threads, capacity * sizeof(double)>>>(gpuPoints.data, threshold,
            rows.data, neighbors.data, degrees.data, members.data, sizes.data, claimed.data,
            scratch.data, overflowRows.data, capacity, winner.data);
        Winner best;
        checkCuda(cudaMemcpy(&best, winner.data, sizeof(best), cudaMemcpyDeviceToHost));
        int seed = INT_MAX - static_cast<int>(static_cast<unsigned>(best));
        int count = static_cast<int>(best >> 32);
        if (count == 1) {
            checkCuda(cudaMemcpy(counts.data(), sizes.data, seeds * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = seed; i < seeds; ++i) {
                if (counts[i]) clusters.push_back({{i}, i});
            }
            break;
        }
        Cluster cluster{{seed}, seed};
        cluster.members.resize(count);
        const int* tail = members.data + offsets[seed];
        checkCuda(cudaMemcpy(cluster.members.data() + 1, tail, (count - 1) * sizeof(int),
                             cudaMemcpyDeviceToHost));
        clusters.push_back(std::move(cluster));
        remaining -= count;
        if (remaining) claim<<<(count + threads - 1) / threads, threads>>>(seed, count, tail, claimed.data, winner.data);
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
}
