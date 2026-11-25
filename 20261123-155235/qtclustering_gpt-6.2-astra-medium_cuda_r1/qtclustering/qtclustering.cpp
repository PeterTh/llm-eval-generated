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
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
        // For small inputs truncation would otherwise keep producing empty groups.
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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
struct DeviceBuffer {
    T* data;
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data, std::max<size_t>(1, count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Bounds {
    double left, right, bottom, top;

    __device__ bool overlaps(const Bounds& other) const {
        return left <= other.right && other.left <= right &&
               bottom <= other.top && other.bottom <= top;
    }
};

struct Candidate {
    double diameter;
    int id;
};

struct ClusterState {
    const Point* points;
    const int* sorted;
    const int2* windows;
    int* removed;
    int* sizes;
    Bounds* bounds;
    Candidate* overflow;
    int overflowStride;
    unsigned long long* winner;
    int* members;
};

constexpr int warpSizeQT = 32;
constexpr int warpsPerBlock = 4;
constexpr int localCapacity = 64;
constexpr unsigned fullWarp = 0xffffffff;

__device__ double pointDistance(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    // Match the native host distance expression, including its rounding.
#ifdef __FMA__
    return sqrt(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
#else
    return sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
#endif
}

// One warp grows each seed. Common small neighborhoods fit in shared memory.
// Larger neighborhoods spill into workspace reused by successive seeds.
template<bool record>
__global__ void growClusters(ClusterState state, int n, double threshold,
                             int previous, int selected, int outputOffset) {
    __shared__ Candidate local[warpsPerBlock][localCapacity];
    int lane = threadIdx.x % warpSizeQT;
    int warp = threadIdx.x / warpSizeQT;
    int worker = blockIdx.x * warpsPerBlock + warp;
    if (record && worker != 0) return;
    Candidate* spill = state.overflow + size_t(worker) * state.overflowStride;
    for (int seed = record ? selected : worker; seed < n;
         seed += gridDim.x * warpsPerBlock) {
        if (!record && state.removed[seed]) continue;
        int size;
        // If no selected member was removed, the greedy sequence is identical.
        if (record || previous < 0 || state.bounds[seed].overlaps(state.bounds[previous])) {
            Point origin = state.points[seed];
            Bounds box{origin.x, origin.x, origin.y, origin.y};
            int2 window = state.windows[seed];
            int count = 0;
            for (int base = window.x; base < window.y; base += warpSizeQT) {
                int id = -1;
                double d = CUDART_INF;
                if (lane < window.y - base) {
                    id = state.sorted[base + lane];
                    if (id != seed && !state.removed[id])
                        d = pointDistance(origin, state.points[id]);
                }
                unsigned mask = __ballot_sync(fullWarp, d < threshold);
                if (d < threshold) {
                    int slot = count + __popc(mask & ((1u << lane) - 1u));
                    Candidate& entry = slot < localCapacity ? local[warp][slot] : spill[slot - localCapacity];
                    entry = {d, id};
                }
                count += __popc(mask);
            }
            __syncwarp();
            int latest = seed;
            size = 0;
            while (true) {
                Point member = state.points[latest];
                box.left = fmin(box.left, member.x);
                box.right = fmax(box.right, member.x);
                box.bottom = fmin(box.bottom, member.y);
                box.top = fmax(box.top, member.y);
                if (record && lane == 0) {
                    state.members[outputOffset + size] = latest;
                    state.removed[latest] = 1;
                }
                ++size;
                Candidate best{CUDART_INF, INT_MAX};
                for (int i = lane; i < count; i += warpSizeQT) {
                    Candidate& entry = i < localCapacity ? local[warp][i] : spill[i - localCapacity];
                    double d = entry.diameter;
                    if (!(d < threshold)) continue;
                    if (latest != seed) {
                        d = latest == entry.id ? CUDART_INF :
                            fmax(d, pointDistance(member, state.points[entry.id]));
                        entry.diameter = d;
                    }
                    if (d < threshold && (d < best.diameter ||
                        (d == best.diameter && entry.id < best.id)))
                        best = {d, entry.id};
                }
                for (int delta = 16; delta; delta /= 2) {
                    double d = __shfl_down_sync(fullWarp, best.diameter, delta);
                    int id = __shfl_down_sync(fullWarp, best.id, delta);
                    if (d < best.diameter || (d == best.diameter && id < best.id))
                        best = {d, id};
                }
                latest = __shfl_sync(fullWarp, best.id, 0);
                if (latest == INT_MAX) break;
            }
            if (!record && lane == 0) {
                state.bounds[seed] = box;
                state.sizes[seed] = size;
            }
            __syncwarp();
        } else {
            size = state.sizes[seed];
        }
        if (record) return;
        if (lane == 0) {
            // Maximizing the low word favors the earliest seed on size ties.
            unsigned long long key = (static_cast<unsigned long long>(size) << 32) |
                                     static_cast<unsigned>(INT_MAX - seed);
            atomicMax(state.winner, key);
        }
    }
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    if (points.empty()) return {};
    int n = static_cast<int>(points.size());
    std::vector<int> sorted(n);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
        return points[a].x < points[b].x;
    });

    // Only points within threshold along x can belong to the seed's cluster.
    std::vector<int2> windows(n);
    int left = 0, right = 0, maxWindow = 0;
    double minY = points.front().y, maxY = minY;
    for (int seed : sorted) {
        minY = std::min(minY, points[seed].y);
        maxY = std::max(maxY, points[seed].y);
        double x = points[seed].x;
        while (left < n && x - points[sorted[left]].x > threshold) ++left;
        while (right < n && points[sorted[right]].x - x <= threshold) ++right;
        windows[seed] = make_int2(left, right);
        maxWindow = std::max(maxWindow, right - left);
    }
    int overflowStride = std::max(0, maxWindow - localCapacity);
    int device, sms;
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int blocks = std::min((n - 1) / warpsPerBlock + 1, sms * 8);
    // A bounding rectangle within the threshold proves that seed zero wins.
    bool allFit = distance({points[sorted.front()].x, minY},
                           {points[sorted.back()].x, maxY}) < threshold;
    if (allFit) blocks = 1;
    if (overflowStride) {
        size_t bytesPerBlock = sizeof(Candidate) * size_t(overflowStride) * warpsPerBlock;
        blocks = std::min(blocks, std::max(1, int((64u << 20) / bytesPerBlock)));
    }

    DeviceBuffer<Point> coords(n);
    DeviceBuffer<int> order(n), removed(n), sizes(n), members(n);
    DeviceBuffer<int2> ranges(n);
    DeviceBuffer<Bounds> bounds(n);
    DeviceBuffer<Candidate> scratch(size_t(blocks) * warpsPerBlock * overflowStride);
    DeviceBuffer<unsigned long long> winner(1);
    ClusterState state{coords.data, order.data, ranges.data, removed.data, sizes.data,
                       bounds.data, scratch.data, overflowStride, winner.data, members.data};
    checkCuda(cudaMemcpy(coords.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(order.data, sorted.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ranges.data, windows.data(), n * sizeof(int2), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(removed.data, 0, n * sizeof(int)));
    std::vector<Cluster> clusters;
    std::vector<int> chosenSizes;
    int used = 0, previous = -1;
    while (used < n) {
        int seed = 0, size = n;
        if (!allFit) {
            checkCuda(cudaMemset(winner.data, 0, sizeof(unsigned long long)));
            growClusters<false><<<blocks, warpsPerBlock * warpSizeQT>>>(state, n, threshold, previous, -1, 0);
            checkCuda(cudaGetLastError());
            unsigned long long key;
            checkCuda(cudaMemcpy(&key, winner.data, sizeof(key), cudaMemcpyDeviceToHost));
            seed = INT_MAX - static_cast<int>(static_cast<unsigned>(key));
            size = static_cast<int>(key >> 32);
        }
        // If the best candidate is a singleton, all remaining seeds are singletons.
        if (size == 1 && !allFit) break;
        growClusters<true><<<1, warpSizeQT>>>(state, n, threshold, previous, seed, used);
        checkCuda(cudaGetLastError());
        clusters.push_back({{}, seed});
        chosenSizes.push_back(size);
        used += size;
        previous = seed;
    }

    std::vector<int> ordered(used);
    if (used)
        checkCuda(cudaMemcpy(ordered.data(), members.data, used * sizeof(int), cudaMemcpyDeviceToHost));
    int offset = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
        auto begin = ordered.begin() + offset;
        clusters[i].members.assign(begin, begin + chosenSizes[i]);
        offset += chosenSizes[i];
    }
    if (used < n) {
        std::vector<bool> assigned(n, false);
        for (int id : ordered) assigned[id] = true;
        for (int id = 0; id < n; ++id)
            if (!assigned[id]) clusters.push_back({{id}, id});
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
