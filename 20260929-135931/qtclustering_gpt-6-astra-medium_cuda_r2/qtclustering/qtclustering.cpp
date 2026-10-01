// QT Clustering Benchmark - CUDA implementation
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
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>

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

// CUDA is required: failures are reported rather than silently running on the CPU.
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int THREADS = 128;

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Store only points within the seed's threshold. Every subsequent cluster
// member must belong to this neighborhood, so no possible member is omitted.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int* counts) {
    const int seed = blockIdx.x;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        count += j != seed && pointDistance(points[seed], points[j]) < threshold;
    __shared__ int totals[THREADS];
    totals[threadIdx.x] = count;
    __syncthreads();
    for (int step = THREADS / 2; step; step /= 2) {
        if (threadIdx.x < step) totals[threadIdx.x] += totals[threadIdx.x + step];
        __syncthreads();
    }
    if (!threadIdx.x) counts[seed] = totals[0] + 1; // Also reserve the seed.
}

__global__ void makeOffsets(const int* counts, size_t* offsets, int n) {
    if (threadIdx.x || blockIdx.x) return;
    size_t total = 0;
    offsets[0] = 0;
    for (int i = 0; i < n; ++i) offsets[i + 1] = (total += counts[i]);
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x;
    __shared__ int cursor;
    if (!threadIdx.x) {
        cursor = 1;
        neighbors[offsets[seed]] = seed;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j != seed && pointDistance(points[seed], points[j]) < threshold) {
            const int slot = atomicAdd(&cursor, 1);
            neighbors[offsets[seed] + slot] = j;
        }
    }
}

// Lexicographic argmin: distances, then original point indices. Comparing
// actual double distances (not squared distances) preserves rounded ties.
__device__ void warpMinimum(double& value, int& index) {
    for (int delta = 16; delta; delta /= 2) {
        const double other = __shfl_down_sync(0xffffffff, value, delta);
        const int otherIndex = __shfl_down_sync(0xffffffff, index, delta);
        if (other < value || (other == value && otherIndex < index)) {
            value = other;
            index = otherIndex;
        }
    }
}

__global__ void buildCandidates(const Point* points, double threshold,
                                const size_t* offsets, const int* neighbors,
                                const unsigned char* clustered, int* sizes,
                                int* members, double* maxima) {
    const int seed = blockIdx.x;
    if (sizes[seed] >= 0) return; // Cached result is still exact.
    const size_t begin = offsets[seed], end = offsets[seed + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ double warpValues[THREADS / 32];
    __shared__ int warpIndices[THREADS / 32];
    __shared__ int next;
    for (size_t k = begin + threadIdx.x; k < end; k += THREADS) {
        const int j = neighbors[k];
        maxima[k] = (clustered[j] || j == seed) ? CUDART_INF : 0.0;
    }
    if (!threadIdx.x) members[begin] = seed;
    int last = seed, count = 1;
    __syncthreads();
    while (true) {
        double best = CUDART_INF;
        int bestIndex = INT_MAX;
        const Point p = points[last];
        for (size_t k = begin + threadIdx.x; k < end; k += THREADS) {
            double value = maxima[k];
            if (value == CUDART_INF) continue;
            const int j = neighbors[k];
            if (j == last) {
                maxima[k] = CUDART_INF;
                continue;
            }
            value = fmax(value, pointDistance(points[j], p));
            // Once outside the diameter bound, a point can never reenter.
            maxima[k] = value < threshold ? value : CUDART_INF;
            if (value < threshold &&
                (value < best || (value == best && j < bestIndex))) {
                best = value;
                bestIndex = j;
            }
        }
        warpMinimum(best, bestIndex);
        if (!lane) {
            warpValues[warp] = best;
            warpIndices[warp] = bestIndex;
        }
        __syncthreads();
        if (!warp) {
            best = lane < THREADS / 32 ? warpValues[lane] : CUDART_INF;
            bestIndex = lane < THREADS / 32 ? warpIndices[lane] : INT_MAX;
            warpMinimum(best, bestIndex);
            if (!lane) next = bestIndex;
        }
        __syncthreads();
        last = next;
        if (last == INT_MAX) break;
        if (!threadIdx.x) members[begin + count] = last;
        ++count;
        // No shared reduction storage may be overwritten until all warps
        // have consumed the selected index.
        __syncthreads();
    }
    if (!threadIdx.x) sizes[seed] = count;
}

__global__ void selectBest(const int* sizes, int n, int* winner) {
    __shared__ int counts[THREADS], seeds[THREADS];
    int count = 0, seed = INT_MAX;
    for (int i = threadIdx.x; i < n; i += THREADS) {
        if (sizes[i] > count || (sizes[i] == count && i < seed)) {
            count = sizes[i];
            seed = i;
        }
    }
    counts[threadIdx.x] = count;
    seeds[threadIdx.x] = seed;
    __syncthreads();
    for (int step = THREADS / 2; step; step /= 2) {
        if (threadIdx.x < step) {
            const int j = threadIdx.x + step;
            if (counts[j] > counts[threadIdx.x] ||
                (counts[j] == counts[threadIdx.x] && seeds[j] < seeds[threadIdx.x])) {
                counts[threadIdx.x] = counts[j];
                seeds[threadIdx.x] = seeds[j];
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        winner[0] = seeds[0];
        winner[1] = counts[0];
    }
}

__global__ void commitCluster(const int* winner, const size_t* offsets,
                              const int* members, unsigned char* clustered,
                              int* output, int outputOffset) {
    const size_t begin = offsets[winner[0]];
    for (int j = threadIdx.x; j < winner[1]; j += THREADS) {
        const int member = members[begin + j];
        clustered[member] = 1;
        output[outputOffset + j] = member;
    }
}

// Removing points outside a cached candidate cannot change any of its greedy
// choices. Only candidates containing a removed member need to be rebuilt.
__global__ void invalidateCandidates(const size_t* offsets, const int* members,
                                     const unsigned char* clustered, int* sizes) {
    const int seed = blockIdx.x;
    if (clustered[seed]) {
        if (!threadIdx.x) sizes[seed] = 0;
        return;
    }
    const int count = sizes[seed];
    int invalid = 0;
    for (int j = threadIdx.x; j < count; j += THREADS)
        invalid |= clustered[members[offsets[seed] + j]];
    if (__syncthreads_or(invalid) && !threadIdx.x) sizes[seed] = -1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    cudaCheck(cudaFree(nullptr)); // Initialize the required CUDA runtime.
    if (!n) return {};
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> sizes(n), winner(2), output(n);
    DeviceBuffer<size_t> offsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<unsigned char> clustered(n);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(clustered.data, 0, n));
    countNeighbors<<<n, THREADS>>>(devicePoints.data, n, threshold, sizes.data);
    makeOffsets<<<1, 1>>>(sizes.data, offsets.data, n);
    cudaCheck(cudaGetLastError());
    size_t entries;
    cudaCheck(cudaMemcpy(&entries, offsets.data + n, sizeof(entries), cudaMemcpyDeviceToHost));
    DeviceBuffer<int> neighbors(entries), members(entries);
    DeviceBuffer<double> maxima(entries);
    fillNeighbors<<<n, THREADS>>>(devicePoints.data, n, threshold, offsets.data, neighbors.data);
    cudaCheck(cudaMemset(sizes.data, 0xff, n * sizeof(int)));
    std::vector<Cluster> clusters;
    int assigned = 0;
    while (assigned < n) {
        buildCandidates<<<n, THREADS>>>(devicePoints.data, threshold, offsets.data,
            neighbors.data, clustered.data, sizes.data, members.data, maxima.data);
        selectBest<<<1, THREADS>>>(sizes.data, n, winner.data);
        commitCluster<<<1, THREADS>>>(winner.data, offsets.data, members.data,
                                     clustered.data, output.data, assigned);
        invalidateCandidates<<<n, THREADS>>>(offsets.data, members.data, clustered.data, sizes.data);
        cudaCheck(cudaGetLastError());
        int selected[2];
        cudaCheck(cudaMemcpy(selected, winner.data, sizeof(selected), cudaMemcpyDeviceToHost));
        clusters.push_back({std::vector<int>(selected[1]), selected[0]});
        assigned += selected[1];
    }
    std::vector<int> hostMembers(n);
    cudaCheck(cudaMemcpy(hostMembers.data(), output.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    size_t offset = 0;
    for (auto& cluster : clusters) {
        std::copy_n(hostMembers.data() + offset, cluster.members.size(), cluster.members.data());
        offset += cluster.members.size();
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
    
    // Initialize the GPU before timing the clustering workload.
    cudaCheck(cudaFree(nullptr));

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
