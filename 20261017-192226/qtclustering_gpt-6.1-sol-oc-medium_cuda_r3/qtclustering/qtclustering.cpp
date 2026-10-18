// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

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

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "CUDA buffer size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK_SIZE = 128;

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Every possible member must be within threshold of its seed. Store only these
// neighbors in CSR form; this avoids dense N-by-N per-seed working storage.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               size_t* offsets) {
    const int seed = blockIdx.x;
    const Point p = points[seed];
    __shared__ int sums[BLOCK_SIZE];
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        count += j == seed || pointDistance(p, points[j]) < threshold;
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) offsets[seed] = sums[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x;
    const Point p = points[seed];
    __shared__ int cursor;
    if (threadIdx.x == 0) cursor = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j == seed || pointDistance(p, points[j]) < threshold) {
            const int slot = atomicAdd(&cursor, 1);
            neighbors[offsets[seed] + slot] = j;
        }
    }
}

// Reduce by (distance, point index). The explicit second key preserves the
// sequential algorithm's first-point-wins rule even for coincident points.
__device__ void closestReduction(double& value, int& index,
                                 double* values, int* indices) {
    const int lane = threadIdx.x;
    values[lane] = value;
    indices[lane] = index;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (lane < stride) {
            const double other = values[lane + stride];
            const int otherIndex = indices[lane + stride];
            if (other < values[lane] ||
                (other == values[lane] && otherIndex < indices[lane])) {
                values[lane] = other;
                indices[lane] = otherIndex;
            }
        }
        __syncthreads();
    }
    value = values[0];
    index = indices[0];
    __syncthreads();
}

__global__ void candidateClusters(const Point* points, double threshold,
                                  const size_t* offsets, const int* neighbors,
                                  const unsigned char* clustered, double* maxima,
                                  int* members, int* sizes) {
    const int seed = blockIdx.x;
    if (clustered[seed]) {
        if (threadIdx.x == 0) sizes[seed] = 0;
        return;
    }
    const size_t begin = offsets[seed], end = offsets[seed + 1];
    __shared__ double values[BLOCK_SIZE];
    __shared__ int indices[BLOCK_SIZE];
    __shared__ int dirty;
    const int oldSize = sizes[seed];
    if (threadIdx.x == 0) dirty = oldSize == 0;
    __syncthreads();
    for (int i = threadIdx.x; i < oldSize; i += blockDim.x)
        if (clustered[members[begin + i]]) atomicExch(&dirty, 1);
    __syncthreads();
    // Deleting points outside a candidate cannot change its greedy sequence.
    if (!dirty) return;

    for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x)
        maxima[i] = (clustered[neighbors[i]] || neighbors[i] == seed) ? CUDART_INF : 0.0;
    if (threadIdx.x == 0) members[begin] = seed;
    int last = seed, count = 1;
    while (true) {
        double best = CUDART_INF;
        int closest = INT_MAX;
        const Point p = points[last];
        for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
            double maximum = maxima[i];
            if (!(maximum < threshold)) continue;
            const int candidate = neighbors[i];
            if (candidate == last) {
                maxima[i] = CUDART_INF;
                continue;
            }
            // Incrementally extend the max over all previously chosen members.
            maximum = fmax(maximum, pointDistance(points[candidate], p));
            maxima[i] = maximum;
            if (maximum < threshold &&
                (maximum < best || (maximum == best && candidate < closest))) {
                best = maximum;
                closest = candidate;
            }
        }
        closestReduction(best, closest, values, indices);
        if (closest == INT_MAX) break;
        if (threadIdx.x == 0) members[begin + count] = closest;
        ++count;
        last = closest;
    }
    if (threadIdx.x == 0) sizes[seed] = count;
}

// Select the largest cluster, with lowest seed index breaking ties, and commit
// it on-device. Other blocks only observe removals on the next kernel launch.
__global__ void commitCluster(int n, const size_t* offsets, const int* members,
                              const int* sizes, unsigned char* clustered, int* result) {
    __shared__ double values[BLOCK_SIZE];
    __shared__ int indices[BLOCK_SIZE];
    double value = CUDART_INF;
    int seed = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (sizes[i] > 0 && (-double(sizes[i]) < value ||
            (-double(sizes[i]) == value && i < seed))) {
            value = -double(sizes[i]);
            seed = i;
        }
    }
    closestReduction(value, seed, values, indices);
    const int count = sizes[seed];
    if (threadIdx.x == 0) {
        result[0] = seed;
        result[1] = count;
    }
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int member = members[offsets[seed] + i];
        clustered[member] = 1;
        result[2 + i] = member;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (n == 0) return clusters;
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<size_t> offsets(size_t(n) + 1);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<int> sizes(n), result(size_t(n) + 2);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(clustered.data, 0, n));
    cudaCheck(cudaMemset(sizes.data, 0, n * sizeof(int)));
    cudaCheck(cudaMemset(offsets.data + n, 0, sizeof(size_t)));
    countNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, offsets.data);
    cudaCheck(cudaGetLastError());
    thrust::device_ptr<size_t> start(offsets.data);
    thrust::exclusive_scan(start, start + size_t(n) + 1, start);
    size_t entries = 0;
    cudaCheck(cudaMemcpy(&entries, offsets.data + n, sizeof(entries), cudaMemcpyDeviceToHost));
    DeviceBuffer<int> neighbors(entries), members(entries);
    DeviceBuffer<double> maxima(entries);
    fillNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    int remaining = n;
    while (remaining > 0) {
        candidateClusters<<<n, BLOCK_SIZE>>>(devicePoints.data, threshold, offsets.data,
            neighbors.data, clustered.data, maxima.data, members.data, sizes.data);
        cudaCheck(cudaGetLastError());
        commitCluster<<<1, BLOCK_SIZE>>>(n, offsets.data, members.data, sizes.data,
                                         clustered.data, result.data);
        cudaCheck(cudaGetLastError());
        int header[2];
        cudaCheck(cudaMemcpy(header, result.data, sizeof(header), cudaMemcpyDeviceToHost));
        // Once the GPU reports a maximum size of one, all remaining clusters
        // are singletons in ascending seed order. Materialize them in one copy.
        if (header[1] == 1) {
            std::vector<int> finalSizes(n);
            cudaCheck(cudaMemcpy(finalSizes.data(), sizes.data, n * sizeof(int), cudaMemcpyDeviceToHost));
            for (int seed = 0; seed < n; ++seed) {
                if (finalSizes[seed] > 0) clusters.push_back(Cluster{{seed}, seed});
            }
            break;
        }
        Cluster cluster;
        cluster.seed_point = header[0];
        cluster.members.resize(header[1]);
        cudaCheck(cudaMemcpy(cluster.members.data(), result.data + 2,
                             cluster.members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        remaining -= header[1];
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

    // Initialize the CUDA context before timing the clustering computation.
    cudaCheck(cudaFree(nullptr));
    
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
