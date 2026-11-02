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
#include <limits>
#include <vector>
#include <cfloat>
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

// Each seed is handled by one block. A candidate's current maximum distance
// only needs to be updated for the member added in the preceding iteration.
constexpr int BLOCK_SIZE = 256;

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> class DeviceArray {
public:
    explicit DeviceArray(size_t count) : ptr_(nullptr) {
        if (count > SIZE_MAX / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), "allocation");
    }
    ~DeviceArray() { cudaFree(ptr_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_;
};

__global__ void calculateDistances(const Point* points, double* distances, int n) {
    size_t total = static_cast<size_t>(n) * n;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int a = static_cast<int>(index / n);
        int b = static_cast<int>(index % n);
        double dx = points[a].x - points[b].x;
        double dy = points[a].y - points[b].y;
        distances[index] = sqrt(dx * dx + dy * dy);
    }
}

__global__ void buildCandidates(const double* distances, const int* clustered,
                                double* maxima, int* members, int* sizes,
                                int n, double threshold) {
    int seed = blockIdx.x;
    int tid = threadIdx.x;
    if (clustered[seed]) {
        if (tid == 0) sizes[seed] = 0;
        return;
    }
    size_t row = static_cast<size_t>(seed) * n;
    // Removing points outside a previously built candidate cannot change any
    // of its greedy choices, so retain that candidate until a member is used.
    __shared__ int invalidated;
    if (tid == 0) invalidated = 0;
    __syncthreads();
    int previous_size = sizes[seed];
    for (int i = tid; i < previous_size; i += BLOCK_SIZE) {
        if (clustered[members[row + i]]) atomicExch(&invalidated, 1);
    }
    __syncthreads();
    if (previous_size > 0 && !invalidated) return;
    __shared__ double best_distance[BLOCK_SIZE];
    __shared__ int best_index[BLOCK_SIZE];
    __shared__ int last_member;
    __shared__ int count;
    if (tid == 0) {
        last_member = seed;
        count = 1;
        members[row] = seed;
    }
    for (int i = tid; i < n; i += BLOCK_SIZE)
        maxima[row + i] = (clustered[i] || i == seed) ? DBL_MAX : 0.0;
    __syncthreads();

    while (true) {
        int last = last_member;
        double local_best = DBL_MAX;
        int local_index = -1;
        for (int i = tid; i < n; i += BLOCK_SIZE) {
            double value = maxima[row + i];
            if (value != DBL_MAX) {
                double d = distances[static_cast<size_t>(last) * n + i];
                if (d > value) value = d;
                // A candidate that has reached the threshold can never become
                // eligible again, since its maximum distance cannot decrease.
                if (!(value < threshold)) value = DBL_MAX;
                maxima[row + i] = value;
                if (value < local_best || (value == local_best && value != DBL_MAX && i < local_index)) {
                    local_best = value;
                    local_index = i;
                }
            }
        }
        best_distance[tid] = local_best;
        best_index[tid] = local_index;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                double other = best_distance[tid + stride];
                int other_index = best_index[tid + stride];
                if (other < best_distance[tid] ||
                    (other == best_distance[tid] && other_index >= 0 &&
                     (best_index[tid] < 0 || other_index < best_index[tid]))) {
                    best_distance[tid] = other;
                    best_index[tid] = other_index;
                }
            }
            __syncthreads();
        }
        int next = best_index[0];
        if (next < 0) break;
        if (tid == 0) {
            maxima[row + next] = DBL_MAX;
            members[row + count] = next;
            ++count;
            last_member = next;
        }
        __syncthreads();
    }
    if (tid == 0) sizes[seed] = count;
}

// Select the largest cluster, breaking ties by seed index as in the CPU loop,
// and commit its membership before the next round starts.
__global__ void commitBest(const int* sizes, const int* members, int* clustered,
                           int* assignments, int* ranks, int* seeds, int* selected_size,
                           int n, int round) {
    __shared__ int best_size[BLOCK_SIZE];
    __shared__ int best_seed[BLOCK_SIZE];
    int tid = threadIdx.x;
    int size = 0, seed = -1;
    for (int i = tid; i < n; i += BLOCK_SIZE) {
        int candidate = sizes[i];
        if (candidate > size || (candidate == size && candidate > 0 && i < seed)) {
            size = candidate;
            seed = i;
        }
    }
    best_size[tid] = size;
    best_seed[tid] = seed;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            int s = best_size[tid + stride], i = best_seed[tid + stride];
            if (s > best_size[tid] || (s == best_size[tid] && i >= 0 &&
                (best_seed[tid] < 0 || i < best_seed[tid]))) {
                best_size[tid] = s;
                best_seed[tid] = i;
            }
        }
        __syncthreads();
    }
    size = best_size[0];
    seed = best_seed[0];
    if (tid == 0) {
        *selected_size = size;
        seeds[round] = seed;
    }
    if (seed >= 0) {
        size_t row = static_cast<size_t>(seed) * n;
        for (int i = tid; i < size; i += BLOCK_SIZE) {
            int point = members[row + i];
            clustered[point] = 1;
            assignments[point] = round;
            ranks[point] = i;
        }
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t matrix_size = static_cast<size_t>(n) * n;
    DeviceArray<Point> device_points(n);
    DeviceArray<double> distances(matrix_size), maxima(matrix_size);
    DeviceArray<int> members(matrix_size), sizes(n), clustered(n), assignments(n), ranks(n), seeds(n), selected_size(1);
    checkCuda(cudaMemcpy(device_points.get(), points.data(), n * sizeof(Point), cudaMemcpyHostToDevice), "copy points");
    checkCuda(cudaMemset(clustered.get(), 0, n * sizeof(int)), "initialize clustered points");
    checkCuda(cudaMemset(sizes.get(), 0, n * sizeof(int)), "initialize candidate sizes");
    int distance_blocks = static_cast<int>(std::min<size_t>((matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535));
    calculateDistances<<<distance_blocks, BLOCK_SIZE>>>(device_points.get(), distances.get(), n);
    checkCuda(cudaGetLastError(), "launch distance calculation");

    int remaining = n, rounds = 0;
    while (remaining > 0) {
        buildCandidates<<<n, BLOCK_SIZE>>>(distances.get(), clustered.get(), maxima.get(),
                                           members.get(), sizes.get(), n, threshold);
        checkCuda(cudaGetLastError(), "launch candidate construction");
        commitBest<<<1, BLOCK_SIZE>>>(sizes.get(), members.get(), clustered.get(),
                                      assignments.get(), ranks.get(), seeds.get(), selected_size.get(), n, rounds);
        checkCuda(cudaGetLastError(), "launch cluster selection");
        int size;
        checkCuda(cudaMemcpy(&size, selected_size.get(), sizeof(size), cudaMemcpyDeviceToHost), "copy cluster size");
        if (size <= 0) break;
        remaining -= size;
        ++rounds;
    }
    std::vector<int> host_assignments(n), host_ranks(n), host_seeds(rounds);
    checkCuda(cudaMemcpy(host_assignments.data(), assignments.get(), n * sizeof(int), cudaMemcpyDeviceToHost), "copy memberships");
    checkCuda(cudaMemcpy(host_ranks.data(), ranks.get(), n * sizeof(int), cudaMemcpyDeviceToHost), "copy member order");
    checkCuda(cudaMemcpy(host_seeds.data(), seeds.get(), rounds * sizeof(int), cudaMemcpyDeviceToHost), "copy seeds");
    std::vector<Cluster> clusters(rounds);
    std::vector<int> cluster_sizes(rounds, 0);
    for (int assignment : host_assignments) ++cluster_sizes[assignment];
    for (int i = 0; i < rounds; ++i) {
        clusters[i].seed_point = host_seeds[i];
        clusters[i].members.resize(cluster_sizes[i]);
    }
    for (int i = 0; i < n; ++i)
        clusters[host_assignments[i]].members[host_ranks[i]] = i;
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
