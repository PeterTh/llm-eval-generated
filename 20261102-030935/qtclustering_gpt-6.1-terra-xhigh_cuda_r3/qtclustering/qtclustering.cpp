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

// CUDA errors must be reported even when validation is disabled.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

template <typename T>
static T* deviceAllocate(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        fprintf(stderr, "CUDA allocation size overflow\n");
        std::exit(EXIT_FAILURE);
    }
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

// Distances are shared by all seeds and all rounds. Keep the same double
// precision square root and arithmetic as the original distance function.
__global__ void calculateDistances(const Point* points, double* distances, int n) {
    size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        int a = static_cast<int>(i / n);
        int b = static_cast<int>(i % n);
        double dx = points[a].x - points[b].x;
        double dy = points[a].y - points[b].y;
        distances[i] = sqrt(dx * dx + dy * dy);
    }
}

// One block grows each seed's cluster. The maximum distance to the existing
// members only needs to be updated for the member added in the last step.
// Ties are resolved by the smallest point index, as in the serial scan.
template <bool SharedMaxima>
__global__ void buildCandidates(const double* distances, const unsigned char* clustered,
                                int* candidate_members, int* candidate_sizes,
                                double* global_maxima, int n, double threshold) {
    const int seed = blockIdx.x;
    if (seed >= n || clustered[seed]) {
        if (threadIdx.x == 0 && seed < n) candidate_sizes[seed] = 0;
        return;
    }
    int* members = candidate_members + static_cast<size_t>(seed) * n;
    // Removing points outside a candidate cannot change its greedy sequence.
    // Rebuild only candidates that intersect the cluster just committed.
    int old_size = candidate_sizes[seed];
    bool intersects = false;
    for (int i = threadIdx.x; i < old_size; i += blockDim.x) {
        if (clustered[members[i]]) intersects = true;
    }
    if (!__syncthreads_or(intersects) && old_size > 0) return;
    extern __shared__ double shared_maxima[];
    double* maxima = SharedMaxima ? shared_maxima : global_maxima + static_cast<size_t>(seed) * n;
    __shared__ double warp_values[32];
    __shared__ int warp_indices[32];
    __shared__ int selected;
    for (int i = threadIdx.x; i < n; i += blockDim.x) maxima[i] = i == seed ? -1.0 : 0.0;
    if (threadIdx.x == 0) members[0] = seed;
    __syncthreads();

    int last = seed;
    int count = 1;
    while (count < n) {
        double best = threshold;
        int best_index = n;
        const double* row = distances + static_cast<size_t>(last) * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double current = maxima[i];
            if (current < 0.0 || clustered[i] || current >= threshold) continue;
            double d = row[i];
            if (d > current) current = d;
            maxima[i] = current;
            if (current < best || (current == best && current < threshold && i < best_index)) {
                best = current;
                best_index = i;
            }
        }
        // Reduce (distance, index) within each warp, then across warps.
        for (int delta = 16; delta > 0; delta /= 2) {
            double value = __shfl_down_sync(0xffffffff, best, delta);
            int index = __shfl_down_sync(0xffffffff, best_index, delta);
            if (value < best || (value == best && index < best_index)) {
                best = value;
                best_index = index;
            }
        }
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) { warp_values[warp] = best; warp_indices[warp] = best_index; }
        __syncthreads();
        if (warp == 0) {
            best = lane < blockDim.x / 32 ? warp_values[lane] : threshold;
            best_index = lane < blockDim.x / 32 ? warp_indices[lane] : n;
            for (int delta = 16; delta > 0; delta /= 2) {
                double value = __shfl_down_sync(0xffffffff, best, delta);
                int index = __shfl_down_sync(0xffffffff, best_index, delta);
                if (value < best || (value == best && index < best_index)) {
                    best = value;
                    best_index = index;
                }
            }
            if (lane == 0) {
                selected = best_index;
                if (best_index < n) {
                    maxima[best_index] = -1.0;
                    members[count] = best_index;
                }
            }
        }
        __syncthreads();
        last = selected;
        if (last == n) break;
        ++count;
    }
    if (threadIdx.x == 0) candidate_sizes[seed] = count;
}

// Select the largest candidate (lowest seed on ties), record it, and remove
// its members before the next round. State is {remaining, output offset, count}.
__global__ void commitCandidate(const int* members, const int* sizes,
                                unsigned char* clustered, int* output_members,
                                int* output_sizes, int* output_seeds, int* state, int n) {
    __shared__ int best_sizes[256];
    __shared__ int best_seeds[256];
    int size = 0, seed = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int candidate_size = sizes[i];
        if (candidate_size > size || (candidate_size == size && i < seed)) {
            size = candidate_size;
            seed = i;
        }
    }
    best_sizes[threadIdx.x] = size;
    best_seeds[threadIdx.x] = seed;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            int other_size = best_sizes[threadIdx.x + step];
            int other_seed = best_seeds[threadIdx.x + step];
            if (other_size > best_sizes[threadIdx.x] ||
                (other_size == best_sizes[threadIdx.x] && other_seed < best_seeds[threadIdx.x])) {
                best_sizes[threadIdx.x] = other_size;
                best_seeds[threadIdx.x] = other_seed;
            }
        }
        __syncthreads();
    }
    size = best_sizes[0];
    seed = best_seeds[0];
    int offset = state[1];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        int point = members[static_cast<size_t>(seed) * n + i];
        output_members[offset + i] = point;
        clustered[point] = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        output_sizes[state[2]] = size;
        output_seeds[state[2]] = seed;
        state[0] -= size;
        state[1] += size;
        ++state[2];
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t matrix_size = static_cast<size_t>(n) * n;
    Point* d_points = deviceAllocate<Point>(n);
    double* d_distances = deviceAllocate<double>(matrix_size);
    int* d_candidates = deviceAllocate<int>(matrix_size);
    int* d_sizes = deviceAllocate<int>(n);
    unsigned char* d_clustered = deviceAllocate<unsigned char>(n);
    int* d_output = deviceAllocate<int>(n);
    int* d_output_sizes = deviceAllocate<int>(n);
    int* d_output_seeds = deviceAllocate<int>(n);
    int* d_state = deviceAllocate<int>(3);
    // Limit per-block shared memory to preserve occupancy; larger inputs use
    // the same algorithm with a global-memory workspace.
    bool use_shared = static_cast<size_t>(n) * sizeof(double) <= 32 * 1024;
    double* d_maxima = use_shared ? nullptr : deviceAllocate<double>(matrix_size);
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_sizes, 0, n * sizeof(int)));
    int state[3] = {n, 0, 0};
    CUDA_CHECK(cudaMemcpy(d_state, state, sizeof(state), cudaMemcpyHostToDevice));
    int blocks = static_cast<int>(std::min<size_t>((matrix_size + 255) / 256, 65535));
    calculateDistances<<<blocks, 256>>>(d_points, d_distances, n);
    CUDA_CHECK(cudaGetLastError());
    while (state[0] > 0) {
        if (use_shared) {
            buildCandidates<true><<<n, 128, static_cast<size_t>(n) * sizeof(double)>>>(
                d_distances, d_clustered, d_candidates, d_sizes, nullptr, n, threshold);
        } else {
            buildCandidates<false><<<n, 128>>>(
                d_distances, d_clustered, d_candidates, d_sizes, d_maxima, n, threshold);
        }
        CUDA_CHECK(cudaGetLastError());
        commitCandidate<<<1, 256>>>(d_candidates, d_sizes, d_clustered, d_output,
                                    d_output_sizes, d_output_seeds, d_state, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state, d_state, sizeof(state), cudaMemcpyDeviceToHost));
    }
    std::vector<int> output(n), sizes(state[2]), seeds(state[2]);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sizes.data(), d_output_sizes, sizes.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seeds.data(), d_output_seeds, seeds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Cluster> clusters;
    clusters.reserve(state[2]);
    int offset = 0;
    for (int i = 0; i < state[2]; ++i) {
        Cluster cluster;
        cluster.seed_point = seeds[i];
        cluster.members.assign(output.begin() + offset, output.begin() + offset + sizes[i]);
        clusters.push_back(std::move(cluster));
        offset += sizes[i];
    }
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_output_sizes));
    CUDA_CHECK(cudaFree(d_output_seeds));
    CUDA_CHECK(cudaFree(d_state));
    if (d_maxima) CUDA_CHECK(cudaFree(d_maxima));
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
