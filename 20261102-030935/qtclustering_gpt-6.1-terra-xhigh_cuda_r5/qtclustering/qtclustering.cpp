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
#include <math_constants.h>

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

// A candidate's diameter only changes when a new member is added.  Each
// block builds one seed's candidate, updating those maxima in parallel.
// Cached candidates remain valid until one of their members is clustered.
constexpr int BLOCK_SIZE = 256;

static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__global__ void calculateDistances(const Point* points, double* distances, int n) {
    size_t total = static_cast<size_t>(n) * n;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < total; k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        int i = static_cast<int>(k / n);
        int j = static_cast<int>(k % n);
        double dx = points[i].x - points[j].x;
        double dy = points[i].y - points[j].y;
        distances[k] = sqrt(dx * dx + dy * dy);
    }
}

__global__ void buildCandidates(const double* distances, const int* assigned,
                                double* maxima, int* members, int* sizes,
                                int n, double threshold) {
    const int seed = blockIdx.x;
    const int tid = threadIdx.x;
    if (assigned[seed] >= 0) return;
    const size_t row = static_cast<size_t>(seed) * n;
    __shared__ int rebuild;
    __shared__ int best_index[BLOCK_SIZE];
    __shared__ double best_value[BLOCK_SIZE];
    __shared__ int selected;
    if (tid == 0) rebuild = sizes[seed] == 0;
    __syncthreads();
    for (int i = tid; i < sizes[seed]; i += blockDim.x) {
        if (assigned[members[row + i]] >= 0) atomicExch(&rebuild, 1);
    }
    __syncthreads();
    if (!rebuild) return;

    for (int i = tid; i < n; i += blockDim.x) {
        double d = distances[row + i];
        maxima[row + i] = (assigned[i] < 0 && i != seed && d < threshold)
                               ? d : CUDART_INF;
    }
    if (tid == 0) members[row] = seed;
    __syncthreads();
    int count = 1;
    while (count < n) {
        double best = CUDART_INF;
        int index = n;
        for (int i = tid; i < n; i += blockDim.x) {
            double d = maxima[row + i];
            if (d < best || (d == best && i < index)) {
                best = d;
                index = i;
            }
        }
        best_value[tid] = best;
        best_index[tid] = index;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                double other = best_value[tid + stride];
                int other_index = best_index[tid + stride];
                if (other < best_value[tid] ||
                    (other == best_value[tid] && other_index < best_index[tid])) {
                    best_value[tid] = other;
                    best_index[tid] = other_index;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            selected = best_value[0] < threshold ? best_index[0] : -1;
            if (selected >= 0) members[row + count] = selected;
        }
        __syncthreads();
        if (selected < 0) break;
        const size_t distance_row = static_cast<size_t>(selected) * n;
        for (int i = tid; i < n; i += blockDim.x) {
            double old = maxima[row + i];
            if (i == selected) {
                maxima[row + i] = CUDART_INF;
            } else if (old < threshold) {
                double d = distances[distance_row + i];
                double updated = d > old ? d : old;
                maxima[row + i] = updated < threshold ? updated : CUDART_INF;
            }
        }
        ++count;
        __syncthreads();
    }
    if (tid == 0) sizes[seed] = count;
}

struct Selection {
    int seed;
    int count;
    int remaining;
};

__global__ void selectCandidate(const int* members, const int* sizes,
                                int* assigned, int* output_members, Selection* selection,
                                int n, int cluster_id) {
    __shared__ int best_size[BLOCK_SIZE];
    __shared__ int best_seed[BLOCK_SIZE];
    __shared__ int output_offset;
    if (threadIdx.x == 0) output_offset = n - selection->remaining;
    int size = 0, seed = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int candidate_size = assigned[i] < 0 ? sizes[i] : 0;
        if (candidate_size > size || (candidate_size == size && i < seed)) {
            size = candidate_size;
            seed = i;
        }
    }
    best_size[threadIdx.x] = size;
    best_seed[threadIdx.x] = seed;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            int other_size = best_size[threadIdx.x + stride];
            int other_seed = best_seed[threadIdx.x + stride];
            if (other_size > best_size[threadIdx.x] ||
                (other_size == best_size[threadIdx.x] && other_seed < best_seed[threadIdx.x])) {
                best_size[threadIdx.x] = other_size;
                best_seed[threadIdx.x] = other_seed;
            }
        }
        __syncthreads();
    }
    seed = best_seed[0];
    size = best_size[0];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        int member = members[static_cast<size_t>(seed) * n + i];
        assigned[member] = cluster_id;
        output_members[output_offset + i] = member;
    }
    if (threadIdx.x == 0) {
        selection->seed = seed;
        selection->count = size;
        selection->remaining -= size;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t matrix_size = static_cast<size_t>(n) * n;
    if (matrix_size > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Point count is too large for CUDA buffers\n");
        std::exit(EXIT_FAILURE);
    }
    Point* d_points;
    double *d_distances, *d_maxima;
    int *d_members, *d_sizes, *d_assigned, *d_output_members;
    Selection* d_selection;
    CUDA_CHECK(cudaMalloc(&d_points, n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_distances, matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxima, matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_members, matrix_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_assigned, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_members, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_selection, sizeof(Selection)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sizes, 0, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_assigned, 0xff, n * sizeof(int)));
    Selection selection{-1, 0, n};
    CUDA_CHECK(cudaMemcpy(d_selection, &selection, sizeof(selection), cudaMemcpyHostToDevice));
    int blocks = static_cast<int>(std::min<size_t>((matrix_size + 255) / 256, 65535));
    calculateDistances<<<blocks, BLOCK_SIZE>>>(d_points, d_distances, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<Cluster> clusters;
    std::vector<int> cluster_sizes;
    while (selection.remaining > 0) {
        buildCandidates<<<n, BLOCK_SIZE>>>(d_distances, d_assigned, d_maxima,
                                            d_members, d_sizes, n, threshold);
        CUDA_CHECK(cudaGetLastError());
        selectCandidate<<<1, BLOCK_SIZE>>>(d_members, d_sizes, d_assigned,
                                            d_output_members, d_selection, n, static_cast<int>(clusters.size()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&selection, d_selection, sizeof(selection), cudaMemcpyDeviceToHost));
        if (selection.count <= 0) {
            fprintf(stderr, "CUDA clustering failed to select a candidate\n");
            std::exit(EXIT_FAILURE);
        }
        clusters.push_back(Cluster{{}, selection.seed});
        cluster_sizes.push_back(selection.count);
    }
    std::vector<int> output_members(n);
    CUDA_CHECK(cudaMemcpy(output_members.data(), d_output_members, n * sizeof(int), cudaMemcpyDeviceToHost));
    int offset = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
        clusters[i].members.assign(output_members.begin() + offset,
                                   output_members.begin() + offset + cluster_sizes[i]);
        offset += cluster_sizes[i];
    }
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_maxima));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_assigned));
    CUDA_CHECK(cudaFree(d_output_members));
    CUDA_CHECK(cudaFree(d_selection));
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
