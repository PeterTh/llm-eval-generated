// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstddef>
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

namespace {

constexpr int CUDA_THREADS_PER_BLOCK = 256;

void checkCuda(const cudaError_t status, const char* operation, const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while %s: %s\n", file, line, operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

bool checkedProduct(const size_t lhs, const size_t rhs, size_t* result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

__device__ __forceinline__ bool isBetterCandidate(const double distance_a, const int index_a,
                                                   const double distance_b, const int index_b) {
    return distance_a < distance_b || (distance_a == distance_b && index_a < index_b);
}

__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances, const int point_count) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t element_count = static_cast<size_t>(point_count) * point_count;
    if (element >= element_count) {
        return;
    }

    const int row = static_cast<int>(element / point_count);
    const int column = static_cast<int>(element - static_cast<size_t>(row) * point_count);
    const double dx = points[row].x - points[column].x;
    const double dy = points[row].y - points[column].y;
    distances[element] = sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ double pointDistance(const Point* __restrict__ points,
                                                const double* __restrict__ distances,
                                                const bool use_distance_matrix,
                                                const int first, const int second,
                                                const int point_count) {
    if (use_distance_matrix) {
        // The matrix is symmetric.  Keeping first fixed makes accesses from a
        // block contiguous when all threads update their candidate distances.
        return distances[static_cast<size_t>(first) * point_count + second];
    }
    const double dx = points[first].x - points[second].x;
    const double dy = points[first].y - points[second].y;
    return sqrt(dx * dx + dy * dy);
}

// One block builds one candidate cluster.  Each candidate's current maximum
// distance is updated only for the newly added member, changing the repeated
// max computation from O(cluster_size^2) to O(cluster_size) per candidate.
__global__ void generateCandidateClusters(const Point* __restrict__ points,
                                          const double* __restrict__ distances,
                                          const bool use_distance_matrix,
                                          const unsigned char* __restrict__ clustered,
                                          const int* __restrict__ seeds,
                                          const int point_count,
                                          const double threshold,
                                          int* __restrict__ member_orders,
                                          double* __restrict__ maximum_distances,
                                          int* __restrict__ cardinalities) {
    const int candidate_slot = blockIdx.x;
    const int thread = threadIdx.x;
    const int seed = seeds[candidate_slot];
    int* const orders = member_orders + static_cast<size_t>(candidate_slot) * point_count;
    double* const candidate_maximums = maximum_distances +
                                       static_cast<size_t>(candidate_slot) * point_count;

    __shared__ double thread_best_distances[CUDA_THREADS_PER_BLOCK];
    __shared__ int thread_best_indices[CUDA_THREADS_PER_BLOCK];
    __shared__ int cluster_size;
    __shared__ int next_point;

    for (int point = thread; point < point_count; point += blockDim.x) {
        orders[point] = -1;
        candidate_maximums[point] = pointDistance(points, distances, use_distance_matrix,
                                                  seed, point, point_count);
    }
    __syncthreads();

    if (thread == 0) {
        orders[seed] = 0;
        candidate_maximums[seed] = 0.0;
        cluster_size = 1;
    }
    __syncthreads();

    while (true) {
        double thread_best_distance = 1.7976931348623157e308;
        int thread_best_index = INT_MAX;
        for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
            const double maximum_distance = candidate_maximums[candidate];
            if (clustered[candidate] == 0 && orders[candidate] < 0 && maximum_distance < threshold &&
                isBetterCandidate(maximum_distance, candidate,
                                  thread_best_distance, thread_best_index)) {
                thread_best_distance = maximum_distance;
                thread_best_index = candidate;
            }
        }

        thread_best_distances[thread] = thread_best_distance;
        thread_best_indices[thread] = thread_best_index;
        __syncthreads();

        if (thread == 0) {
            double next_distance = 1.7976931348623157e308;
            next_point = INT_MAX;
            for (int worker = 0; worker < blockDim.x; ++worker) {
                if (isBetterCandidate(thread_best_distances[worker], thread_best_indices[worker],
                                      next_distance, next_point)) {
                    next_distance = thread_best_distances[worker];
                    next_point = thread_best_indices[worker];
                }
            }

            if (next_point != INT_MAX) {
                orders[next_point] = cluster_size;
                ++cluster_size;
            }
        }
        __syncthreads();

        if (next_point == INT_MAX) {
            break;
        }

        for (int candidate = thread; candidate < point_count; candidate += blockDim.x) {
            if (clustered[candidate] == 0 && orders[candidate] < 0) {
                const double distance_to_new_member = pointDistance(
                    points, distances, use_distance_matrix, next_point, candidate, point_count);
                candidate_maximums[candidate] = fmax(candidate_maximums[candidate],
                                                      distance_to_new_member);
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        cardinalities[candidate_slot] = cluster_size;
    }
}

__global__ void markClustered(const int* __restrict__ member_orders,
                              unsigned char* __restrict__ clustered, const int point_count) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point < point_count && member_orders[point] >= 0) {
        clustered[point] = 1;
    }
}

size_t candidateBatchSize(const int point_count, const size_t free_memory) {
    // Leave substantial headroom for the CUDA runtime and concurrent allocations.
    const size_t bytes_per_candidate = static_cast<size_t>(point_count) *
                                       (sizeof(int) + sizeof(double)) + sizeof(int);
    const size_t usable_memory = free_memory / 2;
    const size_t by_memory = usable_memory / bytes_per_candidate;
    return std::max<size_t>(1, std::min(static_cast<size_t>(point_count), by_memory));
}

} // namespace

// Main QT clustering algorithm. Candidate clusters are generated independently
// on the GPU; cluster selection remains ordered to preserve the original ties.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int point_count = static_cast<int>(points.size());
    const size_t point_count_size = static_cast<size_t>(point_count);
    size_t matrix_elements = 0;
    if (!checkedProduct(point_count_size, point_count_size, &matrix_elements)) {
        std::fprintf(stderr, "CUDA error: point count is too large for addressable device storage\n");
        std::exit(EXIT_FAILURE);
    }

    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_seeds = nullptr;
    int* device_member_orders = nullptr;
    int* device_best_orders = nullptr;
    int* device_cardinalities = nullptr;
    double* device_distances = nullptr;
    double* device_maximum_distances = nullptr;

    CUDA_CHECK(cudaMalloc(&device_points, point_count_size * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&device_clustered, point_count_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_seeds, point_count_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_best_orders, point_count_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(), point_count_size * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_clustered, 0, point_count_size * sizeof(unsigned char)));

    bool use_distance_matrix = false;
    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    const bool matrix_size_valid = matrix_elements <= std::numeric_limits<size_t>::max() / sizeof(double);
    const size_t matrix_bytes = matrix_size_valid ? matrix_elements * sizeof(double) : 0;
    if (matrix_size_valid && matrix_bytes <= free_memory * 3 / 4) {
        const cudaError_t allocation = cudaMalloc(&device_distances, matrix_bytes);
        if (allocation == cudaSuccess) {
            const size_t distance_blocks = (matrix_elements + CUDA_THREADS_PER_BLOCK - 1) /
                                           CUDA_THREADS_PER_BLOCK;
            buildDistanceMatrix<<<static_cast<unsigned int>(distance_blocks), CUDA_THREADS_PER_BLOCK>>>(
                device_points, device_distances, point_count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            use_distance_matrix = true;
        }
    }

    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    const size_t batch_size = candidateBatchSize(point_count, free_memory);
    size_t order_elements = 0;
    if (!checkedProduct(batch_size, point_count_size, &order_elements)) {
        std::fprintf(stderr, "CUDA error: candidate batch is too large for addressable device storage\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMalloc(&device_member_orders, order_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_maximum_distances, order_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_cardinalities, batch_size * sizeof(int)));

    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    std::vector<int> batch_cardinalities(batch_size);
    std::vector<int> best_member_orders(point_count);
    std::vector<Cluster> clusters;
    for (int point = 0; point < point_count; ++point) {
        unclustered_indices[point] = point;
    }

    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;

        for (size_t offset = 0; offset < unclustered_indices.size(); offset += batch_size) {
            const size_t current_batch_size = std::min(batch_size, unclustered_indices.size() - offset);
            CUDA_CHECK(cudaMemcpy(device_seeds, unclustered_indices.data() + offset,
                                  current_batch_size * sizeof(int), cudaMemcpyHostToDevice));
            generateCandidateClusters<<<static_cast<unsigned int>(current_batch_size), CUDA_THREADS_PER_BLOCK>>>(
                device_points, device_distances, use_distance_matrix, device_clustered, device_seeds,
                point_count, threshold, device_member_orders, device_maximum_distances,
                device_cardinalities);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(batch_cardinalities.data(), device_cardinalities,
                                  current_batch_size * sizeof(int), cudaMemcpyDeviceToHost));

            // Strict comparison deliberately retains the first seed on a cardinality tie.
            for (size_t candidate = 0; candidate < current_batch_size; ++candidate) {
                if (batch_cardinalities[candidate] > max_cardinality) {
                    max_cardinality = batch_cardinalities[candidate];
                    best_seed = unclustered_indices[offset + candidate];
                    CUDA_CHECK(cudaMemcpy(device_best_orders,
                                          device_member_orders + candidate * point_count_size,
                                          point_count_size * sizeof(int), cudaMemcpyDeviceToDevice));
                }
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) {
            break;
        }

        CUDA_CHECK(cudaMemcpy(best_member_orders.data(), device_best_orders,
                              point_count_size * sizeof(int), cudaMemcpyDeviceToHost));
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(max_cardinality);
        for (int point = 0; point < point_count; ++point) {
            const int order = best_member_orders[point];
            if (order >= 0) {
                cluster.members[order] = point;
                clustered[point] = 1;
            }
        }
        clusters.push_back(std::move(cluster));

        const int mark_blocks = (point_count + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
        markClustered<<<mark_blocks, CUDA_THREADS_PER_BLOCK>>>(device_best_orders, device_clustered, point_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int index) { return clustered[index] != 0; }),
            unclustered_indices.end());
    }

    CUDA_CHECK(cudaFree(device_distances));
    CUDA_CHECK(cudaFree(device_cardinalities));
    CUDA_CHECK(cudaFree(device_maximum_distances));
    CUDA_CHECK(cudaFree(device_best_orders));
    CUDA_CHECK(cudaFree(device_member_orders));
    CUDA_CHECK(cudaFree(device_seeds));
    CUDA_CHECK(cudaFree(device_clustered));
    CUDA_CHECK(cudaFree(device_points));
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

    // Initialize the CUDA context outside the timed clustering region; this is
    // one-time runtime setup rather than work performed by the algorithm.
    CUDA_CHECK(cudaFree(nullptr));

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
