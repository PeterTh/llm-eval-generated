// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
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

// CUDA failures are not recoverable here: this benchmark intentionally always
// executes its clustering work on the GPU.
void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void buildDistanceMatrix(const Point* points, double* distances, int n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(n) * n;
    if (index >= count) return;

    const int row = static_cast<int>(index / n);
    const int column = static_cast<int>(index - static_cast<size_t>(row) * n);
    const double dx = points[row].x - points[column].x;
    const double dy = points[row].y - points[column].y;
    distances[index] = sqrt(dx * dx + dy * dy);
}

// One block constructs one candidate cluster.  Candidate clusters are
// independent for a fixed clustered set, so this exposes the main QT search
// dimension to the GPU without changing the host's deterministic seed choice.
__global__ void buildCandidateClusters(const int* seeds, int seed_count,
                                       const unsigned char* clustered,
                                       const double* distances, double threshold,
                                       int n, int* all_members, int* cardinalities) {
    const int slot = blockIdx.x;
    if (slot >= seed_count) return;

    const int thread = threadIdx.x;
    const int seed = seeds[slot];
    int* members = all_members + static_cast<size_t>(slot) * n;
    __shared__ int member_count;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];

    if (thread == 0) {
        members[0] = seed;
        member_count = 1;
    }
    __syncthreads();

    while (member_count < n) {
        double local_distance = DBL_MAX;
        int local_index = -1;

        for (int candidate = thread; candidate < n; candidate += blockDim.x) {
            if (clustered[candidate]) continue;

            bool already_member = false;
            double maximum_distance = 0.0;
            for (int m = 0; m < member_count; ++m) {
                const int member = members[m];
                if (candidate == member) {
                    already_member = true;
                    break;
                }
                // Adjacent threads process adjacent candidates.  Indexing this
                // symmetric matrix by member first therefore makes each warp's
                // load contiguous and coalesced.
                const double candidate_distance =
                    distances[static_cast<size_t>(member) * n + candidate];
                if (candidate_distance > maximum_distance) {
                    maximum_distance = candidate_distance;
                }
            }

            // The index tie break exactly matches the original forward scan,
            // whose strict comparison retains the first point encountered.
            if (!already_member && maximum_distance < threshold &&
                (maximum_distance < local_distance ||
                 (maximum_distance == local_distance &&
                  (local_index < 0 || candidate < local_index)))) {
                local_distance = maximum_distance;
                local_index = candidate;
            }
        }

        best_distance[thread] = local_distance;
        best_index[thread] = local_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (thread < offset) {
                const double other_distance = best_distance[thread + offset];
                const int other_index = best_index[thread + offset];
                if (other_distance < best_distance[thread] ||
                    (other_distance == best_distance[thread] && other_index >= 0 &&
                     (best_index[thread] < 0 || other_index < best_index[thread]))) {
                    best_distance[thread] = other_distance;
                    best_index[thread] = other_index;
                }
            }
            __syncthreads();
        }

        if (thread == 0) {
            if (best_index[0] < 0) {
                cardinalities[slot] = member_count;
            } else {
                members[member_count++] = best_index[0];
            }
        }
        __syncthreads();
        // Every thread must take this exit together; otherwise the block would
        // diverge around a later __syncthreads().
        if (best_index[0] < 0) break;
    }

    if (thread == 0 && member_count == n) {
        cardinalities[slot] = member_count;
    }
}

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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    Point* device_points = nullptr;
    double* device_distances = nullptr;
    int* device_seeds = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_members = nullptr;
    int* device_cardinalities = nullptr;
    const size_t point_bytes = static_cast<size_t>(N) * sizeof(Point);
    const size_t matrix_elements = static_cast<size_t>(N) * N;

    cudaCheck(cudaMalloc(&device_points, point_bytes), "allocating points");
    cudaCheck(cudaMalloc(&device_distances, matrix_elements * sizeof(double)),
              "allocating distance matrix");
    cudaCheck(cudaMalloc(&device_seeds, static_cast<size_t>(N) * sizeof(int)),
              "allocating seed list");
    cudaCheck(cudaMalloc(&device_clustered, static_cast<size_t>(N)),
              "allocating clustered flags");
    cudaCheck(cudaMalloc(&device_members, matrix_elements * sizeof(int)),
              "allocating candidate memberships");
    cudaCheck(cudaMalloc(&device_cardinalities, static_cast<size_t>(N) * sizeof(int)),
              "allocating candidate cardinalities");
    cudaCheck(cudaMemcpy(device_points, points.data(), point_bytes, cudaMemcpyHostToDevice),
              "copying points");

    constexpr int distance_threads = 256;
    const int distance_blocks = static_cast<int>(
        (matrix_elements + distance_threads - 1) / distance_threads);
    buildDistanceMatrix<<<distance_blocks, distance_threads>>>(device_points, device_distances, N);
    cudaCheck(cudaGetLastError(), "launching distance-matrix kernel");
    cudaCheck(cudaDeviceSynchronize(), "building distance matrix");

    std::vector<unsigned char> clustered_bytes(N, 0);
    std::vector<int> candidate_cardinalities(N);
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_slot = -1;
        std::vector<int> best_cluster_members;

        for (int i = 0; i < N; ++i) {
            clustered_bytes[i] = clustered[i] ? 1 : 0;
        }
        const int seed_count = static_cast<int>(unclustered_indices.size());
        cudaCheck(cudaMemcpy(device_seeds, unclustered_indices.data(),
                             static_cast<size_t>(seed_count) * sizeof(int), cudaMemcpyHostToDevice),
                  "copying seed list");
        cudaCheck(cudaMemcpy(device_clustered, clustered_bytes.data(), static_cast<size_t>(N),
                             cudaMemcpyHostToDevice), "copying clustered flags");

        constexpr int candidate_threads = 256;
        buildCandidateClusters<<<seed_count, candidate_threads>>>(
            device_seeds, seed_count, device_clustered, device_distances, threshold, N,
            device_members, device_cardinalities);
        cudaCheck(cudaGetLastError(), "launching candidate-cluster kernel");
        cudaCheck(cudaMemcpy(candidate_cardinalities.data(), device_cardinalities,
                             static_cast<size_t>(seed_count) * sizeof(int), cudaMemcpyDeviceToHost),
                  "copying candidate cardinalities");

        // Preserve the original strict cardinality comparison: equal-size
        // candidates retain the first seed in unclustered_indices.
        for (int slot = 0; slot < seed_count; ++slot) {
            if (candidate_cardinalities[slot] > max_cardinality) {
                max_cardinality = candidate_cardinalities[slot];
                best_slot = slot;
            }
        }

        const int best_seed = best_slot >= 0 ? unclustered_indices[best_slot] : -1;
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            best_cluster_members.resize(max_cardinality);
            cudaCheck(cudaMemcpy(best_cluster_members.data(),
                                 device_members + static_cast<size_t>(best_slot) * N,
                                 static_cast<size_t>(max_cardinality) * sizeof(int),
                                 cudaMemcpyDeviceToHost), "copying best candidate membership");
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    cudaCheck(cudaFree(device_cardinalities), "freeing candidate cardinalities");
    cudaCheck(cudaFree(device_members), "freeing candidate memberships");
    cudaCheck(cudaFree(device_clustered), "freeing clustered flags");
    cudaCheck(cudaFree(device_seeds), "freeing seed list");
    cudaCheck(cudaFree(device_distances), "freeing distance matrix");
    cudaCheck(cudaFree(device_points), "freeing points");
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
