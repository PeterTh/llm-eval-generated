// QT Clustering Benchmark - CUDA version
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
        // The original expression is always zero for N <= 30.
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// One block grows one seed cluster. Distances are updated only against the
// latest member: the saved maximum is the exact diameter for each candidate.
// A second invocation records the selected seed's members in greedy order.
__global__ void growClusters(const Point* points, const unsigned char* clustered,
                             int n, double threshold, int first_seed,
                             double* scores, int* cardinalities, int* members) {
    const int seed = first_seed + blockIdx.x;
    if (seed >= n || clustered[seed]) {
        if (cardinalities) cardinalities[seed] = 0;
        return;
    }

    double* row = scores + static_cast<size_t>(blockIdx.x) * n;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];
    __shared__ int current;
    __shared__ int count;
    if (threadIdx.x == 0) {
        current = seed;
        count = 1;
        if (members) members[0] = seed;
    }
    for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x)
        row[candidate] = (clustered[candidate] || candidate == seed) ? -1.0 : 0.0;
    __syncthreads();

    while (true) {
        double local_distance = DBL_MAX;
        int local_index = n;
        const Point member = points[current];
        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            double diameter = row[candidate];
            if (diameter < 0.0 || diameter >= threshold) continue;
            const double dx = points[candidate].x - member.x;
            const double dy = points[candidate].y - member.y;
            const double d = sqrt(dx * dx + dy * dy);
            if (d > diameter) diameter = d;
            row[candidate] = diameter;
            if (diameter < threshold &&
                (diameter < local_distance ||
                 (diameter == local_distance && candidate < local_index))) {
                local_distance = diameter;
                local_index = candidate;
            }
        }
        best_distance[threadIdx.x] = local_distance;
        best_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double other = best_distance[threadIdx.x + stride];
                const int other_index = best_index[threadIdx.x + stride];
                if (other < best_distance[threadIdx.x] ||
                    (other == best_distance[threadIdx.x] && other_index < best_index[threadIdx.x])) {
                    best_distance[threadIdx.x] = other;
                    best_index[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            current = best_index[0] == n ? -1 : best_index[0];
            if (current >= 0) {
                row[current] = -1.0;
                if (members) members[count] = current;
                ++count;
            }
        }
        __syncthreads();
        if (current < 0) break;
    }
    if (threadIdx.x == 0 && cardinalities) cardinalities[seed] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    double* d_scores = nullptr;
    int* d_cardinalities = nullptr;
    int* d_members = nullptr;
    cudaCheck(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)), "allocate points");
    cudaCheck(cudaMalloc(&d_clustered, static_cast<size_t>(n)), "allocate flags");
    cudaCheck(cudaMalloc(&d_cardinalities, static_cast<size_t>(n) * sizeof(int)), "allocate cardinalities");
    cudaCheck(cudaMalloc(&d_members, static_cast<size_t>(n) * sizeof(int)), "allocate members");
    cudaCheck(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point),
                         cudaMemcpyHostToDevice), "copy points");

    size_t free_bytes = 0, total_bytes = 0;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "query memory");
    const size_t scratch_budget = std::min(free_bytes / 3, size_t(512) << 20);
    const int batch_size = static_cast<int>(std::min(static_cast<size_t>(n),
        std::max(size_t(1), scratch_budget / (static_cast<size_t>(n) * sizeof(double)))));
    cudaCheck(cudaMalloc(&d_scores, static_cast<size_t>(batch_size) * n * sizeof(double)),
              "allocate candidate scores");

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> cardinalities(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(n),
                             cudaMemcpyHostToDevice), "copy flags");
        int best_seed = -1;
        int max_cardinality = 0;
        for (int first = 0; first < n; first += batch_size) {
            const int seeds = std::min(batch_size, n - first);
            growClusters<<<seeds, 256>>>(d_points, d_clustered, n, threshold, first,
                                          d_scores, d_cardinalities, nullptr);
            cudaCheck(cudaGetLastError(), "launch seed clusters");
            cudaCheck(cudaMemcpy(cardinalities.data() + first, d_cardinalities + first,
                                 static_cast<size_t>(seeds) * sizeof(int), cudaMemcpyDeviceToHost),
                      "copy cardinalities");
            for (int seed = first; seed < first + seeds; ++seed) {
                if (cardinalities[seed] > max_cardinality) {
                    max_cardinality = cardinalities[seed];
                    best_seed = seed;
                }
            }
        }
        if (best_seed < 0) break;

        growClusters<<<1, 256>>>(d_points, d_clustered, n, threshold, best_seed,
                                  d_scores, nullptr, d_members);
        cudaCheck(cudaGetLastError(), "launch winning cluster");
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(max_cardinality);
        cudaCheck(cudaMemcpy(cluster.members.data(), d_members,
                             static_cast<size_t>(max_cardinality) * sizeof(int),
                             cudaMemcpyDeviceToHost), "copy winning cluster");
        for (int member : cluster.members) {
            clustered[member] = 1;
            --remaining;
        }
        clusters.push_back(std::move(cluster));
    }

    cudaCheck(cudaFree(d_members), "free members");
    cudaCheck(cudaFree(d_cardinalities), "free cardinalities");
    cudaCheck(cudaFree(d_scores), "free candidate scores");
    cudaCheck(cudaFree(d_clustered), "free flags");
    cudaCheck(cudaFree(d_points), "free points");
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
