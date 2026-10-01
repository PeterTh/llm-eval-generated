// QT Clustering Benchmark - CUDA implementation
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
#include <utility>
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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block constructs one candidate cluster. Candidate diameters are kept
// incrementally, so adding a member requires only one distance per point.
__global__ void candidateClusters(const Point* points, const unsigned char* clustered,
                                  double* diameters, int* cardinalities, int* members,
                                  int n, double threshold, int first_seed) {
    const int seed = first_seed + blockIdx.x;
    if (seed >= n || clustered[seed]) return;

    double* row = diameters + static_cast<size_t>(blockIdx.x) * n;
    const int tid = threadIdx.x;
    __shared__ double best_dist[256];
    __shared__ int best_index[256];
    __shared__ int next_member;

    for (int candidate = tid; candidate < n; candidate += blockDim.x) {
        row[candidate] = (clustered[candidate] || candidate == seed) ? DBL_MAX : 0.0;
    }
    if (tid == 0 && members) members[0] = seed;
    __syncthreads();

    int current = seed;
    int count = 1;
    while (count < n) {
        double local_dist = DBL_MAX;
        int local_index = n;
        const Point member = points[current];
        for (int candidate = tid; candidate < n; candidate += blockDim.x) {
            const double previous = row[candidate];
            if (previous == DBL_MAX) continue;
            const Point p = points[candidate];
            const double dx = p.x - member.x;
            const double dy = p.y - member.y;
            const double d = sqrt(dx * dx + dy * dy);
            const double diameter = fmax(previous, d);
            // Diameters only grow; a rejected candidate can never become valid.
            if (diameter >= threshold) {
                row[candidate] = DBL_MAX;
                continue;
            }
            row[candidate] = diameter;
            if (diameter < local_dist || (diameter == local_dist && candidate < local_index)) {
                local_dist = diameter;
                local_index = candidate;
            }
        }
        best_dist[tid] = local_dist;
        best_index[tid] = local_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset &&
                (best_dist[tid + offset] < best_dist[tid] ||
                 (best_dist[tid + offset] == best_dist[tid] &&
                  best_index[tid + offset] < best_index[tid]))) {
                best_dist[tid] = best_dist[tid + offset];
                best_index[tid] = best_index[tid + offset];
            }
            __syncthreads();
        }

        if (tid == 0) {
            next_member = best_index[0];
            if (next_member < n) {
                row[next_member] = DBL_MAX;
                if (members) members[count] = next_member;
            }
        }
        __syncthreads();
        if (next_member == n) break;
        current = next_member;
        ++count;
    }
    if (tid == 0) cardinalities[seed] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    double* d_diameters = nullptr;
    int* d_cardinalities = nullptr;
    int* d_members = nullptr;
    size_t free_bytes = 0, total_bytes = 0;
    checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query memory");
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(double);
    const size_t workspace_budget = std::min<size_t>(256ull << 20, free_bytes / 4);
    const int batch_size = static_cast<int>(std::min<size_t>(n,
        std::max<size_t>(1, workspace_budget / row_bytes)));

    checkCuda(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)), "allocate points");
    checkCuda(cudaMalloc(&d_clustered, n), "allocate cluster mask");
    checkCuda(cudaMalloc(&d_diameters, static_cast<size_t>(batch_size) * row_bytes), "allocate distances");
    checkCuda(cudaMalloc(&d_cardinalities, static_cast<size_t>(n) * sizeof(int)), "allocate cardinalities");
    checkCuda(cudaMalloc(&d_members, static_cast<size_t>(n) * sizeof(int)), "allocate members");
    checkCuda(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point),
                         cudaMemcpyHostToDevice), "copy points");

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> cardinalities(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        checkCuda(cudaMemcpy(d_clustered, clustered.data(), n, cudaMemcpyHostToDevice),
                  "copy cluster mask");
        for (int first = 0; first < n; first += batch_size) {
            const int blocks = std::min(batch_size, n - first);
            candidateClusters<<<blocks, 256>>>(d_points, d_clustered, d_diameters,
                                                 d_cardinalities, nullptr, n, threshold, first);
        }
        checkCuda(cudaGetLastError(), "launch candidate clusters");
        checkCuda(cudaMemcpy(cardinalities.data(), d_cardinalities,
                             static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost),
                  "copy cardinalities");

        int best_seed = -1;
        int max_cardinality = -1;
        for (int seed = 0; seed < n; ++seed) {
            if (!clustered[seed] && cardinalities[seed] > max_cardinality) {
                max_cardinality = cardinalities[seed];
                best_seed = seed;
            }
        }
        if (best_seed < 0) break;

        candidateClusters<<<1, 256>>>(d_points, d_clustered, d_diameters,
                                      d_cardinalities, d_members, n, threshold, best_seed);
        checkCuda(cudaGetLastError(), "launch winning cluster");
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(max_cardinality);
        checkCuda(cudaMemcpy(cluster.members.data(), d_members,
                             static_cast<size_t>(max_cardinality) * sizeof(int), cudaMemcpyDeviceToHost),
                  "copy winning cluster");
        for (int member : cluster.members) {
            clustered[member] = 1;
        }
        remaining -= max_cardinality;
        clusters.push_back(std::move(cluster));
    }

    checkCuda(cudaFree(d_members), "free members");
    checkCuda(cudaFree(d_cardinalities), "free cardinalities");
    checkCuda(cudaFree(d_diameters), "free distances");
    checkCuda(cudaFree(d_clustered), "free cluster mask");
    checkCuda(cudaFree(d_points), "free points");
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
