// QT Clustering Benchmark - CUDA Parallelized Version
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
#include <cuda_runtime.h>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

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
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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

// Calculate Euclidean distance between two points (used for validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// CUDA kernel: compute full pairwise distance matrix
__global__ void computeDistanceMatrixKernel(const Point* points, double* distMatrix, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        double dx = points[i].x - points[j].x;
        double dy = points[i].y - points[j].y;
        distMatrix[i * N + j] = sqrt(dx * dx + dy * dy);
    }
}

// CUDA kernel: for each candidate, compute max distance to cluster members
__global__ void findClosestPointKernel(
    const int* clusterMembers, int clusterSize,
    const double* distMatrix, int N,
    const unsigned char* clustered, const unsigned char* inCluster,
    double threshold,
    double* outMaxDist) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= N) return;
    if (clustered[candidate] || inCluster[candidate]) {
        outMaxDist[candidate] = -1.0;
        return;
    }

    double maxDist = 0.0;
    for (int i = 0; i < clusterSize; i++) {
        double dist = distMatrix[candidate * N + clusterMembers[i]];
        if (dist > maxDist) maxDist = dist;
    }

    outMaxDist[candidate] = (maxDist < threshold) ? maxDist : -1.0;
}

// CUDA kernel: update a single position in inCluster
__global__ void updateInClusterKernel(int index, unsigned char* inCluster) {
    inCluster[index] = 1;
}

// GPU-accelerated candidate cluster generation
int generateCandidateClusterGPU(
    const int seed_point,
    const std::vector<bool>& clustered,
    const double* d_distMatrix, int N,
    unsigned char* d_clustered,
    unsigned char* d_inCluster,
    int* d_clusterMembers,
    double* d_maxDists,
    double* h_maxDists,
    double threshold,
    std::vector<int>* cluster_members = nullptr) {

    std::vector<unsigned char> h_inCluster(N, 0);
    std::vector<int> members;

    h_inCluster[seed_point] = 1;
    members.push_back(seed_point);

    // Upload clustered array (unchanged during this function)
    std::vector<unsigned char> h_clustered(N);
    for (int i = 0; i < N; i++) h_clustered[i] = clustered[i] ? 1 : 0;
    CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N, cudaMemcpyHostToDevice));

    // Upload initial in_cluster
    CUDA_CHECK(cudaMemcpy(d_inCluster, h_inCluster.data(), N, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    while (static_cast<int>(members.size()) < N) {
        // Upload current cluster members
        CUDA_CHECK(cudaMemcpy(d_clusterMembers, members.data(),
                              members.size() * sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel to compute max distance per candidate (parallelized)
        findClosestPointKernel<<<gridSize, blockSize>>>(
            d_clusterMembers, static_cast<int>(members.size()), d_distMatrix, N,
            d_clustered, d_inCluster, threshold, d_maxDists);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download results
        CUDA_CHECK(cudaMemcpy(h_maxDists, d_maxDists, N * sizeof(double), cudaMemcpyDeviceToHost));

        // Find closest point (same logic as original: first candidate with min diameter wins)
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();
        for (int i = 0; i < N; i++) {
            if (h_maxDists[i] >= 0 && h_maxDists[i] < min_diameter) {
                min_diameter = h_maxDists[i];
                closest = i;
            }
        }

        if (closest < 0) break;

        // Update in_cluster on GPU
        updateInClusterKernel<<<1, 1>>>(closest, d_inCluster);
        CUDA_CHECK(cudaGetLastError());

        h_inCluster[closest] = 1;
        members.push_back(closest);
    }

    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// GPU-accelerated QT clustering
std::vector<Cluster> qtClusteringGPU(
    const std::vector<Point>& points,
    double threshold,
    const double* d_distMatrix,
    unsigned char* d_clustered,
    unsigned char* d_inCluster,
    int* d_clusterMembers,
    double* d_maxDists,
    double* h_maxDists) {

    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;

        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateClusterGPU(
                seed, clustered, d_distMatrix, N,
                d_clustered, d_inCluster, d_clusterMembers, d_maxDists, h_maxDists,
                threshold, &candidate_members);

            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }
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

    printf("QT Clustering Benchmark (CUDA)\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    const int N = num_points;

    // --- GPU memory allocation ---
    Point* d_points = nullptr;
    double* d_distMatrix = nullptr;
    unsigned char* d_clustered = nullptr;
    unsigned char* d_inCluster = nullptr;
    int* d_clusterMembers = nullptr;
    double* d_maxDists = nullptr;
    double* h_maxDists = new double[N];

    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_distMatrix, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_inCluster, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_clusterMembers, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_maxDists, N * sizeof(double)));

    // Upload points to GPU
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));

    // Compute pairwise distance matrix on GPU
    const int blockDim = 32;
    const int gridX = (N + blockDim - 1) / blockDim;
    const int gridY = (N + blockDim - 1) / blockDim;
    computeDistanceMatrixKernel<<<dim3(gridX, gridY), dim3(blockDim, blockDim)>>>(
        d_points, d_distMatrix, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform QT clustering on GPU
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringGPU(
        points, threshold, d_distMatrix, d_clustered, d_inCluster,
        d_clusterMembers, d_maxDists, h_maxDists);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_distMatrix));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_inCluster));
    CUDA_CHECK(cudaFree(d_clusterMembers));
    CUDA_CHECK(cudaFree(d_maxDists));
    delete[] h_maxDists;

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
