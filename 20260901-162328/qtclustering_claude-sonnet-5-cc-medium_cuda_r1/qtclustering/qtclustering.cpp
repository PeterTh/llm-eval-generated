// QT Clustering Benchmark - CUDA GPU-Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// GPU parallelization strategy: for every round of the outer clustering
// loop, all still-unclustered points are simultaneously evaluated as
// candidate cluster seeds. One CUDA thread block is assigned per seed
// candidate; the threads within a block cooperatively grow that seed's
// candidate cluster (maintaining a running max-distance-to-cluster value
// per point, updated incrementally as members are added, and reduced in
// parallel to find the next closest eligible point). Once every block has
// produced a candidate cardinality, the host selects the winning seed
// (identical tie-break semantics to the original sequential algorithm:
// first, in ascending index order, to achieve the maximum cardinality),
// commits that cluster, and the next round is launched.

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

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

static constexpr int THREADS_PER_BLOCK = 256;

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

// Calculate Euclidean distance between two points (host-side, used only by validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// One thread block per seed candidate. The block grows the candidate cluster
// for that seed, maintaining a running "max distance from this point to the
// current cluster" value for every point (max_dist_matrix row), updated
// incrementally whenever a new member is added. Each iteration performs a
// parallel argmin reduction over eligible candidates to find the point that
// keeps the cluster diameter smallest while staying below threshold, exactly
// mirroring the sequential findClosestPoint/generateCandidateCluster logic.
__global__ void qtClusterKernel(const double* __restrict__ px,
                                 const double* __restrict__ py,
                                 const unsigned char* __restrict__ clustered,
                                 unsigned char* __restrict__ in_cluster_matrix,
                                 double* __restrict__ max_dist_matrix,
                                 int* __restrict__ cardinality,
                                 const double threshold,
                                 const int N) {
    const int seed = blockIdx.x;

    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinality[seed] = -1;
        return;
    }

    unsigned char* in_cluster = in_cluster_matrix + static_cast<size_t>(seed) * N;
    double* max_dist = max_dist_matrix + static_cast<size_t>(seed) * N;

    const double sx = px[seed];
    const double sy = py[seed];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        in_cluster[i] = (i == seed) ? 1 : 0;
        const double dx = px[i] - sx;
        const double dy = py[i] - sy;
        max_dist[i] = sqrt(dx * dx + dy * dy);
    }

    __shared__ double sh_val[THREADS_PER_BLOCK];
    __shared__ int sh_idx[THREADS_PER_BLOCK];
    __shared__ int s_best_idx;
    __shared__ int s_count;

    if (threadIdx.x == 0) s_count = 1;
    __syncthreads();

    while (s_count < N) {
        double best_val = DBL_MAX;
        int best_idx = -1;

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            if (!clustered[i] && !in_cluster[i]) {
                const double v = max_dist[i];
                if (v < threshold && v < best_val) {
                    best_val = v;
                    best_idx = i;
                }
            }
        }
        sh_val[threadIdx.x] = best_val;
        sh_idx[threadIdx.x] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double ov = sh_val[threadIdx.x + stride];
                const int oi = sh_idx[threadIdx.x + stride];
                const double mv = sh_val[threadIdx.x];
                const int mi = sh_idx[threadIdx.x];
                if (oi >= 0 && (mi < 0 || ov < mv || (ov == mv && oi < mi))) {
                    sh_val[threadIdx.x] = ov;
                    sh_idx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) s_best_idx = sh_idx[0];
        __syncthreads();

        if (s_best_idx < 0) break; // No eligible candidate remains

        if (threadIdx.x == 0) {
            in_cluster[s_best_idx] = 1;
            s_count++;
        }
        __syncthreads();

        const double bx = px[s_best_idx];
        const double by = py[s_best_idx];
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            const double dx = px[i] - bx;
            const double dy = py[i] - by;
            const double d = sqrt(dx * dx + dy * dy);
            if (d > max_dist[i]) max_dist[i] = d;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) cardinality[seed] = s_count;
}

// Main QT clustering algorithm - GPU-parallel across all candidate seeds
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    std::vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
    }

    double *d_px = nullptr, *d_py = nullptr;
    unsigned char* d_clustered = nullptr;
    unsigned char* d_in_cluster_matrix = nullptr;
    double* d_max_dist_matrix = nullptr;
    int* d_cardinality = nullptr;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster_matrix, static_cast<size_t>(N) * N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_max_dist_matrix, static_cast<size_t>(N) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cardinality, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_px, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N * sizeof(unsigned char)));

    std::vector<unsigned char> h_clustered(N, 0);
    std::vector<int> h_cardinality(N);
    std::vector<unsigned char> h_in_cluster_row(N);

    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    while (!unclustered_indices.empty()) {
        qtClusterKernel<<<N, THREADS_PER_BLOCK>>>(d_px, d_py, d_clustered,
                                                    d_in_cluster_matrix, d_max_dist_matrix,
                                                    d_cardinality, threshold, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_cardinality.data(), d_cardinality, N * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_seed = -1;
        for (const int seed : unclustered_indices) {
            if (h_cardinality[seed] > max_cardinality) {
                max_cardinality = h_cardinality[seed];
                best_seed = seed;
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        CUDA_CHECK(cudaMemcpy(h_in_cluster_row.data(),
                              d_in_cluster_matrix + static_cast<size_t>(best_seed) * N, N,
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        for (int i = 0; i < N; ++i) {
            if (h_in_cluster_row[i]) cluster.members.push_back(i);
        }
        clusters.push_back(cluster);

        for (const int member : cluster.members) {
            h_clustered[member] = 1;
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&h_clustered](int idx) { return h_clustered[idx] != 0; }),
            unclustered_indices.end()
        );
    }

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_clustered);
    cudaFree(d_in_cluster_matrix);
    cudaFree(d_max_dist_matrix);
    cudaFree(d_cardinality);

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
