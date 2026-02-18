// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
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

static inline void cudaCheck(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

static __global__ void computeDistanceMatrixKernel(const double* __restrict__ xs,
                                                   const double* __restrict__ ys,
                                                   double* __restrict__ dist,
                                                   int N) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    const double dx = xs[i] - xs[j];
    const double dy = ys[i] - ys[j];
    dist[i * N + j] = sqrt(dx * dx + dy * dy);
}

static __global__ void zeroInClusterKernel(unsigned char* __restrict__ in_cluster, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) in_cluster[idx] = 0;
}

static __global__ void initSeedsKernel(const unsigned char* __restrict__ clustered,
                                      unsigned char* __restrict__ active,
                                      int* __restrict__ counts,
                                      int* __restrict__ members,
                                      unsigned char* __restrict__ in_cluster,
                                      int N) {
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= N) return;
    if (clustered[seed]) {
        active[seed] = 0;
        counts[seed] = 0;
        return;
    }
    active[seed] = 1;
    counts[seed] = 1;
    members[seed * N] = seed;
    in_cluster[seed * N + seed] = 1;
}

static __global__ void findBestCandidateKernel(const double* __restrict__ dist,
                                              const unsigned char* __restrict__ clustered,
                                              const unsigned char* __restrict__ in_cluster,
                                              const unsigned char* __restrict__ active,
                                              const int* __restrict__ counts,
                                              const int* __restrict__ members,
                                              int* __restrict__ best_idx,
                                              double* __restrict__ best_val,
                                              double threshold,
                                              int N) {
    const int seed = blockIdx.x;
    if (seed >= N) return;

    if (!active[seed]) {
        if (threadIdx.x == 0) {
            best_idx[seed] = -1;
            best_val[seed] = HUGE_VAL;
        }
        return;
    }

    const int count = counts[seed];
    const unsigned char* row_in = in_cluster + seed * N;
    const int* row_mem = members + seed * N;

    double local_best = HUGE_VAL;
    int local_idx = INT_MAX;

    for (int cand = threadIdx.x; cand < N; cand += blockDim.x) {
        if (clustered[cand] || row_in[cand]) continue;

        double max_d = 0.0;
        const double* dist_row = dist + cand * N;
        for (int i = 0; i < count; ++i) {
            const int m = row_mem[i];
            const double d = dist_row[m];
            if (d > max_d) max_d = d;
            if (max_d >= local_best) break;
        }

        if (max_d < threshold && (max_d < local_best || (max_d == local_best && cand < local_idx))) {
            local_best = max_d;
            local_idx = cand;
        }
    }

    extern __shared__ unsigned char smem[];
    double* sh_val = reinterpret_cast<double*>(smem);
    int* sh_idx = reinterpret_cast<int*>(sh_val + blockDim.x);
    sh_val[threadIdx.x] = local_best;
    sh_idx[threadIdx.x] = local_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const double ov = sh_val[threadIdx.x + offset];
            const int oi = sh_idx[threadIdx.x + offset];
            const double cv = sh_val[threadIdx.x];
            const int ci = sh_idx[threadIdx.x];
            if (ov < cv || (ov == cv && oi < ci)) {
                sh_val[threadIdx.x] = ov;
                sh_idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (sh_idx[0] == INT_MAX) {
            best_idx[seed] = -1;
            best_val[seed] = HUGE_VAL;
        } else {
            best_idx[seed] = sh_idx[0];
            best_val[seed] = sh_val[0];
        }
    }
}

static __global__ void applyBestKernel(unsigned char* __restrict__ in_cluster,
                                      unsigned char* __restrict__ active,
                                      int* __restrict__ counts,
                                      int* __restrict__ members,
                                      const int* __restrict__ best_idx,
                                      int N) {
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= N) return;
    if (!active[seed]) return;

    const int idx = best_idx[seed];
    if (idx < 0) {
        active[seed] = 0;
        return;
    }

    const int count = counts[seed];
    in_cluster[seed * N + idx] = 1;
    members[seed * N + count] = idx;
    counts[seed] = count + 1;
}

static __global__ void countActiveKernel(const unsigned char* __restrict__ active, int N, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && active[i]) atomicAdd(out, 1);
}

static __global__ void setClusteredKernel(unsigned char* __restrict__ clustered,
                                         const int* __restrict__ member_list,
                                         int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) clustered[member_list[i]] = 1;
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

// Main QT clustering algorithm (CUDA-parallel)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered_host(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    // Device buffers
    double *d_xs = nullptr, *d_ys = nullptr, *d_dist = nullptr;
    unsigned char *d_clustered = nullptr, *d_in_cluster = nullptr, *d_active = nullptr;
    int *d_counts = nullptr, *d_members = nullptr, *d_best_idx = nullptr;
    double *d_best_val = nullptr;
    int *d_active_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_xs, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(double) * N * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, sizeof(unsigned char) * N * N));
    CUDA_CHECK(cudaMalloc(&d_active, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N * N));
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_best_val, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));

    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        xs[i] = points[i].x;
        ys[i] = points[i].y;
    }

    CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, sizeof(unsigned char) * N));

    // Precompute full distance matrix once.
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        computeDistanceMatrixKernel<<<grid, block>>>(d_xs, d_ys, d_dist, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<int> counts_host(N);

    while (!unclustered_indices.empty()) {
        // Initialize per-seed state for this main iteration.
        const int total_in = N * N;
        zeroInClusterKernel<<<(total_in + 255) / 256, 256>>>(d_in_cluster, total_in);
        initSeedsKernel<<<(N + 255) / 256, 256>>>(d_clustered, d_active, d_counts, d_members, d_in_cluster, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Simulate greedy candidate cluster growth for all seeds in parallel.
        const int threads = 256;
        const size_t shmem = sizeof(double) * threads + sizeof(int) * threads;
        for (int step = 1; step < N; ++step) {
            findBestCandidateKernel<<<N, threads, shmem>>>(
                d_dist, d_clustered, d_in_cluster, d_active, d_counts, d_members,
                d_best_idx, d_best_val, threshold, N);
            applyBestKernel<<<(N + 255) / 256, 256>>>(d_in_cluster, d_active, d_counts, d_members, d_best_idx, N);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int)));
            countActiveKernel<<<(N + 255) / 256, 256>>>(d_active, N, d_active_count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int active_count = 0;
            CUDA_CHECK(cudaMemcpy(&active_count, d_active_count, sizeof(int), cudaMemcpyDeviceToHost));
            if (active_count == 0) break;
        }

        CUDA_CHECK(cudaMemcpy(counts_host.data(), d_counts, sizeof(int) * N, cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_seed = -1;
        for (int seed = 0; seed < N; ++seed) {
            if (clustered_host[seed]) continue;
            const int cardinality = counts_host[seed];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        std::vector<int> best_cluster_members(max_cardinality);
        CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                              d_members + best_seed * N,
                              sizeof(int) * max_cardinality,
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_cluster_members;
        clusters.push_back(cluster);

        // Mark clustered on host and device.
        for (int idx : best_cluster_members) clustered_host[idx] = 1;
        CUDA_CHECK(cudaMemcpy(d_best_idx, best_cluster_members.data(), sizeof(int) * max_cardinality, cudaMemcpyHostToDevice));
        setClusteredKernel<<<(max_cardinality + 255) / 256, 256>>>(d_clustered, d_best_idx, max_cardinality);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered_host](int idx) { return clustered_host[idx] != 0; }),
            unclustered_indices.end());
    }

    CUDA_CHECK(cudaFree(d_active_count));
    CUDA_CHECK(cudaFree(d_best_val));
    CUDA_CHECK(cudaFree(d_best_idx));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_active));
    CUDA_CHECK(cudaFree(d_in_cluster));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_xs));

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
