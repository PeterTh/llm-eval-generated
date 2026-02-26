// QT Clustering Benchmark - CUDA GPU Parallel Version
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

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t _err = (call);                                      \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(_err));                               \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static constexpr int CUDA_BLOCK = 256;

__device__ __forceinline__ void pick_better(double& best_val, int& best_idx,
                                            const double cand_val, const int cand_idx) {
    if (cand_idx < 0) return;
    if (best_idx < 0 || cand_val < best_val || (cand_val == best_val && cand_idx < best_idx)) {
        best_val = cand_val;
        best_idx = cand_idx;
    }
}

__global__ void seed_cardinality_kernel(const double* __restrict__ dist,
                                       const unsigned char* __restrict__ clustered,
                                       const int N,
                                       const double threshold,
                                       int* __restrict__ out_card) {
    const int seed = static_cast<int>(blockIdx.x);
    if (seed >= N) return;

    if (clustered[seed]) {
        if (threadIdx.x == 0) out_card[seed] = 0;
        return;
    }

    extern __shared__ double s_dmax[]; // N doubles
    const double INF = 1.0 / 0.0;

    for (int i = static_cast<int>(threadIdx.x); i < N; i += CUDA_BLOCK) {
        const size_t off = static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(seed);
        double v = dist[off];
        if (i == seed || clustered[i]) v = INF;
        s_dmax[i] = v;
    }
    __syncthreads();

    int card = 1;
    __shared__ double s_best_val[CUDA_BLOCK];
    __shared__ int s_best_idx[CUDA_BLOCK];

    for (int step = 1; step < N; ++step) {
        double best_val = INF;
        int best_idx = -1;

        for (int i = static_cast<int>(threadIdx.x); i < N; i += CUDA_BLOCK) {
            const double v = s_dmax[i];
            if (v < threshold) {
                if (best_idx < 0 || v < best_val || (v == best_val && i < best_idx)) {
                    best_val = v;
                    best_idx = i;
                }
            }
        }

        s_best_val[threadIdx.x] = best_val;
        s_best_idx[threadIdx.x] = best_idx;
        __syncthreads();

        for (int offset = CUDA_BLOCK / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < static_cast<unsigned>(offset)) {
                pick_better(s_best_val[threadIdx.x], s_best_idx[threadIdx.x],
                            s_best_val[threadIdx.x + offset], s_best_idx[threadIdx.x + offset]);
            }
            __syncthreads();
        }

        const int selected = s_best_idx[0];
        if (selected < 0) break;

        if (threadIdx.x == 0) {
            s_dmax[selected] = INF;
            ++card;
        }
        __syncthreads();

        for (int i = static_cast<int>(threadIdx.x); i < N; i += CUDA_BLOCK) {
            double cur = s_dmax[i];
            if (cur != INF) {
                const size_t off = static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(selected);
                const double d = dist[off];
                s_dmax[i] = (d > cur) ? d : cur;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) out_card[seed] = card;
}

__global__ void build_cluster_kernel(const double* __restrict__ dist,
                                    const unsigned char* __restrict__ clustered,
                                    const int N,
                                    const double threshold,
                                    const int seed,
                                    int* __restrict__ out_members,
                                    int* __restrict__ out_count) {
    if (clustered[seed]) {
        if (threadIdx.x == 0) *out_count = 0;
        return;
    }

    extern __shared__ double s_dmax[]; // N doubles
    const double INF = 1.0 / 0.0;

    for (int i = static_cast<int>(threadIdx.x); i < N; i += CUDA_BLOCK) {
        const size_t off = static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(seed);
        double v = dist[off];
        if (i == seed || clustered[i]) v = INF;
        s_dmax[i] = v;
    }
    __syncthreads();

    __shared__ double s_best_val[CUDA_BLOCK];
    __shared__ int s_best_idx[CUDA_BLOCK];
    __shared__ int count;

    if (threadIdx.x == 0) {
        out_members[0] = seed;
        count = 1;
    }
    __syncthreads();

    for (int step = 1; step < N; ++step) {
        double best_val = INF;
        int best_idx = -1;

        for (int i = static_cast<int>(threadIdx.x); i < N; i += CUDA_BLOCK) {
            const double v = s_dmax[i];
            if (v < threshold) {
                if (best_idx < 0 || v < best_val || (v == best_val && i < best_idx)) {
                    best_val = v;
                    best_idx = i;
                }
            }
        }

        s_best_val[threadIdx.x] = best_val;
        s_best_idx[threadIdx.x] = best_idx;
        __syncthreads();

        for (int offset = CUDA_BLOCK / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < static_cast<unsigned>(offset)) {
                pick_better(s_best_val[threadIdx.x], s_best_idx[threadIdx.x],
                            s_best_val[threadIdx.x + offset], s_best_idx[threadIdx.x + offset]);
            }
            __syncthreads();
        }

        const int selected = s_best_idx[0];
        if (selected < 0) break;

        if (threadIdx.x == 0) {
            out_members[count++] = selected;
            s_dmax[selected] = INF;
        }
        __syncthreads();

        for (int i = static_cast<int>(threadIdx.x); i < N; i += CUDA_BLOCK) {
            double cur = s_dmax[i];
            if (cur != INF) {
                const size_t off = static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(selected);
                const double d = dist[off];
                s_dmax[i] = (d > cur) ? d : cur;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) *out_count = count;
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
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                       threshold, N, 
                                                       &candidate_members);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
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
    
    return clusters;
}

// CUDA-parallel QT clustering: score all seeds in parallel on the GPU.
std::vector<Cluster> qtClusteringCUDA(const std::vector<Point>& points,
                                     const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    clusters.reserve(N);

    if (N <= 0) return clusters;

    // Precompute pairwise distances on the host (matches original CPU math exactly).
    const size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<double> dist(NN);
    for (int i = 0; i < N; ++i) {
        const size_t row = static_cast<size_t>(i) * static_cast<size_t>(N);
        for (int j = 0; j < N; ++j) {
            dist[row + static_cast<size_t>(j)] = distance(points[i], points[j]);
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    const size_t shmem_bytes = static_cast<size_t>(N) * sizeof(double);
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (shmem_bytes > static_cast<size_t>(max_optin)) {
        fprintf(stderr,
                "Error: N=%d requires %zu bytes dynamic shared memory per block, but device supports %d\n",
                N, shmem_bytes, max_optin);
        std::exit(1);
    }
    CUDA_CHECK(cudaFuncSetAttribute(seed_cardinality_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shmem_bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(build_cluster_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shmem_bytes)));

    double* d_dist = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;
    int* d_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_dist, NN * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), NN * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_card, static_cast<size_t>(N) * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    std::vector<unsigned char> clustered(static_cast<size_t>(N), 0);
    std::vector<int> cardinalities(static_cast<size_t>(N));
    std::vector<int> members(static_cast<size_t>(N));

    int remaining = N;
    while (remaining > 0) {
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                              static_cast<size_t>(N) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        seed_cardinality_kernel<<<static_cast<unsigned>(N), CUDA_BLOCK, shmem_bytes>>>(
            d_dist, d_clustered, N, threshold, d_card);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_card,
                              static_cast<size_t>(N) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int best_seed = -1;
        int best_card = -1;
        for (int i = 0; i < N; ++i) {
            if (clustered[static_cast<size_t>(i)]) continue;
            const int c = cardinalities[static_cast<size_t>(i)];
            if (c > best_card) {
                best_card = c;
                best_seed = i;
            }
        }

        if (best_seed < 0 || best_card <= 0) break;

        build_cluster_kernel<<<1, CUDA_BLOCK, shmem_bytes>>>(
            d_dist, d_clustered, N, threshold, best_seed, d_members, d_count);
        CUDA_CHECK(cudaGetLastError());

        int count = 0;
        CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(members.data(), d_members,
                              static_cast<size_t>(count) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(members.begin(), members.begin() + count);
        clusters.push_back(cluster);

        for (int k = 0; k < count; ++k) {
            const int idx = members[static_cast<size_t>(k)];
            if (!clustered[static_cast<size_t>(idx)]) {
                clustered[static_cast<size_t>(idx)] = 1;
                --remaining;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_dist));

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
    
    const std::vector<Cluster> clusters = qtClusteringCUDA(points, threshold);
    
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
