// QT Clustering Benchmark - CUDA GPU Parallelized Version
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
#include <cfloat>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
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

// CUDA kernel: each block builds a candidate cluster for one seed point.
// Threads within a block cooperate via parallel reduction to find the
// closest valid point at each step.
__global__ void buildCandidateClustersKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ clustered,
    const int* __restrict__ seeds,
    int num_seeds,
    int N,
    double threshold,
    int* __restrict__ cardinalities,
    int* __restrict__ all_members,
    int* __restrict__ all_in_cluster)
{
    int bid = blockIdx.x;
    if (bid >= num_seeds) return;

    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int seed = seeds[bid];

    int* in_cluster = all_in_cluster + (long long)bid * N;
    int* members = all_members + (long long)bid * N;

    // Initialize in_cluster flags
    for (int i = tid; i < N; i += bsz) {
        in_cluster[i] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        in_cluster[seed] = 1;
        members[0] = seed;
    }
    __syncthreads();

    int member_count = 1;

    // Shared memory for parallel min-reduction
    extern __shared__ char smem[];
    double* s_min_dist = (double*)smem;
    int* s_min_idx = (int*)(s_min_dist + bsz);

    while (member_count < N) {
        double my_min_dist = DBL_MAX;
        int my_min_idx = -1;

        // Each thread evaluates a strided subset of candidate points
        for (int c = tid; c < N; c += bsz) {
            if (clustered[c] || in_cluster[c]) continue;

            // Compute max distance from candidate to all current cluster members
            double max_dist = 0.0;
            bool valid = true;
            for (int m = 0; m < member_count; m++) {
                int mi = members[m];
                double dx = px[c] - px[mi];
                double dy = py[c] - py[mi];
                double dist = sqrt(dx * dx + dy * dy);
                if (dist > max_dist) max_dist = dist;
                if (max_dist >= threshold) { valid = false; break; }
            }

            if (valid && max_dist < threshold && max_dist < my_min_dist) {
                my_min_dist = max_dist;
                my_min_idx = c;
            }
        }

        s_min_dist[tid] = my_min_dist;
        s_min_idx[tid] = my_min_idx;
        __syncthreads();

        // Parallel reduction: find the candidate with minimum max-distance,
        // breaking ties by lowest index to match sequential behavior
        for (int stride = bsz / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                int ri = s_min_idx[tid + stride];
                if (ri >= 0) {
                    int li = s_min_idx[tid];
                    if (li < 0 ||
                        s_min_dist[tid + stride] < s_min_dist[tid] ||
                        (s_min_dist[tid + stride] == s_min_dist[tid] && ri < li)) {
                        s_min_dist[tid] = s_min_dist[tid + stride];
                        s_min_idx[tid] = ri;
                    }
                }
            }
            __syncthreads();
        }

        int closest = s_min_idx[0];
        if (closest < 0) break;

        if (tid == 0) {
            in_cluster[closest] = 1;
            members[member_count] = closest;
        }
        __syncthreads();

        member_count++;
    }

    if (tid == 0) {
        cardinalities[bid] = member_count;
    }
}

// Main QT clustering algorithm (GPU-accelerated)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> clustered_host(N, 0);
    std::vector<Cluster> clusters;

    // Structure-of-Arrays layout for coalesced GPU memory access
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    // Allocate GPU memory for point coordinates (constant across rounds)
    double *d_px, *d_py;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    int *d_clustered;
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));

    // Batch processing to bound GPU memory usage
    const int MAX_BATCH = 2048;
    int *d_seeds, *d_cardinalities, *d_all_members, *d_all_in_cluster;
    CUDA_CHECK(cudaMalloc(&d_seeds, MAX_BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, MAX_BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_members, (long long)MAX_BATCH * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_in_cluster, (long long)MAX_BATCH * N * sizeof(int)));

    const int BLOCK_SIZE = 256;
    size_t smem_size = BLOCK_SIZE * (sizeof(double) + sizeof(int));

    std::vector<int> h_cardinalities(MAX_BATCH);
    std::vector<int> h_best_members(N);

    // Initialize unclustered indices
    std::vector<int> unclustered;
    unclustered.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered.push_back(i);
    }

    // Main clustering loop
    while (!unclustered.empty()) {
        int num_seeds = static_cast<int>(unclustered.size());

        CUDA_CHECK(cudaMemcpy(d_clustered, clustered_host.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice));

        int best_cardinality = -1;
        int best_seed = -1;

        // Process seeds in batches
        for (int batch_start = 0; batch_start < num_seeds; batch_start += MAX_BATCH) {
            int batch_sz = std::min(MAX_BATCH, num_seeds - batch_start);

            CUDA_CHECK(cudaMemcpy(d_seeds, unclustered.data() + batch_start,
                                  batch_sz * sizeof(int), cudaMemcpyHostToDevice));

            buildCandidateClustersKernel<<<batch_sz, BLOCK_SIZE, smem_size>>>(
                d_px, d_py, d_clustered, d_seeds, batch_sz, N, threshold,
                d_cardinalities, d_all_members, d_all_in_cluster);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities,
                                  batch_sz * sizeof(int), cudaMemcpyDeviceToHost));

            // Find best seed in this batch; copy its members before next batch overwrites
            for (int i = 0; i < batch_sz; i++) {
                if (h_cardinalities[i] > best_cardinality) {
                    best_cardinality = h_cardinalities[i];
                    best_seed = unclustered[batch_start + i];
                    CUDA_CHECK(cudaMemcpy(h_best_members.data(),
                                          d_all_members + (long long)i * N,
                                          h_cardinalities[i] * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                }
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && best_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.assign(h_best_members.begin(),
                                   h_best_members.begin() + best_cardinality);
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (int i = 0; i < best_cardinality; i++) {
                clustered_host[h_best_members[i]] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered.erase(
                std::remove_if(unclustered.begin(), unclustered.end(),
                              [&clustered_host](int idx) { return clustered_host[idx] != 0; }),
                unclustered.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_all_members));
    CUDA_CHECK(cudaFree(d_all_in_cluster));

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
