// QT Clustering Benchmark - CUDA Parallel Version
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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Maximum cluster members per thread (stored in local memory)
#define MAX_MEMBERS_PER_THREAD 2048

// CUDA kernel: evaluate all seed points in parallel
// One thread per seed. Each thread runs the full greedy cluster generation.
__global__ void evaluateAllSeedsKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ clustered,
    const int* __restrict__ seed_indices,
    const int num_seeds,
    const int point_count,
    const double threshold,
    int* __restrict__ cardinalities
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_seeds) return;

    const int seed = seed_indices[tid];

    // Thread-local storage for cluster members
    int members[MAX_MEMBERS_PER_THREAD];
    int member_count = 0;
    members[member_count++] = seed;

    // Greedy iteration: keep adding closest valid point
    while (member_count < point_count && member_count < MAX_MEMBERS_PER_THREAD) {
        int best_candidate = -1;
        double best_min_diameter = 1e30;

        // Try each point as candidate
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate]) continue;

            // Check if candidate is already in this cluster
            bool already_in = false;
            for (int m = 0; m < member_count; ++m) {
                if (members[m] == candidate) {
                    already_in = true;
                    break;
                }
            }
            if (already_in) continue;

            // Calculate max distance from candidate to all current members
            double max_dist = 0.0;
            const double cx = px[candidate];
            const double cy = py[candidate];
            for (int m = 0; m < member_count; ++m) {
                const int mi = members[m];
                const double dx = cx - px[mi];
                const double dy = cy - py[mi];
                const double dist = sqrt(dx * dx + dy * dy);
                max_dist = fmax(max_dist, dist);
            }

            if (max_dist < threshold && max_dist < best_min_diameter) {
                best_min_diameter = max_dist;
                best_candidate = candidate;
            }
        }

        if (best_candidate < 0) break;
        members[member_count++] = best_candidate;
    }

    cardinalities[tid] = member_count;
}

// Kernel to reconstruct a specific cluster (single block, cooperative)
__global__ void reconstructClusterKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ clustered,
    const int seed_point,
    const int point_count,
    const double threshold,
    int* __restrict__ out_members,
    int* __restrict__ out_count
) {
    __shared__ int members[MAX_MEMBERS_PER_THREAD];
    __shared__ int member_count_shared;

    const int tid = threadIdx.x;

    if (tid == 0) {
        members[0] = seed_point;
        member_count_shared = 1;
    }
    __syncthreads();

    int member_count = 1;

    while (member_count < point_count && member_count < MAX_MEMBERS_PER_THREAD) {
        // Each thread evaluates a subset of candidates
        int local_best_cand = -1;
        double local_best_dist = 1e30;

        for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
            if (clustered[candidate]) continue;

            // Check if in cluster
            bool already_in = false;
            for (int m = 0; m < member_count; ++m) {
                if (members[m] == candidate) {
                    already_in = true;
                    break;
                }
            }
            if (already_in) continue;

            // Max distance to all current members
            double max_dist = 0.0;
            const double cx = px[candidate];
            const double cy = py[candidate];
            for (int m = 0; m < member_count; ++m) {
                const int mi = members[m];
                const double dx = cx - px[mi];
                const double dy = cy - py[mi];
                const double dist = sqrt(dx * dx + dy * dy);
                max_dist = fmax(max_dist, dist);
            }

            if (max_dist < threshold && max_dist < local_best_dist) {
                local_best_dist = max_dist;
                local_best_cand = candidate;
            }
        }

        // Reduction using shared memory
        __shared__ int best_cand_shared[256];
        __shared__ double best_dist_shared[256];
        best_cand_shared[tid] = local_best_cand;
        best_dist_shared[tid] = local_best_dist;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (best_dist_shared[tid + stride] < best_dist_shared[tid]) {
                    best_dist_shared[tid] = best_dist_shared[tid + stride];
                    best_cand_shared[tid] = best_cand_shared[tid + stride];
                }
            }
            __syncthreads();
        }

        const int best_candidate = best_cand_shared[0];
        if (best_candidate < 0) break;

        if (tid == 0) {
            members[member_count] = best_candidate;
            member_count++;
            member_count_shared = member_count;
        }
        __syncthreads();
        member_count = member_count_shared;
    }

    // Copy to output
    for (int i = tid; i < member_count; i += blockDim.x) {
        out_members[i] = members[i];
    }
    if (tid == 0) {
        *out_count = member_count;
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

// Main QT clustering algorithm with CUDA parallelization
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

    // Allocate device memory
    double *d_px, *d_py;
    int *d_clustered, *d_seed_indices, *d_cardinalities;
    int *d_out_members, *d_out_count;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seed_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_members, MAX_MEMBERS_PER_THREAD * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));

    // Prepare host arrays for point coordinates
    std::vector<double> h_px(N), h_py(N);
    for (int i = 0; i < N; ++i) {
        h_px[i] = points[i].x;
        h_py[i] = points[i].y;
    }

    // Copy point coordinates to device
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Prepare clustered array
        std::vector<int> h_clustered(N, 0);
        for (int i = 0; i < N; ++i) {
            h_clustered[i] = clustered[i] ? 1 : 0;
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        // Prepare seed indices
        const int num_seeds = static_cast<int>(unclustered_indices.size());
        std::vector<int> h_seed_indices(num_seeds);
        for (int i = 0; i < num_seeds; ++i) {
            h_seed_indices[i] = unclustered_indices[i];
        }
        CUDA_CHECK(cudaMemcpy(d_seed_indices, h_seed_indices.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel: one thread per seed
        const int threads_per_block = 256;
        const int num_blocks = (num_seeds + threads_per_block - 1) / threads_per_block;

        evaluateAllSeedsKernel<<<num_blocks, threads_per_block>>>(
            d_px, d_py, d_clustered, d_seed_indices, num_seeds, N, threshold, d_cardinalities
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy cardinalities back
        std::vector<int> h_cardinalities(num_seeds);
        CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities, num_seeds * sizeof(int), cudaMemcpyDeviceToHost));

        // Find best seed
        int max_cardinality = -1;
        int best_seed_idx = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (h_cardinalities[i] > max_cardinality) {
                max_cardinality = h_cardinalities[i];
                best_seed_idx = i;
            }
        }

        if (best_seed_idx < 0 || max_cardinality <= 0) break;

        const int best_seed = unclustered_indices[best_seed_idx];

        // Reconstruct the best cluster using GPU
        reconstructClusterKernel<<<1, 256>>>(
            d_px, d_py, d_clustered, best_seed, N, threshold, d_out_members, d_out_count
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy cluster members back
        int h_out_count;
        CUDA_CHECK(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> h_out_members(h_out_count);
        CUDA_CHECK(cudaMemcpy(h_out_members.data(), d_out_members, h_out_count * sizeof(int), cudaMemcpyDeviceToHost));

        // Add cluster
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = h_out_members;
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (int m : h_out_members) {
            clustered[m] = true;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seed_indices));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_out_members));
    CUDA_CHECK(cudaFree(d_out_count));

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
