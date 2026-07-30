// QT Clustering Benchmark - CUDA-Accelerated Version
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
#include <cuda_runtime.h>

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

// ---------------------------------------------------------------------------
// Kernel: evaluate all unclustered seeds in parallel.
//
// One thread block per seed.  All threads in a block cooperate to build the
// candidate cluster greedily: each iteration computes the max distance for
// every unclustered point (parallel across block threads), performs a
// block-level argmin reduction, and the elected thread adds the best point.
//
// Outputs:
//   out_sizes[bid]     — cardinality of the cluster for seed seeds[bid]
//   out_members        — flat array [num_seeds][point_count] of member indices
//   per_block_state    — scratch [num_seeds][point_count] (in_cluster flags)
// ---------------------------------------------------------------------------
__global__ void evaluateAllSeedsKernel(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const int*    __restrict__ clustered,
    double threshold,
    int point_count,
    const int* __restrict__ seeds,
    int num_seeds,
    int* __restrict__ out_sizes,
    int* __restrict__ out_members,
    int* __restrict__ per_block_state)
{
    int bid = blockIdx.x;
    if (bid >= num_seeds) return;

    int seed   = seeds[bid];
    int tid    = threadIdx.x;
    int nlanes = blockDim.x;

    // Per-block arrays in global memory
    int* in_cluster = per_block_state + bid * point_count;
    int* members    = out_members    + bid * point_count;

    // Initialise in_cluster to all-false
    for (int i = tid; i < point_count; i += nlanes)
        in_cluster[i] = 0;
    __syncthreads();

    if (tid == 0) {
        in_cluster[seed] = 1;
        members[0] = seed;
    }
    __syncthreads();

    // Shared memory for loop state and reduction scratch
    __shared__ int    s_num_members;
    __shared__ int    s_done;
    __shared__ double s_dist[256];
    __shared__ int    s_cand[256];

    if (tid == 0) { s_num_members = 1; s_done = 0; }
    __syncthreads();

    while (!s_done) {
        int n = s_num_members;               // current cluster size

        // ---- parallel distance computation across all candidates ----
        double best_max  = 1e308;
        int    best_cand = -1;

        for (int c = tid; c < point_count; c += nlanes) {
            if (clustered[c] || in_cluster[c]) continue;

            double cx = points_x[c];
            double cy = points_y[c];
            double max_dist = 0.0;

            for (int i = 0; i < n; ++i) {
                int m = members[i];
                double dx = cx - points_x[m];
                double dy = cy - points_y[m];
                double dist = sqrt(dx * dx + dy * dy);
                if (dist > max_dist) max_dist = dist;
            }

            if (max_dist < threshold && max_dist < best_max) {
                best_max  = max_dist;
                best_cand = c;
            }
        }

        // ---- block-level argmin reduction ----
        s_dist[tid] = best_max;
        s_cand[tid] = best_cand;
        __syncthreads();

        for (int s = nlanes / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_dist[tid + s] < s_dist[tid]) {
                    s_dist[tid] = s_dist[tid + s];
                    s_cand[tid] = s_cand[tid + s];
                }
            }
            __syncthreads();
        }

        // ---- thread 0 commits the new member (or signals termination) ----
        if (tid == 0) {
            if (s_cand[0] >= 0) {
                members[n]          = s_cand[0];
                in_cluster[s_cand[0]] = 1;
                s_num_members       = n + 1;
            } else {
                s_done = 1;          // no admissible point left
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_sizes[bid] = s_num_members;
    }
}

// ---------------------------------------------------------------------------
// Per-benchmark GPU state  (points + working arrays live on device)
// ---------------------------------------------------------------------------
// Process at most 2048 seeds per kernel launch to keep memory O(N · B).
static const int MAX_BATCH = 2048;

struct GPUState {
    double *d_points_x, *d_points_y;
    int    *d_clustered;
    int    *d_seeds;            // GPU copy of the current seed-batch
    int    *d_out_sizes;        // sizes  [batch_capacity]
    int    *d_out_members;      // members[batch_capacity][N]
    int    *d_per_block_scratch;// scratch[batch_capacity][N] (in_cluster)
    int     N;
    int     batch_cap;          // = min(N, MAX_BATCH)

    GPUState(int N_) : N(N_), batch_cap(std::min(N_, MAX_BATCH)) {
        cudaMalloc(&d_points_x,          N * sizeof(double));
        cudaMalloc(&d_points_y,          N * sizeof(double));
        cudaMalloc(&d_clustered,         N * sizeof(int));
        cudaMalloc(&d_seeds,             batch_cap * sizeof(int));
        cudaMalloc(&d_out_sizes,         batch_cap * sizeof(int));
        cudaMalloc(&d_out_members,       batch_cap * N * sizeof(int));
        cudaMalloc(&d_per_block_scratch, batch_cap * N * sizeof(int));
    }

    ~GPUState() {
        cudaFree(d_points_x);
        cudaFree(d_points_y);
        cudaFree(d_clustered);
        cudaFree(d_seeds);
        cudaFree(d_out_sizes);
        cudaFree(d_out_members);
        cudaFree(d_per_block_scratch);
    }
};

// ---------------------------------------------------------------------------
// Main QT clustering — CUDA accelerated
//
// Each outer iteration evaluates *all* remaining unclustered points as
// candidate seeds in a single grid launch.  Seeds are processed in batches
// of at most MAX_BATCH.  The best cluster (largest cardinality) is then
// selected and its members are removed from further consideration.
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());

    // Host-side flags
    std::vector<int> clustered(N, 0);
    std::vector<int> unclustered;          // indices still available
    std::vector<Cluster> clusters;

    // Upload point data once
    GPUState gpu(N);
    {
        std::vector<double> hx(N), hy(N);
        for (int i = 0; i < N; ++i) {
            hx[i] = points[i].x;
            hy[i] = points[i].y;
        }
        cudaMemcpy(gpu.d_points_x,  hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu.d_points_y,  hy.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(gpu.d_clustered, clustered.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < N; ++i) unclustered.push_back(i);

    // Temporary host buffer for reading kernel output sizes
    std::vector<int> h_sizes(gpu.batch_cap);

    // ---- main loop -------------------------------------------------------
    while (!unclustered.empty()) {
        const int total = static_cast<int>(unclustered.size());
        int best_seed          = -1;
        int best_cardinality   = -1;
        // Storage for the best cluster's member list (saved before next batch
        // overwrites the shared d_out_members buffer).
        std::vector<int> best_members_saved;

        // Push updated clustered flags to device (once per outer iteration)
        cudaMemcpy(gpu.d_clustered, clustered.data(),
                   N * sizeof(int), cudaMemcpyHostToDevice);

        // Process seeds in batches.  Each batch's kernel writes results into
        // d_out_members[bid][N] with bid starting from 0, so a subsequent
        // batch overwrites earlier bids.  We therefore save a copy of the
        // best cluster's members immediately after processing its batch.
        for (int batch_start = 0; batch_start < total;
             batch_start += gpu.batch_cap) {
            const int batch_size = std::min(total - batch_start, gpu.batch_cap);

            cudaMemcpy(gpu.d_seeds, &unclustered[batch_start],
                       batch_size * sizeof(int), cudaMemcpyHostToDevice);

            constexpr int TPB = 256;
            evaluateAllSeedsKernel<<<batch_size, TPB>>>(
                gpu.d_points_x, gpu.d_points_y,
                gpu.d_clustered, threshold, N,
                gpu.d_seeds, batch_size,
                gpu.d_out_sizes,
                gpu.d_out_members,
                gpu.d_per_block_scratch);

            cudaMemcpy(h_sizes.data(), gpu.d_out_sizes,
                       batch_size * sizeof(int), cudaMemcpyDeviceToHost);

            for (int j = 0; j < batch_size; ++j) {
                if (h_sizes[j] <= best_cardinality) continue;

                best_cardinality = h_sizes[j];
                best_seed        = unclustered[batch_start + j];

                // Save the members BEFORE the next batch can overwrite
                // d_out_members[j * N].
                best_members_saved.resize(best_cardinality);
                cudaMemcpy(best_members_saved.data(),
                           gpu.d_out_members + j * N,
                           best_cardinality * sizeof(int),
                           cudaMemcpyDeviceToHost);
            }
        }

        // ---- commit best cluster -----------------------------------------
        if (best_seed < 0 || best_cardinality <= 0) break;

        Cluster cl;
        cl.seed_point = best_seed;
        cl.members.swap(best_members_saved);
        clusters.push_back(cl);

        // Mark members as clustered on the host
        for (int i = 0; i < best_cardinality; ++i)
            clustered[cl.members[i]] = 1;

        // Remove clustered points from the unclustered list
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
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
