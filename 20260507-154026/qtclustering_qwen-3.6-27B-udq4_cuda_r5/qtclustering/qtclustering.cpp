// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   1. Precompute the full N×N distance matrix on GPU (embarrassingly parallel).
//   2. Evaluate every unclustered seed in parallel (one CUDA thread per seed).
//      Each thread independently grows a candidate cluster using the distance
//      matrix, then stores its cardinality and member list.
//   3. The host scans cardinalities to pick the best cluster, then repeats.

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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Kernel 1 – build the full N×N Euclidean distance matrix on the GPU.
// Each thread computes one entry; the 2-D grid gives coalesced access.
__global__ void computeDistanceMatrixKernel(
    const double* points_x, const double* points_y,
    double* dist_matrix, const int N)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        const double dx = points_x[i] - points_x[j];
        const double dy = points_y[i] - points_y[j];
        dist_matrix[i * N + j] = sqrt(dx * dx + dy * dy);
    }
}

// Kernel 2 – evaluate every unclustered seed in parallel.
// Thread idx grows a candidate cluster starting from
// unclustered_indices[idx], using the precomputed distance matrix.
// Results are written to seed_cardinalities[idx] and
// seed_members[idx * N … idx * N + cardinality - 1].
__global__ void evaluateAllSeedsKernel(
    const double* dist_matrix, const int N,
    const char* clustered,
    const double threshold,
    const int* unclustered_indices, const int unclustered_count,
    int* seed_cardinalities,
    int* seed_members)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= unclustered_count) return;

    const int seed = unclustered_indices[idx];
    if (clustered[seed]) {
        seed_cardinalities[idx] = 0;
        return;
    }

    int* members = seed_members + idx * N;
    int count = 0;

    // Add seed point
    members[count++] = seed;

    // Iteratively add the closest point that keeps diameter < threshold.
    while (count < N) {
        int closest = -1;
        double min_max_dist = 1e300;

        for (int candidate = 0; candidate < N; ++candidate) {
            if (clustered[candidate]) continue;

            // Check whether candidate is already in this thread's cluster.
            bool in_cluster = false;
            for (int m = 0; m < count; ++m) {
                if (members[m] == candidate) {
                    in_cluster = true;
                    break;
                }
            }
            if (in_cluster) continue;

            // Max distance from candidate to any current member.
            double max_dist = 0.0;
            for (int m = 0; m < count; ++m) {
                const double d = dist_matrix[candidate * N + members[m]];
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold && max_dist < min_max_dist) {
                min_max_dist = max_dist;
                closest = candidate;
            }
        }

        if (closest < 0) break;
        members[count++] = closest;
    }

    seed_cardinalities[idx] = count;
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

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

// Calculate Euclidean distance between two points (host-only, used by validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Main QT clustering algorithm – parallelised with CUDA
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());

    // ---- allocate device memory ----
    double *d_points_x = nullptr, *d_points_y = nullptr, *d_dist_matrix = nullptr;
    char   *d_clustered = nullptr;
    int    *d_unclustered_indices = nullptr, *d_seed_cardinalities = nullptr,
           *d_seed_members = nullptr;

    cudaMalloc(&d_points_x,             N * sizeof(double));
    cudaMalloc(&d_points_y,             N * sizeof(double));
    cudaMalloc(&d_dist_matrix,          N * N * sizeof(double));
    cudaMalloc(&d_clustered,            N * sizeof(char));
    cudaMalloc(&d_unclustered_indices,  N * sizeof(int));
    cudaMalloc(&d_seed_cardinalities,   N * sizeof(int));
    cudaMalloc(&d_seed_members,         N * N * sizeof(int));

    // ---- copy points to device ----
    std::vector<double> h_px(N), h_py(N);
    for (int i = 0; i < N; ++i) {
        h_px[i] = points[i].x;
        h_py[i] = points[i].y;
    }
    cudaMemcpy(d_points_x, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points_y, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // ---- precompute distance matrix on GPU ----
    {
        const dim3 block(16, 16);
        const dim3 grid((N + 15) / 16, (N + 15) / 16);
        computeDistanceMatrixKernel<<<grid, block>>>(d_points_x, d_points_y, d_dist_matrix, N);
    }

    // ---- host data structures ----
    std::vector<char> clustered(N, 0);
    std::vector<int>  unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    std::vector<int> h_seed_cardinalities(N);
    std::vector<Cluster> clusters;

    // ---- main clustering loop ----
    while (!unclustered_indices.empty()) {
        const int uc = static_cast<int>(unclustered_indices.size());

        // Upload changed state to GPU
        cudaMemcpy(d_clustered, clustered.data(), N * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_unclustered_indices, unclustered_indices.data(),
                   uc * sizeof(int), cudaMemcpyHostToDevice);

        // Launch: one thread per unclustered seed
        const int threads = 256;
        const int blocks  = (uc + threads - 1) / threads;
        evaluateAllSeedsKernel<<<blocks, threads>>>(
            d_dist_matrix, N, d_clustered, threshold,
            d_unclustered_indices, uc,
            d_seed_cardinalities, d_seed_members);
        cudaDeviceSynchronize();

        // Download cardinalities
        cudaMemcpy(h_seed_cardinalities.data(), d_seed_cardinalities,
                   uc * sizeof(int), cudaMemcpyDeviceToHost);

        // Find best seed (largest candidate cluster)
        int max_card = -1, best_idx = -1;
        for (int i = 0; i < uc; ++i) {
            if (h_seed_cardinalities[i] > max_card) {
                max_card = h_seed_cardinalities[i];
                best_idx = i;
            }
        }

        if (best_idx >= 0 && max_card > 0) {
            // Download only the winning seed's member list
            std::vector<int> best_members(max_card);
            cudaMemcpy(best_members.data(),
                       d_seed_members + best_idx * N,
                       max_card * sizeof(int), cudaMemcpyDeviceToHost);

            Cluster c;
            c.seed_point = unclustered_indices[best_idx];
            c.members    = std::move(best_members);
            clusters.push_back(c);

            // Mark members as clustered
            for (const int m : c.members) clustered[m] = 1;

            // Shrink unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                               [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());
        } else {
            break;
        }
    }

    // ---- free device memory ----
    cudaFree(d_points_x);
    cudaFree(d_points_y);
    cudaFree(d_dist_matrix);
    cudaFree(d_clustered);
    cudaFree(d_unclustered_indices);
    cudaFree(d_seed_cardinalities);
    cudaFree(d_seed_members);

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
