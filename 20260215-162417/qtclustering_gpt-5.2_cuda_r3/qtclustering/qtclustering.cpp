// QT Clustering Benchmark - Simplified Sequential Version
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

#include <cstdint>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err__));                                          \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

namespace {
__device__ __forceinline__ double dist2_points(const double* __restrict__ x,
                                               const double* __restrict__ y,
                                               const int a,
                                               const int b) {
    const double dx = x[a] - x[b];
    const double dy = y[a] - y[b];
    return dx * dx + dy * dy;
}

__device__ __forceinline__ bool bit_test(const uint32_t* __restrict__ bits, const int idx) {
    return (bits[idx >> 5] >> (idx & 31)) & 1u;
}

__device__ __forceinline__ void bit_set(uint32_t* __restrict__ bits, const int idx) {
    bits[idx >> 5] |= (1u << (idx & 31));
}

__device__ __forceinline__ void warp_best_reduce(double& best_val, int& best_idx) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const double other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        const int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val < best_val || (other_val == best_val && other_idx < best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
}

__global__ void qt_cardinalities_kernel(const double* __restrict__ x,
                                       const double* __restrict__ y,
                                       const uint8_t* __restrict__ clustered,
                                       const int* __restrict__ seeds,
                                       const int num_seeds,
                                       const int N,
                                       const double threshold2,
                                       int* __restrict__ out_card) {
    const int seed_list_idx = static_cast<int>(blockIdx.x);
    if (seed_list_idx >= num_seeds) return;

    extern __shared__ unsigned char smem[];
    int* const members = reinterpret_cast<int*>(smem);
    const int words = (N + 31) >> 5;
    uint32_t* const in_cluster_bits = reinterpret_cast<uint32_t*>(members + N);

    __shared__ int member_count;
    __shared__ int block_best_idx;
    __shared__ double block_best_val;
    __shared__ double warp_vals[32];
    __shared__ int warp_idxs[32];

    const int tid = static_cast<int>(threadIdx.x);

    for (int w = tid; w < words; w += static_cast<int>(blockDim.x)) {
        in_cluster_bits[w] = 0u;
    }
    __syncthreads();

    if (tid == 0) {
        const int seed = seeds[seed_list_idx];
        members[0] = seed;
        member_count = 1;
        bit_set(in_cluster_bits, seed);
    }
    __syncthreads();

    while (member_count < N) {
        double best_val = 1.0e300;
        int best_idx = 0x7fffffff;

        for (int cand = tid; cand < N; cand += static_cast<int>(blockDim.x)) {
            if (clustered[cand] || bit_test(in_cluster_bits, cand)) continue;

            double max_d2 = 0.0;
            const int mc = member_count;
            for (int mi = 0; mi < mc; ++mi) {
                const int mem = members[mi];
                const double d2 = dist2_points(x, y, cand, mem);
                max_d2 = fmax(max_d2, d2);
                if (max_d2 >= threshold2 || max_d2 >= best_val) break;
            }

            if (max_d2 < threshold2 && (max_d2 < best_val || (max_d2 == best_val && cand < best_idx))) {
                best_val = max_d2;
                best_idx = cand;
            }
        }

        const int lane = tid & 31;
        const int warp = tid >> 5;
        warp_best_reduce(best_val, best_idx);
        if (lane == 0) {
            warp_vals[warp] = best_val;
            warp_idxs[warp] = best_idx;
        }
        __syncthreads();

        if (warp == 0) {
            const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;
            double v = (lane < num_warps) ? warp_vals[lane] : 1.0e300;
            int i = (lane < num_warps) ? warp_idxs[lane] : 0x7fffffff;
            warp_best_reduce(v, i);
            if (lane == 0) {
                block_best_val = v;
                block_best_idx = i;
            }
        }
        __syncthreads();

        if (tid == 0) {
            if (block_best_idx == 0x7fffffff || !(block_best_val < threshold2)) break;
            members[member_count] = block_best_idx;
            bit_set(in_cluster_bits, block_best_idx);
            ++member_count;
        }
        __syncthreads();

        if (block_best_idx == 0x7fffffff) break;
    }

    if (tid == 0) {
        out_card[seed_list_idx] = member_count;
    }
}

__global__ void qt_members_kernel(const double* __restrict__ x,
                                 const double* __restrict__ y,
                                 const uint8_t* __restrict__ clustered,
                                 const int seed,
                                 const int N,
                                 const double threshold2,
                                 int* __restrict__ out_members,
                                 int* __restrict__ out_count) {
    extern __shared__ unsigned char smem[];
    int* const members = reinterpret_cast<int*>(smem);
    const int words = (N + 31) >> 5;
    uint32_t* const in_cluster_bits = reinterpret_cast<uint32_t*>(members + N);

    __shared__ int member_count;
    __shared__ int block_best_idx;
    __shared__ double block_best_val;
    __shared__ double warp_vals[32];
    __shared__ int warp_idxs[32];

    const int tid = static_cast<int>(threadIdx.x);

    for (int w = tid; w < words; w += static_cast<int>(blockDim.x)) {
        in_cluster_bits[w] = 0u;
    }
    __syncthreads();

    if (tid == 0) {
        members[0] = seed;
        member_count = 1;
        bit_set(in_cluster_bits, seed);
    }
    __syncthreads();

    while (member_count < N) {
        double best_val = 1.0e300;
        int best_idx = 0x7fffffff;

        for (int cand = tid; cand < N; cand += static_cast<int>(blockDim.x)) {
            if (clustered[cand] || bit_test(in_cluster_bits, cand)) continue;

            double max_d2 = 0.0;
            const int mc = member_count;
            for (int mi = 0; mi < mc; ++mi) {
                const int mem = members[mi];
                const double d2 = dist2_points(x, y, cand, mem);
                max_d2 = fmax(max_d2, d2);
                if (max_d2 >= threshold2 || max_d2 >= best_val) break;
            }

            if (max_d2 < threshold2 && (max_d2 < best_val || (max_d2 == best_val && cand < best_idx))) {
                best_val = max_d2;
                best_idx = cand;
            }
        }

        const int lane = tid & 31;
        const int warp = tid >> 5;
        warp_best_reduce(best_val, best_idx);
        if (lane == 0) {
            warp_vals[warp] = best_val;
            warp_idxs[warp] = best_idx;
        }
        __syncthreads();

        if (warp == 0) {
            const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;
            double v = (lane < num_warps) ? warp_vals[lane] : 1.0e300;
            int i = (lane < num_warps) ? warp_idxs[lane] : 0x7fffffff;
            warp_best_reduce(v, i);
            if (lane == 0) {
                block_best_val = v;
                block_best_idx = i;
            }
        }
        __syncthreads();

        if (tid == 0) {
            if (block_best_idx == 0x7fffffff || !(block_best_val < threshold2)) break;
            members[member_count] = block_best_idx;
            bit_set(in_cluster_bits, block_best_idx);
            ++member_count;
        }
        __syncthreads();

        if (block_best_idx == 0x7fffffff) break;
    }

    if (tid == 0) {
        *out_count = member_count;
        for (int i = 0; i < member_count; ++i) {
            out_members[i] = members[i];
        }
    }
}

} // namespace

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
    if (N <= 0) return {};

    CUDA_CHECK(cudaSetDevice(0));

    std::vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
    }

    double* dx = nullptr;
    double* dy = nullptr;
    uint8_t* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;
    int* d_count = nullptr;

    CUDA_CHECK(cudaMalloc(&dx, static_cast<size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dy, static_cast<size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(N) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_seeds, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dx, hx.data(), static_cast<size_t>(N) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), static_cast<size_t>(N) * sizeof(double), cudaMemcpyHostToDevice));

    const double threshold2 = threshold * threshold;

    const int block_threads = 256;
    const int words = (N + 31) >> 5;
    const size_t shared_bytes = static_cast<size_t>(N) * sizeof(int) +
                                static_cast<size_t>(words) * sizeof(uint32_t);

    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (shared_bytes > static_cast<size_t>(max_optin)) {
        fprintf(stderr,
                "Error: N=%d requires %zu bytes dynamic shared memory per block, but device supports %d\n",
                N, shared_bytes, max_optin);
        std::exit(1);
    }
    CUDA_CHECK(cudaFuncSetAttribute(qt_cardinalities_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shared_bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(qt_members_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shared_bytes)));

    std::vector<uint8_t> clustered(static_cast<size_t>(N), 0u);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    std::vector<Cluster> clusters;
    clusters.reserve(static_cast<size_t>(N));

    std::vector<int> h_card;

    while (!unclustered_indices.empty()) {
        const int S = static_cast<int>(unclustered_indices.size());
        h_card.resize(static_cast<size_t>(S));

        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(N) * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(), static_cast<size_t>(S) * sizeof(int),
                              cudaMemcpyHostToDevice));

        qt_cardinalities_kernel<<<S, block_threads, shared_bytes>>>(dx, dy, d_clustered, d_seeds, S, N,
                                                                   threshold2, d_card);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_card.data(), d_card, static_cast<size_t>(S) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_seed = -1;

        // Stable selection: first seed in unclustered_indices with maximum cardinality.
        for (int i = 0; i < S; ++i) {
            if (h_card[i] > max_cardinality) {
                max_cardinality = h_card[i];
                best_seed = unclustered_indices[i];
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        qt_members_kernel<<<1, block_threads, shared_bytes>>>(dx, dy, d_clustered, best_seed, N, threshold2,
                                                             d_members, d_count);
        CUDA_CHECK(cudaGetLastError());

        int member_count = 0;
        CUDA_CHECK(cudaMemcpy(&member_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (member_count <= 0) break;

        std::vector<int> best_cluster_members(static_cast<size_t>(member_count));
        CUDA_CHECK(cudaMemcpy(best_cluster_members.data(), d_members,
                              static_cast<size_t>(member_count) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_cluster_members);
        clusters.push_back(std::move(cluster));

        // Mark all members as clustered
        for (int m : clusters.back().members) {
            clustered[static_cast<size_t>(m)] = 1u;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[static_cast<size_t>(idx)] != 0u; }),
            unclustered_indices.end());
    }

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_count));

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
