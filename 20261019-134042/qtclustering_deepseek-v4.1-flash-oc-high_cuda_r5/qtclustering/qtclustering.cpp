// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The original algorithm is sequential over the seed points.  This version
// evaluates every remaining seed concurrently: one CUDA block grows one
// candidate cluster.  Within a block the candidates are examined in
// parallel and the per-seed greedy growth is kept identical to the
// sequential reference (including tie breaking).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
static_assert(sizeof(Point) == sizeof(double2), "Point layout must match double2");

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err__));                  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

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

// Device properties are cached so that the timed clustering region does not
// include CUDA context creation / device interrogation.
static cudaDeviceProp g_device_prop;
static bool g_device_prop_ready = false;

__global__ void warmupKernel() {}

void warmupCuda() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (!g_device_prop_ready) {
        CUDA_CHECK(cudaGetDeviceProperties(&g_device_prop, device));
        g_device_prop_ready = true;
    }
    CUDA_CHECK(cudaFree(nullptr));
    warmupKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Device-side distance.  The exact expression (including the fused
// multiply-add) mirrors the code generated for the sequential reference.
__device__ __forceinline__ double deviceDistance(const double2& p1, const double2& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    const double d2 = __fma_rn(dx, dx, __dmul_rn(dy, dy));
    return __dsqrt_rn(d2);
}

// Grow one candidate cluster.  cur holds, for every point, the maximum
// distance to the members added so far (INFINITY for points already
// clustered or already part of this candidate).  A point can only ever
// become less eligible as the cluster grows, so points at or above the
// threshold can be dropped from future updates.
//
// Only the most recently added member is needed while growing; the full
// member list (when requested through member_out) is written for the
// reconstruction pass.
//
// This produces exactly the same members, in the same order, as the
// reference findClosestPoint/generateCandidateCluster pair.
__device__ __forceinline__ int growCandidate(const double2* __restrict__ points,
                                             const unsigned char* __restrict__ clustered,
                                             const int N,
                                             const double threshold,
                                             const int seed,
                                             double* __restrict__ cur,
                                             int* __restrict__ member_out) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ int s_count;
    __shared__ int s_new_member;
    __shared__ double s_best_dist[32];
    __shared__ int s_best_idx[32];

    const double2 seed_point = points[seed];

    for (int c = tid; c < N; c += nthreads) {
        if (clustered[c] || c == seed) {
            cur[c] = INFINITY;
        } else {
            cur[c] = deviceDistance(points[c], seed_point);
        }
    }

    if (tid == 0) {
        if (member_out) {
            member_out[0] = seed;
        }
        s_count = 1;
    }
    __syncthreads();

    int count = 1;

    while (count < N) {
        double best_dist = INFINITY;
        int best_idx = INT32_MAX;

        for (int c = tid; c < N; c += nthreads) {
            const double d = cur[c];
            if (d < threshold && (d < best_dist || (d == best_dist && c < best_idx))) {
                best_dist = d;
                best_idx = c;
            }
        }

        // Warp reduction.
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const int nwarps = (nthreads + 31) >> 5;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other_dist = __shfl_down_sync(0xffffffffu, best_dist, offset);
            const int other_idx = __shfl_down_sync(0xffffffffu, best_idx, offset);
            if (other_dist < best_dist ||
                (other_dist == best_dist && other_idx < best_idx)) {
                best_dist = other_dist;
                best_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_best_dist[warp] = best_dist;
            s_best_idx[warp] = best_idx;
        }
        __syncthreads();

        if (warp == 0) {
            best_dist = (lane < nwarps) ? s_best_dist[lane] : INFINITY;
            best_idx = (lane < nwarps) ? s_best_idx[lane] : INT32_MAX;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const double other_dist = __shfl_down_sync(0xffffffffu, best_dist, offset);
                const int other_idx = __shfl_down_sync(0xffffffffu, best_idx, offset);
                if (other_dist < best_dist ||
                    (other_dist == best_dist && other_idx < best_idx)) {
                    best_dist = other_dist;
                    best_idx = other_idx;
                }
            }

            if (lane == 0) {
                if (best_idx != INT32_MAX) {
                    if (member_out) {
                        member_out[count] = best_idx;
                    }
                    s_new_member = best_idx;
                    cur[best_idx] = INFINITY;
                    s_count = count + 1;
                } else {
                    s_count = count;
                }
            }
        }
        __syncthreads();

        const int new_count = s_count;
        if (new_count == count) {
            break;
        }

        const int new_member = s_new_member;
        const double2 new_point = points[new_member];

        for (int c = tid; c < N; c += nthreads) {
            const double d = cur[c];
            if (d < threshold) {
                const double nd = deviceDistance(points[c], new_point);
                if (nd > d) {
                    cur[c] = nd;
                }
            }
        }

        count = new_count;
        __syncthreads();
    }

    return count;
}

// One block per remaining seed.  The block grows a candidate cluster and
// publishes its cardinality through an atomic maximum key that encodes
// (cardinality, -seed), giving the same tie break as the sequential loop.
template <bool UseGlobalWorkspace>
__global__ void growClustersKernel(const double2* __restrict__ points,
                                   const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ seeds,
                                   const int seed_offset,
                                   const int num_seeds,
                                   const int N,
                                   const double threshold,
                                   double* __restrict__ global_cur,
                                   unsigned long long* __restrict__ best_key) {
    const int block = blockIdx.x;
    if (block >= num_seeds) {
        return;
    }

    const int seed = seeds[seed_offset + block];

    extern __shared__ unsigned char shared_storage[];
    double* cur;

    if constexpr (UseGlobalWorkspace) {
        cur = global_cur + static_cast<size_t>(block) * N;
    } else {
        cur = reinterpret_cast<double*>(shared_storage);
    }

    const int count = growCandidate(points, clustered, N, threshold, seed, cur, nullptr);

    if (threadIdx.x == 0) {
        const unsigned long long key =
            (static_cast<unsigned long long>(static_cast<unsigned int>(count)) << 32) |
            static_cast<unsigned int>(0xFFFFFFFFu - static_cast<unsigned int>(seed));
        atomicMax(best_key, key);
    }
}

// Decode the winning (cardinality, seed) key on the device.
__global__ void decodeBestKeyKernel(const unsigned long long* __restrict__ best_key,
                                    int* __restrict__ best_seed) {
    const unsigned long long key = *best_key;
    if (key == 0) {
        *best_seed = -1;
        return;
    }
    const unsigned int low = static_cast<unsigned int>(key & 0xFFFFFFFFull);
    *best_seed = static_cast<int>(0xFFFFFFFFu - low);
}

// Regenerate the members of the winning candidate cluster.  Only one seed
// is processed, so this is negligible compared to the main kernel.
template <bool UseGlobalWorkspace>
__global__ void reconstructClusterKernel(const double2* __restrict__ points,
                                         const unsigned char* __restrict__ clustered,
                                         const int N,
                                         const double threshold,
                                         const int* __restrict__ best_seed_ptr,
                                         double* __restrict__ global_cur,
                                         int* __restrict__ out_members,
                                         int* __restrict__ out_count) {
    const int seed = *best_seed_ptr;
    if (seed < 0) {
        if (threadIdx.x == 0) {
            *out_count = 0;
        }
        return;
    }

    extern __shared__ unsigned char shared_storage[];
    double* cur;

    if constexpr (UseGlobalWorkspace) {
        cur = global_cur;
    } else {
        cur = reinterpret_cast<double*>(shared_storage);
    }

    const int count = growCandidate(points, clustered, N, threshold, seed, cur, out_members);

    if (threadIdx.x == 0) {
        *out_count = count;
    }
}

// Main QT clustering algorithm (CUDA).
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    if (g_device_prop_ready) {
        prop = g_device_prop;
    } else {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    }

    const size_t shared_per_block = prop.sharedMemPerBlockOptin;
    const size_t bytes_per_point = sizeof(double);
    const size_t shared_reserve = 1024;
    const int max_shared_points = shared_per_block > shared_reserve
        ? static_cast<int>((shared_per_block - shared_reserve) / bytes_per_point)
        : 0;
    const bool use_shared = N <= max_shared_points;
    const size_t shared_bytes = use_shared
        ? static_cast<size_t>(N) * bytes_per_point
        : 0;

    const int threads = (N <= 1024) ? 128 : 256;

    if (use_shared) {
        CUDA_CHECK(cudaFuncSetAttribute(growClustersKernel<false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(reconstructClusterKernel<false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared_bytes)));
    }

    // For point sets too large for shared memory, process seeds in batches
    // using a per-block global workspace.
    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t workspace_budget = static_cast<size_t>(2) << 30;
    if (workspace_budget > free_mem / 2) {
        workspace_budget = free_mem / 2;
    }
    int max_batch = 1;
    if (!use_shared) {
        const size_t per_seed = static_cast<size_t>(N) * bytes_per_point;
        max_batch = static_cast<int>(std::max<size_t>(1, workspace_budget / std::max<size_t>(1, per_seed)));
    }

    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    unsigned long long* d_best_key = nullptr;
    int* d_best_seed = nullptr;
    int* d_best_out = nullptr;
    double* d_global_cur = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(double2) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_best_key, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_best_seed, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_out, sizeof(int) * static_cast<size_t>(N + 1)));

    if (!use_shared) {
        CUDA_CHECK(cudaMalloc(&d_global_cur,
                              sizeof(double) * static_cast<size_t>(max_batch) * static_cast<size_t>(N)));
    }

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(double2) * static_cast<size_t>(N),
                          cudaMemcpyHostToDevice));

    std::vector<int> best_out(static_cast<size_t>(N) + 1);

    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                              sizeof(int) * static_cast<size_t>(num_seeds),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                              sizeof(unsigned char) * static_cast<size_t>(N),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_best_key, 0, sizeof(unsigned long long)));

        if (use_shared) {
            growClustersKernel<false><<<num_seeds, threads, shared_bytes>>>(
                d_points, d_clustered, d_seeds, 0, num_seeds, N, threshold,
                nullptr, d_best_key);
        } else {
            for (int offset = 0; offset < num_seeds; offset += max_batch) {
                const int batch = std::min(max_batch, num_seeds - offset);
                growClustersKernel<true><<<batch, threads, 0>>>(
                    d_points, d_clustered, d_seeds, offset, batch, N, threshold,
                    d_global_cur, d_best_key);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        decodeBestKeyKernel<<<1, 1>>>(d_best_key, d_best_seed);
        CUDA_CHECK(cudaGetLastError());

        if (use_shared) {
            reconstructClusterKernel<false><<<1, threads, shared_bytes>>>(
                d_points, d_clustered, N, threshold, d_best_seed,
                nullptr, d_best_out, d_best_out + N);
        } else {
            reconstructClusterKernel<true><<<1, threads, 0>>>(
                d_points, d_clustered, N, threshold, d_best_seed,
                d_global_cur, d_best_out, d_best_out + N);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(best_out.data(), d_best_out,
                              sizeof(int) * static_cast<size_t>(N + 1),
                              cudaMemcpyDeviceToHost));

        const int count = best_out[N];
        if (count <= 0) {
            break;
        }

        Cluster cluster;
        cluster.seed_point = best_out[0];
        cluster.members.assign(best_out.begin(), best_out.begin() + count);

        for (int i = 0; i < count; ++i) {
            clustered[cluster.members[i]] = 1;
        }
        clusters.push_back(std::move(cluster));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    CUDA_CHECK(cudaFree(d_global_cur));
    CUDA_CHECK(cudaFree(d_best_out));
    CUDA_CHECK(cudaFree(d_best_seed));
    CUDA_CHECK(cudaFree(d_best_key));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));

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

    // Initialize the CUDA context / device once, outside the timed region.
    warmupCuda();
    
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
