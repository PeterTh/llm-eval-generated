// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (CUDA):
//  * Every unclustered point is tried as a seed. Candidate clusters for
//    different seeds are independent, so seeds are distributed over
//    persistent thread blocks (dynamic work distribution via an atomic
//    counter, because candidate cluster sizes vary a lot).
//  * Inside a block the threads cooperatively grow one candidate cluster.
//    Each thread owns a strided subset of candidate points and keeps the
//    running "max distance to any current member" for them. Adding a member
//    therefore costs O(N) distance evaluations per step instead of
//    O(N * |members|), and the next member is picked with a block-wide
//    argmin reduction that breaks ties on the lowest index, exactly like
//    the sequential scan does.
//  * The host picks the seed with the largest cardinality (lowest index on
//    ties, as in the sequential version), a single block regenerates that
//    one cluster to obtain its member list, and the loop repeats until all
//    points are clustered.

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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
// Device code
// ---------------------------------------------------------------------------

static const int BLOCK_THREADS = 512;
static const int WARPS_PER_BLOCK = BLOCK_THREADS / 32;

__device__ __forceinline__ double devDistance(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Block-wide argmin of (value, index) with the lowest index winning ties.
// Returns the winning index to every thread (-1 if no thread had a candidate).
// Must be called by all threads of the block.
__device__ __forceinline__ int blockArgMin(double val, int idx,
                                           double* s_val, int* s_idx) {
    // Warp level
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, val, off);
        const int oi = __shfl_down_sync(0xffffffffu, idx, off);
        if (oi >= 0 && (idx < 0 || ov < val || (ov == val && oi < idx))) {
            val = ov;
            idx = oi;
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        s_val[warp] = val;
        s_idx[warp] = idx;
    }
    __syncthreads();
    // Final reduction by the first warp
    if (warp == 0) {
        val = (lane < WARPS_PER_BLOCK) ? s_val[lane] : 0.0;
        idx = (lane < WARPS_PER_BLOCK) ? s_idx[lane] : -1;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, val, off);
            const int oi = __shfl_down_sync(0xffffffffu, idx, off);
            if (oi >= 0 && (idx < 0 || ov < val || (ov == val && oi < idx))) {
                val = ov;
                idx = oi;
            }
        }
        if (lane == 0) s_idx[0] = idx;
    }
    __syncthreads();
    const int result = s_idx[0];
    __syncthreads();  // protect s_idx/s_val before the next reduction reuses them
    return result;
}

// Grow one candidate cluster from seed_point with the whole block.
// maxdist[N] is scratch (shared or global) holding, for every candidate, the
// largest distance to any current member (INF = not eligible any more).
// If members != nullptr the member indices are written there in insertion
// order. Returns the cluster cardinality (identical for all threads).
__device__ int growCluster(const int seed_point,
                           const double2* __restrict__ points,
                           const unsigned char* __restrict__ clustered,
                           const double threshold,
                           const int N,
                           double* maxdist,
                           int* members,
                           double* s_val, int* s_idx) {
    const double INF = 1.0 / 0.0;

    // Initialise: unclustered points other than the seed have max distance 0
    for (int c = threadIdx.x; c < N; c += BLOCK_THREADS) {
        maxdist[c] = (clustered[c] || c == seed_point) ? INF : 0.0;
    }
    if (threadIdx.x == 0 && members) members[0] = seed_point;
    __syncthreads();

    int size = 1;
    int newest = seed_point;

    while (size < N) {
        const double2 pm = points[newest];

        // Update running max distance with the newest member and find the
        // local best (smallest max distance, lowest index on ties).
        double best_val = 0.0;
        int best_idx = -1;
        for (int c = threadIdx.x; c < N; c += BLOCK_THREADS) {
            double md = maxdist[c];
            if (md == INF) continue;
            const double d = devDistance(points[c], pm);
            md = fmax(md, d);
            if (md < threshold) {
                maxdist[c] = md;
                // Candidates are visited in increasing index order per
                // thread, so a strict '<' keeps the lowest index on ties.
                if (best_idx < 0 || md < best_val) {
                    best_val = md;
                    best_idx = c;
                }
            } else {
                // Max distance only grows: this point can never qualify
                maxdist[c] = INF;
            }
        }

        const int closest = blockArgMin(best_val, best_idx, s_val, s_idx);
        if (closest < 0) break;  // No more points can be added

        if (threadIdx.x == 0) {
            maxdist[closest] = INF;
            if (members) members[size] = closest;
        }
        newest = closest;
        ++size;
        __syncthreads();
    }
    return size;
}

// Kernel 1: compute the cardinality of the candidate cluster of every seed.
// Persistent blocks pull seeds from a global work counter.
__global__ void __launch_bounds__(BLOCK_THREADS)
candidateCardinalityKernel(const double2* __restrict__ points,
                           const unsigned char* __restrict__ clustered,
                           const int* __restrict__ seeds,
                           const int num_seeds,
                           const double threshold,
                           const int N,
                           double* __restrict__ global_scratch,  // may be nullptr
                           int* __restrict__ work_counter,
                           int* __restrict__ cardinality) {
    extern __shared__ double smem[];
    __shared__ double s_val[WARPS_PER_BLOCK];
    __shared__ int s_idx[WARPS_PER_BLOCK];
    __shared__ int s_work;

    double* maxdist = global_scratch
                          ? global_scratch + static_cast<size_t>(blockIdx.x) * N
                          : smem;

    for (;;) {
        if (threadIdx.x == 0) s_work = atomicAdd(work_counter, 1);
        __syncthreads();
        const int w = s_work;
        __syncthreads();
        if (w >= num_seeds) break;

        const int seed = seeds[w];
        const int size = growCluster(seed, points, clustered, threshold, N,
                                     maxdist, nullptr, s_val, s_idx);
        if (threadIdx.x == 0) cardinality[w] = size;
        __syncthreads();
    }
}

// Kernel 2: regenerate the winning cluster (single block) and emit members.
__global__ void __launch_bounds__(BLOCK_THREADS)
extractClusterKernel(const double2* __restrict__ points,
                     const unsigned char* __restrict__ clustered,
                     const int seed,
                     const double threshold,
                     const int N,
                     double* __restrict__ global_scratch,
                     int* __restrict__ members,
                     int* __restrict__ size_out) {
    extern __shared__ double smem[];
    __shared__ double s_val[WARPS_PER_BLOCK];
    __shared__ int s_idx[WARPS_PER_BLOCK];

    double* maxdist = global_scratch ? global_scratch : smem;
    const int size = growCluster(seed, points, clustered, threshold, N,
                                 maxdist, members, s_val, s_idx);
    if (threadIdx.x == 0) *size_out = size;
}

// Mark members of the newly accepted cluster as clustered.
__global__ void markClusteredKernel(const int* __restrict__ members,
                                    const int count,
                                    unsigned char* __restrict__ clustered) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Main QT clustering algorithm (CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Decide whether the per-block scratch (N doubles) fits in shared memory.
    const size_t scratch_bytes = static_cast<size_t>(N) * sizeof(double);
    const size_t static_smem = sizeof(double) * WARPS_PER_BLOCK +
                               sizeof(int) * WARPS_PER_BLOCK + sizeof(int) + 64;
    const size_t max_dyn_smem = static_cast<size_t>(prop.sharedMemPerBlockOptin) > static_smem
                                    ? prop.sharedMemPerBlockOptin - static_smem : 0;
    const bool use_smem = scratch_bytes <= max_dyn_smem;
    const size_t dyn_smem = use_smem ? scratch_bytes : 0;

    if (use_smem) {
        CUDA_CHECK(cudaFuncSetAttribute(candidateCardinalityKernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(dyn_smem)));
        CUDA_CHECK(cudaFuncSetAttribute(extractClusterKernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(dyn_smem)));
    }

    // Grid size: enough persistent blocks to fill the GPU.
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, candidateCardinalityKernel, BLOCK_THREADS, dyn_smem));
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    const int max_blocks = blocks_per_sm * prop.multiProcessorCount;

    // Device buffers
    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cardinality = nullptr;
    int* d_members = nullptr;
    int* d_size = nullptr;
    int* d_counter = nullptr;
    double* d_scratch = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_cardinality, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    if (!use_smem) {
        CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes * max_blocks));
    }

    {
        std::vector<double2> h_points(N);
        for (int i = 0; i < N; ++i) h_points[i] = make_double2(points[i].x, points[i].y);
        CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), sizeof(double2) * N,
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<int> cardinality(N);
    std::vector<int> members(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpyAsync(d_seeds, unclustered_indices.data(),
                                   sizeof(int) * num_seeds, cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int), 0));

        const int grid = std::min(max_blocks, num_seeds);
        candidateCardinalityKernel<<<grid, BLOCK_THREADS, dyn_smem>>>(
            d_points, d_clustered, d_seeds, num_seeds, threshold, N,
            d_scratch, d_counter, d_cardinality);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cardinality.data(), d_cardinality,
                              sizeof(int) * num_seeds, cudaMemcpyDeviceToHost));

        // Pick the best seed: largest cardinality, first (lowest index) on ties
        int max_cardinality = -1;
        int best_seed = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (cardinality[i] > max_cardinality) {
                max_cardinality = cardinality[i];
                best_seed = unclustered_indices[i];
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            // Regenerate the winning cluster to obtain its member list
            extractClusterKernel<<<1, BLOCK_THREADS, dyn_smem>>>(
                d_points, d_clustered, best_seed, threshold, N,
                d_scratch, d_members, d_size);
            CUDA_CHECK(cudaGetLastError());
            int size = 0;
            CUDA_CHECK(cudaMemcpy(&size, d_size, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(members.data(), d_members, sizeof(int) * size,
                                  cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.assign(members.begin(), members.begin() + size);
            clusters.push_back(cluster);

            // Mark all members as clustered (host + device)
            for (int i = 0; i < size; ++i) clustered[cluster.members[i]] = 1;
            markClusteredKernel<<<(size + 255) / 256, 256>>>(d_members, size, d_clustered);
            CUDA_CHECK(cudaGetLastError());

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                               [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end());
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_cardinality));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_size));
    CUDA_CHECK(cudaFree(d_counter));
    if (d_scratch) CUDA_CHECK(cudaFree(d_scratch));

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

    // Initialise the CUDA context up front so it is not counted as clustering time
    CUDA_CHECK(cudaFree(0));
    
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
