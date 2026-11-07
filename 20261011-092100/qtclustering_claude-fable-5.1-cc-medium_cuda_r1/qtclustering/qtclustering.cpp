// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (GPU):
//   * Every round, all still-unclustered points are compacted and uploaded.
//   * Candidate-cluster growth for one seed is done by one thread block.
//     The block keeps, for every unclustered point, the maximum distance to
//     the current cluster members ("diameter if added"). Adding a member only
//     requires one distance per candidate plus a block-wide arg-min reduction,
//     turning the original O(k^2 * N) per-seed work into O(k * N) with N-way
//     parallelism inside the block, and seed-level parallelism across blocks.
//   * An exact upper bound on each seed's cardinality (number of unclustered
//     points strictly within the threshold of the seed) is used to process the
//     seeds in order of decreasing bound and to stop evaluating seeds that can
//     no longer beat (or tie) the best cardinality found so far. This prunes
//     work without changing the result.
//   * Ties are resolved exactly as in the sequential code: lowest index.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            exit(1);                                                              \
        }                                                                         \
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

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ double dev_distance(double x1, double y1, double x2, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

// Upper bound on the candidate-cluster cardinality of every seed:
// the number of unclustered points (including the seed itself) whose distance
// to the seed is strictly below the threshold. Any member of the seed's
// cluster must satisfy this, so the true cardinality can never exceed it.
__global__ void boundKernel(const double* __restrict__ xs,
                            const double* __restrict__ ys,
                            const int U,
                            const double threshold,
                            int* __restrict__ bounds) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= U) return;
    const double sx = xs[s];
    const double sy = ys[s];
    int cnt = 0;
    for (int c = 0; c < U; ++c) {
        cnt += (dev_distance(xs[c], ys[c], sx, sy) < threshold) ? 1 : 0;
    }
    bounds[s] = cnt;
}

// Lexicographic (value, index) minimum: smaller value wins, ties -> lower index.
__device__ __forceinline__ void better(double& bv, int& bi, const double v, const int i) {
    if (v < bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

// One block per seed. Grows the candidate cluster for seedList[blockIdx.x]
// exactly like the sequential generateCandidateCluster():
// repeatedly add the unclustered candidate whose maximum distance to the
// current members is minimal (ties -> lowest index), as long as it is < threshold.
//
// md[c] holds the maximum distance from candidate c to all current members
// (+inf once c is a member). Each thread owns the candidates
// c = tid, tid + blockDim, ... so md needs no synchronization between passes.
template <bool SHARED>
__global__ void __launch_bounds__(256)
growKernel(const double* __restrict__ xs,
           const double* __restrict__ ys,
           const int U,
           const double threshold,
           const int* __restrict__ seedList,
           int* __restrict__ cardOut,       // [numSeeds]  cardinality per seed
           int* __restrict__ membersOut,    // [numSeeds * U] members in insertion order
           double* __restrict__ gscratch) { // [gridDim.x * U] used when !SHARED
    extern __shared__ double smem[];
    __shared__ double red_val[32];
    __shared__ int red_idx[32];
    __shared__ int s_next;

    double* md = SHARED ? smem : (gscratch + static_cast<size_t>(blockIdx.x) * U);
    int* members = membersOut + static_cast<size_t>(blockIdx.x) * U;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const double INF = HUGE_VAL;

    const int seed = seedList[blockIdx.x];
    int cur = seed;        // member added in the previous step
    int count = 1;         // current cluster size
    if (tid == 0) members[0] = seed;

    // Initial maxima are 0.0 (cluster empty before the seed is "added").
    for (int c = tid; c < U; c += blockDim.x) md[c] = 0.0;

    while (count < U) {
        const double cx = xs[cur];
        const double cy = ys[cur];

        double bv = INF;
        int bi = 0x7fffffff;
        for (int c = tid; c < U; c += blockDim.x) {
            const double d = dev_distance(xs[c], ys[c], cx, cy);
            double v = fmax(md[c], d);
            if (c == cur) v = INF;    // now a member: never a candidate again
            md[c] = v;
            // Within a thread, c strictly increases, so "<" keeps the lowest index.
            if (v < threshold && v < bv) {
                bv = v;
                bi = c;
            }
        }

        // Warp reduction
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            better(bv, bi, ov, oi);
        }
        if (lane == 0) {
            red_val[warp] = bv;
            red_idx[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            double v = (lane < nwarps) ? red_val[lane] : INF;
            int i = (lane < nwarps) ? red_idx[lane] : 0x7fffffff;
            for (int off = 16; off > 0; off >>= 1) {
                const double ov = __shfl_xor_sync(0xffffffffu, v, off);
                const int oi = __shfl_xor_sync(0xffffffffu, i, off);
                better(v, i, ov, oi);
            }
            if (lane == 0) {
                s_next = (v < threshold) ? i : -1;
                if (i >= 0 && v < threshold) members[count] = i;
            }
        }
        __syncthreads();
        const int next = s_next;
        if (next < 0) break;
        cur = next;
        ++count;
    }

    if (tid == 0) cardOut[blockIdx.x] = count;
}

// ---------------------------------------------------------------------------
// Host-side driver
// ---------------------------------------------------------------------------

struct GpuContext {
    int device = 0;
    int numSMs = 0;
    int batchShared = 0;   // seeds per launch when md fits in shared memory
    int batchGlobal = 0;   // seeds per launch when md lives in global memory
    size_t maxShared = 0;  // max dynamic shared memory per block (bytes)
};

static GpuContext initGpu() {
    GpuContext ctx;
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    ctx.numSMs = prop.multiProcessorCount;
    // Dynamic shared memory available = opt-in maximum minus the kernel's
    // static shared usage (reduction scratch).
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, growKernel<true>));
    ctx.maxShared = static_cast<size_t>(prop.sharedMemPerBlockOptin) - attr.sharedSizeBytes;
    ctx.maxShared -= ctx.maxShared % 128;
    CUDA_CHECK(cudaFuncSetAttribute(growKernel<true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(ctx.maxShared)));
    // Enough blocks per launch to fill the GPU a few times over; blocks with
    // different cluster sizes finish at different times, so several waves
    // keep the SMs busy.
    ctx.batchShared = ctx.numSMs * 16;
    ctx.batchGlobal = ctx.numSMs * 8;
    return ctx;
}

// Main QT clustering algorithm (GPU-accelerated, identical results to the
// sequential reference implementation).
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

    GpuContext ctx = initGpu();
    const int blockSize = 256;
    const int maxBatch = std::max(ctx.batchShared, ctx.batchGlobal);

    // Device buffers (sized once for the full problem)
    double *d_xs = nullptr, *d_ys = nullptr, *d_scratch = nullptr;
    int *d_bounds = nullptr, *d_seeds = nullptr, *d_card = nullptr, *d_members = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xs, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_bounds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * maxBatch));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * maxBatch));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * static_cast<size_t>(maxBatch) * N));
    const bool fitsShared = static_cast<size_t>(N) * sizeof(double) <= ctx.maxShared;
    if (!fitsShared) {
        CUDA_CHECK(cudaMalloc(&d_scratch,
                              sizeof(double) * static_cast<size_t>(ctx.batchGlobal) * N));
    }

    // Pinned host staging buffers
    double *h_xs = nullptr, *h_ys = nullptr;
    int *h_bounds = nullptr, *h_seeds = nullptr, *h_card = nullptr, *h_members = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_xs, sizeof(double) * N));
    CUDA_CHECK(cudaMallocHost(&h_ys, sizeof(double) * N));
    CUDA_CHECK(cudaMallocHost(&h_bounds, sizeof(int) * N));
    CUDA_CHECK(cudaMallocHost(&h_seeds, sizeof(int) * maxBatch));
    CUDA_CHECK(cudaMallocHost(&h_card, sizeof(int) * maxBatch));
    CUDA_CHECK(cudaMallocHost(&h_members, sizeof(int) * N));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<int> order;
    order.reserve(N);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        // Compact the unclustered points (ascending global index order, which
        // is exactly the candidate order of the sequential algorithm).
        for (int i = 0; i < U; ++i) {
            h_xs[i] = points[unclustered_indices[i]].x;
            h_ys[i] = points[unclustered_indices[i]].y;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, sizeof(double) * U, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, sizeof(double) * U, cudaMemcpyHostToDevice, stream));

        // Cardinality upper bound per seed
        boundKernel<<<(U + 255) / 256, 256, 0, stream>>>(d_xs, d_ys, U, threshold, d_bounds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_bounds, d_bounds, sizeof(int) * U, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Seeds sorted by decreasing bound, ties by increasing (local) index.
        order.resize(U);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (h_bounds[a] != h_bounds[b]) return h_bounds[a] > h_bounds[b];
            return a < b;
        });

        int max_cardinality = -1;
        int best_seed_local = -1;
        int best_members_count = 0;

        const bool useShared = static_cast<size_t>(U) * sizeof(double) <= ctx.maxShared;
        const int batch = useShared ? ctx.batchShared : ctx.batchGlobal;

        int pos = 0;
        while (pos < U) {
            // A seed whose bound is below the best cardinality found so far can
            // neither beat nor tie it (and on a tie the lower index, already
            // evaluated or in this batch, wins), so it can be skipped.
            if (h_bounds[order[pos]] < max_cardinality) break;
            int nb = 0;
            while (pos < U && nb < batch && h_bounds[order[pos]] >= max_cardinality) {
                h_seeds[nb++] = order[pos++];
            }

            CUDA_CHECK(cudaMemcpyAsync(d_seeds, h_seeds, sizeof(int) * nb, cudaMemcpyHostToDevice, stream));
            if (useShared) {
                growKernel<true><<<nb, blockSize, sizeof(double) * U, stream>>>(
                    d_xs, d_ys, U, threshold, d_seeds, d_card, d_members, nullptr);
            } else {
                growKernel<false><<<nb, blockSize, 0, stream>>>(
                    d_xs, d_ys, U, threshold, d_seeds, d_card, d_members, d_scratch);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_card, d_card, sizeof(int) * nb, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Select best seed of this batch: max cardinality, ties -> lowest index
            int bb = -1;
            for (int i = 0; i < nb; ++i) {
                const int c = h_card[i];
                const int s = h_seeds[i];
                if (c > max_cardinality || (c == max_cardinality && s < best_seed_local)) {
                    max_cardinality = c;
                    best_seed_local = s;
                    bb = i;
                }
            }
            if (bb >= 0) {
                best_members_count = max_cardinality;
                CUDA_CHECK(cudaMemcpyAsync(h_members,
                                           d_members + static_cast<size_t>(bb) * U,
                                           sizeof(int) * best_members_count,
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
        
        // If we found a cluster, add it
        if (best_seed_local >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_seed_local];
            cluster.members.resize(best_members_count);
            for (int i = 0; i < best_members_count; ++i) {
                cluster.members[i] = unclustered_indices[h_members[i]];
            }
            
            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = true;
            }
            clusters.push_back(std::move(cluster));
            
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

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_bounds));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_members));
    if (d_scratch) CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFreeHost(h_xs));
    CUDA_CHECK(cudaFreeHost(h_ys));
    CUDA_CHECK(cudaFreeHost(h_bounds));
    CUDA_CHECK(cudaFreeHost(h_seeds));
    CUDA_CHECK(cudaFreeHost(h_card));
    CUDA_CHECK(cudaFreeHost(h_members));
    
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

    // Initialize the CUDA context up front so that it is not part of the timing
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
