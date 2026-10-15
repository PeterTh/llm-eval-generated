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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int BLOCK = 256;
static const int NWARPS = BLOCK / 32;

// Squared distance without FMA contraction (matches separate mul/add semantics)
__device__ __forceinline__ double dist2(double2 a, double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
}

// Ordering equivalent to comparing (sqrt(d2), index) lexicographically, but only
// evaluates sqrt when the squared distances are nearly equal.
__device__ __forceinline__ bool better(double da, int ia, double db, int ib) {
    if (ib < 0) return ia >= 0;
    if (ia < 0) return false;
    if (da == db) return ia < ib;
    const double diff = fabs(da - db);
    if (diff <= 1e-14 * fmax(da, db)) {
        const double sa = sqrt(da), sb = sqrt(db);
        if (sa == sb) return ia < ib;
        return sa < sb;
    }
    return da < db;
}

// Grow the greedy QT candidate cluster for one seed using a single thread block.
// Only points with dist(seed, p) < threshold (unclustered) can ever join, so the
// block gathers them into a compact scratch list and keeps, per candidate, the
// running maximum distance to the current members (the "diameter if added").
// EXTRACT=false: returns cluster cardinality only. EXTRACT=true: also writes members.
template <bool EXTRACT>
__device__ int growCluster(const int seed, const double2* __restrict__ pts,
                           const int* __restrict__ list, const int U,
                           const double T2,
                           int* nidx, double2* nxy, double* nd,
                           int* members) {
    __shared__ int s_cnt;
    __shared__ double s_d[2][NWARPS];
    __shared__ int s_i[2][NWARPS];
    __shared__ int s_e[2][NWARPS];
    __shared__ int s_off[NWARPS + 1];
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;

    if (tid == 0) s_cnt = 0;
    __syncthreads();

    const double2 sp = pts[seed];
    for (int j = tid; j < U; j += BLOCK) {
        const int p = list[j];
        if (p == seed) continue;
        const double2 q = pts[p];
        if (dist2(q, sp) < T2) {
            const int pos = atomicAdd(&s_cnt, 1);
            nidx[pos] = p;
            nxy[pos] = q;
            nd[pos] = 0.0;
        }
    }
    __syncthreads();
    int n = s_cnt;

    int size = 1;
    if (EXTRACT && tid == 0) members[0] = seed;
    int last = seed;
    double2 lp = sp;
    int parity = 0;
    const double INF = __longlong_as_double(0x7ff0000000000000LL);

    while (true) {
        double bd = 0.0;
        int bi = -1;
        int ne = 0;
        for (int j = tid; j < n; j += BLOCK) {
            double d = nd[j];
            const int id = nidx[j];
            if (id == last) {
                d = INF;
            } else {
                const double e = dist2(nxy[j], lp);
                d = fmax(d, e);
            }
            nd[j] = d;
            if (d < T2) {
                ++ne;
                if (better(d, id, bd, bi)) {
                    bd = d;
                    bi = id;
                }
            }
        }
        // warp reduction
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const double od = __shfl_down_sync(0xffffffffu, bd, o);
            const int oi = __shfl_down_sync(0xffffffffu, bi, o);
            if (better(od, oi, bd, bi)) { bd = od; bi = oi; }
            ne += __shfl_down_sync(0xffffffffu, ne, o);
        }
        if (lane == 0) { s_d[parity][warp] = bd; s_i[parity][warp] = bi; s_e[parity][warp] = ne; }
        __syncthreads();
        double fd = s_d[parity][0];
        int fi = s_i[parity][0];
        int fe = s_e[parity][0];
        #pragma unroll
        for (int w = 1; w < NWARPS; ++w) {
            const double od = s_d[parity][w];
            const int oi = s_i[parity][w];
            fe += s_e[parity][w];
            if (better(od, oi, fd, fi)) { fd = od; fi = oi; }
        }
        parity ^= 1;
        if (fi < 0) break;
        if (EXTRACT && tid == 0) members[size] = fi;
        ++size;
        last = fi;
        lp = pts[fi];

        // Drop candidates that can no longer join (in-place, order-preserving compaction)
        if (fe * 2 < n) {
            int base = 0;
            for (int c0 = 0; c0 < n; c0 += BLOCK) {
                const int j = c0 + tid;
                int keep = 0;
                int ci = 0;
                double2 cxy = make_double2(0.0, 0.0);
                double cd = 0.0;
                if (j < n) {
                    cd = nd[j];
                    keep = (cd < T2) ? 1 : 0;
                    if (keep) { ci = nidx[j]; cxy = nxy[j]; }
                }
                const unsigned bal = __ballot_sync(0xffffffffu, keep);
                if (lane == 0) s_off[warp] = __popc(bal);
                __syncthreads();
                int woff = 0, total = 0;
                for (int w = 0; w < NWARPS; ++w) {
                    const int c = s_off[w];
                    if (w < warp) woff += c;
                    total += c;
                }
                if (keep) {
                    const int pos = base + woff + __popc(bal & ((1u << lane) - 1u));
                    nidx[pos] = ci;
                    nxy[pos] = cxy;
                    nd[pos] = cd;
                }
                base += total;
                __syncthreads();
            }
            n = base;
        }
    }
    // make sure scratch/shared reuse by the next seed is safe
    __syncthreads();
    return size;
}

// Phase 1: every unclustered point is tried as a seed; blocks pull seeds from a queue.
__global__ void seedKernel(const double2* __restrict__ pts, const int* __restrict__ list,
                           const int U, const double T2, const int N,
                           int* __restrict__ card, int* __restrict__ counter,
                           int* __restrict__ scratchIdx, double2* __restrict__ scratchXY,
                           double* __restrict__ scratchD) {
    __shared__ int s_work;
    int* nidx = scratchIdx + (size_t)blockIdx.x * N;
    double2* nxy = scratchXY + (size_t)blockIdx.x * N;
    double* nd = scratchD + (size_t)blockIdx.x * N;
    while (true) {
        if (threadIdx.x == 0) s_work = atomicAdd(counter, 1);
        __syncthreads();
        const int w = s_work;
        __syncthreads();
        if (w >= U) break;
        const int c = growCluster<false>(list[w], pts, list, U, T2, nidx, nxy, nd, nullptr);
        if (threadIdx.x == 0) card[w] = c;
    }
}

// Pick the seed with max cardinality; ties go to the lowest point index.
__global__ void argmaxKernel(const int* __restrict__ card, const int* __restrict__ list,
                             const int U, int* __restrict__ best) {
    __shared__ int s_c[BLOCK];
    __shared__ int s_p[BLOCK];
    int bc = -1, bp = 0x7fffffff;
    for (int j = threadIdx.x; j < U; j += BLOCK) {
        const int c = card[j], p = list[j];
        if (c > bc || (c == bc && p < bp)) { bc = c; bp = p; }
    }
    s_c[threadIdx.x] = bc;
    s_p[threadIdx.x] = bp;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const int c = s_c[threadIdx.x + s], p = s_p[threadIdx.x + s];
            if (c > s_c[threadIdx.x] || (c == s_c[threadIdx.x] && p < s_p[threadIdx.x])) {
                s_c[threadIdx.x] = c;
                s_p[threadIdx.x] = p;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) best[0] = s_p[0];
}

// Phase 2: regenerate the winning cluster with its member list (in insertion order).
__global__ void extractKernel(const double2* __restrict__ pts, const int* __restrict__ list,
                              const int U, const double T2, const int N,
                              const int* __restrict__ best, int* __restrict__ out,
                              int* scratchIdx, double2* scratchXY, double* scratchD) {
    const int c = growCluster<true>(best[0], pts, list, U, T2, scratchIdx, scratchXY,
                                    scratchD, out + 1);
    if (threadIdx.x == 0) out[0] = c;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    // Smallest squared distance whose sqrt is >= threshold: sqrt(d2) < thr  <=>  d2 < T2
    double T2 = threshold * threshold;
    while (std::sqrt(T2) >= threshold) T2 = std::nextafter(T2, 0.0);
    while (std::sqrt(T2) < threshold) T2 = std::nextafter(T2, std::numeric_limits<double>::infinity());

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Concurrent blocks, limited by scratch memory (28 bytes per point per block)
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t perBlock = (size_t)N * (sizeof(int) + sizeof(double2) + sizeof(double));
    size_t maxBlocks = (size_t)prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / BLOCK);
    size_t memBlocks = (freeMem / 2) / perBlock;
    if (memBlocks < 1) memBlocks = 1;
    if (maxBlocks > memBlocks) maxBlocks = memBlocks;

    std::vector<double2> hpts(N);
    for (int i = 0; i < N; ++i) hpts[i] = make_double2(points[i].x, points[i].y);

    double2* d_pts;
    int *d_list, *d_card, *d_counter, *d_best, *d_out, *d_sIdx;
    double2* d_sXY;
    double* d_sD;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_list, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * (N + 1)));
    CUDA_CHECK(cudaMalloc(&d_sIdx, sizeof(int) * N * maxBlocks));
    CUDA_CHECK(cudaMalloc(&d_sXY, sizeof(double2) * N * maxBlocks));
    CUDA_CHECK(cudaMalloc(&d_sD, sizeof(double) * N * maxBlocks));
    CUDA_CHECK(cudaMemcpy(d_pts, hpts.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));

    std::vector<int> unclustered(N);
    for (int i = 0; i < N; ++i) unclustered[i] = i;
    std::vector<char> clustered(N, 0);
    std::vector<int> hout(N + 1);

    while (!unclustered.empty()) {
        const int U = static_cast<int>(unclustered.size());
        CUDA_CHECK(cudaMemcpy(d_list, unclustered.data(), sizeof(int) * U, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int)));

        const int grid = static_cast<int>(std::min<size_t>(maxBlocks, (size_t)U));
        seedKernel<<<grid, BLOCK>>>(d_pts, d_list, U, T2, N, d_card, d_counter,
                                    d_sIdx, d_sXY, d_sD);
        argmaxKernel<<<1, BLOCK>>>(d_card, d_list, U, d_best);
        extractKernel<<<1, BLOCK>>>(d_pts, d_list, U, T2, N, d_best, d_out,
                                    d_sIdx, d_sXY, d_sD);
        CUDA_CHECK(cudaGetLastError());

        int count = 0;
        CUDA_CHECK(cudaMemcpy(&count, d_out, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hout.data(), d_out + 1, sizeof(int) * count, cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = hout[0];
        cluster.members.assign(hout.begin(), hout.begin() + count);
        for (int m : cluster.members) clustered[m] = 1;
        clusters.push_back(std::move(cluster));

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
    }

    cudaFree(d_pts); cudaFree(d_list); cudaFree(d_card); cudaFree(d_counter);
    cudaFree(d_best); cudaFree(d_out); cudaFree(d_sIdx); cudaFree(d_sXY); cudaFree(d_sD);
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
