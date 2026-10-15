// QT Clustering Benchmark - CUDA Version
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

// ---------------------------------------------------------------------------
// CUDA implementation
//
// Key observations that keep results identical to the sequential algorithm:
//  * A point can only join the cluster grown from a seed if it is within the
//    threshold of the seed, so each seed works on a (small) candidate list.
//  * A candidate's "max distance to the cluster" only grows as members are
//    added, so candidates that reach the threshold are dropped for good.
//  * The best cluster of a seed depends only on unclustered points within the
//    threshold of that seed.  After a cluster is removed, only seeds having a
//    removed point within the threshold must be re-evaluated; all other
//    cardinalities are cached.
//  * sqrt is monotone, so max distances are tracked as squared distances; the
//    exact sqrt is only evaluated when two values are too close to decide
//    (ties are broken by lowest index like the sequential scan).
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

static const int BLOCK = 128;
static const int NWARP = BLOCK / 32;

struct Params {
    double thr, lo, hi; // threshold and bounds of the sqrt-ambiguous zone of thr^2
};

__device__ __forceinline__ double dist2(const double2 a, const double2 b) {
    const double dx = __dsub_rn(a.x, b.x);
    const double dy = __dsub_rn(a.y, b.y);
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
}

// Equivalent to sqrt(d2) < thr
__device__ __forceinline__ bool belowThreshold(const double d2, const Params& p) {
    if (d2 < p.lo) return true;
    if (d2 >= p.hi) return false;
    return sqrt(d2) < p.thr;
}

// Ordering by (sqrt(v), idx)
__device__ __forceinline__ bool better(const double av, const int ai, const double bv, const int bi) {
    if (av == bv) return ai < bi;
    const double m = fmax(av, bv);
    if (fabs(av - bv) > 1e-12 * m) return av < bv;
    const double sa = sqrt(av), sb = sqrt(bv);
    if (sa != sb) return sa < sb;
    return ai < bi;
}

struct Scratch {
    int* idx[2];
    double2* pt[2];
    double* d2[2];
};

// Grows the cluster of `seed` with one block. Returns its size (all threads).
// If out != nullptr the members are written to it in insertion order.
__device__ int growCluster(const int seed, const double2* __restrict__ pts,
                           const unsigned char* __restrict__ clustered, const int N,
                           const Params& prm, const Scratch& s, int* out) {
    __shared__ int cnt[2];
    __shared__ double rv[2][NWARP];
    __shared__ int ri[2][NWARP];

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const unsigned full = 0xffffffffu;
    const double2 sp = pts[seed];

    if (tid == 0) {
        cnt[0] = 0;
        cnt[1] = 0;
        if (out) out[0] = seed;
    }
    __syncthreads();

    // Gather candidates within threshold of the seed
    for (int base = 0; base < N; base += BLOCK) {
        const int p = base + tid;
        bool keep = false;
        double d = 0.0;
        double2 q = make_double2(0, 0);
        if (p < N && p != seed && !clustered[p]) {
            q = pts[p];
            d = dist2(sp, q);
            keep = belowThreshold(d, prm);
        }
        const unsigned m = __ballot_sync(full, keep);
        if (m) {
            int b = 0;
            if (lane == (__ffs(m) - 1)) b = atomicAdd(&cnt[0], __popc(m));
            b = __shfl_sync(full, b, __ffs(m) - 1);
            if (keep) {
                const int pos = b + __popc(m & ((1u << lane) - 1));
                s.idx[0][pos] = p;
                s.pt[0][pos] = q;
                s.d2[0][pos] = d;
            }
        }
    }
    __syncthreads();

    int n = cnt[0];
    int cur = 0;
    int size = 1;
    int winner = -1;
    double2 wp = sp;
    int iter = 0;

    while (n > 0) {
        const int nxt = cur ^ 1;
        const int par = iter & 1;
        if (tid == 0) cnt[nxt] = 0; // nxt counter was last read two syncs ago
        __syncthreads();

        double bv = 0.0;
        int bi = 0x7fffffff;
        const int* ci = s.idx[cur];
        const double2* cp = s.pt[cur];
        const double* cd = s.d2[cur];
        for (int base = 0; base < n; base += BLOCK) {
            const int i = base + tid;
            bool keep = false;
            int id = 0;
            double2 q = make_double2(0, 0);
            double d = 0.0;
            if (i < n) {
                id = ci[i];
                if (id != winner) {
                    q = cp[i];
                    d = cd[i];
                    if (winner >= 0) d = fmax(d, dist2(q, wp));
                    keep = (winner < 0) || belowThreshold(d, prm);
                }
            }
            const unsigned m = __ballot_sync(full, keep);
            if (m) {
                int b = 0;
                if (lane == (__ffs(m) - 1)) b = atomicAdd(&cnt[nxt], __popc(m));
                b = __shfl_sync(full, b, __ffs(m) - 1);
                if (keep) {
                    const int pos = b + __popc(m & ((1u << lane) - 1));
                    s.idx[nxt][pos] = id;
                    s.pt[nxt][pos] = q;
                    s.d2[nxt][pos] = d;
                    if (bi == 0x7fffffff || better(d, id, bv, bi)) {
                        bv = d;
                        bi = id;
                    }
                }
            }
        }
        // block-wide argmin
        for (int o = 16; o > 0; o >>= 1) {
            const double ov = __shfl_down_sync(full, bv, o);
            const int oi = __shfl_down_sync(full, bi, o);
            if (oi != 0x7fffffff && (bi == 0x7fffffff || better(ov, oi, bv, bi))) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            rv[par][warp] = bv;
            ri[par][warp] = bi;
        }
        __syncthreads();
        bv = rv[par][0];
        bi = ri[par][0];
        for (int w = 1; w < NWARP; ++w) {
            const double ov = rv[par][w];
            const int oi = ri[par][w];
            if (oi != 0x7fffffff && (bi == 0x7fffffff || better(ov, oi, bv, bi))) {
                bv = ov;
                bi = oi;
            }
        }
        n = cnt[nxt];
        cur = nxt;
        ++iter;
        if (bi == 0x7fffffff || n == 0) break;
        // Add winner (the new member's distances are applied in the next pass)
        winner = bi;
        ++size;
        if (out && tid == 0) out[size - 1] = winner;
        wp = pts[winner];
    }
    __syncthreads();
    return size;
}

// Evaluate the candidate cluster cardinality of every seed in the dirty list
__global__ void evalSeeds(const double2* __restrict__ pts, const unsigned char* clustered,
                          const int N, const Params prm, const int* __restrict__ list,
                          const int* __restrict__ listCount, int* card, Scratch base,
                          const size_t stride) {
    Scratch s;
    for (int k = 0; k < 2; ++k) {
        s.idx[k] = base.idx[k] + (size_t)blockIdx.x * stride;
        s.pt[k] = base.pt[k] + (size_t)blockIdx.x * stride;
        s.d2[k] = base.d2[k] + (size_t)blockIdx.x * stride;
    }
    const int total = *listCount;
    for (int i = blockIdx.x; i < total; i += gridDim.x) {
        const int seed = list[i];
        const int c = growCluster(seed, pts, clustered, N, prm, s, nullptr);
        if (threadIdx.x == 0) card[seed] = c;
    }
}

// Pick the unclustered seed with the largest cardinality (lowest index on ties)
__global__ void pickBest(const unsigned char* clustered, const int* card, const int N, int* bestSeed) {
    __shared__ int sc[32], si[32];
    int bc = -1, bi = 0x7fffffff;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (!clustered[i] && card[i] > bc) {
            bc = card[i];
            bi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const int oc = __shfl_down_sync(0xffffffffu, bc, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (oc > bc || (oc == bc && oi < bi)) { bc = oc; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sc[threadIdx.x >> 5] = bc; si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); ++w) {
            if (sc[w] > bc || (sc[w] == bc && si[w] < bi)) { bc = sc[w]; bi = si[w]; }
        }
        *bestSeed = (bc > 0) ? bi : -1;
    }
}

// Rebuild the winning cluster and store its members
__global__ void buildBest(const double2* __restrict__ pts, const unsigned char* clustered,
                          const int N, const Params prm, const int* bestSeed, int* members,
                          int* memberCount, Scratch s) {
    const int seed = *bestSeed;
    if (seed < 0) {
        if (threadIdx.x == 0) *memberCount = 0;
        return;
    }
    const int c = growCluster(seed, pts, clustered, N, prm, s, members);
    if (threadIdx.x == 0) *memberCount = c;
}

__global__ void markClustered(unsigned char* clustered, const int* members, const int count,
                              int* listCount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
    if (i == 0) *listCount = 0;
}

// Seeds with a removed point within the threshold must be re-evaluated
__global__ void markDirty(const double2* __restrict__ pts, const unsigned char* clustered,
                          const int N, const Params prm, const int* __restrict__ members,
                          const int count, int* list, int* listCount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool dirty = false;
    if (i < N && !clustered[i]) {
        const double2 p = pts[i];
        for (int m = 0; m < count && !dirty; ++m) {
            dirty = belowThreshold(dist2(p, pts[members[m]]), prm);
        }
    }
    if (dirty) list[atomicAdd(listCount, 1)] = i;
}

__global__ void initList(int* list, int* listCount, const int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) list[i] = i;
    if (i == 0) *listCount = N;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    Params prm;
    prm.thr = threshold;
    prm.lo = threshold * threshold * (1.0 - 1e-9);
    prm.hi = threshold * threshold * (1.0 + 1e-9);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Persistent blocks, bounded by scratch memory
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t perBlock = 2 * (sizeof(int) + sizeof(double2) + sizeof(double)) * (size_t)N;
    size_t nb = (size_t)prop.multiProcessorCount * 12;
    nb = std::min(nb, (size_t)(freeMem * 0.5 / perBlock));
    nb = std::max<size_t>(nb, 1);
    nb = std::min<size_t>(nb, N);
    const size_t stride = N;

    std::vector<double2> hpts(N);
    for (int i = 0; i < N; ++i) hpts[i] = make_double2(points[i].x, points[i].y);

    double2* d_pts;
    unsigned char* d_clustered;
    int *d_card, *d_list, *d_listCount, *d_best, *d_members, *d_memberCount;
    Scratch scr;
    char* scratchMem;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_list, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_listCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_memberCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratchMem, perBlock * nb));
    {
        char* p = scratchMem;
        const size_t n = stride * nb;
        for (int k = 0; k < 2; ++k) {
            scr.pt[k] = reinterpret_cast<double2*>(p); p += sizeof(double2) * n;
            scr.d2[k] = reinterpret_cast<double*>(p);  p += sizeof(double) * n;
            scr.idx[k] = reinterpret_cast<int*>(p);    p += sizeof(int) * n;
        }
    }
    // Scratch for the single-block build kernel is the first block's slice
    Scratch scr0;
    for (int k = 0; k < 2; ++k) {
        scr0.idx[k] = scr.idx[k];
        scr0.pt[k] = scr.pt[k];
        scr0.d2[k] = scr.d2[k];
    }

    CUDA_CHECK(cudaMemcpy(d_pts, hpts.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    CUDA_CHECK(cudaMemset(d_card, 0, sizeof(int) * N));

    const int tb = 256;
    const int gridN = (N + tb - 1) / tb;
    initList<<<gridN, tb>>>(d_list, d_listCount, N);

    std::vector<int> hmembers(N);
    int remaining = N;
    while (remaining > 0) {
        evalSeeds<<<(unsigned)nb, BLOCK>>>(d_pts, d_clustered, N, prm, d_list, d_listCount,
                                           d_card, scr, stride);
        pickBest<<<1, 1024>>>(d_clustered, d_card, N, d_best);
        buildBest<<<1, BLOCK>>>(d_pts, d_clustered, N, prm, d_best, d_members, d_memberCount, scr0);
        int count = 0;
        CUDA_CHECK(cudaMemcpy(&count, d_memberCount, sizeof(int), cudaMemcpyDeviceToHost));
        if (count <= 0) break;

        int seedPoint = 0;
        CUDA_CHECK(cudaMemcpy(&seedPoint, d_best, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hmembers.data(), d_members, sizeof(int) * count,
                              cudaMemcpyDeviceToHost));

        markClustered<<<(count + tb - 1) / tb, tb>>>(d_clustered, d_members, count, d_listCount);
        markDirty<<<gridN, tb>>>(d_pts, d_clustered, N, prm, d_members, count, d_list, d_listCount);

        Cluster cluster;
        cluster.seed_point = seedPoint;
        cluster.members.assign(hmembers.begin(), hmembers.begin() + count);
        clusters.push_back(std::move(cluster));
        remaining -= count;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_pts); cudaFree(d_clustered); cudaFree(d_card); cudaFree(d_list);
    cudaFree(d_members); cudaFree(d_listCount); cudaFree(d_best); cudaFree(d_memberCount);
    cudaFree(scratchMem);
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
    
    // Initialize the CUDA context outside of the timed region
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
