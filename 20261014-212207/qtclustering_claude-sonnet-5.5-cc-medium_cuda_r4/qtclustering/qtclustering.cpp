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
#include <math_constants.h>

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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),\
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static const int BLOCK = 128;
static const int NWARPS = BLOCK / 32;

// Per-candidate state: coordinates, running max distance to cluster members, index
struct Cand {
    double x, y, maxd;
    int idx;
    int pad;
};

struct Best {
    double d;
    int idx;
};

__device__ __forceinline__ bool better(double d1, int i1, double d2, int i2) {
    return d1 < d2 || (d1 == d2 && i1 < i2);
}

// Grow one candidate cluster from `seed` with a whole thread block.
// Candidates are only points within threshold of the seed (necessary for membership).
// Each step picks the candidate with the smallest max-distance to the members
// (lowest index on ties, as in the sequential scan), then updates all candidates.
// Returns cluster cardinality. If members != nullptr, the members are written in order.
__device__ int growCluster(const int seed, const double2* __restrict__ pts,
                           const unsigned char* __restrict__ clustered,
                           const double threshold, const int N, Cand* cands,
                           int* members, Best* red, int* shcount) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const double2 sp = pts[seed];

    if (tid == 0) *shcount = 0;
    __syncthreads();
    for (int c = tid; c < N; c += BLOCK) {
        if (c == seed || clustered[c]) continue;
        const double2 p = pts[c];
        const double dx = p.x - sp.x, dy = p.y - sp.y;
        const double d = sqrt(dx * dx + dy * dy);
        if (d < threshold) {
            const int pos = atomicAdd(shcount, 1);
            cands[pos] = Cand{p.x, p.y, d, c, 0};
        }
    }
    __syncthreads();
    const int nc = *shcount;

    int size = 1;
    if (members && tid == 0) members[0] = seed;
    double2 mp = sp;
    int m = seed;
    bool first = true;
    int parity = 0;
    const double INF = CUDART_INF;

    while (true) {
        double bd = INF;
        int bi = 0x7fffffff;
        for (int k = tid; k < nc; k += BLOCK) {
            Cand cd = cands[k];
            double d = cd.maxd;
            if (d == INF) continue;
            if (!first) {
                if (cd.idx == m) {
                    cands[k].maxd = INF;
                    continue;
                }
                const double dx = cd.x - mp.x, dy = cd.y - mp.y;
                const double dd = sqrt(dx * dx + dy * dy);
                if (dd > d) d = dd;
                if (!(d < threshold)) {
                    cands[k].maxd = INF;
                    continue;
                }
                cands[k].maxd = d;
            }
            if (better(d, cd.idx, bd, bi)) { bd = d; bi = cd.idx; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const double od = __shfl_down_sync(0xffffffffu, bd, o);
            const int oi = __shfl_down_sync(0xffffffffu, bi, o);
            if (better(od, oi, bd, bi)) { bd = od; bi = oi; }
        }
        if (lane == 0) red[parity * NWARPS + warp] = Best{bd, bi};
        __syncthreads();
        Best b = red[parity * NWARPS];
        for (int w = 1; w < NWARPS; ++w) {
            const Best o = red[parity * NWARPS + w];
            if (better(o.d, o.idx, b.d, b.idx)) b = o;
        }
        parity ^= 1;
        first = false;
        if (b.d == INF) break;
        m = b.idx;
        mp = pts[m];
        if (members && tid == 0) members[size] = m;
        ++size;
    }
    __syncthreads();
    return size;
}

// Kernel A: evaluate cardinality of the candidate cluster of every unclustered seed
__global__ void cardinalityKernel(const double2* pts, const unsigned char* clustered,
                                  int* card, int* counter, const double threshold,
                                  const int N, Cand* scratch) {
    __shared__ Best red[2 * NWARPS];
    __shared__ int shcount;
    __shared__ int shseed;
    Cand* cands = scratch + (size_t)blockIdx.x * N;
    while (true) {
        if (threadIdx.x == 0) shseed = atomicAdd(counter, 1);
        __syncthreads();
        const int s = shseed;
        __syncthreads();
        if (s >= N) break;
        if (clustered[s]) {
            if (threadIdx.x == 0) card[s] = -1;
            continue;
        }
        const int sz = growCluster(s, pts, clustered, threshold, N, cands, nullptr, red, &shcount);
        if (threadIdx.x == 0) card[s] = sz;
        __syncthreads();
    }
}

// Kernel B (single block): pick best seed (first max), regenerate its cluster,
// append members and mark them clustered. info = {seed, size}
__global__ void selectKernel(const double2* pts, unsigned char* clustered,
                             const int* card, int* members, int* info, int* total,
                             const double threshold, const int N, Cand* scratch) {
    __shared__ Best red[2 * NWARPS];
    __shared__ int shcount;
    __shared__ int bestseed;
    __shared__ int wk[BLOCK];
    __shared__ int ws[BLOCK];
    const int tid = threadIdx.x;
    int bc = -1, bs = 0x7fffffff;
    for (int i = tid; i < N; i += BLOCK) {
        if (card[i] > bc) { bc = card[i]; bs = i; }
    }
    wk[tid] = bc; ws[tid] = bs;
    __syncthreads();
    for (int o = BLOCK / 2; o > 0; o >>= 1) {
        if (tid < o) {
            if (wk[tid + o] > wk[tid] || (wk[tid + o] == wk[tid] && ws[tid + o] < ws[tid])) {
                wk[tid] = wk[tid + o]; ws[tid] = ws[tid + o];
            }
        }
        __syncthreads();
    }
    if (tid == 0) bestseed = (wk[0] > 0) ? ws[0] : -1;
    __syncthreads();
    if (bestseed < 0) {
        if (tid == 0) { info[0] = -1; info[1] = 0; }
        return;
    }
    const int off = *total;
    const int sz = growCluster(bestseed, pts, clustered, threshold, N, scratch,
                               members + off, red, &shcount);
    __syncthreads();
    for (int i = tid; i < sz; i += BLOCK) clustered[members[off + i]] = 1;
    if (tid == 0) {
        info[0] = bestseed;
        info[1] = sz;
        *total = off + sz;
    }
}

// Main QT clustering algorithm (GPU)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    int nblocks = prop.multiProcessorCount * 12;
    const size_t perBlock = sizeof(Cand) * (size_t)N;
    const size_t maxBlocks = (freeB * 7 / 10) / perBlock;
    if ((size_t)nblocks > maxBlocks) nblocks = (int)std::max<size_t>(1, maxBlocks);
    if (nblocks > N) nblocks = N;

    std::vector<double2> hpts(N);
    for (int i = 0; i < N; ++i) hpts[i] = make_double2(points[i].x, points[i].y);

    double2* d_pts; unsigned char* d_clustered; int *d_card, *d_counter, *d_members, *d_info, *d_total;
    Cand* d_scratch;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_info, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scratch, perBlock * nblocks));
    CUDA_CHECK(cudaMemcpy(d_pts, hpts.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(int)));

    int remaining = N;
    int total = 0;
    while (remaining > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int)));
        cardinalityKernel<<<nblocks, BLOCK>>>(d_pts, d_clustered, d_card, d_counter,
                                              threshold, N, d_scratch);
        selectKernel<<<1, BLOCK>>>(d_pts, d_clustered, d_card, d_members, d_info, d_total,
                                   threshold, N, d_scratch);
        int info[2];
        CUDA_CHECK(cudaMemcpy(info, d_info, sizeof(info), cudaMemcpyDeviceToHost));
        if (info[0] < 0) break;
        Cluster c;
        c.seed_point = info[0];
        c.members.resize(info[1]);
        CUDA_CHECK(cudaMemcpy(c.members.data(), d_members + total, sizeof(int) * info[1],
                              cudaMemcpyDeviceToHost));
        total += info[1];
        remaining -= info[1];
        clusters.push_back(std::move(c));
    }
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_pts); cudaFree(d_clustered); cudaFree(d_card); cudaFree(d_counter);
    cudaFree(d_members); cudaFree(d_info); cudaFree(d_total); cudaFree(d_scratch);
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
    
    // Initialize the CUDA context before timing
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
