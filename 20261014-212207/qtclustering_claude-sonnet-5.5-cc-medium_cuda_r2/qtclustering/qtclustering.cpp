// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
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
// Equivalence notes:
//  * A candidate's "max distance to cluster" is maintained incrementally; it
//    only grows, so candidates reaching the threshold are dropped for good.
//    Every candidate must be within the threshold of the seed, so the initial
//    candidate list is the seed's neighbourhood.
//  * The greedy choice is argmin of (max_dist, index), identical to the
//    sequential scan with strict '<'.
//  * A seed's candidate cluster does not change unless one of its members gets
//    clustered, so cardinalities are cached across rounds and only seeds whose
//    neighbourhood lost a point are re-evaluated.
//  * One warp builds one candidate cluster (no block-level synchronisation).
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

struct Counters {
    unsigned int unc_count;   // size of the new unclustered list
    unsigned int work_count;  // number of seeds needing re-evaluation
    unsigned int next_work;   // dynamic work distribution
    unsigned int pad;
    unsigned long long key;   // (cardinality << 32) | (~seed)
};

struct Header {
    int members;
    int seed;
};

// Same arithmetic as the host distance(), without FMA contraction.
__device__ __forceinline__ double dist_dev(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
}

__device__ __forceinline__ unsigned long long make_key(int card, int seed) {
    return (static_cast<unsigned long long>(card) << 32) |
           static_cast<unsigned long long>(0xFFFFFFFFu - static_cast<unsigned int>(seed));
}

// (d, idx) lexicographic minimum across the warp
__device__ __forceinline__ void warpArgMin(double& d, int& idx) {
    for (int off = 16; off > 0; off >>= 1) {
        const double od = __shfl_xor_sync(0xFFFFFFFFu, d, off);
        const int oi = __shfl_xor_sync(0xFFFFFFFFu, idx, off);
        if (od < d || (od == d && oi < idx)) {
            d = od;
            idx = oi;
        }
    }
}

// Grow a candidate cluster from `seed` using a single warp.
// Returns the cardinality (valid in all lanes). Optionally records members.
__device__ int growCluster(const int seed, const int* __restrict__ unc, const int U,
                           int* __restrict__ sidx, double* __restrict__ smd,
                           const double2* __restrict__ pts, const double thr,
                           int* __restrict__ out_members) {
    const int lane = threadIdx.x & 31;
    const unsigned lt_mask = (1u << lane) - 1u;
    const double lim = thr * 1.000001;  // conservative prefilter margin
    const double2 sp = pts[seed];

    int L = 0;
    double bd = DBL_MAX;
    int bi = INT_MAX;

    // Pass 1: neighbourhood of the seed
    for (int base = 0; base < U; base += 32) {
        const int t = base + lane;
        bool keep = false;
        int c = 0;
        double d = 0.0;
        if (t < U) {
            c = unc[t];
            if (c != seed) {
                const double2 p = pts[c];
                if (fabs(p.x - sp.x) <= lim && fabs(p.y - sp.y) <= lim) {
                    d = dist_dev(p, sp);
                    keep = d < thr;
                }
            }
        }
        const unsigned mask = __ballot_sync(0xFFFFFFFFu, keep);
        if (keep) {
            const int pos = L + __popc(mask & lt_mask);
            sidx[pos] = c;
            smd[pos] = d;
            if (d < bd || (d == bd && c < bi)) {
                bd = d;
                bi = c;
            }
        }
        L += __popc(mask);
    }
    warpArgMin(bd, bi);

    int count = 1;
    if (out_members && lane == 0) out_members[0] = seed;

    while (bi != INT_MAX) {
        const int m = bi;
        if (out_members && lane == 0) out_members[count] = m;
        ++count;
        const double2 mp = pts[m];
        __syncwarp();

        int newL = 0;
        bd = DBL_MAX;
        bi = INT_MAX;
        for (int base = 0; base < L; base += 32) {
            const int t = base + lane;
            bool keep = false;
            int c = 0;
            double nd = 0.0;
            if (t < L) {
                c = sidx[t];
                nd = smd[t];
                if (c != m) {
                    const double2 p = pts[c];
                    if (fabs(p.x - mp.x) <= lim && fabs(p.y - mp.y) <= lim) {
                        nd = fmax(nd, dist_dev(p, mp));
                    } else {
                        nd = DBL_MAX;
                    }
                    keep = nd < thr;
                }
            }
            __syncwarp();
            const unsigned mask = __ballot_sync(0xFFFFFFFFu, keep);
            if (keep) {
                const int pos = newL + __popc(mask & lt_mask);
                sidx[pos] = c;
                smd[pos] = nd;
                if (nd < bd || (nd == bd && c < bi)) {
                    bd = nd;
                    bi = c;
                }
            }
            newL += __popc(mask);
        }
        L = newL;
        warpArgMin(bd, bi);
        __syncwarp();
    }
    return count;
}

// Evaluate cardinalities of the seeds on the work list (persistent warps).
__global__ void evalKernel(const int* __restrict__ unc, const int* __restrict__ work,
                           int* __restrict__ card, int* __restrict__ scratch_idx,
                           double* __restrict__ scratch_md, const double2* __restrict__ pts,
                           Counters* __restrict__ ctr, const int U, const int N,
                           const double thr) {
    const int lane = threadIdx.x & 31;
    const size_t warp = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    int* sidx = scratch_idx + warp * N;
    double* smd = scratch_md + warp * N;
    const unsigned W = ctr->work_count;
    while (true) {
        unsigned w = 0;
        if (lane == 0) w = atomicAdd(&ctr->next_work, 1u);
        w = __shfl_sync(0xFFFFFFFFu, w, 0);
        if (w >= W) break;
        const int seed = work[w];
        const int c = growCluster(seed, unc, U, sidx, smd, pts, thr, nullptr);
        if (lane == 0) {
            card[seed] = c;
            atomicMax(&ctr->key, make_key(c, seed));
        }
    }
}

// Drop clustered points from the unclustered list, invalidate cached
// cardinalities of seeds near the previously removed cluster, and build the
// work list plus the best key among still-valid cached seeds.
#define UPD_THREADS 256
__global__ void updateKernel(const int* __restrict__ unc_old, int* __restrict__ unc_new,
                             int* __restrict__ work, const int* __restrict__ card,
                             const char* __restrict__ clustered,
                             const int* __restrict__ prev_members, const int prevM,
                             const double2* __restrict__ pts, Counters* __restrict__ ctr,
                             const int Uold, const double thr) {
    __shared__ double2 tile[UPD_THREADS];
    __shared__ unsigned s_cu, s_cw, s_bu, s_bw;
    __shared__ unsigned long long s_key;
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_cu = 0;
        s_cw = 0;
        s_key = 0;
    }
    const int t = blockIdx.x * UPD_THREADS + tid;
    int s = -1;
    double2 sp = make_double2(0.0, 0.0);
    bool alive = false;
    if (t < Uold) {
        s = unc_old[t];
        alive = !clustered[s];
        sp = pts[s];
    }
    bool inv = alive && card[s] < 0;
    const double lim = thr * 1.000001;
    for (int base = 0; base < prevM; base += UPD_THREADS) {
        __syncthreads();
        if (base + tid < prevM) tile[tid] = pts[prev_members[base + tid]];
        __syncthreads();
        if (alive && !inv) {
            const int n = min(UPD_THREADS, prevM - base);
            for (int j = 0; j < n; ++j) {
                const double2 q = tile[j];
                if (fabs(q.x - sp.x) <= lim && fabs(q.y - sp.y) <= lim &&
                    dist_dev(q, sp) < thr) {
                    inv = true;
                    break;
                }
            }
        }
    }
    __syncthreads();
    unsigned lu = 0, lw = 0;
    if (alive) {
        lu = atomicAdd(&s_cu, 1u);
        if (inv) lw = atomicAdd(&s_cw, 1u);
        else atomicMax(&s_key, make_key(card[s], s));
    }
    __syncthreads();
    if (tid == 0) {
        s_bu = s_cu ? atomicAdd(&ctr->unc_count, s_cu) : 0;
        s_bw = s_cw ? atomicAdd(&ctr->work_count, s_cw) : 0;
        if (s_key) atomicMax(&ctr->key, s_key);
    }
    __syncthreads();
    if (alive) {
        unc_new[s_bu + lu] = s;
        if (inv) work[s_bw + lw] = s;
    }
}

// Re-grow the winning cluster, record members, and mark them clustered.
__global__ void finalizeKernel(const int* __restrict__ unc, const unsigned int* __restrict__ ucount,
                               int* __restrict__ scratch_idx, double* __restrict__ scratch_md,
                               const double2* __restrict__ pts, const Counters* __restrict__ ctr,
                               int* __restrict__ out_members, char* __restrict__ clustered,
                               Header* __restrict__ hdr, const double thr) {
    const int lane = threadIdx.x;
    const int seed = static_cast<int>(0xFFFFFFFFu - static_cast<unsigned int>(ctr->key & 0xFFFFFFFFull));
    const int M = growCluster(seed, unc, static_cast<int>(*ucount), scratch_idx, scratch_md,
                              pts, thr, out_members);
    __syncwarp();
    for (int i = lane; i < M; i += 32) clustered[out_members[i]] = 1;
    if (lane == 0) {
        hdr->members = M;
        hdr->seed = seed;
    }
}

__global__ void iotaKernel(int* a, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = i;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Number of persistent warps limited by scratch memory
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t per_warp = static_cast<size_t>(N) * (sizeof(int) + sizeof(double));
    const size_t fixed = static_cast<size_t>(N) * (sizeof(double2) + 4 * sizeof(int) + 2) + (1 << 20);
    size_t budget = free_mem > fixed ? (free_mem - fixed) * 6 / 10 : 0;
    size_t max_warps = std::max<size_t>(budget / per_warp, 4);
    size_t want_warps = static_cast<size_t>(prop.multiProcessorCount) * 32;
    int blocks = static_cast<int>(std::max<size_t>(1, std::min(want_warps, max_warps) / 4));
    const int warps = blocks * 4;

    double2* d_pts;
    int *d_unc[2], *d_work, *d_card, *d_members, *d_sidx;
    double* d_smd;
    char* d_clustered;
    Counters* d_ctr;
    Header* d_hdr;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_unc[0], sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_unc[1], sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_sidx, sizeof(int) * static_cast<size_t>(N) * warps));
    CUDA_CHECK(cudaMalloc(&d_smd, sizeof(double) * static_cast<size_t>(N) * warps));
    CUDA_CHECK(cudaMalloc(&d_ctr, sizeof(Counters)));
    CUDA_CHECK(cudaMalloc(&d_hdr, sizeof(Header)));

    std::vector<double2> hp(N);
    for (int i = 0; i < N; ++i) hp[i] = make_double2(points[i].x, points[i].y);
    CUDA_CHECK(cudaMemcpy(d_pts, hp.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_card, 0xFF, sizeof(int) * N));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    iotaKernel<<<(N + 255) / 256, 256>>>(d_unc[0], N);

    int cur = 0;
    int U = N;            // size of current unclustered list (before update)
    int prevM = 0;        // members removed in the previous round
    int prevOffset = 0;
    int offset = 0;
    std::vector<int> seeds;
    std::vector<int> sizes;

    while (U > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_ctr, 0, sizeof(Counters)));
        updateKernel<<<(U + UPD_THREADS - 1) / UPD_THREADS, UPD_THREADS>>>(
            d_unc[cur], d_unc[cur ^ 1], d_work, d_card, d_clustered, d_members + prevOffset,
            prevM, d_pts, d_ctr, U, threshold);
        cur ^= 1;
        U -= prevM;
        evalKernel<<<blocks, 128>>>(d_unc[cur], d_work, d_card, d_sidx, d_smd, d_pts, d_ctr,
                                    U, N, threshold);
        finalizeKernel<<<1, 32>>>(d_unc[cur], &d_ctr->unc_count, d_sidx, d_smd, d_pts, d_ctr,
                                  d_members + offset, d_clustered, d_hdr, threshold);
        Header h;
        CUDA_CHECK(cudaMemcpy(&h, d_hdr, sizeof(Header), cudaMemcpyDeviceToHost));
        seeds.push_back(h.seed);
        sizes.push_back(h.members);
        prevM = h.members;
        prevOffset = offset;
        offset += h.members;
        if (offset >= N) break;
    }

    std::vector<int> members(N);
    CUDA_CHECK(cudaMemcpy(members.data(), d_members, sizeof(int) * N, cudaMemcpyDeviceToHost));
    int pos = 0;
    for (size_t c = 0; c < seeds.size(); ++c) {
        Cluster cl;
        cl.seed_point = seeds[c];
        cl.members.assign(members.begin() + pos, members.begin() + pos + sizes[c]);
        pos += sizes[c];
        clusters.push_back(std::move(cl));
    }

    cudaFree(d_pts);
    cudaFree(d_unc[0]);
    cudaFree(d_unc[1]);
    cudaFree(d_work);
    cudaFree(d_card);
    cudaFree(d_members);
    cudaFree(d_clustered);
    cudaFree(d_sidx);
    cudaFree(d_smd);
    cudaFree(d_ctr);
    cudaFree(d_hdr);
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
    CUDA_CHECK(cudaFree(nullptr));

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
