// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
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
#include <mpi.h>
#include <omp.h>

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Calculate Euclidean distance between two points
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

static const int BLOCK = 256;
static const int NWARPS = BLOCK / 32;

// Block-wide argmin of (value, index), ties broken towards the smaller index.
// Every thread receives the result.
__device__ inline void blockArgMin(double& v, int& idx, double* sv, int* si) {
    const unsigned full = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        double ov = __shfl_down_sync(full, v, off);
        int oi = __shfl_down_sync(full, idx, off);
        if (ov < v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = v; si[warp] = idx; }
    __syncthreads();
    v = sv[0]; idx = si[0];
    for (int w = 1; w < NWARPS; ++w) {
        if (sv[w] < v || (sv[w] == v && si[w] < idx)) { v = sv[w]; idx = si[w]; }
    }
}

// Greedy QT candidate cluster for one seed, built cooperatively by one block.
// Only unclustered points within the threshold of the seed can ever join, so
// they are gathered once; cd[] holds each candidate's max distance to the
// current members (INF once it can no longer join). Semantics are identical to
// the sequential search: smallest max-distance wins, lowest index on ties.
// Returns the cluster cardinality; optionally records members in order.
__device__ int buildCluster(const int seed, const Point* __restrict__ pts, const int N,
                            const double thr, const char* __restrict__ clustered,
                            int* cidx, double* cd, int* members,
                            int* s_cnt, double* sv, int* si, int* s_best) {
    const double INF = 1e300;
    const int tid = threadIdx.x;
    if (tid == 0) *s_cnt = 0;
    __syncthreads();
    const Point sp = pts[seed];
    for (int c = tid; c < N; c += BLOCK) {
        if (c == seed || clustered[c]) continue;
        const double d = distance(sp, pts[c]);
        if (d < thr) {
            int pos = atomicAdd(s_cnt, 1);
            cidx[pos] = c;
            cd[pos] = d;
        }
    }
    __syncthreads();
    const int M = *s_cnt;
    int size = 1;
    if (members && tid == 0) members[1] = seed;
    while (true) {
        double bv = INF;
        int bi = 0x7fffffff;
        for (int j = tid; j < M; j += BLOCK) {
            const double d = cd[j];
            if (d < bv || (d == bv && cidx[j] < bi)) { bv = d; bi = cidx[j]; }
        }
        blockArgMin(bv, bi, sv, si);
        if (bv >= INF) break;
        ++size;
        if (members && tid == 0) members[size] = bi;
        const Point bp = pts[bi];
        for (int j = tid; j < M; j += BLOCK) {
            double d = cd[j];
            if (d >= INF) continue;
            if (cidx[j] == bi) { cd[j] = INF; continue; }
            const double dd = distance(pts[cidx[j]], bp);
            if (dd > d) d = dd;
            cd[j] = (d < thr) ? d : INF;
        }
        __syncthreads();
    }
    (void)s_best;
    return size;
}

// Batch kernel: each block repeatedly grabs a dirty seed and computes its
// candidate cluster cardinality.
__global__ void seedKernel(const Point* pts, const int N, const double thr,
                           const char* clustered, const int* dirty,
                           const int* dirtyCount, int* workCtr,
                           int* scratchIdx, double* scratchD, int* card) {
    __shared__ int s_cnt, s_work, s_best[1];
    __shared__ double sv[NWARPS];
    __shared__ int si[NWARPS];
    int* cidx = scratchIdx + (size_t)blockIdx.x * N;
    double* cd = scratchD + (size_t)blockIdx.x * N;
    const int total = *dirtyCount;
    while (true) {
        if (threadIdx.x == 0) s_work = atomicAdd(workCtr, 1);
        __syncthreads();
        const int w = s_work;
        __syncthreads();
        if (w >= total) break;
        const int seed = dirty[w];
        const int sz = buildCluster(seed, pts, N, thr, clustered, cidx, cd, nullptr,
                                    &s_cnt, sv, si, s_best);
        if (threadIdx.x == 0) card[seed] = sz;
        __syncthreads();
    }
}

// Single-seed kernel: recompute the winning cluster and emit its members
// (members[0] = size, members[1..] in insertion order).
__global__ void memberKernel(const Point* pts, const int N, const double thr,
                             const char* clustered, const int seed,
                             int* scratchIdx, double* scratchD, int* members) {
    __shared__ int s_cnt, s_best[1];
    __shared__ double sv[NWARPS];
    __shared__ int si[NWARPS];
    const int sz = buildCluster(seed, pts, N, thr, clustered, scratchIdx, scratchD,
                                members, &s_cnt, sv, si, s_best);
    if (threadIdx.x == 0) members[0] = sz;
}

// Mark the members of the new cluster as clustered, and drop their cached card.
__global__ void commitKernel(const int* members, char* clustered, int* card) {
    const int n = members[0];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int m = members[1 + i];
        clustered[m] = 1;
        card[m] = -1;
    }
}

// A seed's candidate cluster only changes if a removed point lies within the
// threshold of it. Collect owned unclustered seeds that are affected.
__global__ void markDirtyKernel(const Point* pts, const int N, const double thr,
                                const char* clustered, const int* members,
                                const int rank, const int nranks,
                                int* dirty, int* dirtyCount) {
    const int n = members[0];
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < N; p += gridDim.x * blockDim.x) {
        if (p % nranks != rank || clustered[p]) continue;
        const Point pp = pts[p];
        bool hit = false;
        for (int i = 0; i < n && !hit; ++i) hit = distance(pp, pts[members[1 + i]]) < thr;
        if (hit) dirty[atomicAdd(dirtyCount, 1)] = p;
    }
}

// Initially every owned seed is dirty.
__global__ void initDirtyKernel(const int N, const int rank, const int nranks,
                                int* dirty, int* dirtyCount) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < N; p += gridDim.x * blockDim.x) {
        if (p % nranks == rank) dirty[atomicAdd(dirtyCount, 1)] = p;
    }
}

// Best owned seed: max cardinality, lowest index on ties, packed in one key.
__global__ void argmaxKernel(const int N, const char* clustered, const int* card,
                             unsigned long long* key) {
    unsigned long long best = 0;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < N; p += gridDim.x * blockDim.x) {
        if (clustered[p] || card[p] <= 0) continue;
        const unsigned long long k =
            ((unsigned long long)card[p] << 32) | (unsigned)(0x7fffffff - p);
        if (k > best) best = k;
    }
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long o = __shfl_down_sync(0xffffffffu, best, off);
        if (o > best) best = o;
    }
    if ((threadIdx.x & 31) == 0 && best) atomicMax(key, best);
}

// Main QT clustering algorithm: seeds are distributed cyclically over MPI ranks
// (one GPU each); the global winner is selected with an Allreduce per round.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold,
                                  const int rank, const int nranks, const int local_rank) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { fprintf(stderr, "No CUDA device\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % ndev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, local_rank % ndev));

    Point* d_pts; char* d_clustered; int *d_card, *d_dirty, *d_dirtyCount, *d_work, *d_members;
    unsigned long long* d_key;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_dirty, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * (N + 1)));
    CUDA_CHECK(cudaMalloc(&d_dirtyCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_key, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_pts, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    CUDA_CHECK(cudaMemset(d_card, 0xff, sizeof(int) * N));

    // Per-block scratch (candidate list + running max distances)
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    size_t per_block = (sizeof(int) + sizeof(double)) * (size_t)N;
    int blocks = prop.multiProcessorCount * 4;
    size_t maxBlocks = (freeB / 2) / per_block;
    if (maxBlocks < 1) maxBlocks = 1;
    if ((size_t)blocks > maxBlocks) blocks = (int)maxBlocks;
    int* d_sidx; double* d_sd;
    CUDA_CHECK(cudaMalloc(&d_sidx, sizeof(int) * (size_t)N * blocks));
    CUDA_CHECK(cudaMalloc(&d_sd, sizeof(double) * (size_t)N * blocks));

    const int gridN = std::min((N + BLOCK - 1) / BLOCK, prop.multiProcessorCount * 8);
    const int gridM = std::min((N + BLOCK - 1) / BLOCK, prop.multiProcessorCount * 8);

    std::vector<int> h_members(N + 1);
    int remaining = N;
    bool first = true;
    while (remaining > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_dirtyCount, 0, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_work, 0, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_key, 0, sizeof(unsigned long long)));
        if (first) {
            initDirtyKernel<<<gridN, BLOCK>>>(N, rank, nranks, d_dirty, d_dirtyCount);
            first = false;
        } else {
            markDirtyKernel<<<gridN, BLOCK>>>(d_pts, N, threshold, d_clustered, d_members,
                                              rank, nranks, d_dirty, d_dirtyCount);
        }
        seedKernel<<<blocks, BLOCK>>>(d_pts, N, threshold, d_clustered, d_dirty, d_dirtyCount,
                                      d_work, d_sidx, d_sd, d_card);
        argmaxKernel<<<gridM, BLOCK>>>(N, d_clustered, d_card, d_key);
        unsigned long long key = 0;
        CUDA_CHECK(cudaMemcpy(&key, d_key, sizeof(key), cudaMemcpyDeviceToHost));
        unsigned long long gkey = 0;
        MPI_Allreduce(&key, &gkey, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (gkey == 0) break;
        const int best_seed = 0x7fffffff - static_cast<int>(gkey & 0xffffffffu);

        memberKernel<<<1, BLOCK>>>(d_pts, N, threshold, d_clustered, best_seed, d_sidx, d_sd,
                                   d_members);
        commitKernel<<<gridN, BLOCK>>>(d_members, d_clustered, d_card);
        CUDA_CHECK(cudaMemcpy(h_members.data(), d_members, sizeof(int), cudaMemcpyDeviceToHost));
        const int sz = h_members[0];
        CUDA_CHECK(cudaMemcpy(h_members.data() + 1, d_members + 1, sizeof(int) * sz,
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(h_members.begin() + 1, h_members.begin() + 1 + sz);
        clusters.push_back(std::move(cluster));
        remaining -= sz;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_pts); cudaFree(d_clustered); cudaFree(d_card); cudaFree(d_dirty);
    cudaFree(d_members); cudaFree(d_dirtyCount); cudaFree(d_work); cudaFree(d_key);
    cudaFree(d_sidx); cudaFree(d_sd);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute diameters in parallel (OpenMP), report sequentially
    std::vector<double> diam(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (long c = 0; c < static_cast<long>(clusters.size()); ++c) {
        const auto& m = clusters[c].members;
        double max_diameter = 0.0;
        for (size_t i = 0; i < m.size(); ++i) {
            for (size_t j = i + 1; j < m.size(); ++j) {
                max_diameter = std::max(max_diameter, distance(points[m[i]], points[m[j]]));
            }
        }
        diam[c] = max_diameter;
    }

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diam[c];

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

int realMain(int argc, char** argv, const int rank, const int nranks, const int local_rank) {
    if (rank != 0 && !freopen("/dev/null", "w", stdout)) return 1; // only rank 0 reports
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
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks, local_rank);
    
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
    if (printResults && rank == 0) {
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
    if (validate && rank == 0) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1, local_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    const int rc = realMain(argc, argv, rank, nranks, local_rank);
    MPI_Finalize();
    return rc;
}
