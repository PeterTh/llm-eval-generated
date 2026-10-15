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

#include <mpi.h>
#include <omp.h>
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
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

static const int BLOCK = 256;
static const double INF_D = std::numeric_limits<double>::infinity();

// Exact (non-fused) distance so results match the sequential reference
__device__ __forceinline__ double dist_dev(double x1, double y1, double x2, double y2) {
    const double dx = __dsub_rn(x1, x2);
    const double dy = __dsub_rn(y1, y2);
    return __dsqrt_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
}

// Persistent-block kernel. Each block pulls seeds (owned indices) from a work
// queue and greedily grows the candidate cluster, maintaining for every point
// the maximum distance to the current cluster members (incremental update).
//  - card:  per-owned-seed cardinality output (nullable)
//  - bits:  per-owned-seed membership bitset rows (nullable)
//  - outMembers/outCount: member list output (nullable)
__global__ void growKernel(const double2* __restrict__ pts,
                           const unsigned char* __restrict__ clustered,
                           double* __restrict__ work,
                           const int* __restrict__ seedList,
                           const int* __restrict__ listCount,
                           int* __restrict__ ticket,
                           int* __restrict__ card,
                           unsigned long long* __restrict__ bits,
                           int* __restrict__ outMembers,
                           int* __restrict__ outCount,
                           const int N, const double threshold,
                           const int rank, const int nranks) {
    __shared__ int s_ticket;
    __shared__ int s_next;
    __shared__ double s_val[BLOCK / 32];
    __shared__ int s_idx[BLOCK / 32];

    double* md = work + static_cast<size_t>(blockIdx.x) * N;
    const size_t words = (static_cast<size_t>(N) + 63) / 64;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;

    while (true) {
        if (tid == 0) s_ticket = atomicAdd(ticket, 1);
        __syncthreads();
        const int t = s_ticket;
        __syncthreads();
        if (t >= *listCount) break;
        const int oi = seedList[t];
        const int seed = rank + oi * nranks;
        unsigned long long* row = bits ? bits + static_cast<size_t>(oi) * words : nullptr;

        for (int c = tid; c < N; c += BLOCK) md[c] = clustered[c] ? INF_D : 0.0;
        if (row) for (size_t w = tid; w < words; w += BLOCK) row[w] = 0ULL;
        __syncthreads();

        int member = seed;
        int count = 0;
        while (true) {
            if (tid == 0) {
                if (row) row[member >> 6] |= 1ULL << (member & 63);
                if (outMembers) outMembers[count] = member;
            }
            count++;
            const double2 pm = pts[member];
            double bv = INF_D;
            int bi = -1;
            for (int c = tid; c < N; c += BLOCK) {
                double m = md[c];
                if (m == INF_D) continue;
                if (c == member) { md[c] = INF_D; continue; }
                const double2 p = pts[c];
                const double d = dist_dev(p.x, p.y, pm.x, pm.y);
                if (d > m) { m = d; md[c] = m; }
                if (m < threshold && m < bv) { bv = m; bi = c; }
            }
            // block argmin (ties -> lowest index)
            for (int o = 16; o > 0; o >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, bv, o);
                const int oidx = __shfl_down_sync(0xffffffffu, bi, o);
                if (oidx >= 0 && (bi < 0 || ov < bv || (ov == bv && oidx < bi))) { bv = ov; bi = oidx; }
            }
            if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
            __syncthreads();
            if (tid == 0) {
                double v = s_val[0]; int idx = s_idx[0];
                for (int w = 1; w < BLOCK / 32; ++w) {
                    if (s_idx[w] >= 0 && (idx < 0 || s_val[w] < v || (s_val[w] == v && s_idx[w] < idx))) {
                        v = s_val[w]; idx = s_idx[w];
                    }
                }
                s_next = idx;
            }
            __syncthreads();
            member = s_next;
            __syncthreads();
            if (member < 0) break;
        }
        if (tid == 0) {
            if (card) card[oi] = count;
            if (outCount) *outCount = count;
        }
        __syncthreads();
    }
}

// Mark newly clustered points
__global__ void markClusteredKernel(unsigned char* clustered, const int* members, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) clustered[members[i]] = 1;
}

// Decide which owned seeds need (re)computation: those never computed, or whose
// cached candidate cluster intersects the newly extracted cluster. Removing
// points that were never selected cannot change a greedy candidate cluster.
__global__ void markStaleKernel(const unsigned char* __restrict__ clustered,
                                const unsigned long long* __restrict__ bits,
                                const int* __restrict__ newMembers, const int K,
                                int* __restrict__ card, int* __restrict__ seedList,
                                int* __restrict__ listCount,
                                const int nOwn, const int N, const int rank, const int nranks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nOwn) return;
    const int g = rank + i * nranks;
    if (clustered[g]) { card[i] = -2; return; }
    bool need = (card[i] == -1) || (bits == nullptr);
    if (!need) {
        const size_t words = (static_cast<size_t>(N) + 63) / 64;
        const unsigned long long* row = bits + static_cast<size_t>(i) * words;
        for (int k = 0; k < K; ++k) {
            const int m = newMembers[k];
            if ((row[m >> 6] >> (m & 63)) & 1ULL) { need = true; break; }
        }
    }
    if (need) {
        card[i] = -1;
        seedList[atomicAdd(listCount, 1)] = i;
    }
}

// Main QT clustering algorithm (MPI over seeds, CUDA for cluster growth,
// OpenMP for host-side reductions)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int nranks) {
    const int N = static_cast<int>(points.size());
    const int nOwn = (N - rank + nranks - 1) / nranks;  // seeds g = rank + i*nranks

    // Device selection: one GPU per local rank (round robin)
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int lrank = 0, lsize = 1;
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_size(local, &lsize);
    MPI_Comm_free(&local);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { fprintf(stderr, "No CUDA device found\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(lrank % ndev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, lrank % ndev));

    // Ranks sharing a GPU split its free memory
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const int perGpu = std::max(1, (lsize + ndev - 1) / ndev);
    const size_t budget = freeB / 8 * 7 / perGpu;

    const size_t words = (static_cast<size_t>(N) + 63) / 64;
    // Bitset cache enables incremental re-evaluation if it fits in ~half the memory
    const size_t bitsBytes = static_cast<size_t>(std::max(nOwn, 1)) * words * 8;
    const bool useCache = bitsBytes <= budget / 2;
    const size_t remain = budget - (useCache ? bitsBytes : 0);
    // Work buffers: md[N] doubles per resident block
    int nblocks = prop.multiProcessorCount * 4;
    const size_t perBlock = static_cast<size_t>(N) * sizeof(double);
    const size_t maxBlocks = std::max<size_t>(1, (remain / 2) / perBlock);
    if (static_cast<size_t>(nblocks) > maxBlocks) nblocks = static_cast<int>(maxBlocks);

    std::vector<double2> hp(N);
    for (int i = 0; i < N; ++i) hp[i] = make_double2(points[i].x, points[i].y);

    double2* d_pts; unsigned char* d_clustered; double* d_work;
    int *d_card, *d_list, *d_cnt, *d_ticket, *d_newm, *d_outm, *d_outc;
    unsigned long long* d_bits = nullptr;
    const int ownAlloc = std::max(nOwn, 1);
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_work, perBlock * nblocks));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * ownAlloc));
    CUDA_CHECK(cudaMalloc(&d_list, sizeof(int) * ownAlloc));
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_newm, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_outm, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_outc, sizeof(int)));
    if (useCache) CUDA_CHECK(cudaMalloc(&d_bits, bitsBytes));
    CUDA_CHECK(cudaMemcpy(d_pts, hp.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    {
        std::vector<int> init(ownAlloc, -1);
        CUDA_CHECK(cudaMemcpy(d_card, init.data(), sizeof(int) * ownAlloc, cudaMemcpyHostToDevice));
    }

    std::vector<int> hcard(ownAlloc);
    std::vector<Cluster> clusters;
    int remaining = N;
    int K = 0;
    std::vector<int> members;

    while (remaining > 0) {
        // Refresh stale seeds
        CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_ticket, 0, sizeof(int)));
        if (nOwn > 0) {
            markStaleKernel<<<(nOwn + 255) / 256, 256>>>(d_clustered, d_bits, d_newm, K, d_card,
                                                         d_list, d_cnt, nOwn, N, rank, nranks);
            growKernel<<<nblocks, BLOCK>>>(d_pts, d_clustered, d_work, d_list, d_cnt, d_ticket,
                                           d_card, d_bits, nullptr, nullptr, N, threshold,
                                           rank, nranks);
            CUDA_CHECK(cudaMemcpy(hcard.data(), d_card, sizeof(int) * nOwn, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaGetLastError());

        // Local best: max cardinality, lowest seed index on ties
        long long bestKey = -1;
        #pragma omp parallel
        {
            long long lk = -1;
            #pragma omp for nowait
            for (int i = 0; i < nOwn; ++i) {
                if (hcard[i] < 0) continue;
                const long long g = rank + static_cast<long long>(i) * nranks;
                const long long key = (static_cast<long long>(hcard[i]) << 32) | (N - 1 - g);
                lk = std::max(lk, key);
            }
            #pragma omp critical
            bestKey = std::max(bestKey, lk);
        }
        long long gKey = -1;
        MPI_Allreduce(&bestKey, &gKey, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (gKey < 0) break;
        const int bestCard = static_cast<int>(gKey >> 32);
        const int bestSeed = N - 1 - static_cast<int>(gKey & 0xffffffffLL);
        const int owner = bestSeed % nranks;

        members.assign(bestCard, 0);
        if (rank == owner) {
            // Re-grow the winning seed to obtain its member list
            const int oi = bestSeed / nranks;
            const int one = 1;
            CUDA_CHECK(cudaMemcpy(d_list, &oi, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cnt, &one, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_ticket, 0, sizeof(int)));
            growKernel<<<1, BLOCK>>>(d_pts, d_clustered, d_work, d_list, d_cnt, d_ticket,
                                     nullptr, nullptr, d_outm, d_outc, N, threshold, rank, nranks);
            CUDA_CHECK(cudaMemcpy(members.data(), d_outm, sizeof(int) * bestCard,
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), bestCard, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = bestSeed;
        cluster.members = members;
        clusters.push_back(std::move(cluster));

        K = bestCard;
        remaining -= K;
        CUDA_CHECK(cudaMemcpy(d_newm, members.data(), sizeof(int) * K, cudaMemcpyHostToDevice));
        markClusteredKernel<<<(K + 255) / 256, 256>>>(d_clustered, d_newm, K);
    }

    cudaFree(d_pts); cudaFree(d_clustered); cudaFree(d_work); cudaFree(d_card);
    cudaFree(d_list); cudaFree(d_cnt); cudaFree(d_ticket); cudaFree(d_newm);
    cudaFree(d_outm); cudaFree(d_outc);
    if (d_bits) cudaFree(d_bits);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute diameters in parallel
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (long c = 0; c < static_cast<long>(clusters.size()); ++c) {
        const auto& m = clusters[c].members;
        double md = 0.0;
        for (size_t i = 0; i < m.size(); ++i) {
            for (size_t j = i + 1; j < m.size(); ++j) {
                md = std::max(md, distance(points[m[i]], points[m[j]]));
            }
        }
        diameters[c] = md;
    }
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
        
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                              num_points, threshold);
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
    
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
