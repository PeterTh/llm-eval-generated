// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

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
// GPU kernel
//
// One thread block processes one seed at a time (persistent blocks grab seeds
// from a work counter). For a seed, only unclustered points closer than the
// threshold to the seed can ever join its cluster, so a candidate list is
// built first. Each candidate keeps its running maximum distance to the
// cluster members (incremental update gives bit-identical values to the
// original full recomputation). Each growth step selects the candidate with
// minimum (max-distance, index) -- identical to the sequential tie-breaking.
// ---------------------------------------------------------------------------
static const int BLOCK = 128;
static const int NWARPS = BLOCK / 32;

__device__ __forceinline__ double dDist(const double2 a, const double2 b) {
    const double dx = __dsub_rn(a.x, b.x);
    const double dy = __dsub_rn(a.y, b.y);
    return __dsqrt_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
}

__global__ void __launch_bounds__(BLOCK)
qtKernel(const double2* __restrict__ pts, const int N, const double thr,
         unsigned char* clustered, const int* __restrict__ seeds, const int nseeds,
         int* counter, unsigned long long* best, int* idxScratch, double* mdScratch,
         int* outMembers, int* outCount, const int finalMode) {
    int* idx = idxScratch + (size_t)blockIdx.x * N;
    double* md = mdScratch + (size_t)blockIdx.x * N;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;

    __shared__ int s_seedi, s_list, s_skip;
    __shared__ double s_v[2][NWARPS];
    __shared__ int s_j[2][NWARPS];
    __shared__ int s_p[2][NWARPS];

    bool first = true;
    for (;;) {
        if (tid == 0) {
            s_seedi = finalMode ? (first ? 0 : nseeds) : atomicAdd(counter, 1);
            s_list = 0;
        }
        __syncthreads();
        const int si = s_seedi;
        first = false;
        if (si >= nseeds) break;
        const int seed = seeds[si];
        const double2 sp = pts[seed];

        // Build candidate list
        for (int j = tid; j < N; j += BLOCK) {
            if (clustered[j] || j == seed) continue;
            const double d = dDist(pts[j], sp);
            if (d < thr) {
                const int pos = atomicAdd(&s_list, 1);
                idx[pos] = j;
                md[pos] = d;
            }
        }
        __syncthreads();
        const int n = s_list;
        if (tid == 0) {
            s_skip = 0;
            if (!finalMode) {
                const unsigned long long cur = *((volatile unsigned long long*)best);
                // upper bound on cardinality is n + 1
                if ((unsigned long long)(n + 1) < (cur >> 32)) s_skip = 1;
            } else {
                outMembers[0] = seed;
            }
        }
        __syncthreads();
        if (s_skip) {
            __syncthreads();
            continue;
        }

        int size = 1;
        bool havenew = false;
        double2 np = sp;
        for (int it = 0;; ++it) {
            double lmin = 1e300;
            int lj = INT_MAX, lp = -1;
            for (int p = tid; p < n; p += BLOCK) {
                double v = md[p];
                if (v < thr) {
                    const int j = idx[p];
                    if (havenew) {
                        const double d = dDist(pts[j], np);
                        if (d > v) { v = d; md[p] = v; }
                    }
                    if (v < thr && (v < lmin || (v == lmin && j < lj))) {
                        lmin = v; lj = j; lp = p;
                    }
                }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, lmin, o);
                const int oj = __shfl_down_sync(0xffffffffu, lj, o);
                const int op = __shfl_down_sync(0xffffffffu, lp, o);
                if (ov < lmin || (ov == lmin && oj < lj)) { lmin = ov; lj = oj; lp = op; }
            }
            const int b = it & 1;
            if (lane == 0) { s_v[b][wid] = lmin; s_j[b][wid] = lj; s_p[b][wid] = lp; }
            __syncthreads();
            double wv = s_v[b][0];
            int wj = s_j[b][0], wp = s_p[b][0];
            #pragma unroll
            for (int w = 1; w < NWARPS; ++w) {
                const double ov = s_v[b][w];
                const int oj = s_j[b][w];
                if (ov < wv || (ov == wv && oj < wj)) { wv = ov; wj = oj; wp = s_p[b][w]; }
            }
            if (wp < 0) break;
            if (tid == (wp % BLOCK)) md[wp] = 1e300;  // remove from candidates
            if (finalMode && tid == 0) outMembers[size] = wj;
            ++size;
            np = pts[wj];
            havenew = true;
        }

        if (finalMode) {
            if (tid == 0) *outCount = size;
            __syncthreads();
            for (int i = tid; i < size; i += BLOCK) clustered[outMembers[i]] = 1;
        } else if (tid == 0) {
            atomicMax(best, ((unsigned long long)size << 32) |
                                (unsigned long long)(0x7fffffff - seed));
        }
        __syncthreads();
    }
}

// Bind this rank to a GPU (by node-local rank) and create the CUDA context
static void initDevice(const int rank) {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int lrank = 0;
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_free(&local);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(lrank % ndev));
    CUDA_CHECK(cudaFree(0));
}

// Main QT clustering algorithm: seeds are distributed over MPI ranks (one GPU
// per rank); the per-round winner is found with an Allreduce, and every rank
// regenerates the winning cluster identically so all ranks stay in sync.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int nranks) {
    const int N = static_cast<int>(points.size());

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const size_t perBlock = (size_t)N * (sizeof(int) + sizeof(double));
    size_t maxBlocks = (freeB / 2) / perBlock;
    int nblocks = (int)std::min<size_t>(std::max<size_t>(maxBlocks, 1), (size_t)prop.multiProcessorCount * 8);

    double2* d_pts;
    unsigned char* d_clustered;
    int *d_seeds, *d_counter, *d_idx, *d_members, *d_count;
    double* d_md;
    unsigned long long* d_best;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_idx, perBlock / 3 * 1 * nblocks > 0 ? sizeof(int) * (size_t)N * nblocks : 1));
    CUDA_CHECK(cudaMalloc(&d_md, sizeof(double) * (size_t)N * nblocks));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pts, points.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));

    std::vector<int> unclustered(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) unclustered[i] = i;
    std::vector<char> clusteredH(N, 0);
    std::vector<int> mySeeds;
    std::vector<int> members(N);
    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        const int U = static_cast<int>(unclustered.size());
        mySeeds.clear();
        for (int i = rank; i < U; i += nranks) mySeeds.push_back(unclustered[i]);
        const int ns = static_cast<int>(mySeeds.size());

        unsigned long long localBest = 0;
        if (ns > 0) {
            CUDA_CHECK(cudaMemcpy(d_seeds, mySeeds.data(), sizeof(int) * ns, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_best, 0, sizeof(unsigned long long)));
            qtKernel<<<std::min(nblocks, ns), BLOCK>>>(d_pts, N, threshold, d_clustered, d_seeds, ns,
                                                       d_counter, d_best, d_idx, d_md,
                                                       d_members, d_count, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&localBest, d_best, sizeof(localBest), cudaMemcpyDeviceToHost));
        }
        unsigned long long globalBest = 0;
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (globalBest == 0) break;
        const int bestSeed = 0x7fffffff - static_cast<int>(globalBest & 0xffffffffULL);

        // Regenerate the winning cluster (and mark it clustered) on every rank
        CUDA_CHECK(cudaMemcpy(d_seeds, &bestSeed, sizeof(int), cudaMemcpyHostToDevice));
        qtKernel<<<1, BLOCK>>>(d_pts, N, threshold, d_clustered, d_seeds, 1, d_counter, d_best,
                               d_idx, d_md, d_members, d_count, 1);
        CUDA_CHECK(cudaGetLastError());
        int cnt = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(members.data(), d_members, sizeof(int) * cnt, cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = bestSeed;
        cluster.members.assign(members.begin(), members.begin() + cnt);
        for (int i = 0; i < cnt; ++i) clusteredH[members[i]] = 1;
        clusters.push_back(std::move(cluster));

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clusteredH](int idx) { return clusteredH[idx] != 0; }),
            unclustered.end());
    }

    cudaFree(d_pts); cudaFree(d_clustered); cudaFree(d_seeds); cudaFree(d_counter);
    cudaFree(d_best); cudaFree(d_idx); cudaFree(d_md); cudaFree(d_members); cudaFree(d_count);
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

int realMain(int argc, char** argv, int rank, int nranks);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const int rc = realMain(argc, argv, rank, nranks);
    MPI_Finalize();
    return rc;
}

int realMain(int argc, char** argv, const int rank, const int nranks) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Only rank 0 produces output
    if (rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) return 1;
    }

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
    initDevice(rank);
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    printf("MPI ranks: %d\n", nranks);
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
