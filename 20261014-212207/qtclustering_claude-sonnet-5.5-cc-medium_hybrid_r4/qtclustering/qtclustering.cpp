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
#include <climits>
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

// Calculate Euclidean distance (no FMA contraction so host and device agree bit-for-bit)
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
#ifdef __CUDA_ARCH__
    return sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
#else
    volatile double sx = dx * dx;
    volatile double sy = dy * dy;
    return std::sqrt(sx + sy);
#endif
}

#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),      \
                    __FILE__, __LINE__);                                               \
            MPI_Abort(MPI_COMM_WORLD, 2);                                              \
        }                                                                              \
    } while (0)

static const int BLOCK = 256;
static const int NWARPS = BLOCK / 32;

struct Best {
    double d;
    int idx;
};

__device__ inline bool better(double d1, int i1, double d2, int i2) {
    return d1 < d2 || (d1 == d2 && i1 < i2);
}

// Grow the candidate cluster of one seed with one thread block.
//
// The greedy rule of the original (add the unclustered point whose maximum distance
// to the current members is smallest, lowest index on ties, only if < threshold) only
// depends on the unclustered neighbours of the seed within the threshold. For those
// we keep the running maximum distance to the members, which is updated incrementally
// with every newly added member.
// Returns the cardinality; if members_out is given it receives the members in order.
__device__ int growCluster(const int seed, const Point* __restrict__ pts,
                           const unsigned char* __restrict__ clustered, const int N,
                           const double threshold, int* cidx, double* cd,
                           int* members_out, int* sh_n, Best (*red)[NWARPS]) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const Point sp = pts[seed];

    if (tid == 0) *sh_n = 0;
    __syncthreads();
    for (int c = tid; c < N; c += BLOCK) {
        if (c == seed || clustered[c]) continue;
        const double d = distance(pts[c], sp);
        if (d < threshold) {
            const int pos = atomicAdd(sh_n, 1);
            cidx[pos] = c;
            cd[pos] = d;
        }
    }
    __syncthreads();
    const int n = *sh_n;

    int card = 1;
    if (members_out && tid == 0) members_out[0] = seed;

    // Initial selection
    double lb = INFINITY;
    int li = INT_MAX;
    for (int k = tid; k < n; k += BLOCK) {
        if (better(cd[k], cidx[k], lb, li)) { lb = cd[k]; li = cidx[k]; }
    }
    int par = 0;
    while (true) {
        for (int o = 16; o > 0; o >>= 1) {
            const double od = __shfl_xor_sync(0xffffffffu, lb, o);
            const int oi = __shfl_xor_sync(0xffffffffu, li, o);
            if (better(od, oi, lb, li)) { lb = od; li = oi; }
        }
        if (lane == 0) { red[par][warp].d = lb; red[par][warp].idx = li; }
        __syncthreads();
        double bd = red[par][0].d;
        int bi = red[par][0].idx;
        for (int w = 1; w < NWARPS; ++w) {
            if (better(red[par][w].d, red[par][w].idx, bd, bi)) { bd = red[par][w].d; bi = red[par][w].idx; }
        }
        par ^= 1;
        if (!(bd < threshold)) break;
        const int m = bi;
        if (members_out && tid == 0) members_out[card] = m;
        ++card;
        const Point mp = pts[m];

        lb = INFINITY;
        li = INT_MAX;
        for (int k = tid; k < n; k += BLOCK) {
            const int c = cidx[k];
            double d = cd[k];
            if (c == m) { cd[k] = INFINITY; continue; }
            if (!(d < threshold)) continue;
            const double nd = distance(pts[c], mp);
            if (nd > d) d = nd;
            if (!(d < threshold)) d = INFINITY;
            cd[k] = d;
            if (better(d, c, lb, li)) { lb = d; li = c; }
        }
    }
    __syncthreads();  // protect shared state before the block is reused
    return card;
}

struct Ctrl {
    unsigned long long key;
    int work_count;
    int work_next;
    int member_count;
};

// Per-seed cached cardinality (0 = needs recomputation); owned seeds are rank, rank+P, ...
// Collect owned seeds that need (re)computation. A cached cardinality stays valid unless a
// newly clustered point lies within the threshold of the seed.
__global__ void invalidateKernel(const Point* __restrict__ pts, const unsigned char* clustered,
                                 int* card, int* work, Ctrl* ctrl, const int* members,
                                 const int rank, const int P, const int N,
                                 const double threshold) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = rank + j * P;
    if (i >= N || clustered[i]) return;
    bool need = card[i] == 0;
    if (!need) {
        const int cnt = ctrl->member_count;
        const Point p = pts[i];
        for (int t = 0; t < cnt; ++t) {
            if (distance(p, pts[members[t]]) < threshold) { need = true; break; }
        }
        if (need) card[i] = 0;
    }
    if (need) work[atomicAdd(&ctrl->work_count, 1)] = i;
}

__global__ void computeKernel(const Point* __restrict__ pts, const unsigned char* clustered,
                              int* card, const int* work, Ctrl* ctrl, int* scratch_i,
                              double* scratch_d, const int N, const double threshold) {
    __shared__ int sh_n;
    __shared__ int sh_work;
    __shared__ Best red[2][NWARPS];
    int* cidx = scratch_i + (size_t)blockIdx.x * N;
    double* cd = scratch_d + (size_t)blockIdx.x * N;
    while (true) {
        if (threadIdx.x == 0) sh_work = atomicAdd(&ctrl->work_next, 1);
        __syncthreads();
        const int w = sh_work;
        __syncthreads();
        if (w >= ctrl->work_count) break;
        const int seed = work[w];
        const int c = growCluster(seed, pts, clustered, N, threshold, cidx, cd, nullptr, &sh_n, red);
        if (threadIdx.x == 0) card[seed] = c;
    }
}

__global__ void argmaxKernel(const int* card, const unsigned char* clustered, Ctrl* ctrl,
                             const int rank, const int P, const int N) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = rank + j * P;
    if (i >= N || clustered[i]) return;
    const unsigned long long key = ((unsigned long long)(unsigned)card[i] << 32) |
                                   (0xFFFFFFFFull - (unsigned)i);
    atomicMax(&ctrl->key, key);
}

// Recompute the winning cluster, record its members and mark them as clustered.
__global__ void winnerKernel(const Point* __restrict__ pts, unsigned char* clustered,
                             Ctrl* ctrl, int* members, int* scratch_i, double* scratch_d,
                             const int seed, const int N, const double threshold) {
    __shared__ int sh_n;
    __shared__ Best red[2][NWARPS];
    const int c = growCluster(seed, pts, clustered, N, threshold, scratch_i, scratch_d,
                              members, &sh_n, red);
    if (threadIdx.x == 0) ctrl->member_count = c;
    __syncthreads();
    for (int t = threadIdx.x; t < c; t += BLOCK) clustered[members[t]] = 1;
}

// Main QT clustering algorithm. Seeds are distributed over MPI ranks (one GPU each);
// every rank keeps the full clustering state and the winner is agreed upon by an allreduce.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int nranks) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    Point* d_pts;
    unsigned char* d_clustered;
    int *d_card, *d_work, *d_members, *d_scratch_i;
    double* d_scratch_d;
    Ctrl* d_ctrl;
    const int nOwned = (N - rank + nranks - 1) / nranks;

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    size_t nblocks = (size_t)prop.multiProcessorCount * 4;
    const size_t per_block = (size_t)N * (sizeof(int) + sizeof(double));
    const size_t cap = (size_t)1 << 30;
    if (nblocks * per_block > cap) nblocks = std::max<size_t>(1, cap / per_block);
    if (nblocks > (size_t)std::max(nOwned, 1)) nblocks = std::max(nOwned, 1);

    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(int) * std::max(nOwned, 1)));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_scratch_i, (size_t)N * sizeof(int) * nblocks));
    CUDA_CHECK(cudaMalloc(&d_scratch_d, (size_t)N * sizeof(double) * nblocks));
    CUDA_CHECK(cudaMalloc(&d_ctrl, sizeof(Ctrl)));
    CUDA_CHECK(cudaMemcpy(d_pts, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    CUDA_CHECK(cudaMemset(d_card, 0, sizeof(int) * N));
    CUDA_CHECK(cudaMemset(d_ctrl, 0, sizeof(Ctrl)));

    const int tb = 128;
    const int ownGrid = (nOwned + tb - 1) / tb;
    int remaining = N;
    std::vector<int> h_members(N);

    while (remaining > 0) {
        // 1. refresh stale cardinalities
        CUDA_CHECK(cudaMemsetAsync(&d_ctrl->work_count, 0, 2 * sizeof(int), 0));
        if (ownGrid > 0)
            invalidateKernel<<<ownGrid, tb>>>(d_pts, d_clustered, d_card, d_work, d_ctrl,
                                              d_members, rank, nranks, N, threshold);
        if (nOwned > 0)
            computeKernel<<<(unsigned)nblocks, BLOCK>>>(d_pts, d_clustered, d_card, d_work, d_ctrl,
                                                         d_scratch_i, d_scratch_d, N, threshold);
        // 2. local best, then global best (max cardinality, lowest seed index)
        CUDA_CHECK(cudaMemsetAsync(&d_ctrl->key, 0, sizeof(unsigned long long), 0));
        if (ownGrid > 0)
            argmaxKernel<<<ownGrid, tb>>>(d_card, d_clustered, d_ctrl, rank, nranks, N);
        unsigned long long key = 0, gkey = 0;
        CUDA_CHECK(cudaMemcpy(&key, &d_ctrl->key, sizeof(key), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&key, &gkey, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if ((gkey >> 32) == 0) break;
        const int seed = static_cast<int>(0xFFFFFFFFull - (gkey & 0xFFFFFFFFull));

        // 3. materialise the winning cluster on every rank
        winnerKernel<<<1, BLOCK>>>(d_pts, d_clustered, d_ctrl, d_members, d_scratch_i,
                                   d_scratch_d, seed, N, threshold);
        int cnt = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, &d_ctrl->member_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_members.data(), d_members, sizeof(int) * cnt, cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.assign(h_members.begin(), h_members.begin() + cnt);
        clusters.push_back(std::move(cluster));
        remaining -= cnt;
    }

    cudaFree(d_pts); cudaFree(d_clustered); cudaFree(d_card); cudaFree(d_work);
    cudaFree(d_members); cudaFree(d_scratch_i); cudaFree(d_scratch_d); cudaFree(d_ctrl);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties.
// Clusters are distributed over MPI ranks and threads; only rank 0 prints.
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold, int rank, int nranks) {
    bool valid = true;
    const int nc = static_cast<int>(clusters.size());

    if (rank == 0) printf("Validating clusters:\n");

    std::vector<double> diam(nc, 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = rank; c < nc; c += nranks) {
        const auto& m = clusters[c].members;
        double max_diameter = 0.0;
        for (size_t i = 0; i < m.size(); ++i) {
            for (size_t j = i + 1; j < m.size(); ++j) {
                max_diameter = std::max(max_diameter, distance(points[m[i]], points[m[j]]));
            }
        }
        diam[c] = max_diameter;
    }
    if (nc > 0)
        MPI_Allreduce(MPI_IN_PLACE, diam.data(), nc, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for (int c = 0; c < nc; ++c) {
        if (rank == 0 && c < 10) { // Print first 10 clusters
            printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n",
                   c, clusters[c].members.size(), clusters[c].seed_point, diam[c]);
        }
        if (diam[c] > threshold * 1.001) { // Allow small numerical error
            if (rank == 0)
                printf("ERROR: Cluster %d has diameter %.4f > threshold %.4f\n",
                       c, diam[c], threshold);
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (int c = 0; c < nc; ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                if (rank == 0)
                    printf("ERROR: Point %d appears in multiple clusters (%d and %d)\n",
                           member, membership[member], c);
                valid = false;
            }
            membership[member] = c;
        }
    }

    int clustered_count = 0;
    #pragma omp parallel for reduction(+ : clustered_count)
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }

    if (rank == 0)
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

static int run(int argc, char** argv, int rank, int nranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (identical on every rank)
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

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", (long)cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        // Calculate statistics and performance metrics
        long long total_clustered = 0;
        int max_cluster_size = 0;
        const int nc = static_cast<int>(clusters.size());
        #pragma omp parallel for reduction(+ : total_clustered) reduction(max : max_cluster_size)
        for (int i = 0; i < nc; ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %lld / %d (%.1f%%)\n",
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
            std::vector<double> membershipData(num_points, -1.0);
            #pragma omp parallel for
            for (int c = 0; c < nc; ++c) {
                for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                    membershipData[clusters[c].members[i]] = static_cast<double>(c);
                }
            }
            print_results(membershipData, "ClusterMembership");
        }
    }

    // Validation (collective)
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold, rank, nranks);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // One GPU per rank (round-robin over the GPUs of the node)
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int lrank;
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_free(&local);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(lrank % ndev));

    const int ret = run(argc, argv, rank, nranks);
    MPI_Finalize();
    return ret;
}
