// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA
//
// This preserves the original greedy QT clustering semantics:
//  - At each iteration, evaluate candidate clusters for every unclustered seed,
//    choose the maximum-cardinality cluster (ties broken by smallest seed index).
//  - Candidate cluster growth repeatedly adds the point with the minimal induced
//    cluster diameter (< threshold), where diameter is max pairwise distance.
//
// Parallelization:
//  - MPI: distribute seed evaluations across ranks each iteration.
//  - CUDA: accelerate the dominant operation inside candidate growth
//          (find closest admissible point) on each rank.
//  - OpenMP: parallelize remaining host-side metric/validation loops.

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

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

static inline void mpi_abort_if(bool cond, const char* msg) {
    if (cond) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            fprintf(stderr, "%s\n", msg);
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            fprintf(stderr, "%s\n", msg);
            std::abort();
        }
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

// Generate synthetic 2D point data in clusters (rank 0 only, then broadcast).
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

        // Prevent stalling when N is very small; keep original behavior otherwise.
        if (N < 30 && group_cnt < 1) group_cnt = 1;

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

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------- CUDA accelerated closest-point search ----------------

struct DeviceContext {
    bool available = false;
    int device = 0;
    int N = 0;

    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;   // N
    unsigned char* d_in_cluster = nullptr;  // N (per candidate-cluster generation)
    int* d_members = nullptr;              // N

    double* d_candidate_max = nullptr;     // N
    double* d_block_min = nullptr;         // ceil(N / REDUCE_THREADS)
    int* d_block_idx = nullptr;

    std::vector<double> h_block_min;
    std::vector<int> h_block_idx;
};

__global__ void set_flag_kernel(unsigned char* flags, int idx) {
    flags[idx] = 1;
}

__global__ void compute_candidate_maxdist_kernel(
    const double2* __restrict__ pts,
    const unsigned char* __restrict__ clustered,
    const unsigned char* __restrict__ in_cluster,
    const int* __restrict__ members,
    int members_count,
    double* __restrict__ out_candidate_max,
    int N,
    double threshold)
{
    const int cand = (int)blockIdx.x;
    if (cand >= N) return;

    if (clustered[cand] || in_cluster[cand]) {
        out_candidate_max[cand] = INFINITY;
        return;
    }

    const double2 pc = pts[cand];
    double local_max = 0.0;

    for (int i = (int)threadIdx.x; i < members_count; i += (int)blockDim.x) {
        const int m = members[i];
        const double2 pm = pts[m];
        const double dx = pc.x - pm.x;
        const double dy = pc.y - pm.y;
        const double d2 = dx * dx + dy * dy;
        const double dist = sqrt(d2);
        local_max = (dist > local_max) ? dist : local_max;
    }

    __shared__ double smax[256];
    const int tid = (int)threadIdx.x;
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double v = smax[tid + stride];
            smax[tid] = (v > smax[tid]) ? v : smax[tid];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double maxd = smax[0];
        out_candidate_max[cand] = (maxd < threshold) ? maxd : INFINITY;
    }
}

__global__ void reduce_min_kernel(
    const double* __restrict__ in,
    int N,
    double* __restrict__ out_val,
    int* __restrict__ out_idx)
{
    const int tid = (int)threadIdx.x;
    const int i = (int)blockIdx.x * (int)blockDim.x + tid;

    double v = (i < N) ? in[i] : INFINITY;
    int idx = (i < N) ? i : -1;

    __shared__ double sval[256];
    __shared__ int sidx[256];
    sval[tid] = v;
    sidx[tid] = idx;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double v2 = sval[tid + stride];
            const int idx2 = sidx[tid + stride];
            const double v1 = sval[tid];
            const int idx1 = sidx[tid];

            if (v2 < v1 || (v2 == v1 && idx2 >= 0 && (idx1 < 0 || idx2 < idx1))) {
                sval[tid] = v2;
                sidx[tid] = idx2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_val[blockIdx.x] = sval[0];
        out_idx[blockIdx.x] = sidx[0];
    }
}

static DeviceContext initDeviceContext(const std::vector<Point>& points, int rank) {
    DeviceContext ctx;
    ctx.N = (int)points.size();

    int deviceCount = 0;
    cudaError_t ce = cudaGetDeviceCount(&deviceCount);
    if (ce != cudaSuccess || deviceCount <= 0) {
        ctx.available = false;
        return ctx;
    }

    ctx.available = true;
    ctx.device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(ctx.device));
    CUDA_CHECK(cudaFree(0));

    std::vector<double2> h_pts(ctx.N);
    for (int i = 0; i < ctx.N; ++i) {
        h_pts[i] = make_double2(points[i].x, points[i].y);
    }

    CUDA_CHECK(cudaMalloc((void**)&ctx.d_points, sizeof(double2) * ctx.N));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_clustered, sizeof(unsigned char) * ctx.N));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_in_cluster, sizeof(unsigned char) * ctx.N));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_members, sizeof(int) * ctx.N));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_candidate_max, sizeof(double) * ctx.N));

    const int REDUCE_THREADS = 256;
    const int reduceBlocks = (ctx.N + REDUCE_THREADS - 1) / REDUCE_THREADS;
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_block_min, sizeof(double) * reduceBlocks));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_block_idx, sizeof(int) * reduceBlocks));
    ctx.h_block_min.resize(reduceBlocks);
    ctx.h_block_idx.resize(reduceBlocks);

    CUDA_CHECK(cudaMemcpy(ctx.d_points, h_pts.data(), sizeof(double2) * ctx.N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx.d_clustered, 0, sizeof(unsigned char) * ctx.N));
    CUDA_CHECK(cudaMemset(ctx.d_in_cluster, 0, sizeof(unsigned char) * ctx.N));

    return ctx;
}

static void freeDeviceContext(DeviceContext& ctx) {
    if (!ctx.available) return;
    cudaFree(ctx.d_points);
    cudaFree(ctx.d_clustered);
    cudaFree(ctx.d_in_cluster);
    cudaFree(ctx.d_members);
    cudaFree(ctx.d_candidate_max);
    cudaFree(ctx.d_block_min);
    cudaFree(ctx.d_block_idx);
    ctx = DeviceContext{};
}

static int findClosestPointGPU(DeviceContext& ctx, int members_count, double threshold) {
    const int N = ctx.N;
    const int THREADS = 256;

    compute_candidate_maxdist_kernel<<<(unsigned)N, THREADS>>>(
        ctx.d_points,
        ctx.d_clustered,
        ctx.d_in_cluster,
        ctx.d_members,
        members_count,
        ctx.d_candidate_max,
        N,
        threshold);
    CUDA_CHECK(cudaGetLastError());

    const int reduceBlocks = (N + THREADS - 1) / THREADS;
    reduce_min_kernel<<<(unsigned)reduceBlocks, THREADS>>>(ctx.d_candidate_max, N, ctx.d_block_min, ctx.d_block_idx);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(ctx.h_block_min.data(), ctx.d_block_min, sizeof(double) * reduceBlocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_block_idx.data(), ctx.d_block_idx, sizeof(int) * reduceBlocks, cudaMemcpyDeviceToHost));

    double best = std::numeric_limits<double>::infinity();
    int best_idx = -1;
    for (int b = 0; b < reduceBlocks; ++b) {
        const double v = ctx.h_block_min[b];
        const int idx = ctx.h_block_idx[b];
        if (v < best || (v == best && idx >= 0 && (best_idx < 0 || idx < best_idx))) {
            best = v;
            best_idx = idx;
        }
    }

    return (best_idx >= 0 && std::isfinite(best)) ? best_idx : -1;
}

static int generateCandidateClusterGPU(
    const int seed_point,
    DeviceContext& ctx,
    const double threshold,
    const int point_count,
    std::vector<int>* cluster_members)
{
    (void)point_count;

    CUDA_CHECK(cudaMemset(ctx.d_in_cluster, 0, sizeof(unsigned char) * ctx.N));
    set_flag_kernel<<<1, 1>>>(ctx.d_in_cluster, seed_point);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(ctx.d_members, &seed_point, sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> members;
    members.reserve(ctx.N);
    members.push_back(seed_point);

    while ((int)members.size() < ctx.N) {
        const int closest = findClosestPointGPU(ctx, (int)members.size(), threshold);
        if (closest < 0) break;

        members.push_back(closest);
        set_flag_kernel<<<1, 1>>>(ctx.d_in_cluster, closest);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(ctx.d_members + ((int)members.size() - 1), &closest, sizeof(int), cudaMemcpyHostToDevice));
    }

    if (cluster_members) *cluster_members = std::move(members);
    return cluster_members ? (int)cluster_members->size() : (int)members.size();
}

// CPU fallback (also used when no CUDA device is available)
static int findClosestPointCPU(const std::vector<int>& cluster_members,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<unsigned char>& in_cluster,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }

        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

static int generateCandidateClusterCPU(const int seed_point,
                                      const std::vector<unsigned char>& clustered,
                                      const std::vector<Point>& points,
                                      const double threshold,
                                      const int point_count,
                                      std::vector<int>* cluster_members) {
    std::vector<unsigned char> in_cluster(point_count, 0);
    std::vector<int> members;

    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    while ((int)members.size() < point_count) {
        const int closest = findClosestPointCPU(members, clustered, in_cluster, points, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = 1;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return (int)members.size();
}

// ---------------- MPI + CUDA + OpenMP hybrid QT clustering ----------------

static std::vector<Cluster> qtClusteringHybridMPI(const std::vector<Point>& points,
                                                 const double threshold,
                                                 int rank,
                                                 int size,
                                                 DeviceContext& ctx) {
    const int N = (int)points.size();

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    std::vector<Cluster> clusters;

    while (!unclustered_indices.empty()) {
        int local_best_card = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        if (ctx.available) {
            CUDA_CHECK(cudaMemcpy(ctx.d_clustered, clustered.data(), sizeof(unsigned char) * (size_t)N, cudaMemcpyHostToDevice));
        }

        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            if ((seed % size) != rank) continue;

            std::vector<int> candidate_members;
            int card = 0;
            if (ctx.available) {
                card = generateCandidateClusterGPU(seed, ctx, threshold, N, &candidate_members);
            } else {
                card = generateCandidateClusterCPU(seed, clustered, points, threshold, N, &candidate_members);
            }

            if (card > local_best_card || (card == local_best_card && seed >= 0 && (local_best_seed < 0 || seed < local_best_seed))) {
                local_best_card = card;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        uint64_t local_key = 0;
        if (local_best_seed >= 0 && local_best_card > 0) {
            const uint32_t seed_u = (uint32_t)local_best_seed;
            local_key = (uint64_t)((uint32_t)local_best_card) << 32;
            local_key |= (uint64_t)(0xFFFFFFFFu - seed_u);
        }

        uint64_t global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

        const int global_card = (int)(global_key >> 32);
        if (global_card <= 0) break;

        const uint32_t inv_seed = (uint32_t)(global_key & 0xFFFFFFFFu);
        const int best_seed = (int)(0xFFFFFFFFu - inv_seed);
        const int root = (size > 0) ? (best_seed % size) : 0;

        int member_count = 0;
        std::vector<int> best_members;
        if (rank == root) {
            member_count = (int)local_best_members.size();
            best_members = local_best_members;
        }

        MPI_Bcast(&member_count, 1, MPI_INT, root, MPI_COMM_WORLD);
        mpi_abort_if(member_count <= 0, "Invalid cluster broadcast size");

        if (rank != root) best_members.resize((size_t)member_count);
        MPI_Bcast(best_members.data(), member_count, MPI_INT, root, MPI_COMM_WORLD);

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_members;
            clusters.push_back(std::move(cluster));
        }

        for (int m : best_members) clustered[m] = 1;

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties (rank 0 only)
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = (int)c;
        }
    }

    int clustered_count = 0;
#pragma omp parallel for reduction(+ : clustered_count)
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
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark (MPI ranks=%d, OMP threads=%d)\n", size, omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);

    // Broadcast points to all ranks for deterministic semantics.
    std::vector<double> packed(2ull * (size_t)num_points);
    if (rank == 0) {
        for (int i = 0; i < num_points; ++i) {
            packed[2ull * (size_t)i + 0] = points[i].x;
            packed[2ull * (size_t)i + 1] = points[i].y;
        }
    }
    MPI_Bcast(packed.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (int i = 0; i < num_points; ++i) {
            points[i].x = packed[2ull * (size_t)i + 0];
            points[i].y = packed[2ull * (size_t)i + 1];
        }
    }

    DeviceContext ctx = initDeviceContext(points, rank);
    if (rank == 0) {
        printf("CUDA: %s\n", ctx.available ? "enabled" : "no device (CPU fallback)" );
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringHybridMPI(points, threshold, rank, size, ctx);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();

    freeDeviceContext(ctx);

    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        auto cluster_time = std::chrono::milliseconds(max_cluster_time);
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
#pragma omp parallel for reduction(+ : total_clustered) reduction(max : max_cluster_size)
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = (int)clusters[i].members.size();
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }

        const double avg_cluster_size = clusters.empty() ? 0.0 : (double)total_clustered / (double)clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * (double)total_clustered / (double)num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = (time_sec > 0) ? (clusters.size() / time_sec) : 0.0;
        const double points_per_sec = (time_sec > 0) ? (num_points / time_sec) : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve((size_t)num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                    membership[clusters[c].members[i]] = (int)c;
                }
            }
            for (int m : membership) membershipData.push_back((double)m);
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool ok = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
