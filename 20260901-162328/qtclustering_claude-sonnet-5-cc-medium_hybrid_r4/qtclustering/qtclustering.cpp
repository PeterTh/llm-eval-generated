// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - MPI distributes the set of candidate seed points (one per unclustered
//    point) across ranks/nodes.
//  - OpenMP, within a rank, drives one host thread per locally-visible GPU
//    so that all accelerators on a node are kept busy concurrently.
//  - CUDA performs the actual candidate-cluster growth: one CUDA block
//    grows one candidate cluster, using an in-block parallel reduction to
//    find (each growth step) the closest remaining point, exactly
//    reproducing the sequential algorithm's tie-breaking semantics.
// The best candidate cluster of a round is found via a deterministic,
// associative (max cardinality, min seed) reduction across CUDA blocks,
// GPUs, and MPI ranks, so results are bit-for-bit equivalent to the
// original sequential algorithm regardless of how work is partitioned.

#include <algorithm>
#include <cfloat>
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
static const int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_err));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                        \
    } while (0)

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
// CUDA kernels
// ---------------------------------------------------------------------------

// Compute the full pairwise distance matrix once (points never change).
__global__ void computeDistMatrixKernel(const double* __restrict__ px,
                                         const double* __restrict__ py,
                                         double* __restrict__ dist,
                                         int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        double dx = px[i] - px[j];
        double dy = py[i] - py[j];
        dist[static_cast<size_t>(i) * N + j] = sqrt(dx * dx + dy * dy);
    }
}

// Grow one candidate cluster per CUDA block, starting from seeds[blockIdx.x].
// Reproduces the sequential "closest point" tie-break exactly: among all
// candidates whose max-distance-to-cluster is < threshold, pick the one
// with the smallest max-distance; ties broken by smallest point index.
__global__ void growClusterKernel(const double* __restrict__ dist,
                                   const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ seeds,
                                   int num_seeds,
                                   int N,
                                   double threshold,
                                   unsigned char* __restrict__ in_cluster_scratch,
                                   int* __restrict__ members_scratch,
                                   int* __restrict__ out_cardinality) {
    int b = blockIdx.x;
    if (b >= num_seeds) return;

    unsigned char* in_cluster = in_cluster_scratch + static_cast<size_t>(b) * N;
    int* members = members_scratch + static_cast<size_t>(b) * N;

    __shared__ int s_count;
    __shared__ int s_best_idx;
    __shared__ double s_best_val;
    __shared__ double sh_val[BLOCK_SIZE];
    __shared__ int sh_idx[BLOCK_SIZE];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        in_cluster[i] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int seed = seeds[b];
        in_cluster[seed] = 1;
        members[0] = seed;
        s_count = 1;
    }
    __syncthreads();

    while (s_count < N) {
        int count = s_count;

        double local_best_val = DBL_MAX;
        int local_best_idx = -1;

        for (int cand = threadIdx.x; cand < N; cand += blockDim.x) {
            if (clustered[cand] || in_cluster[cand]) continue;

            const double* row = dist + static_cast<size_t>(cand) * N;
            double max_dist = 0.0;
            for (int m = 0; m < count; ++m) {
                double d = row[members[m]];
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold &&
                (max_dist < local_best_val ||
                 (max_dist == local_best_val && cand < local_best_idx))) {
                local_best_val = max_dist;
                local_best_idx = cand;
            }
        }

        sh_val[threadIdx.x] = local_best_val;
        sh_idx[threadIdx.x] = local_best_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                double ov = sh_val[threadIdx.x + stride];
                int oi = sh_idx[threadIdx.x + stride];
                double mv = sh_val[threadIdx.x];
                int mi = sh_idx[threadIdx.x];
                if (oi >= 0 && (mi < 0 || ov < mv || (ov == mv && oi < mi))) {
                    sh_val[threadIdx.x] = ov;
                    sh_idx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            s_best_idx = sh_idx[0];
            s_best_val = sh_val[0];
        }
        __syncthreads();

        if (s_best_idx < 0) break; // no more points can be added

        if (threadIdx.x == 0) {
            in_cluster[s_best_idx] = 1;
            members[count] = s_best_idx;
            s_count = count + 1;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_cardinality[b] = s_count;
    }
}

// ---------------------------------------------------------------------------
// Per-GPU context: holds device buffers reused across all rounds.
// ---------------------------------------------------------------------------
struct GpuContext {
    int device = 0;
    double* d_dist = nullptr;                 // N x N
    unsigned char* d_clustered = nullptr;      // N
    int* d_seeds = nullptr;                    // capacity (== N)
    unsigned char* d_in_cluster_scratch = nullptr; // capacity * N
    int* d_members_scratch = nullptr;          // capacity * N
    int* d_out_cardinality = nullptr;          // capacity
    std::vector<int> host_cardinality;
};

static void setupGpuContext(GpuContext& ctx, int device, const std::vector<Point>& points, int N) {
    ctx.device = device;
    CUDA_CHECK(cudaSetDevice(device));

    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }
    double *d_px, *d_py;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&ctx.d_dist, static_cast<size_t>(N) * N * sizeof(double)));

    dim3 block2d(16, 16);
    dim3 grid2d((N + block2d.x - 1) / block2d.x, (N + block2d.y - 1) / block2d.y);
    computeDistMatrixKernel<<<grid2d, block2d>>>(d_px, d_py, ctx.d_dist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));

    CUDA_CHECK(cudaMalloc(&ctx.d_clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&ctx.d_seeds, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_in_cluster_scratch, static_cast<size_t>(N) * N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&ctx.d_members_scratch, static_cast<size_t>(N) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_out_cardinality, static_cast<size_t>(N) * sizeof(int)));
    ctx.host_cardinality.resize(N);
}

static void destroyGpuContext(GpuContext& ctx) {
    CUDA_CHECK(cudaSetDevice(ctx.device));
    cudaFree(ctx.d_dist);
    cudaFree(ctx.d_clustered);
    cudaFree(ctx.d_seeds);
    cudaFree(ctx.d_in_cluster_scratch);
    cudaFree(ctx.d_members_scratch);
    cudaFree(ctx.d_out_cardinality);
}

// Result of evaluating a batch of candidate seeds on one GPU.
struct BestCandidate {
    int cardinality = -1;
    int seed = std::numeric_limits<int>::max();
    int local_block = -1; // index into the seed batch that produced it
};

static BestCandidate combineBest(const BestCandidate& a, const BestCandidate& b) {
    if (b.cardinality > a.cardinality) return b;
    if (b.cardinality == a.cardinality && b.seed < a.seed) return b;
    return a;
}

// Evaluate all seeds assigned to this GPU for the current round; returns the
// best candidate found on this GPU, and (if requested) copies its member
// list into out_members.
static BestCandidate evaluateSeedsOnGpu(GpuContext& ctx,
                                         const std::vector<int>& seeds,
                                         const std::vector<unsigned char>& clustered,
                                         double threshold,
                                         int N,
                                         std::vector<int>* out_members) {
    BestCandidate best;
    if (seeds.empty()) return best;

    CUDA_CHECK(cudaSetDevice(ctx.device));
    CUDA_CHECK(cudaMemcpy(ctx.d_clustered, clustered.data(), N * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_seeds, seeds.data(), seeds.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    int num_seeds = static_cast<int>(seeds.size());
    growClusterKernel<<<num_seeds, BLOCK_SIZE>>>(
        ctx.d_dist, ctx.d_clustered, ctx.d_seeds, num_seeds, N, threshold,
        ctx.d_in_cluster_scratch, ctx.d_members_scratch, ctx.d_out_cardinality);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(ctx.host_cardinality.data(), ctx.d_out_cardinality,
                          num_seeds * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_seeds; ++i) {
        BestCandidate cand{ctx.host_cardinality[i], seeds[i], i};
        best = combineBest(best, cand);
    }

    if (out_members && best.local_block >= 0) {
        out_members->resize(best.cardinality);
        CUDA_CHECK(cudaMemcpy(out_members->data(),
                              ctx.d_members_scratch + static_cast<size_t>(best.local_block) * N,
                              best.cardinality * sizeof(int), cudaMemcpyDeviceToHost));
    }

    return best;
}

// ---------------------------------------------------------------------------
// Main hybrid QT clustering algorithm
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  int mpi_rank, int mpi_size,
                                  std::vector<GpuContext>& gpus) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    const int num_gpus = static_cast<int>(gpus.size());

    while (!unclustered_indices.empty()) {
        const int total = static_cast<int>(unclustered_indices.size());

        // Block-distribute the unclustered seeds across MPI ranks.
        const int base = total / mpi_size;
        const int rem = total % mpi_size;
        const int my_start = mpi_rank * base + std::min(mpi_rank, rem);
        const int my_count = base + (mpi_rank < rem ? 1 : 0);

        std::vector<int> rank_seeds(unclustered_indices.begin() + my_start,
                                    unclustered_indices.begin() + my_start + my_count);

        // Distribute this rank's seeds across its local GPUs.
        std::vector<BestCandidate> gpu_best(num_gpus);
        std::vector<std::vector<int>> gpu_members(num_gpus);

        const int gcount = static_cast<int>(rank_seeds.size());
        const int gbase = num_gpus > 0 ? gcount / num_gpus : 0;
        const int grem = num_gpus > 0 ? gcount % num_gpus : 0;

        #pragma omp parallel for num_threads(num_gpus > 0 ? num_gpus : 1) schedule(static)
        for (int g = 0; g < num_gpus; ++g) {
            const int start = g * gbase + std::min(g, grem);
            const int count = gbase + (g < grem ? 1 : 0);
            std::vector<int> seeds_for_gpu(rank_seeds.begin() + start,
                                           rank_seeds.begin() + start + count);
            gpu_best[g] = evaluateSeedsOnGpu(gpus[g], seeds_for_gpu, clustered, threshold, N,
                                             &gpu_members[g]);
        }

        BestCandidate rank_best;
        int best_gpu = -1;
        for (int g = 0; g < num_gpus; ++g) {
            if (gpu_best[g].cardinality > rank_best.cardinality ||
                (gpu_best[g].cardinality == rank_best.cardinality &&
                 gpu_best[g].seed < rank_best.seed)) {
                rank_best = gpu_best[g];
                best_gpu = g;
            }
        }

        // Gather every rank's best (cardinality, seed) to determine the
        // global winner deterministically on all ranks.
        struct RankBest { int cardinality; int seed; };
        RankBest my_rb{rank_best.cardinality, rank_best.seed};
        std::vector<RankBest> all_rb(mpi_size);
        MPI_Allgather(&my_rb, 2, MPI_INT, all_rb.data(), 2, MPI_INT, MPI_COMM_WORLD);

        int winner_rank = 0;
        RankBest global_best = all_rb[0];
        for (int r = 1; r < mpi_size; ++r) {
            if (all_rb[r].cardinality > global_best.cardinality ||
                (all_rb[r].cardinality == global_best.cardinality &&
                 all_rb[r].seed < global_best.seed)) {
                global_best = all_rb[r];
                winner_rank = r;
            }
        }

        if (global_best.cardinality <= 0) break; // no more clusters can be formed

        std::vector<int> best_members;
        if (mpi_rank == winner_rank) {
            best_members = gpu_members[best_gpu];
        } else {
            best_members.resize(global_best.cardinality);
        }
        MPI_Bcast(best_members.data(), global_best.cardinality, MPI_INT, winner_rank, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global_best.seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int m : best_members) {
            clustered[m] = 1;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

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
    int provided = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Determine this rank's node-local rank/size so nodes with multiple
    // ranks share the local GPUs instead of oversubscribing them.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (mpi_rank == 0) {
            fprintf(stderr, "Error: no CUDA-capable devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign a contiguous, load-balanced slice of the node's GPUs to this
    // rank based on its node-local rank.
    const int gbase = device_count / local_size;
    const int grem = device_count % local_size;
    const int my_gpu_start = local_rank * gbase + std::min(local_rank, grem);
    int my_gpu_count = gbase + (local_rank < grem ? 1 : 0);
    if (my_gpu_count <= 0) {
        // More ranks than GPUs on this node: fall back to sharing one GPU.
        my_gpu_count = 1;
    }

    std::vector<int> my_devices;
    for (int i = 0; i < my_gpu_count; ++i) {
        int dev = (my_gpu_start + i) % device_count;
        my_devices.push_back(dev);
    }

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI + OpenMP + CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Data generation is a deterministic pure function of num_points, so
    // every rank independently reproduces the identical point set without
    // any communication.
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    // Set up per-GPU device contexts (distance matrix + scratch buffers).
    std::vector<GpuContext> gpus(my_devices.size());
    for (size_t g = 0; g < my_devices.size(); ++g) {
        setupGpuContext(gpus[g], my_devices[g], points, num_points);
    }

    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size, gpus);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time = cluster_time.count();
    long global_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &global_cluster_time, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    for (auto& g : gpus) destroyGpuContext(g);

    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", global_cluster_time);
        printf("Clusters found: %zu\n", clusters.size());

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

        const double time_sec = global_cluster_time / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
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

        int exit_code = 0;
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return exit_code;
    } else {
        int exit_code = 0;
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return exit_code;
    }
}
