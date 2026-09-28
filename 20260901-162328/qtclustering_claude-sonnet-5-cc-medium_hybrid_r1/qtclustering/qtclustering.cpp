// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - MPI distributes the set of candidate seed points (the outer "try
//    every unclustered point as a cluster seed" search) across ranks/nodes.
//  - OpenMP distributes each rank's share of seeds across the GPUs local
//    to that rank (a thread per GPU manages that device).
//  - CUDA grows one candidate cluster per thread block in parallel: each
//    block iteratively finds, via an on-device reduction, the closest
//    remaining point that keeps the cluster diameter below the threshold,
//    exactly mirroring the sequential greedy growth rule (and its tie
//    break: smallest point index wins).
// The globally best candidate (largest cardinality, smallest seed index
// on ties - identical tie-break to the original sequential algorithm) is
// selected via an MPI_MAXLOC reduction and broadcast to all ranks each
// outer iteration, so all ranks stay in lock-step with identical state.

#include <algorithm>
#include <chrono>
#include <cfloat>
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
static const int THREADS = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Compute the full pairwise distance matrix D[i*N+j] = dist(points[i], points[j])
__global__ void computeDistanceMatrix(const Point* __restrict__ points, int N,
                                       double* __restrict__ D) {
    int total = N * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        int i = idx / N;
        int j = idx % N;
        double dx = points[i].x - points[j].x;
        double dy = points[i].y - points[j].y;
        D[idx] = sqrt(dx * dx + dy * dy);
    }
}

// Grow one candidate cluster per block, for each seed in seed_ids[blockIdx.x].
// status: 0 = eligible candidate, 1 = member of this candidate cluster,
//         2 = already assigned to a previously finalized cluster (excluded).
// dist_to_cluster[c] tracks the running max distance from candidate c to the
// cluster grown so far (matches the sequential "max_dist" computation).
__global__ void growClusters(const double* __restrict__ D, int N,
                              const unsigned char* __restrict__ global_clustered,
                              const int* __restrict__ seed_ids,
                              double* __restrict__ dist_to_cluster,
                              unsigned char* __restrict__ status,
                              int* __restrict__ members,
                              int* __restrict__ cardinality,
                              double threshold) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    double* my_dist = dist_to_cluster + static_cast<size_t>(b) * N;
    unsigned char* my_status = status + static_cast<size_t>(b) * N;
    int* my_members = members + static_cast<size_t>(b) * N;
    const int seed = seed_ids[b];

    for (int c = tid; c < N; c += nthreads) {
        my_status[c] = global_clustered[c] ? 2 : 0;
        my_dist[c] = 0.0;
    }
    __syncthreads();
    if (tid == 0) {
        my_status[seed] = 1;
        my_members[0] = seed;
    }
    __syncthreads();

    // Seed the running max-distance-to-cluster with distances to the seed
    // point itself, since it is the first member but was not added via the
    // regular growth-update path below.
    {
        const double* SeedRow = D + static_cast<size_t>(seed) * N;
        for (int c = tid; c < N; c += nthreads) {
            if (my_status[c] == 0) my_dist[c] = SeedRow[c];
        }
    }
    __syncthreads();

    __shared__ double s_dist[THREADS];
    __shared__ int s_idx[THREADS];

    int members_count = 1;
    while (members_count < N) {
        double local_d = DBL_MAX;
        int local_i = -1;
        for (int c = tid; c < N; c += nthreads) {
            if (my_status[c] == 0) {
                double d = my_dist[c];
                if (d < threshold) {
                    if (local_i < 0 || d < local_d || (d == local_d && c < local_i)) {
                        local_d = d;
                        local_i = c;
                    }
                }
            }
        }
        s_dist[tid] = local_d;
        s_idx[tid] = local_i;
        __syncthreads();

        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                double od = s_dist[tid + stride];
                int oi = s_idx[tid + stride];
                double md = s_dist[tid];
                int mi = s_idx[tid];
                if (mi < 0 || (oi >= 0 && (od < md || (od == md && oi < mi)))) {
                    s_dist[tid] = od;
                    s_idx[tid] = oi;
                }
            }
            __syncthreads();
        }

        int best_idx = s_idx[0];
        if (best_idx < 0) break;

        if (tid == 0) {
            my_status[best_idx] = 1;
            my_members[members_count] = best_idx;
        }
        __syncthreads();
        members_count++;

        const double* Drow = D + static_cast<size_t>(best_idx) * N;
        for (int c = tid; c < N; c += nthreads) {
            if (my_status[c] == 0) {
                double dd = Drow[c];
                if (dd > my_dist[c]) my_dist[c] = dd;
            }
        }
        __syncthreads();
    }

    if (tid == 0) cardinality[b] = members_count;
}

// ---------------------------------------------------------------------------
// Per-GPU device state
// ---------------------------------------------------------------------------

struct GpuContext {
    int device_id = -1;
    int N = 0;
    int batch_capacity = 0;
    Point* d_points = nullptr;
    double* d_D = nullptr;
    unsigned char* d_global_clustered = nullptr;
    int* d_seed_ids = nullptr;
    double* d_dist_to_cluster = nullptr;
    unsigned char* d_status = nullptr;
    int* d_members = nullptr;
    int* d_cardinality = nullptr;
};

void initGpuContext(GpuContext& ctx, int device_id, const std::vector<Point>& points, int N) {
    ctx.device_id = device_id;
    ctx.N = N;
    CUDA_CHECK(cudaSetDevice(device_id));

    CUDA_CHECK(cudaMalloc(&ctx.d_points, static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMemcpy(ctx.d_points, points.data(), static_cast<size_t>(N) * sizeof(Point),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&ctx.d_D, static_cast<size_t>(N) * N * sizeof(double)));

    int total = N * N;
    int blocks = std::min(65535, (total + THREADS - 1) / THREADS);
    blocks = std::max(blocks, 1);
    computeDistanceMatrix<<<blocks, THREADS>>>(ctx.d_points, N, ctx.d_D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMalloc(&ctx.d_global_clustered, static_cast<size_t>(N) * sizeof(unsigned char)));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t per_seed_bytes = static_cast<size_t>(N) * (sizeof(double) + sizeof(unsigned char) + sizeof(int));
    size_t budget = free_mem / 2; // leave headroom for the distance matrix and driver overhead
    int cap = static_cast<int>(std::min<size_t>(budget / std::max<size_t>(per_seed_bytes, 1),
                                                 static_cast<size_t>(N)));
    cap = std::max(1, std::min(cap, 4096));
    ctx.batch_capacity = cap;

    CUDA_CHECK(cudaMalloc(&ctx.d_seed_ids, static_cast<size_t>(cap) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_dist_to_cluster, static_cast<size_t>(cap) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_status, static_cast<size_t>(cap) * N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&ctx.d_members, static_cast<size_t>(cap) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_cardinality, static_cast<size_t>(cap) * sizeof(int)));
}

void freeGpuContext(GpuContext& ctx) {
    cudaSetDevice(ctx.device_id);
    cudaFree(ctx.d_points);
    cudaFree(ctx.d_D);
    cudaFree(ctx.d_global_clustered);
    cudaFree(ctx.d_seed_ids);
    cudaFree(ctx.d_dist_to_cluster);
    cudaFree(ctx.d_status);
    cudaFree(ctx.d_members);
    cudaFree(ctx.d_cardinality);
}

// Evaluate every seed in `seeds` on this GPU (batching to fit in scratch
// memory) and return the best (largest cardinality, smallest seed on ties).
void processSeedsOnGpu(GpuContext& ctx, const std::vector<int>& seeds,
                        const std::vector<unsigned char>& clustered_host, double threshold,
                        int& best_card, int& best_seed, std::vector<int>& best_members) {
    best_card = -1;
    best_seed = -1;
    best_members.clear();
    if (seeds.empty()) return;

    CUDA_CHECK(cudaSetDevice(ctx.device_id));
    CUDA_CHECK(cudaMemcpy(ctx.d_global_clustered, clustered_host.data(),
                           static_cast<size_t>(ctx.N) * sizeof(unsigned char), cudaMemcpyHostToDevice));

    const int N = ctx.N;
    std::vector<int> host_card(ctx.batch_capacity);

    for (size_t offset = 0; offset < seeds.size(); offset += ctx.batch_capacity) {
        int cur = static_cast<int>(std::min<size_t>(ctx.batch_capacity, seeds.size() - offset));

        CUDA_CHECK(cudaMemcpy(ctx.d_seed_ids, seeds.data() + offset, cur * sizeof(int),
                               cudaMemcpyHostToDevice));

        growClusters<<<cur, THREADS>>>(ctx.d_D, N, ctx.d_global_clustered, ctx.d_seed_ids,
                                        ctx.d_dist_to_cluster, ctx.d_status, ctx.d_members,
                                        ctx.d_cardinality, threshold);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_card.data(), ctx.d_cardinality, cur * sizeof(int),
                               cudaMemcpyDeviceToHost));

        for (int i = 0; i < cur; ++i) {
            int seed = seeds[offset + i];
            int card = host_card[i];
            if (card > best_card || (card == best_card && (best_seed < 0 || seed < best_seed))) {
                best_card = card;
                best_seed = seed;
                best_members.resize(card);
                CUDA_CHECK(cudaMemcpy(best_members.data(),
                                       ctx.d_members + static_cast<size_t>(i) * N,
                                       card * sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
    }
}

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

// Calculate Euclidean distance between two points (host-side, used by validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA seed evaluation
// ---------------------------------------------------------------------------

struct BestResult {
    int cardinality = -1;
    int seed = -1;
    std::vector<int> members;
};

// Evaluate every currently-unclustered point as a candidate cluster seed and
// return the globally best one (largest cardinality; smallest seed index on
// ties), identical to what the sequential algorithm would have selected.
BestResult hybridBestCluster(const std::vector<bool>& clustered,
                              const std::vector<int>& unclustered_indices, double threshold, int N,
                              std::vector<GpuContext>& gpu_ctxs, int world_rank, int world_size) {
    std::vector<unsigned char> clustered_host(N);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        clustered_host[i] = clustered[i] ? 1 : 0;
    }

    // Distribute this rank's share of seeds (strided for load balance).
    std::vector<int> my_seeds;
    for (size_t i = world_rank; i < unclustered_indices.size(); i += world_size) {
        my_seeds.push_back(unclustered_indices[i]);
    }

    const int ngpus = static_cast<int>(gpu_ctxs.size());
    std::vector<int> thread_best_card(ngpus, -1);
    std::vector<int> thread_best_seed(ngpus, -1);
    std::vector<std::vector<int>> thread_best_members(ngpus);

#pragma omp parallel for schedule(static) num_threads(ngpus)
    for (int t = 0; t < ngpus; ++t) {
        std::vector<int> chunk;
        for (size_t i = t; i < my_seeds.size(); i += ngpus) {
            chunk.push_back(my_seeds[i]);
        }
        processSeedsOnGpu(gpu_ctxs[t], chunk, clustered_host, threshold, thread_best_card[t],
                           thread_best_seed[t], thread_best_members[t]);
    }

    int rank_best_card = -1, rank_best_seed = -1;
    std::vector<int> rank_best_members;
    for (int t = 0; t < ngpus; ++t) {
        if (thread_best_card[t] > rank_best_card ||
            (thread_best_card[t] == rank_best_card && thread_best_seed[t] >= 0 &&
             (rank_best_seed < 0 || thread_best_seed[t] < rank_best_seed))) {
            rank_best_card = thread_best_card[t];
            rank_best_seed = thread_best_seed[t];
            rank_best_members = std::move(thread_best_members[t]);
        }
    }

    // Encode (cardinality, seed) into a single key so MPI_MAXLOC picks the
    // rank with the largest cardinality, and (via the "-seed" term) the
    // smallest seed index on ties.
    double key = (rank_best_seed >= 0)
                     ? static_cast<double>(rank_best_card) * (static_cast<double>(N) + 1.0) -
                           static_cast<double>(rank_best_seed)
                     : -std::numeric_limits<double>::max();

    struct {
        double val;
        int rank;
    } in{key, world_rank}, out{};
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    BestResult result;
    if (out.rank == world_rank) {
        result.cardinality = rank_best_card;
        result.seed = rank_best_seed;
        result.members = std::move(rank_best_members);
    }
    MPI_Bcast(&result.cardinality, 1, MPI_INT, out.rank, MPI_COMM_WORLD);
    MPI_Bcast(&result.seed, 1, MPI_INT, out.rank, MPI_COMM_WORLD);
    if (result.cardinality > 0) {
        result.members.resize(result.cardinality);
        MPI_Bcast(result.members.data(), result.cardinality, MPI_INT, out.rank, MPI_COMM_WORLD);
    }
    return result;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold,
                                   std::vector<GpuContext>& gpu_ctxs, int world_rank,
                                   int world_size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<Cluster> clusters;

    while (!unclustered_indices.empty()) {
        BestResult best =
            hybridBestCluster(clustered, unclustered_indices, threshold, N, gpu_ctxs, world_rank,
                               world_size);

        if (best.seed >= 0 && best.cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best.seed;
            cluster.members = best.members;
            clusters.push_back(cluster);

            for (int m : best.members) {
                clustered[m] = true;
            }

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());
        } else {
            // No more clusters can be formed
            break;
        }
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points,
                       const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        // Check diameter (max distance between any two points)
        const int M = static_cast<int>(cluster.members.size());
#pragma omp parallel for reduction(max : max_diameter) schedule(dynamic)
        for (int i = 0; i < M; ++i) {
            double local_max = 0.0;
            for (int j = i + 1; j < M; ++j) {
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
                local_max = std::max(local_max, dist);
            }
            max_diameter = std::max(max_diameter, local_max);
        }

        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(),
                   cluster.seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter,
                   threshold);
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member,
                       membership[member], c);
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

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count,
           points.size() - clustered_count);

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
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (world_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points,
                   threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
    }

    // Generate synthetic data - deterministic given the fixed seed, so every
    // rank independently produces an identical point set with no communication.
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Determine which GPU(s) this rank is responsible for, splitting the
    // node's devices evenly across the ranks that share the node.
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &shmcomm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_size(shmcomm, &local_size);
    MPI_Comm_free(&shmcomm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int gpus_per_rank = std::max(1, device_count / local_size);
    std::vector<int> my_gpu_ids;
    for (int g = 0; g < gpus_per_rank; ++g) {
        my_gpu_ids.push_back((local_rank * gpus_per_rank + g) % device_count);
    }

    std::vector<GpuContext> gpu_ctxs(my_gpu_ids.size());
#pragma omp parallel for num_threads(my_gpu_ids.size())
    for (size_t t = 0; t < my_gpu_ids.size(); ++t) {
        initGpuContext(gpu_ctxs[t], my_gpu_ids[t], points, num_points);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, gpu_ctxs, world_rank, world_size);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);

    for (auto& ctx : gpu_ctxs) freeGpuContext(ctx);

    if (world_rank == 0) {
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

        const double avg_cluster_size =
            clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

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
    }

    MPI_Finalize();
    return 0;
}
