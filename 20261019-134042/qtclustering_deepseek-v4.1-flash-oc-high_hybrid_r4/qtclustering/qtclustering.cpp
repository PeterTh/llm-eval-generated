// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering builds clusters by starting from a seed
// point and iteratively adding the closest point that keeps the cluster's
// diameter below a threshold.
//
// Parallelization strategy:
//   * MPI distributes the candidate seeds of every clustering round among
//     ranks and performs the global best-seed reduction.
//   * OpenMP builds the distance matrix and drives several independent CUDA
//     streams per rank.
//   * CUDA evaluates all candidate clusters of a round in parallel (one
//     thread block per seed), using an incremental maximum-distance array so
//     that each greedy step is a linear scan instead of a full recomputation.

#include <algorithm>
#include <chrono>
#include <climits>
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

// Sentinels stored in the per-seed max-distance array.
static const double DIST_IN_CLUSTER = -1.0;
static const double DIST_CLUSTERED = -2.0;

static int g_device = 0;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
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

// Evaluate the greedy candidate cluster for every seed in one launch.
// Grid: one thread block per seed. The per-seed "max distance to any member"
// array is updated incrementally: adding a member only ever increases it.
__global__ void evaluateSeedsKernel(const double* __restrict__ dist,
                                    const unsigned char* __restrict__ clustered,
                                    const int* __restrict__ seeds,
                                    const int num_seeds,
                                    const int point_count,
                                    const double threshold,
                                    double* __restrict__ max_dist,
                                    int* __restrict__ cardinality) {
    const int s = blockIdx.x;
    if (s >= num_seeds) return;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int seed = seeds[s];
    double* md = max_dist + static_cast<size_t>(s) * static_cast<size_t>(point_count);

    __shared__ double red_val[256];
    __shared__ int red_idx[256];
    __shared__ int chosen;

    for (int c = tid; c < point_count; c += nt) {
        md[c] = (clustered[c] != 0)
                    ? DIST_CLUSTERED
                    : dist[static_cast<size_t>(seed) * point_count + c];
    }
    __syncthreads();
    if (tid == 0) md[seed] = DIST_IN_CLUSTER;
    __syncthreads();

    int count = 1;
    for (;;) {
        double best_val = threshold;
        int best_idx = -1;
        for (int c = tid; c < point_count; c += nt) {
            const double v = md[c];
            if (v >= 0.0 && v < best_val) {
                best_val = v;
                best_idx = c;
            }
        }

        red_val[tid] = best_val;
        red_idx[tid] = best_idx;
        __syncthreads();
        for (int offset = nt >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const double v2 = red_val[tid + offset];
                const int i2 = red_idx[tid + offset];
                const double v1 = red_val[tid];
                const int i1 = red_idx[tid];
                if (v2 < v1 ||
                    (v2 == v1 && i2 >= 0 && (i1 < 0 || i2 < i1))) {
                    red_val[tid] = v2;
                    red_idx[tid] = i2;
                }
            }
            __syncthreads();
        }
        if (tid == 0) chosen = red_idx[0];
        __syncthreads();

        const int m = chosen;
        if (m < 0) break;

        if (tid == 0) md[m] = DIST_IN_CLUSTER;
        ++count;
        __syncthreads();

        const double* col = dist + static_cast<size_t>(m) * point_count;
        for (int c = tid; c < point_count; c += nt) {
            const double cur = md[c];
            if (cur >= 0.0) {
                const double d = col[c];
                if (d > cur) md[c] = d;
            }
        }
        __syncthreads();
    }

    if (tid == 0) cardinality[s] = count;
}

// Recompute a single candidate cluster and record its members in order.
__global__ void computeOneSeedKernel(const double* __restrict__ dist,
                                     const unsigned char* __restrict__ clustered,
                                     const int seed,
                                     const int point_count,
                                     const double threshold,
                                     double* __restrict__ max_dist,
                                     int* __restrict__ members,
                                     int* __restrict__ cardinality) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    double* md = max_dist;

    __shared__ double red_val[256];
    __shared__ int red_idx[256];
    __shared__ int chosen;

    for (int c = tid; c < point_count; c += nt) {
        md[c] = (clustered[c] != 0)
                    ? DIST_CLUSTERED
                    : dist[static_cast<size_t>(seed) * point_count + c];
    }
    __syncthreads();
    if (tid == 0) {
        md[seed] = DIST_IN_CLUSTER;
        members[0] = seed;
    }
    __syncthreads();

    int count = 1;
    for (;;) {
        double best_val = threshold;
        int best_idx = -1;
        for (int c = tid; c < point_count; c += nt) {
            const double v = md[c];
            if (v >= 0.0 && v < best_val) {
                best_val = v;
                best_idx = c;
            }
        }

        red_val[tid] = best_val;
        red_idx[tid] = best_idx;
        __syncthreads();
        for (int offset = nt >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const double v2 = red_val[tid + offset];
                const int i2 = red_idx[tid + offset];
                const double v1 = red_val[tid];
                const int i1 = red_idx[tid];
                if (v2 < v1 ||
                    (v2 == v1 && i2 >= 0 && (i1 < 0 || i2 < i1))) {
                    red_val[tid] = v2;
                    red_idx[tid] = i2;
                }
            }
            __syncthreads();
        }
        if (tid == 0) chosen = red_idx[0];
        __syncthreads();

        const int m = chosen;
        if (m < 0) break;

        if (tid == 0) {
            md[m] = DIST_IN_CLUSTER;
            members[count] = m;
        }
        ++count;
        __syncthreads();

        const double* col = dist + static_cast<size_t>(m) * point_count;
        for (int c = tid; c < point_count; c += nt) {
            const double cur = md[c];
            if (cur >= 0.0) {
                const double d = col[c];
                if (d > cur) md[c] = d;
            }
        }
        __syncthreads();
    }

    if (tid == 0) cardinality[0] = count;
}

// ---------------------------------------------------------------------------
// Per-rank GPU resources
// ---------------------------------------------------------------------------

struct GpuWorkspace {
    int point_count = 0;
    int num_threads = 1;
    int capacity = 0;
    double* dist = nullptr;
    unsigned char* clustered = nullptr;
    int* members = nullptr;
    int* cardinality = nullptr;
    std::vector<cudaStream_t> stream;
    std::vector<double*> max_dist;
    std::vector<int*> seeds;
    std::vector<int*> cards;
};

static void createGpuWorkspace(GpuWorkspace& g, const int point_count,
                               const int num_threads) {
    g.point_count = point_count;
    g.num_threads = num_threads;
    g.capacity = (point_count + num_threads - 1) / num_threads + 1;

    CUDA_CHECK(cudaMalloc(&g.dist,
                          static_cast<size_t>(point_count) * point_count *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.clustered,
                          static_cast<size_t>(point_count) *
                              sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&g.members,
                          static_cast<size_t>(point_count) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.cardinality, sizeof(int)));

    g.stream.resize(num_threads);
    g.max_dist.resize(num_threads);
    g.seeds.resize(num_threads);
    g.cards.resize(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        CUDA_CHECK(cudaStreamCreate(&g.stream[t]));
        CUDA_CHECK(cudaMalloc(&g.max_dist[t],
                              static_cast<size_t>(g.capacity) * point_count *
                                  sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.seeds[t],
                              static_cast<size_t>(g.capacity) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&g.cards[t],
                              static_cast<size_t>(g.capacity) * sizeof(int)));
    }
}

static void destroyGpuWorkspace(GpuWorkspace& g) {
    for (int t = 0; t < g.num_threads; ++t) {
        if (g.stream[t]) cudaStreamDestroy(g.stream[t]);
        if (g.max_dist[t]) cudaFree(g.max_dist[t]);
        if (g.seeds[t]) cudaFree(g.seeds[t]);
        if (g.cards[t]) cudaFree(g.cards[t]);
    }
    if (g.dist) cudaFree(g.dist);
    if (g.clustered) cudaFree(g.clustered);
    if (g.members) cudaFree(g.members);
    if (g.cardinality) cudaFree(g.cardinality);
}

// Evaluate all seeds in the range [begin, end) of the unclustered list and
// return the best (maximum cardinality, lowest seed index) candidate.
static void evaluateSeedRange(const GpuWorkspace& g,
                              const std::vector<int>& unclustered,
                              const size_t begin, const size_t end,
                              const double threshold,
                              std::vector<int>& host_cards,
                              int& best_cardinality, int& best_seed) {
    const size_t count = end - begin;
    const int num_threads = g.num_threads;

#pragma omp parallel num_threads(num_threads)
    {
        cudaSetDevice(g_device);
        const int t = omp_get_thread_num();
        const size_t ts = begin + (count * static_cast<size_t>(t)) / num_threads;
        const size_t te =
            begin + (count * static_cast<size_t>(t + 1)) / num_threads;
        const int n = static_cast<int>(te - ts);
        if (n > 0) {
            CUDA_CHECK(cudaMemcpyAsync(g.seeds[t], unclustered.data() + ts,
                                       static_cast<size_t>(n) * sizeof(int),
                                       cudaMemcpyHostToDevice, g.stream[t]));
            evaluateSeedsKernel<<<n, 256, 0, g.stream[t]>>>(
                g.dist, g.clustered, g.seeds[t], n, g.point_count, threshold,
                g.max_dist[t], g.cards[t]);
            CUDA_CHECK(cudaMemcpyAsync(host_cards.data() + ts, g.cards[t],
                                       static_cast<size_t>(n) * sizeof(int),
                                       cudaMemcpyDeviceToHost, g.stream[t]));
            CUDA_CHECK(cudaStreamSynchronize(g.stream[t]));
        }
    }

    best_cardinality = -1;
    best_seed = INT_MAX;
    for (size_t i = begin; i < end; ++i) {
        const int card = host_cards[i];
        const int seed = unclustered[i];
        if (card > best_cardinality ||
            (card == best_cardinality && seed < best_seed)) {
            best_cardinality = card;
            best_seed = seed;
        }
    }
}

static std::vector<int> recoverCluster(const GpuWorkspace& g, const int seed,
                                       const double threshold) {
    CUDA_CHECK(cudaSetDevice(g_device));
    int cardinality = 0;
    computeOneSeedKernel<<<1, 256, 0, g.stream[0]>>>(
        g.dist, g.clustered, seed, g.point_count, threshold, g.max_dist[0],
        g.members, g.cardinality);
    CUDA_CHECK(cudaMemcpyAsync(&cardinality, g.cardinality, sizeof(int),
                               cudaMemcpyDeviceToHost, g.stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(g.stream[0]));

    std::vector<int> members(static_cast<size_t>(cardinality));
    if (cardinality > 0) {
        CUDA_CHECK(cudaMemcpy(members.data(), g.members,
                              static_cast<size_t>(cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }
    return members;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, const int rank,
                                  const int world_size, GpuWorkspace& gpu) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<int> host_cards(N);

    while (!unclustered_indices.empty()) {
        CUDA_CHECK(cudaSetDevice(g_device));
        CUDA_CHECK(cudaMemcpy(gpu.clustered, clustered.data(),
                              static_cast<size_t>(N) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        const size_t total = unclustered_indices.size();
        const size_t begin =
            (total * static_cast<size_t>(rank)) / world_size;
        const size_t end =
            (total * static_cast<size_t>(rank + 1)) / world_size;

        int local_cardinality = -1;
        int local_seed = INT_MAX;
        evaluateSeedRange(gpu, unclustered_indices, begin, end, threshold,
                          host_cards, local_cardinality, local_seed);

        unsigned long long local_key = 0;
        if (local_cardinality > 0) {
            local_key =
                (static_cast<unsigned long long>(
                     static_cast<unsigned int>(local_cardinality))
                 << 32) |
                static_cast<unsigned long long>(0xFFFFFFFFu -
                                                static_cast<unsigned int>(
                                                    local_seed));
        }

        unsigned long long global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD);

        const int best_cardinality = static_cast<int>(global_key >> 32);
        const int best_seed =
            static_cast<int>(0xFFFFFFFFu -
                             static_cast<unsigned int>(global_key & 0xFFFFFFFFu));

        if (best_cardinality <= 0) {
            break;
        }

        const std::vector<int> best_members =
            recoverCluster(gpu, best_seed, threshold);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (size_t i = 0; i < best_members.size(); ++i) {
            clustered[best_members[i]] = 1;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(),
                           unclustered_indices.end(),
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // One GPU per rank where possible.
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess || device_count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: no CUDA device available: %s\n",
                    cudaGetErrorString(dev_err));
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    g_device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(g_device));

    // Choose a sensible number of OpenMP threads per rank.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int procs = omp_get_num_procs();
        const int per_rank = std::max(1, procs / world_size);
        omp_set_num_threads(per_rank);
    }
    const int max_threads = omp_get_max_threads();
    const int eval_threads = std::max(1, std::min(max_threads, 8));

    // Generate synthetic data on rank 0 and broadcast it.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(reinterpret_cast<double*>(points.data()),
              static_cast<int>(2 * static_cast<size_t>(num_points)),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Build the distance matrix (stored member-major). OpenMP parallelizes
    // this over all cores of the rank. Distances are computed exactly like
    // the sequential reference so results are bit-for-bit comparable.
    std::vector<double> host_dist(
        static_cast<size_t>(num_points) * static_cast<size_t>(num_points));

    const auto cluster_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
    for (int j = 0; j < num_points; ++j) {
        double* row =
            host_dist.data() + static_cast<size_t>(j) * num_points;
        for (int i = 0; i < num_points; ++i) {
            row[i] = distance(points[i], points[j]);
        }
    }

    GpuWorkspace gpu;
    createGpuWorkspace(gpu, num_points, eval_threads);
    CUDA_CHECK(cudaMemcpy(gpu.dist, host_dist.data(),
                          static_cast<size_t>(num_points) * num_points *
                              sizeof(double),
                          cudaMemcpyHostToDevice));
    std::vector<double>().swap(host_dist);

    MPI_Barrier(MPI_COMM_WORLD);
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, world_size, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);
    const std::chrono::milliseconds cluster_time(max_cluster_time);

    destroyGpuWorkspace(gpu);
    MPI_Comm_free(&node_comm);

    if (rank == 0) {
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
