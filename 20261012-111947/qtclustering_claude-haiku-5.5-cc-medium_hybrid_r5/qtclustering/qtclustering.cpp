// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization:
//   MPI    - every outer iteration the candidate seeds are dealt out cyclically
//            across ranks; one allreduce selects the global best seed.
//   OpenMP - each rank runs one host thread per GPU worker. The threads pull
//            seed batches from a shared counter (dynamic load balancing).
//            Validation cluster diameters are computed in parallel.
//   CUDA   - one thread block grows the candidate cluster of one seed; the
//            threads of the block cooperate on the per-step candidate scan.
//
// The results are identical to the sequential algorithm: the running maximum
// distance of a candidate to the cluster equals the per-step recomputation of
// the original (a maximum is exact), and ties are broken by lowest point index.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

static const int BLOCK_THREADS = 256;
static const int WARPS_PER_BLOCK = BLOCK_THREADS / 32;

// Device scratch budget per MPI rank (split evenly between its OpenMP workers)
static const size_t SCRATCH_BYTES_PER_RANK = size_t(1) << 30;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
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

__device__ __forceinline__ double deviceDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Keep the lexicographically smallest (value, index) pair
__device__ __forceinline__ void keepMin(double& v, int& i, double ov, int oi) {
    if (ov < v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

// Grow the candidate cluster of seeds[blockIdx.x]. Equivalent to the sequential
// generateCandidateCluster(): at each step the candidate with the smallest maximum
// distance to the current members (below threshold, lowest index on ties) is added.
// A candidate whose maximum distance reaches the threshold can never be added later
// (maxima only grow), so it leaves the active list for good. The cluster members (in
// insertion order) are written to members when it is non-null.
__global__ void __launch_bounds__(BLOCK_THREADS)
growClusterKernel(const Point* __restrict__ points,
                  const unsigned char* __restrict__ clustered,
                  const int* __restrict__ seeds,
                  int* __restrict__ cards,
                  int* __restrict__ members,
                  double* __restrict__ maxd_scratch,
                  int* __restrict__ list_scratch,
                  const int N,
                  const double threshold)
{
    __shared__ double s_val[WARPS_PER_BLOCK];
    __shared__ int s_idx[WARPS_PER_BLOCK];
    __shared__ int s_cnt[2];   // active-list lengths, double-buffered

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int seed = seeds[b];
    double* maxd = maxd_scratch + static_cast<size_t>(b) * N;
    int* src = list_scratch + static_cast<size_t>(2 * b) * N;
    int* dst = list_scratch + static_cast<size_t>(2 * b + 1) * N;
    int* mem = members ? members + static_cast<size_t>(b) * N : nullptr;

    // Initial active list: unclustered points other than the seed
    if (tid == 0) s_cnt[0] = 0;
    __syncthreads();
    for (int c = tid; c < N; c += BLOCK_THREADS) {
        if (!clustered[c] && c != seed) {
            maxd[c] = 0.0;
            src[atomicAdd(&s_cnt[0], 1)] = c;
        }
    }
    if (tid == 0 && mem) mem[0] = seed;
    __syncthreads();

    int cur = seed;   // newest member
    int count = 1;
    int cur_buf = 0;  // s_cnt index holding the length of the source list
    while (true) {
        const int n_src = s_cnt[cur_buf];
        if (tid == 0) s_cnt[cur_buf ^ 1] = 0;
        __syncthreads();

        // Fold the newest member into the running maximum distance of every active
        // candidate; keep those still below threshold and track the best of them
        const Point pc = points[cur];
        double best_v = DBL_MAX;
        int best_i = INT_MAX;
        for (int i = tid; i < n_src; i += BLOCK_THREADS) {
            const int c = src[i];
            if (c == cur) continue;   // the newest member leaves the candidate pool
            const double m = fmax(maxd[c], deviceDistance(points[c], pc));
            maxd[c] = m;
            if (m < threshold) {
                dst[atomicAdd(&s_cnt[cur_buf ^ 1], 1)] = c;
                keepMin(best_v, best_i, m, c);
            }
        }

        // Block-wide argmin: warp shuffle, then one warp over the warp results
        for (int off = 16; off > 0; off >>= 1) {
            keepMin(best_v, best_i, __shfl_down_sync(0xffffffffu, best_v, off),
                    __shfl_down_sync(0xffffffffu, best_i, off));
        }
        if (lane == 0) {
            s_val[warp] = best_v;
            s_idx[warp] = best_i;
        }
        __syncthreads();
        if (warp == 0) {
            best_v = lane < WARPS_PER_BLOCK ? s_val[lane] : DBL_MAX;
            best_i = lane < WARPS_PER_BLOCK ? s_idx[lane] : INT_MAX;
            for (int off = 16; off > 0; off >>= 1) {
                keepMin(best_v, best_i, __shfl_down_sync(0xffffffffu, best_v, off),
                        __shfl_down_sync(0xffffffffu, best_i, off));
            }
            if (lane == 0) {
                s_idx[0] = best_i;
            }
        }
        __syncthreads();

        const int pick = s_idx[0];
        if (pick == INT_MAX) break;   // no candidate keeps the diameter below threshold

        int* const tmp = src;
        src = dst;
        dst = tmp;
        cur_buf ^= 1;
        cur = pick;
        if (tid == 0 && mem) mem[count] = pick;
        count++;
    }

    if (tid == 0) cards[b] = count;
}

// Per-worker CUDA context: one stream and its scratch buffers on the rank's device.
// Each OpenMP thread of a rank owns one worker.
struct GpuWorker {
    int device = 0;
    int capacity = 0;   // seeds per launch
    cudaStream_t stream = nullptr;
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    int* d_members = nullptr;   // one cluster's members
    double* d_maxd = nullptr;
    int* d_list = nullptr;      // two active-candidate lists per block
};

static void initWorker(GpuWorker& w, int device, const std::vector<Point>& points,
                       const int N, const size_t scratch_bytes) {
    const size_t per_block = static_cast<size_t>(N) * (sizeof(double) + 2 * sizeof(int)) + 2 * sizeof(int);
    w.device = device;
    w.capacity = static_cast<int>(std::max<size_t>(1, std::min<size_t>(scratch_bytes / per_block, size_t(1) << 20)));

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreate(&w.stream));
    CUDA_CHECK(cudaMalloc(&w.d_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMemcpy(w.d_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&w.d_clustered, N));
    CUDA_CHECK(cudaMalloc(&w.d_seeds, sizeof(int) * w.capacity));
    CUDA_CHECK(cudaMalloc(&w.d_cards, sizeof(int) * w.capacity));
    CUDA_CHECK(cudaMalloc(&w.d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&w.d_maxd, sizeof(double) * static_cast<size_t>(w.capacity) * N));
    CUDA_CHECK(cudaMalloc(&w.d_list, sizeof(int) * 2 * static_cast<size_t>(w.capacity) * N));
}

static void freeWorker(GpuWorker& w) {
    CUDA_CHECK(cudaSetDevice(w.device));
    CUDA_CHECK(cudaFree(w.d_points));
    CUDA_CHECK(cudaFree(w.d_clustered));
    CUDA_CHECK(cudaFree(w.d_seeds));
    CUDA_CHECK(cudaFree(w.d_cards));
    CUDA_CHECK(cudaFree(w.d_members));
    CUDA_CHECK(cudaFree(w.d_maxd));
    CUDA_CHECK(cudaFree(w.d_list));
    CUDA_CHECK(cudaStreamDestroy(w.stream));
}

// Create the GPU workers of this rank. The node's GPUs are shared out among its ranks:
// with at least as many GPUs as ranks, each rank drives every GPU it owns (round-robin);
// otherwise ranks share GPUs round-robin. Each worker is driven by one OpenMP thread.
static std::vector<GpuWorker> createWorkers(const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    int rank = 0, local_rank = 0, local_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> devices;
    if (local_size <= num_devices) {
        for (int d = local_rank; d < num_devices; d += local_size) devices.push_back(d);
    } else {
        devices.push_back(local_rank % num_devices);
    }

    // One OpenMP thread (and GPU worker) per device, within this rank's share of the cores
    omp_set_dynamic(0);
    const int threads_per_rank = std::max(1, omp_get_max_threads() / local_size);
    const int num_workers = std::min(static_cast<int>(devices.size()), threads_per_rank);

    std::vector<GpuWorker> workers(num_workers);
    for (int t = 0; t < num_workers; ++t) {
        initWorker(workers[t], devices[t], points, N, SCRATCH_BYTES_PER_RANK / num_workers);
    }
    return workers;
}

static void freeWorkers(std::vector<GpuWorker>& workers) {
    for (GpuWorker& w : workers) {
        freeWorker(w);
    }
    workers.clear();
}

// Main QT clustering algorithm (identical result on every rank)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  std::vector<GpuWorker>& workers) {
    const int N = static_cast<int>(points.size());
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int num_workers = static_cast<int>(workers.size());

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<int> local_seeds;
    std::vector<int> local_cards;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Deal the candidate seeds out cyclically across ranks
        const int U = static_cast<int>(unclustered_indices.size());
        local_seeds.clear();
        for (int p = rank; p < U; p += size) {
            local_seeds.push_back(unclustered_indices[p]);
        }
        const int local_count = static_cast<int>(local_seeds.size());
        local_cards.assign(local_count, 0);

        // Grow candidate clusters for the local seeds, batches shared among the workers
        // One launch per worker where scratch allows: the GPU block scheduler balances the seeds
        const int batch = std::max(1, std::min(workers[0].capacity,
                                               (local_count + num_workers - 1) / num_workers));
        std::atomic<int> next_pos(0);
        #pragma omp parallel num_threads(num_workers)
        {
            GpuWorker& w = workers[omp_get_thread_num()];
            CUDA_CHECK(cudaSetDevice(w.device));
            CUDA_CHECK(cudaMemcpyAsync(w.d_clustered, clustered.data(), N,
                                       cudaMemcpyHostToDevice, w.stream));
            while (true) {
                const int start = next_pos.fetch_add(batch);
                if (start >= local_count) break;
                const int count = std::min(batch, local_count - start);
                CUDA_CHECK(cudaMemcpyAsync(w.d_seeds, local_seeds.data() + start,
                                           sizeof(int) * count, cudaMemcpyHostToDevice, w.stream));
                growClusterKernel<<<count, BLOCK_THREADS, 0, w.stream>>>(
                    w.d_points, w.d_clustered, w.d_seeds, w.d_cards, nullptr,
                    w.d_maxd, w.d_list, N, threshold);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(local_cards.data() + start, w.d_cards,
                                           sizeof(int) * count, cudaMemcpyDeviceToHost, w.stream));
                CUDA_CHECK(cudaStreamSynchronize(w.stream));
            }
        }

        // Best local candidate: largest cardinality, then lowest seed index
        // (the unclustered list is ascending, so this matches the sequential first-wins rule)
        uint64_t local_key = 0;
        for (int i = 0; i < local_count; ++i) {
            const uint64_t key = (static_cast<uint64_t>(local_cards[i]) << 32) |
                                 static_cast<uint32_t>(0xFFFFFFFFu - static_cast<uint32_t>(local_seeds[i]));
            local_key = std::max(local_key, key);
        }
        uint64_t global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

        const int max_cardinality = static_cast<int>(global_key >> 32);
        const int best_seed = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(global_key & 0xFFFFFFFFu));

        // If we found a cluster, add it
        if (global_key == 0 || max_cardinality <= 0) {
            // No more clusters can be formed
            break;
        }

        // Rebuild the winning cluster's members (identical on every rank)
        GpuWorker& w = workers[0];
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaMemcpyAsync(w.d_clustered, clustered.data(), N, cudaMemcpyHostToDevice, w.stream));
        CUDA_CHECK(cudaMemcpyAsync(w.d_seeds, &best_seed, sizeof(int), cudaMemcpyHostToDevice, w.stream));
        growClusterKernel<<<1, BLOCK_THREADS, 0, w.stream>>>(
            w.d_points, w.d_clustered, w.d_seeds, w.d_cards, w.d_members,
            w.d_maxd, w.d_list, N, threshold);
        CUDA_CHECK(cudaGetLastError());
        int cardinality = 0;
        CUDA_CHECK(cudaMemcpyAsync(&cardinality, w.d_cards, sizeof(int), cudaMemcpyDeviceToHost, w.stream));
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
        if (cardinality != max_cardinality) {
            fprintf(stderr, "Internal error: cluster size mismatch for seed %d\n", best_seed);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(cardinality);
        CUDA_CHECK(cudaMemcpyAsync(cluster.members.data(), w.d_members, sizeof(int) * cardinality,
                                   cudaMemcpyDeviceToHost, w.stream));
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (int m : cluster.members) {
            clustered[m] = 1;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end()
        );
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Cluster diameters are independent: compute them in parallel, report in order
    const int num_clusters = static_cast<int>(clusters.size());
    std::vector<double> diameters(num_clusters, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < num_clusters; ++c) {
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
        diameters[c] = max_diameter;
    }

    // Check each cluster
    for (int c = 0; c < num_clusters; ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %d has diameter %.4f > threshold %.4f\n",
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

static int runBenchmark(int argc, char** argv, const int rank) {
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
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // GPU setup (context creation, data upload) is one-time work, kept out of the timed region
    std::vector<GpuWorker> workers = createWorkers(points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, workers);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    const long local_cluster_time = cluster_time.count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);
    if (rank == 0) cluster_time = std::chrono::milliseconds(max_cluster_time);

    freeWorkers(workers);

    if (rank != 0) {
        return 0;   // Rank 0 reports; every rank holds the same clusters
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int rc = runBenchmark(argc, argv, rank);

    MPI_Finalize();
    return rc;
}
