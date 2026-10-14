// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   - MPI:    candidate seeds are distributed cyclically across ranks. Each
//             round, every rank reports its best candidate seed and the
//             global choice is made identically on all ranks (MPI_Allgather).
//   - OpenMP: each rank drives all of its local GPUs with one host thread per
//             device, and validation diameters are computed in parallel.
//   - CUDA:   one thread block grows the candidate cluster of one seed; the
//             per-step closest-point search is a block-wide reduction.
//
// Arithmetic and tie-breaking follow the sequential algorithm exactly, so the
// clusters produced are identical to the sequential version.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per block for the candidate-growth kernel (must be a power of two)
constexpr int BLOCK_SIZE = 256;

// Upper bound on per-device scratch memory used for candidate growth state
constexpr size_t SCRATCH_BUDGET_BYTES = size_t{256} << 20;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Best candidate found so far: a cluster cardinality and the seed producing it
struct Candidate {
    int cardinality;
    int seed;
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

// Squared Euclidean distance. Shared by host and device code so that both
// evaluate exactly the same floating-point expression.
__host__ __device__ inline double squaredDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Calculate Euclidean distance between two points
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    return sqrt(squaredDistance(p1, p2));
}

// Grow a candidate cluster from seed_point. At each step the unclustered point
// with the smallest maximum distance to the current members is added (lowest
// index wins ties), as long as that maximum distance is below the threshold.
// max_dist[c] is maintained incrementally, which yields exactly the value that
// a full recomputation over all members would give. Returns the members in
// insertion order.
std::vector<int> growCandidateCluster(const int seed_point,
                                      const std::vector<uint8_t>& clustered,
                                      const std::vector<Point>& points,
                                      const double threshold) {
    const int N = static_cast<int>(points.size());

    // A negative value marks a point that can no longer be a candidate
    std::vector<double> max_dist(N, 0.0);
    for (int c = 0; c < N; ++c) {
        if (clustered[c]) max_dist[c] = -1.0;
    }
    max_dist[seed_point] = -1.0;

    std::vector<int> members{seed_point};
    int added = seed_point;
    while (true) {
        for (int c = 0; c < N; ++c) {
            if (max_dist[c] >= 0.0) {
                max_dist[c] = std::max(max_dist[c], distance(points[c], points[added]));
            }
        }

        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();
        for (int c = 0; c < N; ++c) {
            if (max_dist[c] >= 0.0 && max_dist[c] < threshold && max_dist[c] < min_diameter) {
                min_diameter = max_dist[c];
                closest = c;
            }
        }

        if (closest < 0) break; // No more points can be added

        members.push_back(closest);
        max_dist[closest] = -1.0;
        added = closest;
    }

    return members;
}

// Device-side counterpart of growCandidateCluster(): block b grows the candidate
// cluster of seeds[b] and writes its cardinality to cardinalities[b].
//
// Only live candidates are visited at each step. A candidate whose maximum
// distance to the members has reached the threshold can never become admissible
// again (that maximum only grows), so it is dropped from the live list for good.
// Block b uses max_dist_scratch[b * N ...] and list_scratch[2 * b * N ...].
__global__ void candidateCardinalityKernel(const Point* points, const uint8_t* clustered,
                                           const int N, const double threshold,
                                           const int* seeds, double* max_dist_scratch,
                                           int* list_scratch, int* cardinalities) {
    __shared__ double red_dist[BLOCK_SIZE];
    __shared__ int red_idx[BLOCK_SIZE];
    __shared__ int added;
    __shared__ int live_len;
    __shared__ int next_len;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int seed = seeds[b];
    double* max_dist = max_dist_scratch + static_cast<size_t>(b) * N;
    int* live = list_scratch + static_cast<size_t>(b) * 2 * N;
    int* next = live + N;

    // Initially every unclustered point other than the seed is live
    if (tid == 0) {
        added = seed;
        live_len = 0;
        next_len = 0;
    }
    __syncthreads();
    for (int c = tid; c < N; c += BLOCK_SIZE) {
        if (!clustered[c] && c != seed) {
            max_dist[c] = 0.0;
            live[atomicAdd(&live_len, 1)] = c;
        }
    }
    __syncthreads();

    int cardinality = 1;
    while (true) {
        // Fold the most recently added member into each live candidate's max
        // distance; candidates that stay admissible are kept for the next step
        const Point& last = points[added];
        const int len = live_len;
        double best_dist = 0.0;
        int best = -1;
        for (int k = tid; k < len; k += BLOCK_SIZE) {
            const int c = live[k];
            if (c == added) continue; // The newest member is no longer a candidate

            const double d = fmax(max_dist[c], distance(points[c], last));
            max_dist[c] = d;
            if (d < threshold) {
                next[atomicAdd(&next_len, 1)] = c;
                if (best < 0 || d < best_dist || (d == best_dist && c < best)) {
                    best_dist = d;
                    best = c;
                }
            }
        }
        red_dist[tid] = best_dist;
        red_idx[tid] = best;
        __syncthreads();

        // Block-wide reduction: smallest distance wins, lowest index breaks ties
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const int mi = red_idx[tid];
                const int oi = red_idx[tid + s];
                const double md = red_dist[tid];
                const double od = red_dist[tid + s];
                if (oi >= 0 && (mi < 0 || od < md || (od == md && oi < mi))) {
                    red_dist[tid] = od;
                    red_idx[tid] = oi;
                }
            }
            __syncthreads();
        }

        if (red_idx[0] < 0) break; // No more points can be added

        if (tid == 0) {
            added = red_idx[0];
            live_len = next_len;
            next_len = 0;
        }
        cardinality++;
        int* tmp = live;
        live = next;
        next = tmp;
        __syncthreads();
    }

    if (tid == 0) cardinalities[b] = cardinality;
}

// Per-GPU state, owned by exactly one OpenMP thread during evaluation
struct DeviceWorker {
    int device = 0;
    Point* d_points = nullptr;
    uint8_t* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cardinalities = nullptr;
    double* d_scratch = nullptr;
    int* d_lists = nullptr;
    size_t batch_capacity = 0;
    std::vector<int> h_cardinalities;
};

// Choose the GPUs this MPI rank drives. Ranks sharing a node split the node's
// devices round-robin; every rank gets at least one device.
std::vector<int> selectDevices(MPI_Comm node_comm) {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    std::vector<int> devices;
    for (int d = local_rank; d < ndev; d += local_size) {
        devices.push_back(d);
    }
    if (devices.empty()) {
        devices.push_back(local_rank % ndev);
    }
    return devices;
}

// Allocate device buffers and upload the (read-only) point set to each device
std::vector<DeviceWorker> createWorkers(const std::vector<int>& devices,
                                        const std::vector<Point>& points,
                                        const size_t max_owned) {
    const size_t N = points.size();
    // Per block: max_dist (N doubles) plus two live lists (2 * N ints)
    const size_t bytes_per_seed = N * (sizeof(double) + 2 * sizeof(int));
    const size_t max_batch = std::max<size_t>(1, SCRATCH_BUDGET_BYTES / bytes_per_seed);
    const size_t batch = std::max<size_t>(1, std::min(max_batch, max_owned));

    std::vector<DeviceWorker> workers(devices.size());
    for (size_t t = 0; t < devices.size(); ++t) {
        DeviceWorker& w = workers[t];
        w.device = devices[t];
        w.batch_capacity = batch;
        w.h_cardinalities.assign(batch, 0);

        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaMalloc(&w.d_points, N * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&w.d_clustered, N * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&w.d_seeds, batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.d_cardinalities, batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.d_scratch, batch * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.d_lists, batch * 2 * N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(w.d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    }
    return workers;
}

void destroyWorkers(std::vector<DeviceWorker>& workers) {
    for (DeviceWorker& w : workers) {
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaFree(w.d_points));
        CUDA_CHECK(cudaFree(w.d_clustered));
        CUDA_CHECK(cudaFree(w.d_seeds));
        CUDA_CHECK(cudaFree(w.d_cardinalities));
        CUDA_CHECK(cudaFree(w.d_scratch));
        CUDA_CHECK(cudaFree(w.d_lists));
    }
}

// True if (card, seed) should replace the current best: larger cardinality wins
// and ties go to the lower seed index, matching the sequential scan order
inline bool isBetter(const int cardinality, const int seed, const Candidate& best) {
    return cardinality > best.cardinality ||
           (cardinality == best.cardinality && seed < best.seed);
}

// Evaluate the cardinality of the candidate cluster for every seed in `active`,
// split across this rank's GPUs (one OpenMP thread per device), and return the
// best candidate found.
Candidate evaluateActiveSeeds(std::vector<DeviceWorker>& workers,
                              const std::vector<int>& active,
                              const std::vector<uint8_t>& clustered,
                              const double threshold) {
    Candidate result{-1, -1};
    if (active.empty()) return result;

    const int N = static_cast<int>(clustered.size());
    const int T = static_cast<int>(workers.size());
    const size_t n = active.size();
    std::vector<Candidate> per_worker(T, Candidate{-1, -1});

    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        DeviceWorker& w = workers[t];
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaMemcpy(w.d_clustered, clustered.data(), N * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));

        const size_t lo = n * t / T;
        const size_t hi = n * (t + 1) / T;
        Candidate local{-1, -1};
        for (size_t off = lo; off < hi; off += w.batch_capacity) {
            const int m = static_cast<int>(std::min(w.batch_capacity, hi - off));
            CUDA_CHECK(cudaMemcpy(w.d_seeds, active.data() + off, m * sizeof(int),
                                  cudaMemcpyHostToDevice));
            candidateCardinalityKernel<<<m, BLOCK_SIZE>>>(w.d_points, w.d_clustered, N, threshold,
                                                          w.d_seeds, w.d_scratch, w.d_lists,
                                                          w.d_cardinalities);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(w.h_cardinalities.data(), w.d_cardinalities, m * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            for (int i = 0; i < m; ++i) {
                const int seed = active[off + i];
                if (isBetter(w.h_cardinalities[i], seed, local)) {
                    local = Candidate{w.h_cardinalities[i], seed};
                }
            }
        }
        per_worker[t] = local;
    }

    for (const Candidate& c : per_worker) {
        if (c.seed >= 0 && isBetter(c.cardinality, c.seed, result)) {
            result = c;
        }
    }
    return result;
}

// Main QT clustering algorithm. `owned` lists the seed indices this rank is
// responsible for, in ascending order.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const std::vector<int>& owned,
                                  std::vector<DeviceWorker>& workers,
                                  const int nranks) {
    const int N = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(N, 0);
    std::vector<Cluster> clusters;
    std::vector<int> active;
    active.reserve(owned.size());
    std::vector<int> gathered(2 * nranks);

    // Main clustering loop
    while (true) {
        // Unclustered seeds owned by this rank
        active.clear();
        for (int seed : owned) {
            if (!clustered[seed]) active.push_back(seed);
        }

        // Best candidate over all ranks; every rank derives the same result
        const Candidate local = evaluateActiveSeeds(workers, active, clustered, threshold);
        const int send[2] = {local.cardinality, local.seed};
        MPI_Allgather(send, 2, MPI_INT, gathered.data(), 2, MPI_INT, MPI_COMM_WORLD);

        Candidate best{-1, -1};
        for (int r = 0; r < nranks; ++r) {
            const Candidate c{gathered[2 * r], gathered[2 * r + 1]};
            if (c.seed >= 0 && isBetter(c.cardinality, c.seed, best)) {
                best = c;
            }
        }

        // No more clusters can be formed
        if (best.seed < 0 || best.cardinality <= 0) break;

        Cluster cluster;
        cluster.seed_point = best.seed;
        cluster.members = growCandidateCluster(best.seed, clustered, points, threshold);
        if (static_cast<int>(cluster.members.size()) != best.cardinality) {
            fprintf(stderr, "Internal error: device/host cluster cardinality mismatch\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Mark all members as clustered
        for (int member : cluster.members) {
            clustered[member] = 1;
        }
        clusters.push_back(std::move(cluster));
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Diameters are independent per cluster, so compute them in parallel
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
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
        diameters[c] = max_diameter;
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

// Benchmark driver; runs identically on every rank, with output from rank 0 only
int runBenchmark(int argc, char** argv, int rank, int nranks) {
    const bool root = rank == 0;
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }

    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Map ranks onto the GPUs of their node
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    const std::vector<int> devices = selectDevices(node_comm);
    MPI_Comm_free(&node_comm);

    // Generate synthetic data (deterministic, so identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Cyclic distribution of seeds across ranks for load balance
    std::vector<int> owned;
    for (int i = rank; i < num_points; i += nranks) {
        owned.push_back(i);
    }

    std::vector<DeviceWorker> workers = createWorkers(devices, points, owned.size());

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, owned, workers, nranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    destroyWorkers(workers);

    if (root) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }

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

    if (root) {
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
    }

    // Print results for external validation
    if (printResults && root) {
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

    // Validation (performed on rank 0; result shared so all ranks agree on exit status)
    if (validate) {
        int valid = 0;
        if (root) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (root) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Fixed thread count per parallel region: one thread per device
    omp_set_dynamic(0);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int ret = runBenchmark(argc, argv, rank, nranks);
    MPI_Finalize();
    return ret;
}
