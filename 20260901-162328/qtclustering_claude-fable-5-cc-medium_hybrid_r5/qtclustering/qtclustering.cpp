// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - Each round, every unclustered point is tried as a seed for greedy
//    cluster growth. The seeds are partitioned across MPI ranks (one GPU
//    per rank), and each rank grows its candidate clusters on the GPU with
//    one thread block per seed. Growth uses an incremental per-candidate
//    max-distance array, so each seed costs O(k*n) instead of O(k^2*n)
//    while producing bit-identical distances (max is exact).
//  - The globally best (cardinality, seed) pair is combined with
//    MPI_MAXLOC, whose lowest-index tie-break matches the sequential
//    "first seed with strictly larger cardinality wins" semantics.
//  - Every rank deterministically regrows the winning cluster on the host
//    using OpenMP-parallel argmin/update sweeps, avoiding a member
//    broadcast. FP contraction is disabled on host and device so both
//    compute identical distances.

#include <algorithm>
#include <chrono>
#include <cfloat>
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

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

constexpr int TPB = 256;

// Grow one candidate cluster per thread block. dmax[j] tracks the maximum
// distance from unclustered candidate j to the current cluster members
// (DBL_MAX marks excluded points). Each step selects the candidate with the
// smallest dmax below the threshold (ties -> lowest index, matching the
// sequential scan) and folds the new member's distances into dmax.
// best_seen holds the largest cardinality reached by any block so far this
// round (updated with atomicMax as clusters grow). A block may stop early
// once current size + remaining admissible candidates < best_seen: it can
// then never win the round, and the strict inequality preserves the
// lowest-seed tie-break of the sequential scan.
__global__ void growCandidateClusters(const Point* __restrict__ points,
                                      const unsigned char* __restrict__ clustered,
                                      const int* __restrict__ seeds,
                                      const int point_count,
                                      const double threshold,
                                      double* __restrict__ dmax_all,
                                      int* __restrict__ cardinalities,
                                      int* __restrict__ best_seen) {
    const int seed = seeds[blockIdx.x];
    double* dmax = dmax_all + static_cast<size_t>(blockIdx.x) * point_count;
    const Point sp = points[seed];

    for (int j = threadIdx.x; j < point_count; j += blockDim.x) {
        dmax[j] = (clustered[j] || j == seed) ? DBL_MAX
                                              : distance(points[j], sp);
    }

    __shared__ double sval[TPB];
    __shared__ int sidx[TPB];
    __shared__ int scnt[TPB];
    __shared__ int chosen;

    int cardinality = 1;
    while (cardinality < point_count) {
        __syncthreads();
        // Per-thread strided argmin + admissible count over candidates
        double best = DBL_MAX;
        int best_idx = INT_MAX;
        int cnt = 0;
        for (int j = threadIdx.x; j < point_count; j += blockDim.x) {
            const double v = dmax[j];
            if (v < threshold) {
                cnt++;
                if (v < best || (v == best && j < best_idx)) {
                    best = v;
                    best_idx = j;
                }
            }
        }
        sval[threadIdx.x] = best;
        sidx[threadIdx.x] = best_idx;
        scnt[threadIdx.x] = cnt;
        __syncthreads();
        // Block-level (min value, min index) and sum reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double ov = sval[threadIdx.x + stride];
                const int oi = sidx[threadIdx.x + stride];
                if (ov < sval[threadIdx.x] ||
                    (ov == sval[threadIdx.x] && oi < sidx[threadIdx.x])) {
                    sval[threadIdx.x] = ov;
                    sidx[threadIdx.x] = oi;
                }
                scnt[threadIdx.x] += scnt[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int prev_best = atomicMax(best_seen, cardinality);
            if (cardinality + scnt[0] < max(prev_best, cardinality)) {
                chosen = -1; // Cannot win this round; stop early
            } else {
                chosen = (sidx[0] != INT_MAX) ? sidx[0] : -1;
            }
            if (chosen >= 0) dmax[chosen] = DBL_MAX;
        }
        __syncthreads();
        if (chosen < 0) break; // No more points can be added

        cardinality++;
        const Point np = points[chosen];
        for (int j = threadIdx.x; j < point_count; j += blockDim.x) {
            const double v = dmax[j];
            if (v != DBL_MAX) {
                const double d = distance(points[j], np);
                if (d > v) dmax[j] = d;
            }
        }
    }

    if (threadIdx.x == 0) {
        cardinalities[blockIdx.x] = cardinality;
    }
}

// Deterministically regrow the winning candidate cluster on the host,
// returning its members in insertion order (identical to the sequential
// algorithm). The argmin and update sweeps are OpenMP-parallel.
static std::vector<int> growClusterHost(const int seed,
                                        const std::vector<unsigned char>& clustered,
                                        const std::vector<Point>& points,
                                        const double threshold,
                                        const int point_count) {
    std::vector<double> dmax(point_count);
    const Point sp = points[seed];

    std::vector<int> members;
    members.reserve(point_count);
    members.push_back(seed);

    const int nthreads = omp_get_max_threads();
    std::vector<double> tbest(nthreads);
    std::vector<int> tidx(nthreads);
    Point next = sp;
    bool done = false;

    // One persistent parallel region for the whole growth loop; the
    // sequential decision per step happens in a single block.
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int j = 0; j < point_count; ++j) {
            dmax[j] = (clustered[j] || j == seed) ? DBL_MAX
                                                  : distance(points[j], sp);
        }

        while (true) {
            double lbest = DBL_MAX;
            int lidx = INT_MAX;
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < point_count; ++j) {
                const double v = dmax[j];
                if (v < threshold && (v < lbest || (v == lbest && j < lidx))) {
                    lbest = v;
                    lidx = j;
                }
            }
            tbest[tid] = lbest;
            tidx[tid] = lidx;
            #pragma omp barrier

            #pragma omp single
            {
                double best = DBL_MAX;
                int best_idx = INT_MAX;
                for (int t = 0; t < nthreads; ++t) {
                    if (tbest[t] < best || (tbest[t] == best && tidx[t] < best_idx)) {
                        best = tbest[t];
                        best_idx = tidx[t];
                    }
                }
                if (best_idx == INT_MAX) {
                    done = true; // No more points can be added
                } else {
                    members.push_back(best_idx);
                    dmax[best_idx] = DBL_MAX;
                    next = points[best_idx];
                    if (static_cast<int>(members.size()) >= point_count) {
                        done = true;
                    }
                }
            } // implicit barrier

            if (done) break;

            #pragma omp for schedule(static)
            for (int j = 0; j < point_count; ++j) {
                const double v = dmax[j];
                if (v != DBL_MAX) {
                    const double d = distance(points[j], next);
                    if (d > v) dmax[j] = d;
                }
            }
        }
    }

    return members;
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Device buffers: points, clustered flags, seed batch, per-seed
    // max-distance workspace (batched to fit memory), cardinalities.
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    double* d_dmax = nullptr;
    int* d_cards = nullptr;
    int* d_best = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t batch_cap = (free_mem / 10 * 8) / (sizeof(double) * static_cast<size_t>(N));
    if (batch_cap < 1) batch_cap = 1;
    if (batch_cap > static_cast<size_t>(N)) batch_cap = N;

    CUDA_CHECK(cudaMalloc(&d_dmax, sizeof(double) * batch_cap * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * batch_cap));
    CUDA_CHECK(cudaMalloc(&d_cards, sizeof(int) * batch_cap));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(int)));

    std::vector<int> h_cards(batch_cap);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const size_t total = unclustered_indices.size();
        // Contiguous partition of the (ascending) seed list across ranks
        const size_t begin = total * static_cast<size_t>(rank) / nranks;
        const size_t end = total * static_cast<size_t>(rank + 1) / nranks;

        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), N,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_best, 0, sizeof(int)));

        int max_cardinality = -1;
        int best_seed = INT_MAX;

        // Try each seed in this rank's partition, in GPU batches
        for (size_t b = begin; b < end; b += batch_cap) {
            const int count = static_cast<int>(std::min(batch_cap, end - b));
            CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data() + b,
                                  sizeof(int) * count, cudaMemcpyHostToDevice));
            growCandidateClusters<<<count, TPB>>>(d_points, d_clustered, d_seeds,
                                                  N, threshold, d_dmax, d_cards,
                                                  d_best);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_cards.data(), d_cards, sizeof(int) * count,
                                  cudaMemcpyDeviceToHost));

            for (int i = 0; i < count; ++i) {
                if (h_cards[i] > max_cardinality) {
                    max_cardinality = h_cards[i];
                    best_seed = unclustered_indices[b + i];
                }
            }
        }

        // Global best: max cardinality, ties broken by lowest seed index
        // (MPI_MAXLOC), matching the sequential scan order.
        int best_pair[2] = {max_cardinality, best_seed};
        MPI_Allreduce(MPI_IN_PLACE, best_pair, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        // If we found a cluster, add it
        if (best_pair[1] != INT_MAX && best_pair[0] > 0) {
            Cluster cluster;
            cluster.seed_point = best_pair[1];
            cluster.members = growClusterHost(best_pair[1], clustered, points,
                                              threshold, N);

            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = 1;
            }

            clusters.push_back(std::move(cluster));

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_dmax));
    CUDA_CHECK(cudaFree(d_cards));
    CUDA_CHECK(cudaFree(d_best));

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
        #pragma omp parallel for schedule(dynamic) reduction(max:max_diameter)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // One GPU per rank on each node (round-robin over local ranks)
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    // Share the node's cores between the ranks on it unless the user
    // explicitly requested a thread count.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_max_threads() / local_size));
    }

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
        printf("MPI ranks: %d, OpenMP threads: %d, GPUs/node: %d\n",
               nranks, omp_get_max_threads(), device_count);
    }

    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    int rc = 0;
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
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
