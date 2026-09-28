// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - MPI: candidate seeds are statically partitioned across ranks
//    (seed % nranks); each round the globally best candidate cluster is
//    selected with an MPI_MAXLOC allreduce (ties broken by lowest seed
//    index, exactly as in the sequential scan) and its members are
//    broadcast from the owning rank.
//  - CUDA: each rank grows all of its candidate clusters on its GPU, one
//    thread block per seed. Every block keeps an incremental
//    "max distance to current members" value per candidate point, so each
//    greedy step is a parallel argmin over N candidates instead of an
//    O(N * |cluster|) rescan. Ties in the argmin are resolved
//    lexicographically (distance, index), matching the sequential loop.
//  - OpenMP: host-side bookkeeping (marking members, seed-cache
//    invalidation) and result validation.
//
// A per-seed cardinality cache avoids regrowing candidate clusters that
// cannot have changed: a seed's greedy growth only ever selects points
// strictly within `threshold` of the seed, so if none of the points removed
// in a round lie within that radius, the seed's candidate cluster (and
// cardinality) is unchanged. This is a pure optimization; results are
// identical to the sequential algorithm.

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

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err__), __FILE__, __LINE__);         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
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

// Calculate Euclidean distance between two points (host)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA kernel: grow one candidate cluster per thread block
// ---------------------------------------------------------------------------

static const int BLOCK = 256;

__device__ inline double devDistance(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Block b grows the candidate cluster seeded at seeds[b]. scratch holds one
// double per (block, point): the maximum distance from that point to the
// current cluster members, or DBL_MAX once the point is either in the
// cluster, already globally clustered, or permanently inadmissible (its max
// distance reached the threshold; it can only grow, so it never becomes
// admissible again). Cardinalities are written per block; members_out (used
// only for single-seed regeneration launches) records the insertion order.
__global__ void growClustersKernel(const double2* __restrict__ points,
                                   const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ seeds,
                                   const int point_count,
                                   const double threshold,
                                   double* __restrict__ scratch,
                                   int* __restrict__ cardinalities,
                                   int* __restrict__ members_out) {
    const int tid = threadIdx.x;
    const int seed = seeds[blockIdx.x];
    double* best = scratch + static_cast<size_t>(blockIdx.x) * point_count;

    __shared__ double sdist[BLOCK];
    __shared__ int sidx[BLOCK];
    __shared__ double2 added;

    const double2 sp = points[seed];
    for (int c = tid; c < point_count; c += BLOCK) {
        double d = DBL_MAX;
        if (!clustered[c] && c != seed) {
            const double dd = devDistance(points[c], sp);
            if (dd < threshold) d = dd;
        }
        best[c] = d;
    }
    if (members_out != nullptr && tid == 0) {
        members_out[0] = seed;
    }
    __syncthreads();

    int count = 1;
    while (count < point_count) {
        // Parallel argmin over admissible candidates. Each thread scans its
        // strided slice in ascending index order with a strict '<', then the
        // block reduction breaks distance ties by lower index - identical to
        // the sequential candidate scan.
        double md = DBL_MAX;
        int mi = point_count;  // sentinel: no admissible candidate
        for (int c = tid; c < point_count; c += BLOCK) {
            const double d = best[c];
            if (d < md) {
                md = d;
                mi = c;
            }
        }
        sdist[tid] = md;
        sidx[tid] = mi;
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (sdist[tid + s] < sdist[tid] ||
                    (sdist[tid + s] == sdist[tid] && sidx[tid + s] < sidx[tid])) {
                    sdist[tid] = sdist[tid + s];
                    sidx[tid] = sidx[tid + s];
                }
            }
            __syncthreads();
        }
        const int chosen = sidx[0];
        if (chosen >= point_count) break;  // no more points can be added

        if (tid == 0) {
            best[chosen] = DBL_MAX;
            added = points[chosen];
            if (members_out != nullptr) {
                members_out[count] = chosen;
            }
        }
        __syncthreads();

        // Incrementally update each candidate's max distance to the cluster.
        const double2 np = added;
        for (int c = tid; c < point_count; c += BLOCK) {
            const double b = best[c];
            if (b < DBL_MAX) {
                const double d = devDistance(points[c], np);
                if (d > b) {
                    best[c] = (d < threshold) ? d : DBL_MAX;
                }
            }
        }
        ++count;
        __syncthreads();
    }

    if (tid == 0) {
        cardinalities[blockIdx.x] = count;
    }
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<Cluster> clusters;  // populated on rank 0 only

    // Per-seed cached cardinality of the candidate cluster grown from that
    // seed against the current clustered set (owned seeds only).
    std::vector<int> card_cache(N, 0);
    std::vector<unsigned char> cache_valid(N, 0);

    // --- Device setup ---
    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    int* d_members = nullptr;
    double* d_scratch = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));

    {
        std::vector<double2> h_pts(N);
        for (int i = 0; i < N; ++i) h_pts[i] = make_double2(points[i].x, points[i].y);
        CUDA_CHECK(cudaMemcpy(d_points, h_pts.data(), sizeof(double2) * N,
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), sizeof(unsigned char) * N,
                          cudaMemcpyHostToDevice));

    // Scratch memory bounds how many seeds one kernel launch can grow.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = free_bytes / 2;
    int batch_cap = static_cast<int>(std::min<size_t>(
        static_cast<size_t>(N),
        std::max<size_t>(1, budget / (sizeof(double) * static_cast<size_t>(N)))));
    CUDA_CHECK(cudaMalloc(&d_scratch,
                          sizeof(double) * static_cast<size_t>(batch_cap) * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * batch_cap));
    CUDA_CHECK(cudaMalloc(&d_cards, sizeof(int) * batch_cap));

    std::vector<int> h_batch_seeds(batch_cap);
    std::vector<int> h_batch_cards(batch_cap);
    std::vector<int> best_members(N);

    // Slightly enlarged radius for the conservative cache-invalidation test
    // so host-side rounding can never keep a stale entry alive.
    const double inval_r2 = threshold * threshold * (1.0 + 1e-9);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Grow (or reuse cached) candidate clusters for the seeds this rank
        // owns, batched to fit the GPU scratch buffer.
        std::vector<int> to_compute;
        for (const int s : unclustered_indices) {
            if (s % nranks == rank && !cache_valid[s]) to_compute.push_back(s);
        }

        for (size_t off = 0; off < to_compute.size();
             off += static_cast<size_t>(batch_cap)) {
            const int nb = static_cast<int>(
                std::min<size_t>(batch_cap, to_compute.size() - off));
            std::copy(to_compute.begin() + off, to_compute.begin() + off + nb,
                      h_batch_seeds.begin());
            CUDA_CHECK(cudaMemcpy(d_seeds, h_batch_seeds.data(), sizeof(int) * nb,
                                  cudaMemcpyHostToDevice));
            growClustersKernel<<<nb, BLOCK>>>(d_points, d_clustered, d_seeds, N,
                                              threshold, d_scratch, d_cards,
                                              nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_batch_cards.data(), d_cards, sizeof(int) * nb,
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < nb; ++i) {
                const int s = h_batch_seeds[i];
                card_cache[s] = h_batch_cards[i];
                cache_valid[s] = 1;
            }
        }

        // Local best candidate: max cardinality, ties -> lowest seed index
        // (unclustered_indices stays sorted, so the strict '>' does this).
        int best_card = -1;
        int best_seed = INT_MAX;
        for (const int s : unclustered_indices) {
            if (s % nranks != rank) continue;
            if (card_cache[s] > best_card) {
                best_card = card_cache[s];
                best_seed = s;
            }
        }

        // Global best: MPI_MAXLOC picks the maximum cardinality and, on
        // ties, the minimum seed index - matching the sequential seed scan.
        struct {
            int card;
            int seed;
        } local_best = {best_card, best_seed}, global_best;
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        const int winner_seed = global_best.seed;
        const int winner_card = global_best.card;
        const int owner = winner_seed % nranks;

        // The owning rank regenerates the winning cluster once more to
        // recover the member list, then broadcasts it.
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(d_seeds, &winner_seed, sizeof(int),
                                  cudaMemcpyHostToDevice));
            growClustersKernel<<<1, BLOCK>>>(d_points, d_clustered, d_seeds, N,
                                             threshold, d_scratch, d_cards,
                                             d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(best_members.data(), d_members,
                                  sizeof(int) * winner_card,
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(best_members.data(), winner_card, MPI_INT, owner,
                  MPI_COMM_WORLD);

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = winner_seed;
            cluster.members.assign(best_members.begin(),
                                   best_members.begin() + winner_card);
            clusters.push_back(std::move(cluster));
        }

        // Mark all members as clustered
        for (int i = 0; i < winner_card; ++i) {
            clustered[best_members[i]] = 1;
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                              sizeof(unsigned char) * N, cudaMemcpyHostToDevice));

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());

        // Invalidate cached cardinalities for owned seeds whose candidate
        // cluster may contain a removed point: growth only ever selects
        // points strictly within `threshold` of the seed, so seeds farther
        // than that from every removed point are unaffected.
        const int remaining = static_cast<int>(unclustered_indices.size());
        const long inval_work = static_cast<long>(remaining) * winner_card;
        #pragma omp parallel for schedule(dynamic, 64) if (inval_work > 65536)
        for (int u = 0; u < remaining; ++u) {
            const int s = unclustered_indices[u];
            if (s % nranks != rank || !cache_valid[s]) continue;
            const double sx = points[s].x;
            const double sy = points[s].y;
            for (int i = 0; i < winner_card; ++i) {
                const double dx = points[best_members[i]].x - sx;
                const double dy = points[best_members[i]].y - sy;
                if (dx * dx + dy * dy <= inval_r2) {
                    cache_valid[s] = 0;
                    break;
                }
            }
        }
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_cards));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_scratch));

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
        const int sz = static_cast<int>(cluster.members.size());

        // Check diameter (max distance between any two points)
        #pragma omp parallel for reduction(max : max_diameter) schedule(dynamic, 16)
        for (int i = 0; i < sz; ++i) {
            for (int j = i + 1; j < sz; ++j) {
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

    // Bind each rank to a GPU (round-robin over the node-local ranks).
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &local_comm);
        int local_rank = 0;
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);

        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev < 1) {
            if (rank == 0) fprintf(stderr, "Error: no CUDA device found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % ndev));
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
    }

    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

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
    }

    // Validation (performed on rank 0, verdict shared for a uniform exit code)
    int exit_code = 0;
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exit_code = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exit_code;
}
