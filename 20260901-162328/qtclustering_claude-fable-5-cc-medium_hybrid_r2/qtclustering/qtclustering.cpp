// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - Each round, every unclustered point is tried as a seed. Seeds are
//    distributed cyclically across MPI ranks (points are replicated).
//  - Within a rank, seeds are processed by a work-stealing pool: OpenMP
//    thread 0 drives the local GPU (one CUDA block grows one candidate
//    cluster), while the remaining OpenMP threads grow candidate clusters
//    on the CPU.
//  - The globally best (cardinality, seed) pair is selected with an
//    MPI_MAXLOC reduction, which matches the sequential tie-break exactly
//    (largest cluster, smallest seed index on ties).
//
// Candidate growth uses incremental max-distance tracking and pre-filters
// candidates to points within `threshold` of the seed; both are exact
// transformations of the original greedy algorithm (identical selections
// and tie-breaks). Host code is compiled with -ffp-contract=off and device
// code with -fmad=false so CPU and GPU distances are bit-identical, making
// the result independent of the CPU/GPU work split.

#include <algorithm>
#include <atomic>
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

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA kernel: one block grows one candidate cluster from one seed.
//
// Per block workspace (in global memory): md[] holds, for each surviving
// candidate, the maximum distance to the cluster members so far; cand[]
// holds the candidate's point index. Candidates are pre-filtered to points
// within `thr` of the seed (any other point can never join). Each growth
// step fuses the max-distance update with a lexicographic argmin reduction
// over (max_dist, point index), which reproduces the sequential selection
// including its tie-break, independent of candidate ordering.
// ---------------------------------------------------------------------------
constexpr int TPB = 256;

__global__ void growClustersKernel(const double2* __restrict__ pts,
                                   const unsigned char* __restrict__ clustered,
                                   const int N, const double thr,
                                   const int* __restrict__ seeds,
                                   double* __restrict__ workMd,
                                   int* __restrict__ workCand,
                                   int* __restrict__ cards) {
    const int tid = threadIdx.x;
    const int seed = seeds[blockIdx.x];
    double* md = workMd + static_cast<size_t>(blockIdx.x) * N;
    int* cand = workCand + static_cast<size_t>(blockIdx.x) * N;

    __shared__ int sL;      // number of surviving candidates
    __shared__ int sCard;   // current cluster cardinality
    __shared__ int sCur;    // most recently added member
    __shared__ double rd[TPB];
    __shared__ int ri[TPB];
    __shared__ int rj[TPB];

    if (tid == 0) {
        sL = 0;
        sCard = 1;
        sCur = seed;
    }
    __syncthreads();

    // Pre-filter: only unclustered points within thr of the seed can ever join
    const double2 sp = pts[seed];
    for (int c = tid; c < N; c += TPB) {
        if (clustered[c] || c == seed) continue;
        const double dx = pts[c].x - sp.x;
        const double dy = pts[c].y - sp.y;
        const double d = sqrt(dx * dx + dy * dy);
        if (d < thr) {
            const int j = atomicAdd(&sL, 1);
            cand[j] = c;
            md[j] = d;
        }
    }
    __syncthreads();

    while (true) {
        // Fused pass: fold in distances to the newest member, then select the
        // candidate with the smallest max-distance (ties -> smallest index).
        const int L = sL;
        const double2 mp = pts[sCur];
        double bd = 0.0;
        int bi = INT_MAX;
        int bj = -1;
        for (int j = tid; j < L; j += TPB) {
            double m = md[j];
            if (m >= thr) continue;  // permanently dead candidate
            const int c = cand[j];
            const double dx = pts[c].x - mp.x;
            const double dy = pts[c].y - mp.y;
            const double d = sqrt(dx * dx + dy * dy);
            if (d > m) {
                m = d;
                md[j] = m;
            }
            if (m < thr && (bi == INT_MAX || m < bd || (m == bd && c < bi))) {
                bd = m;
                bi = c;
                bj = j;
            }
        }
        rd[tid] = bd;
        ri[tid] = bi;
        rj[tid] = bj;
        __syncthreads();
        for (int s = TPB / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const bool other = (ri[tid + s] != INT_MAX) &&
                    (ri[tid] == INT_MAX || rd[tid + s] < rd[tid] ||
                     (rd[tid + s] == rd[tid] && ri[tid + s] < ri[tid]));
                if (other) {
                    rd[tid] = rd[tid + s];
                    ri[tid] = ri[tid + s];
                    rj[tid] = rj[tid + s];
                }
            }
            __syncthreads();
        }
        if (ri[0] == INT_MAX) break;  // no candidate can be added

        if (tid == 0) {
            sCard++;
            sCur = ri[0];
            // swap-remove the selected candidate
            const int last = sL - 1;
            const int j = rj[0];
            md[j] = md[last];
            cand[j] = cand[last];
            sL = last;
        }
        __syncthreads();
    }

    if (tid == 0) cards[blockIdx.x] = sCard;
}

// ---------------------------------------------------------------------------
// CPU candidate-cluster growth, bit-identical to the GPU kernel (same
// pre-filter, incremental max-distance, and (max_dist, index) tie-break).
// Returns the cardinality; optionally records the member list.
// ---------------------------------------------------------------------------
static int growClusterCPU(const int seed,
                          const unsigned char* clustered,
                          const Point* pts,
                          const int N,
                          const double thr,
                          double* md, int* cand,
                          std::vector<int>* members = nullptr) {
    int L = 0;
    for (int c = 0; c < N; ++c) {
        if (clustered[c] || c == seed) continue;
        const double d = distance(pts[c], pts[seed]);
        if (d < thr) {
            cand[L] = c;
            md[L] = d;
            ++L;
        }
    }

    if (members) {
        members->clear();
        members->push_back(seed);
    }

    int card = 1;
    int cur = seed;
    while (true) {
        double bd = 0.0;
        int bi = INT_MAX;
        int bj = -1;
        for (int j = 0; j < L; ++j) {
            double m = md[j];
            if (m >= thr) continue;
            const int c = cand[j];
            const double d = distance(pts[c], pts[cur]);
            if (d > m) {
                m = d;
                md[j] = m;
            }
            if (m < thr && (bi == INT_MAX || m < bd || (m == bd && c < bi))) {
                bd = m;
                bi = c;
                bj = j;
            }
        }
        if (bi == INT_MAX) break;

        ++card;
        cur = bi;
        if (members) members->push_back(bi);
        const int last = L - 1;
        md[bj] = md[last];
        cand[bj] = cand[last];
        L = last;
    }
    return card;
}

// Per-rank GPU context (points uploaded once; workspaces reused each round)
struct GpuContext {
    double2* d_pts = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    double* d_workMd = nullptr;
    int* d_workCand = nullptr;
    int chunkCap = 0;  // max concurrent seeds (blocks) per kernel launch
};

static void initGpu(GpuContext& gpu, const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());

    CUDA_CHECK(cudaMalloc(&gpu.d_pts, static_cast<size_t>(N) * sizeof(double2)));
    CUDA_CHECK(cudaMemcpy(gpu.d_pts, points.data(),
                          static_cast<size_t>(N) * sizeof(double2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&gpu.d_clustered, N));

    // Size the per-seed workspace to available memory (md: 8B + cand: 4B per
    // point per concurrent block), capped for reasonable launch granularity.
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t perSeed = static_cast<size_t>(N) * 12 + 8;
    size_t cap = (freeMem * 7 / 10) / perSeed;
    cap = std::min(cap, static_cast<size_t>(4096));
    cap = std::min(cap, static_cast<size_t>(N));
    cap = std::max(cap, static_cast<size_t>(1));
    gpu.chunkCap = static_cast<int>(cap);

    CUDA_CHECK(cudaMalloc(&gpu.d_seeds, cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_cards, cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_workMd, cap * static_cast<size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_workCand, cap * static_cast<size_t>(N) * sizeof(int)));
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  GpuContext& gpu,
                                  const int mpi_rank, const int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    CUDA_CHECK(cudaMemcpy(gpu.d_clustered, clustered.data(), N,
                          cudaMemcpyHostToDevice));

    // Reusable host-side buffers
    std::vector<int> mySeeds;
    std::vector<int> h_cards(gpu.chunkCap);
    std::vector<double> regrowMd(N);
    std::vector<int> regrowCand(N);
    std::vector<int> best_members;

    // Main clustering loop: one iteration extracts one cluster
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        // This rank's share of the seeds (cyclic distribution)
        mySeeds.clear();
        for (int p = mpi_rank; p < U; p += mpi_size) {
            mySeeds.push_back(unclustered_indices[p]);
        }
        const int M = static_cast<int>(mySeeds.size());

        int local_card = -1;
        int local_seed = INT_MAX;

        std::atomic<int> next(0);
        const int gpuChunk = std::min(gpu.chunkCap, 1024);

        #pragma omp parallel
        {
            int t_card = -1;
            int t_seed = INT_MAX;

            if (omp_get_thread_num() == 0) {
                // GPU driver thread: grab chunks of seeds, one CUDA block each.
                // Chunks shrink as the queue drains so CPU workers can absorb
                // the tail instead of waiting on one large final kernel.
                while (true) {
                    const int remaining = M - next.load(std::memory_order_relaxed);
                    if (remaining <= 0) break;
                    const int want = std::max(32, std::min(gpuChunk, remaining / 3));
                    const int start = next.fetch_add(want);
                    if (start >= M) break;
                    const int n = std::min(want, M - start);
                    CUDA_CHECK(cudaMemcpy(gpu.d_seeds, mySeeds.data() + start,
                                          n * sizeof(int), cudaMemcpyHostToDevice));
                    growClustersKernel<<<n, TPB>>>(gpu.d_pts, gpu.d_clustered,
                                                   N, threshold, gpu.d_seeds,
                                                   gpu.d_workMd, gpu.d_workCand,
                                                   gpu.d_cards);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpy(h_cards.data(), gpu.d_cards,
                                          n * sizeof(int), cudaMemcpyDeviceToHost));
                    for (int k = 0; k < n; ++k) {
                        const int card = h_cards[k];
                        const int seed = mySeeds[start + k];
                        if (card > t_card || (card == t_card && seed < t_seed)) {
                            t_card = card;
                            t_seed = seed;
                        }
                    }
                }
            } else {
                // CPU worker threads: grow candidate clusters host-side
                std::vector<double> md(N);
                std::vector<int> cand(N);
                int start;
                while ((start = next.fetch_add(2)) < M) {
                    const int end = std::min(start + 2, M);
                    for (int k = start; k < end; ++k) {
                        const int seed = mySeeds[k];
                        const int card = growClusterCPU(seed, clustered.data(),
                                                        points.data(), N, threshold,
                                                        md.data(), cand.data());
                        if (card > t_card || (card == t_card && seed < t_seed)) {
                            t_card = card;
                            t_seed = seed;
                        }
                    }
                }
            }

            #pragma omp critical
            {
                if (t_card > local_card ||
                    (t_card == local_card && t_seed < local_seed)) {
                    local_card = t_card;
                    local_seed = t_seed;
                }
            }
        }

        // Global best: max cardinality, ties broken by smallest seed index
        // (MPI_MAXLOC picks the smaller "location" on equal values), which is
        // exactly the sequential iteration order's tie-break.
        struct { int card; int seed; } best = { local_card, local_seed };
        MPI_Allreduce(MPI_IN_PLACE, &best, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        // Regrow the winning cluster deterministically on every rank to
        // obtain its member list (avoids broadcasting variable-length data).
        best_members.clear();
        const int regrown = growClusterCPU(best.seed, clustered.data(),
                                           points.data(), N, threshold,
                                           regrowMd.data(), regrowCand.data(),
                                           &best_members);
        if (regrown != best.card && mpi_rank == 0) {
            fprintf(stderr,
                    "WARNING: regrown cardinality %d != selected %d (seed %d)\n",
                    regrown, best.card, best.seed);
        }

        Cluster cluster;
        cluster.seed_point = best.seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (size_t i = 0; i < best_members.size(); ++i) {
            clustered[best_members[i]] = 1;
        }
        CUDA_CHECK(cudaMemcpy(gpu.d_clustered, clustered.data(), N,
                              cudaMemcpyHostToDevice));

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

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        // Check diameter (max distance between any two points)
        #pragma omp parallel for reduction(max:max_diameter) schedule(dynamic)
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

    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int rc = 0;
    bool exitEarly = false;
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
            exitEarly = true;
            break;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            rc = 1;
            exitEarly = true;
            break;
        }
    }

    if (!exitEarly && (num_points <= 0 || threshold <= 0.0)) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        rc = 1;
        exitEarly = true;
    }

    if (exitEarly) {
        MPI_Finalize();
        return rc;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n",
               mpi_size, omp_get_max_threads());
    }

    // Select one GPU per rank (round-robin over local ranks)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaSetDevice(local_rank % deviceCount));

    // Generate synthetic data (deterministic; replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    GpuContext gpu;
    initGpu(gpu, points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, gpu, mpi_rank, mpi_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (mpi_rank == 0) {
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

    // Free GPU resources
    cudaFree(gpu.d_pts);
    cudaFree(gpu.d_clustered);
    cudaFree(gpu.d_seeds);
    cudaFree(gpu.d_cards);
    cudaFree(gpu.d_workMd);
    cudaFree(gpu.d_workCand);

    // All ranks return the same exit code
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
