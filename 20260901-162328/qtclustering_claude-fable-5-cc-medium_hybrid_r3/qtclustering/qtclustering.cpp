// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - Each round, every unclustered point is evaluated as a candidate seed.
//    Seeds are distributed round-robin across MPI ranks (one GPU per rank).
//  - On each rank, a CUDA kernel grows one candidate cluster per thread
//    block; threads within a block cooperate on the argmin reduction and
//    the distance updates over all unclustered points.
//  - The globally best cluster (max cardinality, ties broken by lowest seed
//    index, exactly as in the sequential code) is selected with an
//    MPI_MAXLOC all-reduce. Its member list is then regenerated redundantly
//    on every rank on the CPU using OpenMP, so no member data needs to be
//    communicated.
//  - Device code is compiled with -fmad=false and host code with
//    -ffp-contract=off so CPU and GPU distances are bit-identical and the
//    result matches the sequential algorithm exactly.

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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
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

// Sentinel marking a point that is already a member of the growing cluster
// (or otherwise ineligible). Always >= any real distance and any threshold.
static const double EXCLUDED = DBL_MAX;

static const int TPB = 256;       // threads per block
static const int MAX_GRID = 512;  // max concurrent candidate clusters per GPU

// Grow one candidate cluster per thread block. Points are the compacted
// (unclustered-only) set; coordinates in ux/uy. Block k handles compact seed
// index firstSeed + k*seedStride (round-robin MPI distribution), grid-striding
// over the rank's seeds. dist[j] tracks the max distance from point j to all
// current cluster members; each step adds the point with minimal dist[j]
// subject to dist[j] < threshold (ties -> lowest index), matching the
// sequential algorithm. Only the resulting cardinality is stored.
__global__ void candidateClusterKernel(const double* __restrict__ ux,
                                       const double* __restrict__ uy,
                                       const int U,
                                       const double threshold,
                                       const int firstSeed,
                                       const int seedStride,
                                       const int numLocalSeeds,
                                       double* __restrict__ work,
                                       int* __restrict__ cards) {
    __shared__ double sdist[TPB];
    __shared__ int sidx[TPB];
    __shared__ double bestX, bestY;

    double* dist = work + static_cast<size_t>(blockIdx.x) * U;

    for (int k = blockIdx.x; k < numLocalSeeds; k += gridDim.x) {
        const int seed = firstSeed + k * seedStride;
        const double sx = ux[seed];
        const double sy = uy[seed];

        for (int j = threadIdx.x; j < U; j += blockDim.x) {
            if (j == seed) {
                dist[j] = EXCLUDED;
            } else {
                const double dx = ux[j] - sx;
                const double dy = uy[j] - sy;
                dist[j] = sqrt(dx * dx + dy * dy);
            }
        }
        __syncthreads();

        int count = 1;
        while (count < U) {
            // Block-wide argmin over eligible candidates
            double bd = DBL_MAX;
            int bi = INT_MAX;
            for (int j = threadIdx.x; j < U; j += blockDim.x) {
                const double d = dist[j];
                if (d < threshold && d < bd) {
                    bd = d;
                    bi = j;
                }
            }
            sdist[threadIdx.x] = bd;
            sidx[threadIdx.x] = bi;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    const double od = sdist[threadIdx.x + s];
                    const int oi = sidx[threadIdx.x + s];
                    if (od < sdist[threadIdx.x] ||
                        (od == sdist[threadIdx.x] && oi < sidx[threadIdx.x])) {
                        sdist[threadIdx.x] = od;
                        sidx[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }

            const int best = sidx[0];
            if (best == INT_MAX) break;  // no candidate keeps diameter < threshold

            if (threadIdx.x == 0) {
                bestX = ux[best];
                bestY = uy[best];
            }
            __syncthreads();

            // Fold the new member's distances into the running maxima
            for (int j = threadIdx.x; j < U; j += blockDim.x) {
                if (j == best) {
                    dist[j] = EXCLUDED;
                    continue;
                }
                const double d = dist[j];
                if (d != EXCLUDED) {
                    const double dx = ux[j] - bestX;
                    const double dy = uy[j] - bestY;
                    const double nd = sqrt(dx * dx + dy * dy);
                    if (nd > d) dist[j] = nd;
                }
            }
            ++count;
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            cards[k] = count;
        }
        __syncthreads();
    }
}

// Regenerate the winning cluster on the CPU (OpenMP-parallel greedy growth
// over the compacted point set). Same selection rules as the CUDA kernel and
// the original sequential code; returns members as original point indices.
static std::vector<int> growClusterHost(const std::vector<double>& ux,
                                        const std::vector<double>& uy,
                                        const std::vector<int>& orig,
                                        const int U,
                                        const int seed,
                                        const double threshold) {
    std::vector<double> dist(U);
    const double sx = ux[seed];
    const double sy = uy[seed];

    // For small working sets the OpenMP fork/join overhead per growth step
    // outweighs the parallel work; run those serially.
    const bool par = U >= 4096;

    #pragma omp parallel for schedule(static) if(par)
    for (int j = 0; j < U; ++j) {
        if (j == seed) {
            dist[j] = EXCLUDED;
        } else {
            const double dx = ux[j] - sx;
            const double dy = uy[j] - sy;
            dist[j] = std::sqrt(dx * dx + dy * dy);
        }
    }

    std::vector<int> members;
    members.push_back(orig[seed]);

    while (static_cast<int>(members.size()) < U) {
        double bd = DBL_MAX;
        int bi = INT_MAX;
        #pragma omp parallel if(par)
        {
            double lbd = DBL_MAX;
            int lbi = INT_MAX;
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < U; ++j) {
                const double d = dist[j];
                if (d < threshold && d < lbd) {
                    lbd = d;
                    lbi = j;
                }
            }
            #pragma omp critical
            {
                if (lbd < bd || (lbd == bd && lbi < bi)) {
                    bd = lbd;
                    bi = lbi;
                }
            }
        }

        if (bi == INT_MAX) break;

        const double bx = ux[bi];
        const double by = uy[bi];
        #pragma omp parallel for schedule(static) if(par)
        for (int j = 0; j < U; ++j) {
            if (j == bi) {
                dist[j] = EXCLUDED;
                continue;
            }
            const double d = dist[j];
            if (d != EXCLUDED) {
                const double dx = ux[j] - bx;
                const double dy = uy[j] - by;
                const double nd = std::sqrt(dx * dx + dy * dy);
                if (nd > d) dist[j] = nd;
            }
        }

        members.push_back(orig[bi]);
    }

    return members;
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;

    // Compacted unclustered point set (rebuilt each round, ascending order)
    std::vector<double> ux(N), uy(N);
    std::vector<int> orig(N);

    // Device buffers sized once for the worst case
    double *d_ux = nullptr, *d_uy = nullptr, *d_work = nullptr;
    int* d_cards = nullptr;
    const int maxLocalSeeds = (N + nprocs - 1) / nprocs;
    CUDA_CHECK(cudaMalloc(&d_ux, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_uy, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(double) * static_cast<size_t>(MAX_GRID) * N));
    CUDA_CHECK(cudaMalloc(&d_cards, sizeof(int) * std::max(maxLocalSeeds, 1)));
    std::vector<int> h_cards(std::max(maxLocalSeeds, 1));

    int remaining = N;
    while (remaining > 0) {
        // Compact the unclustered points (ascending original index)
        int U = 0;
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) {
                ux[U] = points[i].x;
                uy[U] = points[i].y;
                orig[U] = i;
                ++U;
            }
        }

        // This rank evaluates compact seeds rank, rank+nprocs, ...
        const int numLocalSeeds = (rank < U) ? (U - rank + nprocs - 1) / nprocs : 0;

        // Local best: max cardinality, ties -> lowest original seed index.
        // Encoded as (cardinality, seed index) for MPI_MAXLOC, which picks the
        // max value and the minimum index on ties - exactly the order in which
        // the sequential code scans seeds.
        int localBest[2] = {-1, INT_MAX};

        if (numLocalSeeds > 0) {
            CUDA_CHECK(cudaMemcpy(d_ux, ux.data(), sizeof(double) * U,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_uy, uy.data(), sizeof(double) * U,
                                  cudaMemcpyHostToDevice));

            const int grid = std::min(MAX_GRID, numLocalSeeds);
            candidateClusterKernel<<<grid, TPB>>>(d_ux, d_uy, U, threshold,
                                                  rank, nprocs, numLocalSeeds,
                                                  d_work, d_cards);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_cards.data(), d_cards,
                                  sizeof(int) * numLocalSeeds,
                                  cudaMemcpyDeviceToHost));

            // Compact seed indices are ascending, so the first maximum is the
            // one with the lowest original seed index.
            for (int k = 0; k < numLocalSeeds; ++k) {
                if (h_cards[k] > localBest[0]) {
                    localBest[0] = h_cards[k];
                    localBest[1] = orig[rank + k * nprocs];
                }
            }
        }

        int globalBest[2];
        MPI_Allreduce(localBest, globalBest, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        if (globalBest[0] <= 0) break;  // no more clusters can be formed

        // Regenerate the winning cluster on every rank (deterministic)
        const int bestSeed = globalBest[1];
        const int compactSeed = static_cast<int>(
            std::lower_bound(orig.begin(), orig.begin() + U, bestSeed) -
            orig.begin());

        Cluster cluster;
        cluster.seed_point = bestSeed;
        cluster.members = growClusterHost(ux, uy, orig, U, compactSeed, threshold);

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            clustered[cluster.members[i]] = true;
        }
        remaining -= static_cast<int>(cluster.members.size());
        clusters.push_back(std::move(cluster));
    }

    CUDA_CHECK(cudaFree(d_ux));
    CUDA_CHECK(cudaFree(d_uy));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_cards));

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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Split the node's cores among its local ranks unless the user pinned the
    // thread count explicitly; otherwise every rank spawns nproc threads and
    // the node is heavily oversubscribed.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &nodeComm);
    int localSize = 1;
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
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

    // Bind one GPU per rank (round-robin over the node's devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "Error: no CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    CUDA_CHECK(cudaFree(0));  // establish context before timing

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs/node: %d, OpenMP threads: %d\n",
               nprocs, deviceCount, omp_get_max_threads());
    }

    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time_ms = cluster_time.count();
    long max_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &max_cluster_time_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", max_cluster_time_ms);
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
        const double time_sec = max_cluster_time_ms / 1000.0;
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

    // Validation (rank 0 validates; result broadcast so all ranks agree)
    int exit_code = 0;
    if (validate) {
        if (rank == 0) {
            const bool valid = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exit_code;
}
