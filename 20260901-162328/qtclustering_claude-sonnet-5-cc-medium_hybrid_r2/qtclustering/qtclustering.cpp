// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - MPI distributes the candidate seed points of each clustering round
//    across ranks (e.g. across nodes of a cluster).
//  - OpenMP distributes each rank's share of seed points across the GPUs
//    that are locally visible to that rank.
//  - CUDA grows every candidate cluster (one CUDA block per seed point) in
//    parallel on the GPU, using the classic "running maximum distance"
//    formulation of QT clustering so that the result is bit-identical to
//    the original sequential algorithm.

#include <algorithm>
#include <chrono>
#include <cfloat>
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
static const int THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
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

// Calculate Euclidean distance between two points (host-side, used for validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Compute the [begin, end) range of a contiguous, balanced split of `total`
// items into `parts` parts, for part index `idx`.
static void computeRange(int total, int parts, int idx, int* begin, int* end) {
    const int base = total / parts;
    const int rem = total % parts;
    *begin = idx * base + std::min(idx, rem);
    *end = *begin + base + (idx < rem ? 1 : 0);
}

// CUDA kernel: compute the full pairwise Euclidean distance matrix.
__global__ void computeDistanceMatrixKernel(const Point* pts, double* dist, int N) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        const double dx = pts[i].x - pts[j].x;
        const double dy = pts[i].y - pts[j].y;
        dist[static_cast<size_t>(i) * N + j] = std::sqrt(dx * dx + dy * dy);
    }
}

// CUDA kernel: grow a candidate QT cluster for every seed point in seedList,
// one CUDA block per seed. Uses the running-maximum-distance formulation:
// runMax[i] tracks the maximum distance from point i to every point already
// added to the cluster, so the next point to add is simply the unclustered
// point with the smallest runMax[i] that still satisfies runMax[i] < threshold
// (ties broken by smallest point index, matching the sequential scan order
// of the original algorithm).
__global__ void growClustersKernel(const double* __restrict__ dist,
                                    const unsigned char* __restrict__ clustered,
                                    const int* __restrict__ seedList,
                                    int numSeeds,
                                    int N,
                                    double threshold,
                                    double* __restrict__ runningMaxScratch,
                                    int* __restrict__ cardOut,
                                    unsigned char* __restrict__ membershipOut) {
    const int b = blockIdx.x;
    if (b >= numSeeds) return;

    const int seed = seedList[b];
    double* runMax = runningMaxScratch + static_cast<size_t>(b) * N;
    unsigned char* inClus = membershipOut + static_cast<size_t>(b) * N;
    const double* distSeedRow = dist + static_cast<size_t>(seed) * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        inClus[i] = 0;
        runMax[i] = distSeedRow[i];
    }
    __syncthreads();

    extern __shared__ double smem[];
    double* sMinVal = smem;
    int* sMinIdx = reinterpret_cast<int*>(sMinVal + blockDim.x);

    __shared__ int sCount;
    if (threadIdx.x == 0) {
        inClus[seed] = 1;
        sCount = 1;
    }
    __syncthreads();

    while (sCount < N) {
        double localMin = DBL_MAX;
        int localIdx = -1;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            if (!inClus[i] && !clustered[i] && runMax[i] < threshold) {
                if (runMax[i] < localMin) {
                    localMin = runMax[i];
                    localIdx = i;
                }
            }
        }
        sMinVal[threadIdx.x] = localMin;
        sMinIdx[threadIdx.x] = localIdx;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                const double otherVal = sMinVal[threadIdx.x + s];
                const int otherIdx = sMinIdx[threadIdx.x + s];
                const double curVal = sMinVal[threadIdx.x];
                const int curIdx = sMinIdx[threadIdx.x];
                if (otherIdx >= 0 &&
                    (curIdx < 0 || otherVal < curVal ||
                     (otherVal == curVal && otherIdx < curIdx))) {
                    sMinVal[threadIdx.x] = otherVal;
                    sMinIdx[threadIdx.x] = otherIdx;
                }
            }
            __syncthreads();
        }

        const int winner = sMinIdx[0];
        if (winner < 0) break;

        if (threadIdx.x == 0) {
            inClus[winner] = 1;
            sCount++;
        }
        __syncthreads();

        const double* distWinnerRow = dist + static_cast<size_t>(winner) * N;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            if (!inClus[i]) {
                const double dv = distWinnerRow[i];
                if (dv > runMax[i]) runMax[i] = dv;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) cardOut[b] = sCount;
}

// Per-GPU persistent device buffers.
struct GpuContext {
    int device = -1;
    Point* d_points = nullptr;
    double* d_dist = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seedList = nullptr;
    int* d_cardOut = nullptr;
    double* d_runningMax = nullptr;
    unsigned char* d_membershipOut = nullptr;
};

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
        const int m = static_cast<int>(cluster.members.size());
        #pragma omp parallel for reduction(max : max_diameter) schedule(static)
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
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
    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpiRank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpiRank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", mpiSize);
    }

    // Generate synthetic data (deterministic; identical on every rank)
    const int N = num_points;
    std::vector<Point> points(N);
    generateSyntheticData(points, N);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices visible to rank %d\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<GpuContext> gpuCtx(deviceCount);
    const size_t sharedBytes = THREADS_PER_BLOCK * (sizeof(double) + sizeof(int));

    auto cluster_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(deviceCount) schedule(static)
    for (int d = 0; d < deviceCount; ++d) {
        GpuContext& ctx = gpuCtx[d];
        ctx.device = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&ctx.d_points, sizeof(Point) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_dist, sizeof(double) * static_cast<size_t>(N) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_clustered, sizeof(unsigned char) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_seedList, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_cardOut, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_runningMax, sizeof(double) * static_cast<size_t>(N) * N));
        CUDA_CHECK(cudaMalloc(&ctx.d_membershipOut, sizeof(unsigned char) * static_cast<size_t>(N) * N));

        CUDA_CHECK(cudaMemcpy(ctx.d_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));

        dim3 block2d(16, 16);
        dim3 grid2d((N + block2d.x - 1) / block2d.x, (N + block2d.y - 1) / block2d.y);
        computeDistanceMatrixKernel<<<grid2d, block2d>>>(ctx.d_points, ctx.d_dist, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Main clustering loop

    std::vector<unsigned char> clusteredU8(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<Cluster> clusters;

    while (!unclustered_indices.empty()) {
        const int totalUnclustered = static_cast<int>(unclustered_indices.size());

        int rankBegin, rankEnd;
        computeRange(totalUnclustered, mpiSize, mpiRank, &rankBegin, &rankEnd);
        const int rankChunkSize = rankEnd - rankBegin;

        std::vector<int> devBestCard(deviceCount, -1);
        std::vector<int> devBestSeed(deviceCount, -1);
        std::vector<std::vector<unsigned char>> devBestMembership(deviceCount);

        #pragma omp parallel for num_threads(deviceCount) schedule(static)
        for (int d = 0; d < deviceCount; ++d) {
            int devBegin, devEnd;
            computeRange(rankChunkSize, deviceCount, d, &devBegin, &devEnd);
            const int chunkSize = devEnd - devBegin;
            if (chunkSize <= 0) continue;

            GpuContext& ctx = gpuCtx[d];
            CUDA_CHECK(cudaSetDevice(d));

            std::vector<int> seedChunk(chunkSize);
            for (int i = 0; i < chunkSize; ++i) {
                seedChunk[i] = unclustered_indices[rankBegin + devBegin + i];
            }

            CUDA_CHECK(cudaMemcpy(ctx.d_clustered, clusteredU8.data(), sizeof(unsigned char) * N,
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx.d_seedList, seedChunk.data(), sizeof(int) * chunkSize,
                                   cudaMemcpyHostToDevice));

            growClustersKernel<<<chunkSize, THREADS_PER_BLOCK, sharedBytes>>>(
                ctx.d_dist, ctx.d_clustered, ctx.d_seedList, chunkSize, N, threshold,
                ctx.d_runningMax, ctx.d_cardOut, ctx.d_membershipOut);
            CUDA_CHECK(cudaGetLastError());

            std::vector<int> cardHost(chunkSize);
            CUDA_CHECK(cudaMemcpy(cardHost.data(), ctx.d_cardOut, sizeof(int) * chunkSize,
                                   cudaMemcpyDeviceToHost));

            int bestPos = -1, bestCard = -1, bestSeed = -1;
            for (int i = 0; i < chunkSize; ++i) {
                const int seedId = seedChunk[i];
                if (cardHost[i] > bestCard ||
                    (cardHost[i] == bestCard && bestPos >= 0 && seedId < bestSeed)) {
                    bestCard = cardHost[i];
                    bestSeed = seedId;
                    bestPos = i;
                }
            }

            if (bestPos >= 0 && bestCard > 0) {
                devBestCard[d] = bestCard;
                devBestSeed[d] = bestSeed;
                devBestMembership[d].resize(N);
                CUDA_CHECK(cudaMemcpy(devBestMembership[d].data(),
                                       ctx.d_membershipOut + static_cast<size_t>(bestPos) * N,
                                       sizeof(unsigned char) * N, cudaMemcpyDeviceToHost));
            }
        }

        int rankBestCard = -1, rankBestSeed = -1, rankBestDev = -1;
        for (int d = 0; d < deviceCount; ++d) {
            if (devBestCard[d] < 0) continue;
            if (devBestCard[d] > rankBestCard ||
                (devBestCard[d] == rankBestCard && devBestSeed[d] < rankBestSeed)) {
                rankBestCard = devBestCard[d];
                rankBestSeed = devBestSeed[d];
                rankBestDev = d;
            }
        }

        struct BestInfo { int card; int seed; int rank; };
        BestInfo local{rankBestCard, rankBestSeed, mpiRank};
        std::vector<BestInfo> allBest(mpiSize);
        MPI_Allgather(&local, 3, MPI_INT, allBest.data(), 3, MPI_INT, MPI_COMM_WORLD);

        int winCard = -1, winSeed = -1, winRank = -1;
        for (const auto& b : allBest) {
            if (b.card <= 0) continue;
            if (b.card > winCard || (b.card == winCard && b.seed < winSeed)) {
                winCard = b.card;
                winSeed = b.seed;
                winRank = b.rank;
            }
        }

        if (winCard <= 0) break; // No more clusters can be formed

        std::vector<unsigned char> membership(N);
        if (mpiRank == winRank) {
            membership = devBestMembership[rankBestDev];
        }
        MPI_Bcast(membership.data(), N, MPI_UNSIGNED_CHAR, winRank, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = winSeed;
        cluster.members.reserve(winCard);
        for (int i = 0; i < N; ++i) {
            if (membership[i]) {
                cluster.members.push_back(i);
                clusteredU8[i] = 1;
            }
        }
        clusters.push_back(std::move(cluster));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clusteredU8](int idx) { return clusteredU8[idx] != 0; }),
            unclustered_indices.end()
        );
    }

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    const long local_cluster_time = cluster_time.count();
    long global_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &global_cluster_time, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    #pragma omp parallel for num_threads(deviceCount) schedule(static)
    for (int d = 0; d < deviceCount; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        GpuContext& ctx = gpuCtx[d];
        cudaFree(ctx.d_points);
        cudaFree(ctx.d_dist);
        cudaFree(ctx.d_clustered);
        cudaFree(ctx.d_seedList);
        cudaFree(ctx.d_cardOut);
        cudaFree(ctx.d_runningMax);
        cudaFree(ctx.d_membershipOut);
    }

    if (mpiRank == 0) {
        printf("Clustering time: %ld ms\n", global_cluster_time);
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
        const double time_sec = global_cluster_time / 1000.0;
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
