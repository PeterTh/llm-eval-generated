// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold. Every unclustered
// point is tried as a seed each round; the largest candidate cluster (lowest
// seed index on ties) is committed, and the process repeats until all points
// are clustered.
//
// Parallelization:
//  - CUDA:   one thread block grows the candidate cluster of one seed. Distance
//            scans and the closest-candidate argmin are performed block-wide in
//            shared memory.
//  - OpenMP: each MPI rank runs one host worker thread per CUDA stream it owns
//            (STREAMS_PER_DEVICE per local GPU). Workers pull seed chunks from a
//            shared queue, so GPUs are kept busy regardless of seed cost.
//            Cluster validation is also parallelized.
//  - MPI:    the seeds of each round are distributed cyclically over ranks.
//            Each round the best (cardinality, seed) pair is combined with
//            MPI_Allreduce so every rank commits the identical cluster.
//
// Equivalence with the sequential algorithm:
//  - A candidate's maximum distance to the cluster is maintained incrementally.
//    Max over a set is order-independent, so this equals recomputing it.
//  - A candidate whose maximum distance reaches the threshold can never become
//    eligible again (the maximum only grows), so it is dropped permanently.
//  - Selection is "minimum max-distance, lowest point index on ties", identical
//    to the sequential scan order.
//  - Seeds are pruned only when an upper bound on their cardinality (1 + number
//    of eligible neighbours, since every member lies within the threshold of the
//    seed) is strictly below a cardinality already achieved. Such seeds cannot
//    win or tie, so the selected cluster is unchanged.
//  - Floating-point contraction is disabled on host and device (-ffp-contract=off,
//    --fmad=false) so CPU and GPU evaluate distances bit-identically.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per seed-growth block. Must be a power of two.
static constexpr int BLOCK_THREADS = 256;
// Host worker threads (and CUDA streams) per local GPU.
static constexpr int STREAMS_PER_DEVICE = 2;
// Device scratch memory budget per worker for the growth workspaces.
static constexpr size_t WORKSPACE_BUDGET = size_t(512) << 20;

// Whether this process prints (rank 0 only).
static bool g_root = true;

static void rootPrintf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void rootPrintf(const char* fmt, ...) {
    if (!g_root) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
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

// Euclidean distance (same operation order on host and device)
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
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

// Lexicographic candidate order: smaller max-distance first, then lower index.
__device__ __forceinline__ bool candidateLess(double d1, int c1, double d2, int c2) {
    return d1 < d2 || (d1 == d2 && c1 < c2);
}

// Upper bound per seed: number of unclustered points (other than the seed) within
// the threshold. Every member of the seed's cluster lies within the threshold of
// the seed, so cardinality <= 1 + outCount[i].
__global__ void countCandidatesKernel(const int* __restrict__ seeds, int n,
                                      const Point* __restrict__ pts,
                                      const unsigned char* __restrict__ clustered,
                                      int N, double threshold, int* __restrict__ outCount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int seed = seeds[i];
    const Point ps = pts[seed];
    int cnt = 0;
    for (int c = 0; c < N; ++c) {
        if (c == seed || clustered[c]) continue;
        if (distance(pts[c], ps) < threshold) ++cnt;
    }
    outCount[i] = cnt;
}

// Grow the candidate cluster of seeds[blockIdx.x]. One block per seed.
//
// Each step adds the eligible candidate with the smallest maximum distance to the
// current members (ties: lowest index), exactly as the sequential greedy growth.
// Candidates are kept in a compact active list (double-buffered in global memory)
// with their running maximum distance to the members; candidates reaching the
// threshold are removed permanently.
//
// outCard[blk] receives the cluster cardinality. If outMembers is non-null (used
// for a single seed), the members are written in the order they were added.
__global__ void growClusterKernel(const int* __restrict__ seeds, int n,
                                  const Point* __restrict__ pts,
                                  const unsigned char* __restrict__ clustered,
                                  int N, double threshold,
                                  int* __restrict__ wsIdx, double* __restrict__ wsDist,
                                  int* __restrict__ outCard,
                                  int* __restrict__ outMembers) {
    __shared__ int cnt[2];
    __shared__ int redI[BLOCK_THREADS];
    __shared__ double redD[BLOCK_THREADS];

    const int blk = blockIdx.x;
    if (blk >= n) return;
    const int tid = threadIdx.x;
    const int seed = seeds[blk];
    const Point ps = pts[seed];
    const bool record = outMembers != nullptr;

    int* const idxBase = wsIdx + size_t(blk) * 2 * N;
    double* const distBase = wsDist + size_t(blk) * 2 * N;

    if (tid == 0) {
        cnt[0] = 0;
        cnt[1] = 0;
    }
    __syncthreads();

    // Initial active list: unclustered points (other than the seed) within the threshold
    for (int c = tid; c < N; c += BLOCK_THREADS) {
        if (c == seed || clustered[c]) continue;
        const double d = distance(pts[c], ps);
        if (d < threshold) {
            const int pos = atomicAdd(&cnt[0], 1);
            idxBase[pos] = c;
            distBase[pos] = d;
        }
    }
    __syncthreads();

    int cur = 0;
    int card = 1;  // the seed
    if (record && tid == 0) outMembers[0] = seed;

    while (true) {
        const int A = cnt[cur];  // uniform across the block
        if (A == 0) break;

        const int nxt = cur ^ 1;
        const int* ci = idxBase + cur * N;
        const double* cd = distBase + cur * N;
        int* ni = idxBase + nxt * N;
        double* nd = distBase + nxt * N;

        // Block-wide argmin of (max distance, index)
        double bd = INFINITY;
        int bi = INT_MAX;
        for (int k = tid; k < A; k += BLOCK_THREADS) {
            const double d = cd[k];
            const int c = ci[k];
            if (candidateLess(d, c, bd, bi)) {
                bd = d;
                bi = c;
            }
        }
        redD[tid] = bd;
        redI[tid] = bi;
        __syncthreads();
        for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s && candidateLess(redD[tid + s], redI[tid + s], redD[tid], redI[tid])) {
                redD[tid] = redD[tid + s];
                redI[tid] = redI[tid + s];
            }
            __syncthreads();
        }
        const int chosen = redI[0];
        const Point pc = pts[chosen];

        // Update running maxima with the new member and keep eligible candidates
        for (int k = tid; k < A; k += BLOCK_THREADS) {
            const int c = ci[k];
            if (c == chosen) continue;
            const double m = fmax(cd[k], distance(pts[c], pc));
            if (m < threshold) {
                const int pos = atomicAdd(&cnt[nxt], 1);
                ni[pos] = c;
                nd[pos] = m;
            }
        }
        __syncthreads();

        if (tid == 0) {
            if (record) outMembers[card] = chosen;
            cnt[cur] = 0;  // consumed buffer becomes the next output buffer
        }
        ++card;
        cur = nxt;
        __syncthreads();
    }

    if (tid == 0) {
        outCard[blk] = card;
    }
}

// Pack (cardinality, seed) so that a larger key is a larger cardinality, and on
// equal cardinality a smaller seed index. Key 0 means "no candidate".
static uint64_t encodeCandidate(int card, int seed) {
    return (uint64_t(card) << 32) | uint64_t(0xFFFFFFFFu - uint32_t(seed));
}

static void decodeCandidate(uint64_t key, int& card, int& seed) {
    card = static_cast<int>(key >> 32);
    seed = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(key & 0xFFFFFFFFull));
}

// Per-device execution context. Each worker owns one non-blocking stream and its
// own device buffers, so several workers can share a device concurrently.
struct GpuWorker {
    int device = -1;
    int chunkCap = 0;                   // seeds per growth launch
    cudaStream_t stream = nullptr;
    Point* pts = nullptr;               // N points
    unsigned char* clustered = nullptr; // N flags
    int* seeds = nullptr;               // N seed ids
    int* out = nullptr;                 // N cardinalities / upper bounds
    int* ws_idx = nullptr;              // chunkCap * 2N
    double* ws_dist = nullptr;          // chunkCap * 2N
    int* members = nullptr;             // N (single-seed record)
};

static GpuWorker makeWorker(int device, const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    GpuWorker g;
    g.device = device;
    CUDA_CHECK(cudaSetDevice(device));

    int sms = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

    const size_t bytesPerBlock = size_t(2) * N * (sizeof(int) + sizeof(double));
    const size_t budgetBlocks = std::max<size_t>(1, WORKSPACE_BUDGET / bytesPerBlock);
    const size_t wanted = std::max<size_t>(64, size_t(sms) * 4);
    g.chunkCap = static_cast<int>(std::min({wanted, budgetBlocks, size_t(N)}));

    CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&g.pts, size_t(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&g.clustered, size_t(N)));
    CUDA_CHECK(cudaMalloc(&g.seeds, size_t(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.out, size_t(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.members, size_t(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.ws_idx, size_t(g.chunkCap) * 2 * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.ws_dist, size_t(g.chunkCap) * 2 * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(g.pts, points.data(), size_t(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));
    return g;
}

static void freeWorker(GpuWorker& g) {
    CUDA_CHECK(cudaSetDevice(g.device));
    cudaFree(g.pts);
    cudaFree(g.clustered);
    cudaFree(g.seeds);
    cudaFree(g.out);
    cudaFree(g.members);
    cudaFree(g.ws_idx);
    cudaFree(g.ws_dist);
    cudaStreamDestroy(g.stream);
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         const double threshold,
                                         std::vector<GpuWorker>& workers,
                                         const int rank, const int nranks) {
    const int N = static_cast<int>(points.size());
    const int W = static_cast<int>(workers.size());
    GpuWorker& primary = workers[0];

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<Cluster> clusters;

    std::vector<int> mySeeds;
    std::vector<int> sortedSeed;
    std::vector<int> sortedUB;

    while (!unclustered.empty()) {
        // Current clustered mask on every device
        for (GpuWorker& g : workers) {
            CUDA_CHECK(cudaSetDevice(g.device));
            CUDA_CHECK(cudaMemcpyAsync(g.clustered, clustered.data(), size_t(N),
                                       cudaMemcpyHostToDevice, g.stream));
        }

        // Seeds owned by this rank (cyclic over the unclustered list, which is the
        // same on every rank)
        mySeeds.clear();
        for (size_t i = size_t(rank); i < unclustered.size(); i += size_t(nranks)) {
            mySeeds.push_back(unclustered[i]);
        }
        const int nMine = static_cast<int>(mySeeds.size());

        // Cardinality upper bounds for this rank's seeds
        std::vector<int> ub(nMine, 1);
        if (nMine > 0) {
            CUDA_CHECK(cudaSetDevice(primary.device));
            CUDA_CHECK(cudaMemcpyAsync(primary.seeds, mySeeds.data(), size_t(nMine) * sizeof(int),
                                       cudaMemcpyHostToDevice, primary.stream));
            const int threads = 128;
            countCandidatesKernel<<<(nMine + threads - 1) / threads, threads, 0, primary.stream>>>(
                primary.seeds, nMine, primary.pts, primary.clustered, N, threshold, primary.out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(ub.data(), primary.out, size_t(nMine) * sizeof(int),
                                       cudaMemcpyDeviceToHost, primary.stream));
            CUDA_CHECK(cudaStreamSynchronize(primary.stream));
            for (int& u : ub) u += 1;  // count of neighbours + the seed itself
        }

        // Visit seeds by decreasing bound so good clusters are found early and the
        // remaining seeds can be pruned.
        std::vector<int> order(nMine);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return ub[a] != ub[b] ? ub[a] > ub[b] : mySeeds[a] < mySeeds[b];
        });
        sortedSeed.resize(nMine);
        sortedUB.resize(nMine);
        for (int k = 0; k < nMine; ++k) {
            sortedSeed[k] = mySeeds[order[k]];
            sortedUB[k] = ub[order[k]];
        }

        // Evaluate seed cardinalities on all local workers
        std::atomic<size_t> nextPos{0};
        std::atomic<int> globalBest{0};
        std::vector<uint64_t> workerKey(W, 0);
        const size_t M = static_cast<size_t>(nMine);

#pragma omp parallel num_threads(W)
        {
            const int tw = omp_get_thread_num();
            if (tw < W) {
                GpuWorker& g = workers[tw];
                CUDA_CHECK(cudaSetDevice(g.device));
                std::vector<int> cards(g.chunkCap);
                int bestCard = 0;
                int bestSeed = -1;

                for (;;) {
                    const size_t p = nextPos.fetch_add(size_t(g.chunkCap));
                    if (p >= M) break;

                    const int best = globalBest.load();
                    const size_t end = std::min(M, p + size_t(g.chunkCap));
                    size_t q = p;
                    while (q < end && sortedUB[q] >= best) ++q;
                    const int n = static_cast<int>(q - p);
                    // Bounds are sorted descending: everything from p on is pruned
                    if (n == 0) break;

                    CUDA_CHECK(cudaMemcpyAsync(g.seeds, sortedSeed.data() + p,
                                               size_t(n) * sizeof(int),
                                               cudaMemcpyHostToDevice, g.stream));
                    growClusterKernel<<<n, BLOCK_THREADS, 0, g.stream>>>(
                        g.seeds, n, g.pts, g.clustered, N, threshold,
                        g.ws_idx, g.ws_dist, g.out, nullptr);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpyAsync(cards.data(), g.out, size_t(n) * sizeof(int),
                                               cudaMemcpyDeviceToHost, g.stream));
                    CUDA_CHECK(cudaStreamSynchronize(g.stream));

                    int chunkBest = 0;
                    for (int j = 0; j < n; ++j) {
                        const int card = cards[j];
                        const int seed = sortedSeed[p + size_t(j)];
                        chunkBest = std::max(chunkBest, card);
                        if (card > bestCard || (card == bestCard && seed < bestSeed)) {
                            bestCard = card;
                            bestSeed = seed;
                        }
                    }
                    int cur = globalBest.load();
                    while (cur < chunkBest && !globalBest.compare_exchange_weak(cur, chunkBest)) {
                    }
                }
                workerKey[tw] = bestCard > 0 ? encodeCandidate(bestCard, bestSeed) : 0;
            }
        }

        uint64_t localKey = 0;
        for (uint64_t k : workerKey) localKey = std::max(localKey, k);
        uint64_t globalKey = 0;
        MPI_Allreduce(&localKey, &globalKey, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

        int bestCard = 0;
        int bestSeed = -1;
        decodeCandidate(globalKey, bestCard, bestSeed);
        if (bestCard <= 0) break;  // No more clusters can be formed

        // Recompute the winning cluster's members (identical on every rank)
        CUDA_CHECK(cudaSetDevice(primary.device));
        CUDA_CHECK(cudaMemcpyAsync(primary.seeds, &bestSeed, sizeof(int),
                                   cudaMemcpyHostToDevice, primary.stream));
        growClusterKernel<<<1, BLOCK_THREADS, 0, primary.stream>>>(
            primary.seeds, 1, primary.pts, primary.clustered, N, threshold,
            primary.ws_idx, primary.ws_dist, primary.out, primary.members);
        CUDA_CHECK(cudaGetLastError());
        int recordedCard = 0;
        CUDA_CHECK(cudaMemcpyAsync(&recordedCard, primary.out, sizeof(int),
                                   cudaMemcpyDeviceToHost, primary.stream));
        CUDA_CHECK(cudaStreamSynchronize(primary.stream));
        if (recordedCard != bestCard) {
            fprintf(stderr, "internal error: cluster size mismatch (%d vs %d)\n",
                    recordedCard, bestCard);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<int> members(bestCard);
        CUDA_CHECK(cudaMemcpyAsync(members.data(), primary.members,
                                   size_t(bestCard) * sizeof(int),
                                   cudaMemcpyDeviceToHost, primary.stream));
        CUDA_CHECK(cudaStreamSynchronize(primary.stream));

        // Commit the cluster and remove its members from the unclustered list
        for (int m : members) clustered[m] = 1;
        Cluster cluster;
        cluster.members = std::move(members);
        cluster.seed_point = bestSeed;
        clusters.push_back(std::move(cluster));

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Diameters are independent per cluster: compute them in parallel
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

    // Report in cluster order
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

static void printUsage(const char* progName) {
    rootPrintf("Usage: %s [options]\n", progName);
    rootPrintf("Options:\n");
    rootPrintf("  -n <num>     Number of points (default: 1000)\n");
    rootPrintf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    rootPrintf("  -v           Enable validation\n");
    rootPrintf("  -r           Print results for external validation\n");
    rootPrintf("  -h           Show this help message\n");
}

static int runBenchmark(int argc, char** argv, int rank, int nranks) {
    g_root = (rank == 0);

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
            printUsage(argv[0]);
            return 0;
        } else {
            rootPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        rootPrintf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        return 1;
    }

    rootPrintf("QT Clustering Benchmark\n");
    rootPrintf("Number of points: %d\n", num_points);
    rootPrintf("Distance threshold: %.2f\n", threshold);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Assign local GPUs: ranks on a node share the node's devices
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int lrank = 0;
    int lsize = 1;
    MPI_Comm_rank(nodeComm, &lrank);
    MPI_Comm_size(nodeComm, &lsize);
    MPI_Comm_free(&nodeComm);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    if (lsize <= ndev) {
        for (int d = lrank; d < ndev; d += lsize) devices.push_back(d);
    } else {
        devices.push_back(lrank % ndev);
    }

    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    std::vector<GpuWorker> workers;
    const int numWorkers = STREAMS_PER_DEVICE * static_cast<int>(devices.size());
    for (int w = 0; w < numWorkers; ++w) {
        workers.push_back(makeWorker(devices[w % devices.size()], points));
    }

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, workers, rank, nranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Allreduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);

    for (GpuWorker& g : workers) freeWorker(g);

    rootPrintf("Clustering time: %ld ms\n", cluster_time.count());
    rootPrintf("Clusters found: %zu\n", clusters.size());

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

    rootPrintf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
    rootPrintf("Average cluster size: %.2f\n", avg_cluster_size);
    rootPrintf("Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    rootPrintf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

    // Print results for external validation
    if (printResults && g_root) {
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

    // Validation (performed on rank 0; result shared with all ranks)
    if (validate) {
        int valid = 0;
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (valid) {
            rootPrintf("Validation: PASSED\n");
            return 0;
        } else {
            rootPrintf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int rc = runBenchmark(argc, argv, rank, nranks);

    MPI_Finalize();
    return rc;
}
