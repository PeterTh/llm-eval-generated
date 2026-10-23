// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * MPI     : every rank holds the full (small) point set and the global
//               "clustered" state.  The candidate seeds of each clustering
//               round are distributed across ranks; a single packed integer
//               all-reduce elects the globally best seed every round.
//   * CUDA    : candidate cluster cardinalities are evaluated on the GPU with
//               one CUDA block per seed (dynamic work queue).  Each block
//               maintains the running maximum squared distance from every
//               unclustered point to the cluster, turning each greedy step
//               into an O(U) scan/update instead of O(U * |cluster|).
//   * OpenMP  : the host concurrently evaluates a share of the seeds while the
//               GPU is busy, and parallelizes the round reduction and the
//               result validation.  The CPU/GPU seed split is feedback-tuned.
//
// The algorithm is numerically identical to the sequential reference: the
// same squared Euclidean distances are computed, only maxima (which are exact
// and order independent) and strict "less than" comparisons are used.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

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
// CUDA device state
// ---------------------------------------------------------------------------

static double* g_dxs = nullptr;
static double* g_dys = nullptr;
static int* g_dactive = nullptr;    // full list of unclustered point indices
static int* g_dcards = nullptr;     // per-seed cardinalities
static int* g_dcounter = nullptr;   // dynamic work queue counter
static double* g_dmaxd = nullptr;   // per-block running max distance (fallback)
static size_t g_maxd_capacity = 0;  // doubles allocated in g_dmaxd
static bool g_cuda = false;
static int g_block = 256;
static int g_sm_count = 0;
static size_t g_shared_optin = 0;

// Compute the maximum squared distance from each active point to the growing
// cluster, one CUDA block per seed.  Blocks pull seeds from a shared atomic
// work queue so that long/short candidate clusters are load balanced.
__global__ void computeCardsKernel(const double* __restrict__ xs,
                                   const double* __restrict__ ys,
                                   const int* __restrict__ active,
                                   int U,
                                   int N,
                                   double threshold2,
                                   int posBase,
                                   int posStride,
                                   int localStart,
                                   int S,
                                   int* __restrict__ cards,
                                   int* __restrict__ counter,
                                   double* __restrict__ globalMaxd,
                                   int useShared) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int tid = threadIdx.x;
    const int bd = blockDim.x;
    const int words = (N + 31) >> 5;

    double* s_val = reinterpret_cast<double*>(smem_raw);          // bd doubles
    double* s_max = s_val + bd;                                   // U doubles (shared mode)
    int* s_int = reinterpret_cast<int*>(s_max + (useShared ? U : 0));
    int* s_idx = s_int;                                           // bd ints
    int* s_bits = s_idx + bd;                                     // words ints
    int* s_size = s_bits + words;                                 // 1 int
    int* s_work = s_size + 1;                                     // 1 int

    double* maxd = useShared ? s_max
                             : (globalMaxd + (size_t)blockIdx.x * (size_t)U);

    for (;;) {
        if (tid == 0) *s_work = atomicAdd(counter, 1);
        __syncthreads();
        const int i = *s_work;
        if (i >= S) break;

        const int pos = posBase + (localStart + i) * posStride;
        const int seed = active[pos];

        for (int w = tid; w < words; w += bd) s_bits[w] = 0;
        __syncthreads();
        if (tid == 0) {
            s_bits[seed >> 5] |= (1u << (seed & 31));
            *s_size = 1;
        }
        __syncthreads();

        const double xseed = xs[seed];
        const double yseed = ys[seed];
        for (int p = tid; p < U; p += bd) {
            const int c = active[p];
            const double dx = xs[c] - xseed;
            const double dy = ys[c] - yseed;
            maxd[p] = dx * dx + dy * dy;
        }
        __syncthreads();

        for (;;) {
            const int size = *s_size;
            if (size >= U) break;

            double bestv = threshold2;
            int bestc = -1;
            for (int p = tid; p < U; p += bd) {
                const int c = active[p];
                if (s_bits[c >> 5] & (1u << (c & 31))) continue;
                const double v = maxd[p];
                if (v < bestv) {
                    bestv = v;
                    bestc = c;
                }
            }

            s_val[tid] = bestv;
            s_idx[tid] = bestc;
            __syncthreads();
            for (int stride = bd >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    const double v2 = s_val[tid + stride];
                    const int i2 = s_idx[tid + stride];
                    const double v1 = s_val[tid];
                    const int i1 = s_idx[tid];
                    if (i2 >= 0 && (i1 < 0 || v2 < v1 || (v2 == v1 && i2 < i1))) {
                        s_val[tid] = v2;
                        s_idx[tid] = i2;
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                const int m = s_idx[0];
                if (m >= 0) {
                    s_bits[m >> 5] |= (1u << (m & 31));
                    *s_size = size + 1;
                }
            }
            __syncthreads();

            const int m = s_idx[0];
            if (m < 0) break;
            if (*s_size >= U) break;

            const double xm = xs[m];
            const double ym = ys[m];
            for (int p = tid; p < U; p += bd) {
                const int c = active[p];
                if (s_bits[c >> 5] & (1u << (c & 31))) continue;
                const double dx = xs[c] - xm;
                const double dy = ys[c] - ym;
                const double d = dx * dx + dy * dy;
                if (d > maxd[p]) maxd[p] = d;
            }
            __syncthreads();
        }

        if (tid == 0) cards[i] = *s_size;
        __syncthreads();
    }
}

static cudaStream_t g_stream = nullptr;
static cudaEvent_t g_ev_start = nullptr;
static cudaEvent_t g_ev_stop = nullptr;

// Launch the cardinality kernel asynchronously on the internal stream.
static void launchGpuCards(int U, int N, double threshold2,
                           int posBase, int posStride, int localStart, int S) {
    cudaMemsetAsync(g_dcounter, 0, sizeof(int), g_stream);

    const int wordsN = (N + 31) >> 5;
    const size_t doublesShared = static_cast<size_t>(U) + g_block;
    size_t sharedBytes = doublesShared * sizeof(double) +
                         static_cast<size_t>(wordsN + g_block + 2) * sizeof(int);
    const size_t smallShared =
        static_cast<size_t>(g_block) * sizeof(double) +
        static_cast<size_t>(wordsN + g_block + 2) * sizeof(int);

    const int useShared = (sharedBytes <= g_shared_optin) ? 1 : 0;
    if (!useShared) sharedBytes = smallShared;

    int maxBlocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, computeCardsKernel, g_block, static_cast<int>(sharedBytes));
    if (maxBlocks < 1) maxBlocks = 1;

    int grid = g_sm_count * maxBlocks;
    if (!useShared) {
        const size_t budget = static_cast<size_t>(512) << 20;
        int cap = static_cast<int>(budget / (static_cast<size_t>(U) * sizeof(double) + 1));
        if (cap < 1) cap = 1;
        if (grid > cap) grid = cap;
    }
    if (grid > S) grid = S;
    if (grid < 1) grid = 1;

    if (!useShared) {
        const size_t need = static_cast<size_t>(grid) * static_cast<size_t>(U);
        if (need > g_maxd_capacity) {
            if (g_dmaxd) cudaFree(g_dmaxd);
            g_dmaxd = nullptr;
            cudaMalloc(&g_dmaxd, need * sizeof(double));
            g_maxd_capacity = need;
        }
    }

    cudaEventRecord(g_ev_start, g_stream);
    computeCardsKernel<<<grid, g_block, sharedBytes, g_stream>>>(
        g_dxs, g_dys, g_dactive, U, N, threshold2,
        posBase, posStride, localStart, S,
        g_dcards, g_dcounter, g_dmaxd, useShared);
    cudaEventRecord(g_ev_stop, g_stream);
}

// Wait for the kernel, return its duration and copy the cardinalities back.
static void finishGpuCards(int S, int* out, float* ms) {
    cudaStreamSynchronize(g_stream);
    cudaEventElapsedTime(ms, g_ev_start, g_ev_stop);
    cudaMemcpy(out, g_dcards, static_cast<size_t>(S) * sizeof(int),
               cudaMemcpyDeviceToHost);
}

// Host-side helper: evaluate one candidate cluster with the incremental
// maximum-distance formulation.  Used for the CPU share of the work and for
// materializing the elected cluster's member list.
static int generateCandidateCPU(int seed,
                                int U,
                                const int* active,
                                const double* xs,
                                const double* ys,
                                double threshold2,
                                int N,
                                uint32_t* bits,
                                double* maxd,
                                std::vector<int>* members) {
    const int words = (N + 31) >> 5;
    std::memset(bits, 0, static_cast<size_t>(words) * sizeof(uint32_t));
    bits[seed >> 5] |= (1u << (seed & 31));

    if (members) {
        members->clear();
        members->push_back(seed);
    }

    const double xseed = xs[seed];
    const double yseed = ys[seed];
    for (int p = 0; p < U; ++p) {
        const int c = active[p];
        const double dx = xs[c] - xseed;
        const double dy = ys[c] - yseed;
        maxd[p] = dx * dx + dy * dy;
    }

    int size = 1;
    while (size < U) {
        double bestv = threshold2;
        int bestc = -1;
        for (int p = 0; p < U; ++p) {
            const int c = active[p];
            if (bits[c >> 5] & (1u << (c & 31))) continue;
            const double v = maxd[p];
            if (v < bestv) {
                bestv = v;
                bestc = c;
            }
        }
        if (bestc < 0) break;

        bits[bestc >> 5] |= (1u << (bestc & 31));
        ++size;
        if (members) members->push_back(bestc);
        if (size >= U) break;

        const double xm = xs[bestc];
        const double ym = ys[bestc];
        for (int p = 0; p < U; ++p) {
            const int c = active[p];
            if (bits[c >> 5] & (1u << (c & 31))) continue;
            const double dx = xs[c] - xm;
            const double dy = ys[c] - ym;
            const double d = dx * dx + dy * dy;
            if (d > maxd[p]) maxd[p] = d;
        }
    }
    return size;
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  MPI_Comm comm,
                                  int rank,
                                  int nranks,
                                  int omp_threads) {
    const int N = static_cast<int>(points.size());
    const int words = (N + 31) >> 5;
    const double threshold2 = threshold * threshold;

    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> active(N);
    std::iota(active.begin(), active.end(), 0);
    std::vector<Cluster> clusters;

    // Host point arrays (SoA layout gives the CPU path sequential access).
    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        xs[i] = points[i].x;
        ys[i] = points[i].y;
    }

    // Upload the immutable point coordinates once.
    if (g_cuda) {
        cudaMemcpy(g_dxs, xs.data(), (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(g_dys, ys.data(), (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Per-thread CPU scratch (max clamp size U <= N).
    const int T = std::max(1, omp_threads);
    std::vector<uint32_t> tbits((size_t)T * words);
    std::vector<double> tmaxd((size_t)T * N);

    std::vector<int> gpu_cards(N);
    std::vector<int> cpu_cards(N);

    // Feedback-tuned CPU/GPU cost per seed (seconds).
    double gpu_cost = 1.0;
    double cpu_cost = 1.0;

    if (g_cuda) {
        cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
        cudaEventCreate(&g_ev_start);
        cudaEventCreate(&g_ev_stop);
    }

    // One-time calibration of the per-seed cost of the CPU and GPU paths.
    if (g_cuda) {
        const int K = std::min(static_cast<int>(active.size()), 32);
        if (K >= 8) {
            const int U0 = static_cast<int>(active.size());
            const double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < K; ++i) {
                const int t = omp_get_thread_num();
                gpu_cards[i] = generateCandidateCPU(
                    active[i], U0, active.data(), xs.data(), ys.data(),
                    threshold2, N, tbits.data() + (size_t)t * words,
                    tmaxd.data() + (size_t)t * N, nullptr);
            }
            const double cdt = omp_get_wtime() - t0;

            cudaMemcpy(g_dactive, active.data(), (size_t)U0 * sizeof(int),
                       cudaMemcpyHostToDevice);
            float gms = 0.0f;
            launchGpuCards(U0, N, threshold2, 0, 1, 0, K);
            finishGpuCards(K, gpu_cards.data(), &gms);

            gpu_cost = (gms / 1000.0) / K;
            cpu_cost = cdt / K;
            if (gpu_cost <= 0.0) gpu_cost = 1e-9;
            if (cpu_cost <= 0.0) cpu_cost = 1e-9;
        }
    }

    while (!active.empty()) {
        const int U = static_cast<int>(active.size());

        // Contiguous/strided seed partition across ranks.
        // Rank r owns local indices i with position = r + i * nranks.
        const int locCount = (U > rank) ? (U - rank + nranks - 1) / nranks : 0;

        int G = 0;
        if (g_cuda && locCount > 0) {
            double frac = cpu_cost / (gpu_cost + cpu_cost);
            G = static_cast<int>(std::llround(static_cast<double>(locCount) * frac));
            if (G < 0) G = 0;
            if (G > locCount) G = locCount;
        }
        const int C = locCount - G;

        float gpu_ms = 0.0f;
        double cpu_dt = 0.0;

        if (G > 0) {
            // Upload the full unclustered list; all ranks use identical data.
            cudaMemcpy(g_dactive, active.data(), (size_t)U * sizeof(int),
                       cudaMemcpyHostToDevice);
            launchGpuCards(U, N, threshold2, rank, nranks, 0, G);
        }

        // CPU share runs concurrently with the GPU kernel.
        if (C > 0) {
            const double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1) if (C > 8)
            for (int j = 0; j < C; ++j) {
                const int t = omp_get_thread_num();
                const int pos = rank + (G + j) * nranks;
                cpu_cards[j] = generateCandidateCPU(
                    active[pos], U, active.data(), xs.data(), ys.data(),
                    threshold2, N, tbits.data() + (size_t)t * words,
                    tmaxd.data() + (size_t)t * N, nullptr);
            }
            cpu_dt = omp_get_wtime() - t0;
        }

        if (G > 0) {
            finishGpuCards(G, gpu_cards.data(), &gpu_ms);
            gpu_cost = 0.5 * gpu_cost + 0.5 * (gpu_ms / 1000.0) / G;
        }
        if (C > 0) {
            cpu_cost = 0.5 * cpu_cost + 0.5 * cpu_dt / C;
        }

        // Elect the globally best seed: largest cardinality, lowest position
        // on ties (matching the sequential "first maximum wins" rule).
        uint64_t local_key = 0;
        for (int i = 0; i < locCount; ++i) {
            const int card = (i < G) ? gpu_cards[i] : cpu_cards[i - G];
            const int pos = rank + i * nranks;
            const uint64_t k = (static_cast<uint64_t>(static_cast<uint32_t>(card)) << 32) |
                               static_cast<uint32_t>(U - 1 - pos);
            if (k > local_key) local_key = k;
        }

        uint64_t global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UINT64_T, MPI_MAX, comm);

        const int best_pos = U - 1 - static_cast<int>(global_key & 0xffffffffu);
        const int best_seed = active[best_pos];

        // Materialize the elected cluster once on the root, then broadcast.
        std::vector<int> best_members;
        int member_count = 0;
        if (rank == 0) {
            best_members.reserve(N);
            generateCandidateCPU(best_seed, U, active.data(), xs.data(), ys.data(),
                                 threshold2, N, tbits.data(), tmaxd.data(),
                                 &best_members);
            member_count = static_cast<int>(best_members.size());
        }
        MPI_Bcast(&member_count, 1, MPI_INT, 0, comm);
        if (rank != 0) best_members.resize(member_count);
        MPI_Bcast(best_members.data(), member_count, MPI_INT, 0, comm);

        if (member_count > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_members;
            clusters.push_back(cluster);

            for (int i = 0; i < member_count; ++i) {
                clustered[best_members[i]] = 1;
            }
            active.erase(
                std::remove_if(active.begin(), active.end(),
                               [&clustered](int idx) { return clustered[idx] != 0; }),
                active.end());
        } else {
            break;
        }
    }

    if (g_cuda) {
        cudaEventDestroy(g_ev_start);
        cudaEventDestroy(g_ev_stop);
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
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

    // Count clustered points (parallel reduction)
    int clustered_count = 0;
#pragma omp parallel for reduction(+ : clustered_count) schedule(static)
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Shared-memory (node) communicator: pick one GPU per local rank and give
    // each rank a fair share of the node's CPU cores for OpenMP.
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &node_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    int omp_threads = std::max(1, omp_get_num_procs() / std::max(1, local_size));
    omp_set_num_threads(omp_threads);
    omp_set_dynamic(0);

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

    // Initialize CUDA on each rank (one GPU per local rank when available).
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) {
        if (cudaSetDevice(local_rank % ndev) == cudaSuccess) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, local_rank % ndev);
            g_sm_count = prop.multiProcessorCount;
            g_shared_optin = prop.sharedMemPerBlockOptin;

            cudaMalloc(&g_dxs, (size_t)num_points * sizeof(double));
            cudaMalloc(&g_dys, (size_t)num_points * sizeof(double));
            cudaMalloc(&g_dactive, (size_t)num_points * sizeof(int));
            cudaMalloc(&g_dcards, (size_t)num_points * sizeof(int));
            cudaMalloc(&g_dcounter, sizeof(int));
            cudaFuncSetAttribute(computeCardsKernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(g_shared_optin));
            g_cuda = true;
        }
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (identical on every rank).
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, MPI_COMM_WORLD, rank, nranks, omp_threads);

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
        const double clusters_per_sec = time_sec > 0.0 ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = time_sec > 0.0 ? num_points / time_sec : 0.0;
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

    int rc = 0;
    // Validation
    if (validate) {
        bool valid = true;
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (g_cuda) {
        cudaFree(g_dxs);
        cudaFree(g_dys);
        cudaFree(g_dactive);
        cudaFree(g_dcards);
        cudaFree(g_dcounter);
        if (g_dmaxd) cudaFree(g_dmaxd);
    }

    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return rc;
}
