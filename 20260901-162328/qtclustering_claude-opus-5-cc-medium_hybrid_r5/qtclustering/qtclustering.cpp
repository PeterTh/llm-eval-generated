// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy
// ------------------------
// The dominant cost of the algorithm is the "candidate cluster" scan: in every
// round, each still-unclustered point is used as a seed and a candidate cluster
// is grown from it; the largest candidate cluster wins the round.  Growing a
// candidate cluster from one seed is completely independent from every other
// seed, which gives a large amount of embarrassingly parallel work per round.
//
//   * MPI    - the seeds of a round are distributed across the ranks
//              (round-robin).  The round winner is determined with a single
//              MPI_MAXLOC all-reduce, and every rank replays the winning seed
//              so that the global clustering state stays replicated (no bulk
//              communication is needed at all).
//   * CUDA   - each rank drives one GPU.  One CUDA block grows one candidate
//              cluster; the threads of the block cooperatively scan the
//              candidate list, prune it and perform the arg-min reduction.
//   * OpenMP - the remaining CPU cores of a rank process seeds concurrently
//              with the GPU (dynamic, self-balancing work queue shared between
//              the GPU driver thread and the CPU worker threads), and the
//              validation/statistics phases are parallelized as well.
//
// Algorithmic note: the candidate list of a growing cluster is compacted on the
// fly.  A candidate whose distance to the cluster has reached the threshold can
// never become admissible again (the max-distance is monotonically increasing),
// so it is dropped permanently.  This keeps the arithmetic bit-for-bit
// identical to the sequential reference while removing dead candidates from
// all later scans.

#include <algorithm>
#include <chrono>
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

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err_ = (call);                                       \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Synthetic data generation (kept strictly sequential: the RNG stream and the
// rejection loop define the data set and must not be reordered).
// ---------------------------------------------------------------------------
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
// CPU candidate cluster growth
// ---------------------------------------------------------------------------

// Per-thread scratch space for growing a candidate cluster.
struct GrowScratch {
    std::vector<int> idx;      // candidate point ids
    std::vector<double> md;    // max distance of candidate to the cluster
    std::vector<int> members;  // cluster members, in insertion order
};

// Grow a candidate cluster from 'seed'.  'ulist'/'ucount' is the list of all
// currently unclustered points (the candidate set of the round).  Returns the
// cardinality of the candidate cluster; the member list is stored in
// scratch.members.
static int growCluster(const int seed, const Point* __restrict__ pts,
                       const int* __restrict__ ulist, const int ucount,
                       const double threshold, GrowScratch& scratch) {
    int* __restrict__ aidx = scratch.idx.data();
    double* __restrict__ amd = scratch.md.data();

    int ac = 0;
    for (int j = 0; j < ucount; ++j) {
        const int i = ulist[j];
        if (i != seed) {
            aidx[ac] = i;
            amd[ac] = 0.0;
            ++ac;
        }
    }

    scratch.members.clear();
    scratch.members.push_back(seed);

    int cur = seed;
    int count = 1;

    while (ac > 0) {
        const double mx = pts[cur].x;
        const double my = pts[cur].y;
        double best_val = std::numeric_limits<double>::max();
        int best_idx = -1;
        int w = 0;

        // Update the max-distance of every candidate with the freshly added
        // member, drop candidates that exceeded the threshold, and track the
        // admissible candidate with the smallest resulting diameter
        // (lowest point index wins ties, as in the reference code).
        for (int j = 0; j < ac; ++j) {
            const int i = aidx[j];
            if (i == cur) continue;  // the point just added to the cluster
            double v = amd[j];
            const double dx = pts[i].x - mx;
            const double dy = pts[i].y - my;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d > v) v = d;
            if (v < threshold) {
                aidx[w] = i;
                amd[w] = v;
                ++w;
                if (v < best_val || (v == best_val && i < best_idx)) {
                    best_val = v;
                    best_idx = i;
                }
            }
        }
        ac = w;

        if (best_idx < 0) break;  // no more points can be added

        cur = best_idx;
        ++count;
        scratch.members.push_back(best_idx);
    }

    return count;
}

// ---------------------------------------------------------------------------
// GPU candidate cluster growth: one block per seed
// ---------------------------------------------------------------------------

#define QT_BLOCK 256
#define QT_WARPS (QT_BLOCK / 32)

__device__ __forceinline__ void mergeBest(double& bv, int& bi, double ov, int oi) {
    if (oi >= 0 && (bi < 0 || ov < bv || (ov == bv && oi < bi))) {
        bv = ov;
        bi = oi;
    }
}

__global__ __launch_bounds__(QT_BLOCK) void growKernel(
    const double2* __restrict__ pts,
    const int* __restrict__ ulist, const int ucount,
    const int* __restrict__ seeds, const int nseeds,
    const double threshold,
    int* __restrict__ card,
    int* __restrict__ idxbuf, double* __restrict__ mdbuf,
    const int stride) {

    __shared__ double s_val[QT_WARPS];
    __shared__ int s_idx[QT_WARPS];
    __shared__ int s_cnt;
    __shared__ int s_ac;
    __shared__ int s_best;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Double-buffered candidate list, private to this block.
    int* const idx0 = idxbuf + static_cast<size_t>(blockIdx.x) * 2 * stride;
    int* const idx1 = idx0 + stride;
    double* const md0 = mdbuf + static_cast<size_t>(blockIdx.x) * 2 * stride;
    double* const md1 = md0 + stride;

    for (int s = blockIdx.x; s < nseeds; s += gridDim.x) {
        const int seed = seeds[s];

        int* Ai = idx0;
        double* Am = md0;
        int* Bi = idx1;
        double* Bm = md1;

        if (tid == 0) s_cnt = 0;
        __syncthreads();

        // Initial candidate set: all unclustered points except the seed.
        for (int j0 = 0; j0 < ucount; j0 += QT_BLOCK) {
            const int j = j0 + tid;
            int i = -1;
            const bool keep = (j < ucount) && ((i = ulist[j]) != seed);
            const unsigned ballot = __ballot_sync(0xffffffffu, keep);
            int base = 0;
            const int leader = __ffs(ballot) - 1;
            if (ballot) {
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(ballot));
                base = __shfl_sync(0xffffffffu, base, leader);
                if (keep) {
                    const int k = base + __popc(ballot & ((1u << lane) - 1u));
                    Ai[k] = i;
                    Am[k] = 0.0;
                }
            }
        }
        __syncthreads();
        if (tid == 0) { s_ac = s_cnt; s_cnt = 0; }
        __syncthreads();

        int ac = s_ac;
        int cur = seed;
        double mx = pts[seed].x;
        double my = pts[seed].y;
        int count = 1;

        while (ac > 0) {
            double bv = 0.0;
            int bi = -1;

            for (int j0 = 0; j0 < ac; j0 += QT_BLOCK) {
                const int j = j0 + tid;
                int i = -1;
                double v = 0.0;
                bool keep = false;
                if (j < ac) {
                    i = Ai[j];
                    if (i != cur) {
                        v = Am[j];
                        const double2 p = pts[i];
                        const double dx = p.x - mx;
                        const double dy = p.y - my;
                        const double d = sqrt(dx * dx + dy * dy);
                        if (d > v) v = d;
                        keep = (v < threshold);
                    }
                }
                const unsigned ballot = __ballot_sync(0xffffffffu, keep);
                if (ballot) {
                    int base = 0;
                    const int leader = __ffs(ballot) - 1;
                    if (lane == leader) base = atomicAdd(&s_cnt, __popc(ballot));
                    base = __shfl_sync(0xffffffffu, base, leader);
                    if (keep) {
                        const int k = base + __popc(ballot & ((1u << lane) - 1u));
                        Bi[k] = i;
                        Bm[k] = v;
                        mergeBest(bv, bi, v, i);
                    }
                }
            }

            // Warp-level arg-min reduction, then one warp-sized final merge.
            for (int off = 16; off > 0; off >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                mergeBest(bv, bi, ov, oi);
            }
            if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
            __syncthreads();
            if (tid == 0) {
                double rv = s_val[0];
                int ri = s_idx[0];
                for (int w = 1; w < QT_WARPS; ++w) mergeBest(rv, ri, s_val[w], s_idx[w]);
                s_best = ri;
                s_ac = s_cnt;
                s_cnt = 0;
            }
            __syncthreads();

            ac = s_ac;
            const int best = s_best;

            // swap candidate buffers
            int* ti = Ai; Ai = Bi; Bi = ti;
            double* tm = Am; Am = Bm; Bm = tm;

            if (best < 0) break;
            cur = best;
            mx = pts[best].x;
            my = pts[best].y;
            ++count;
        }

        if (tid == 0) card[s] = count;
        __syncthreads();
    }
}

// GPU resources of this rank (one device per rank).
struct GpuContext {
    double2* d_pts = nullptr;
    int* d_ulist = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_idxbuf = nullptr;
    double* d_mdbuf = nullptr;
    int* h_card = nullptr;   // pinned
    int* h_seeds = nullptr;  // pinned
    int blocks = 0;
    int stride = 0;
    cudaStream_t stream = nullptr;
};

static void gpuInit(GpuContext& g, const std::vector<Point>& points, int rank) {
    const int n = static_cast<int>(points.size());
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % devCount));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, rank % devCount));

    g.stride = n;
    // Per-block workspace: two int + two double arrays of 'stride' entries.
    const size_t perBlock = static_cast<size_t>(g.stride) * 2 * (sizeof(int) + sizeof(double));

    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t budget = static_cast<size_t>(freeMem * 0.7);

    int blocks = 4 * prop.multiProcessorCount;
    if (perBlock * static_cast<size_t>(blocks) > budget) {
        blocks = static_cast<int>(budget / perBlock);
    }
    if (blocks < 1) blocks = 1;
    if (blocks > n) blocks = n;
    g.blocks = blocks;

    CUDA_CHECK(cudaStreamCreate(&g.stream));
    CUDA_CHECK(cudaMalloc(&g.d_pts, sizeof(double2) * n));
    CUDA_CHECK(cudaMalloc(&g.d_ulist, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&g.d_seeds, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&g.d_card, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&g.d_idxbuf, static_cast<size_t>(g.stride) * 2 * sizeof(int) * blocks));
    CUDA_CHECK(cudaMalloc(&g.d_mdbuf, static_cast<size_t>(g.stride) * 2 * sizeof(double) * blocks));
    CUDA_CHECK(cudaHostAlloc(&g.h_card, sizeof(int) * n, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&g.h_seeds, sizeof(int) * n, cudaHostAllocDefault));

    std::vector<double2> hp(n);
    for (int i = 0; i < n; ++i) { hp[i].x = points[i].x; hp[i].y = points[i].y; }
    CUDA_CHECK(cudaMemcpy(g.d_pts, hp.data(), sizeof(double2) * n, cudaMemcpyHostToDevice));
}

static void gpuFree(GpuContext& g) {
    cudaFree(g.d_pts);
    cudaFree(g.d_ulist);
    cudaFree(g.d_seeds);
    cudaFree(g.d_card);
    cudaFree(g.d_idxbuf);
    cudaFree(g.d_mdbuf);
    cudaFreeHost(g.h_card);
    cudaFreeHost(g.h_seeds);
    if (g.stream) cudaStreamDestroy(g.stream);
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  GpuContext& gpu,
                                  const int rank, const int nranks) {
    const int N = static_cast<int>(points.size());
    const Point* pts = points.data();
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    const int nthreads = omp_get_max_threads();
    // Measured throughput of the two engines, in "work units" (one work unit =
    // growing one candidate cluster out of a candidate set of one point) per
    // second.  They are used to decide, per round, whether engaging the GPU
    // pays for its launch latency and how many CPU threads to spin up.  The
    // initial guesses only matter for the very first (largest) round.
    double gpu_work = 0.0, gpu_secs = 0.0;
    double cpu_work = 0.0, cpu_secs = 0.0;
    const double kGpuLatency = 6.0e-5;  // kernel launch + sync
    const double kOmpLatency = 2.0e-5;  // parallel region fork/join
    std::vector<GrowScratch> scratch(nthreads);
    for (auto& s : scratch) { s.idx.resize(N); s.md.resize(N); s.members.reserve(N); }

    std::vector<int> local_seeds(N);
    std::vector<int> local_card(N);
    GrowScratch winner_scratch;
    winner_scratch.idx.resize(N);
    winner_scratch.md.resize(N);
    winner_scratch.members.reserve(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        // Round-robin distribution of this round's seeds across the ranks.
        int nlocal = 0;
        for (int k = rank; k < U; k += nranks) local_seeds[nlocal++] = unclustered_indices[k];

        int best_card = -1;
        int best_seed = std::numeric_limits<int>::max();

        if (nlocal > 0) {
            // Late rounds are tiny; running them on the CPU only avoids the
            // kernel launch / synchronization latency.
            const double round_work = static_cast<double>(nlocal) * static_cast<double>(U);
            const double grate = (gpu_secs > 0.0) ? gpu_work / gpu_secs : 1.0e9;
            const double crate = (cpu_secs > 0.0) ? cpu_work / cpu_secs : 3.0e8;
            const double t_gpu = round_work / grate + kGpuLatency;
            const double t_cpu = round_work / (crate * nthreads) + kOmpLatency;
            // Engage the GPU unless it is clearly not worth its launch latency.
            const bool use_gpu = (t_gpu < 2.0 * t_cpu);
            // Don't spin up more threads than the round can keep busy.
            const int round_threads =
                use_gpu ? nthreads
                        : std::max(1, std::min(std::min(nthreads, nlocal),
                                               static_cast<int>(round_work /
                                                                (crate * kOmpLatency))));

            // Shared work queue: the GPU driver thread (thread 0) and the CPU
            // worker threads pull seed ranges from it until the round is done.
            int next = 0;
            // A GPU batch holds several seeds per block so that the grid-stride
            // loop evens out the (widely varying) per-seed cost, but never more
            // than half of what is left, so that the CPU workers keep a share.
            const int gpu_batch = std::max(1, std::min(4 * gpu.blocks, nlocal / 2));

            if (use_gpu) {
                CUDA_CHECK(cudaMemcpyAsync(gpu.d_ulist, unclustered_indices.data(),
                                           sizeof(int) * U, cudaMemcpyHostToDevice, gpu.stream));
                std::memcpy(gpu.h_seeds, local_seeds.data(), sizeof(int) * nlocal);
                CUDA_CHECK(cudaMemcpyAsync(gpu.d_seeds, gpu.h_seeds, sizeof(int) * nlocal,
                                           cudaMemcpyHostToDevice, gpu.stream));
                CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
            }

            int* const card = local_card.data();
            const int* const seeds = local_seeds.data();

            double r_gpu_work = 0.0, r_gpu_secs = 0.0, r_cpu_work = 0.0, r_cpu_secs = 0.0;

            #pragma omp parallel num_threads(round_threads) \
                reduction(+ : r_gpu_work, r_gpu_secs, r_cpu_work, r_cpu_secs)
            {
                const int tid = omp_get_thread_num();
                const bool gpu_thread = use_gpu && (tid == 0);
                // Chunk size: the GPU takes large batches (one block per seed),
                // the CPU workers take single seeds for fine-grained balancing.
                for (;;) {
                    int lo, hi;
                    if (gpu_thread) {
                        #pragma omp atomic capture
                        { lo = next; next += gpu_batch; }
                        hi = std::min(lo + gpu_batch, nlocal);
                    } else {
                        #pragma omp atomic capture
                        { lo = next; next += 1; }
                        hi = std::min(lo + 1, nlocal);
                    }
                    if (lo >= nlocal) break;

                    const double t0 = omp_get_wtime();
                    if (gpu_thread) {
                        const int cnt = hi - lo;
                        const int grid = std::min(cnt, gpu.blocks);
                        growKernel<<<grid, QT_BLOCK, 0, gpu.stream>>>(
                            gpu.d_pts, gpu.d_ulist, U, gpu.d_seeds + lo, cnt,
                            threshold, gpu.d_card + lo, gpu.d_idxbuf, gpu.d_mdbuf,
                            gpu.stride);
                        CUDA_CHECK(cudaMemcpyAsync(gpu.h_card + lo, gpu.d_card + lo,
                                                   sizeof(int) * cnt, cudaMemcpyDeviceToHost,
                                                   gpu.stream));
                        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
                        std::memcpy(card + lo, gpu.h_card + lo, sizeof(int) * cnt);
                        r_gpu_work += static_cast<double>(cnt) * U;
                        r_gpu_secs += omp_get_wtime() - t0;
                    } else {
                        for (int s = lo; s < hi; ++s) {
                            card[s] = growCluster(seeds[s], pts, unclustered_indices.data(),
                                                  U, threshold, scratch[tid]);
                        }
                        r_cpu_work += static_cast<double>(hi - lo) * U;
                        r_cpu_secs += omp_get_wtime() - t0;
                    }
                }
            }

            gpu_work += r_gpu_work; gpu_secs += r_gpu_secs;
            cpu_work += r_cpu_work; cpu_secs += r_cpu_secs;

            // Local winner: highest cardinality, lowest seed index on ties.
            for (int s = 0; s < nlocal; ++s) {
                const int c = local_card[s];
                if (c > best_card || (c == best_card && local_seeds[s] < best_seed)) {
                    best_card = c;
                    best_seed = local_seeds[s];
                }
            }
        }

        // Global winner (MAXLOC picks the smallest seed index among ties).
        int in[2] = {best_card, best_seed};
        int out[2];
        MPI_Allreduce(in, out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int win_card = out[0];
        const int win_seed = out[1];

        if (win_seed >= 0 && win_card > 0 && win_seed != std::numeric_limits<int>::max()) {
            // Replay the winning seed on every rank (cheap: a single seed) so
            // that the cluster list stays replicated without communication.
            growCluster(win_seed, pts, unclustered_indices.data(), U, threshold,
                        winner_scratch);

            Cluster cluster;
            cluster.seed_point = win_seed;
            cluster.members = winner_scratch.members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = true;
            }

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

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    const int nc = static_cast<int>(clusters.size());
    std::vector<double> diameters(nc, 0.0);

    // Check diameter (max distance between any two points) of each cluster
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < nc; ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }

    for (int c = 0; c < nc; ++c) {
        const double max_diameter = diameters[c];

        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n",
                   c, clusters[c].members.size(), clusters[c].seed_point, max_diameter);
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    // Generate synthetic data (replicated: identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    GpuContext gpu;
    gpuInit(gpu, points, rank);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, gpu, rank, nranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    gpuFree(gpu);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
