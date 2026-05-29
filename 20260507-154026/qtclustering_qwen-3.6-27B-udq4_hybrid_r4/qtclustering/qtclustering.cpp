// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   MPI    - distribute seed-point evaluation across ranks
//   OpenMP - parallelize seed evaluation within each rank
//   CUDA   - accelerate max-distance reduction in findClosestPoint
//
// Each OpenMP thread has its own CUDA buffers (d_cand, d_mem, d_mdist)
// to avoid shared-buffer races.  GPU operations are serialized with
// #pragma omp critical to avoid concurrent-kernel GPU races.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cstdint>

#include "../common/results_output.hpp"

static const double MAX_WIDTH  = 20.0;
static const double MAX_HEIGHT = 20.0;

// ---- data structures ------------------------------------------------

struct Point   { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

// ============================================================
//  CUDA KERNEL
// ============================================================
// For every candidate point compute the maximum Euclidean
// distance to any cluster member  (one reduction per block).

__global__ void computeMaxDistancesKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int*    __restrict__ cand_idx,
    const int*    __restrict__ mem_idx,
    double*       __restrict__ max_dists,
    int nc, int nm)
{
    int ci = blockIdx.x;
    double md = 0.0;
    if (ci < nc) {
        int    c  = cand_idx[ci];
        double cx = px[c];
        double cy = py[c];

        // each thread walks a slice of the member list
        for (int m = threadIdx.x; m < nm; m += blockDim.x) {
            int    mi = mem_idx[m];
            double dx = cx - px[mi];
            double dy = cy - py[mi];
            double d  = sqrt(dx * dx + dy * dy);
            if (d > md) md = d;
        }
    }

    // shared-memory max-reduction (all 256 threads participate)
    __shared__ double sd[256];
    sd[threadIdx.x] = md;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sd[threadIdx.x] = fmax(sd[threadIdx.x], sd[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0 && ci < nc)
        max_dists[ci] = sd[0];
}

// ============================================================
//  CUDA CONTEXT  (one per MPI rank, per-thread buffers)
// ============================================================

struct PerThreadBufs {
    int*    d_cand  = nullptr;
    int*    d_mem   = nullptr;
    double* d_mdist = nullptr;
    int*    h_cand  = nullptr;
    int*    h_mem   = nullptr;
    double* h_mdist = nullptr;

    void alloc(int n) {
        cudaMalloc(&d_cand,  n * sizeof(int));
        cudaMalloc(&d_mem,   n * sizeof(int));
        cudaMalloc(&d_mdist, n * sizeof(double));
        cudaMallocHost(&h_cand,  n * sizeof(int));
        cudaMallocHost(&h_mem,   n * sizeof(int));
        cudaMallocHost(&h_mdist, n * sizeof(double));
    }
    ~PerThreadBufs() {
        cudaFree(d_cand); cudaFree(d_mem); cudaFree(d_mdist);
        cudaFreeHost(h_cand); cudaFreeHost(h_mem); cudaFreeHost(h_mdist);
    }
};

struct CUDAContext {
    double*       d_px     = nullptr;   // shared point coords
    double*       d_py     = nullptr;
    std::vector<PerThreadBufs> bufs;    // per-thread
    int           max_points  = 0;
    int           num_threads = 0;
    int           device_id   = 0;

    void init(const std::vector<Point>& pts, int threads, int dev) {
        max_points  = static_cast<int>(pts.size());
        num_threads = threads;
        device_id   = dev;

        // extract x / y
        std::vector<double> px(max_points), py(max_points);
        for (int i = 0; i < max_points; ++i) {
            px[i] = pts[i].x;
            py[i] = pts[i].y;
        }

        cudaMalloc(&d_px, max_points * sizeof(double));
        cudaMalloc(&d_py, max_points * sizeof(double));
        cudaMemcpy(d_px, px.data(), max_points * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_py, py.data(), max_points * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        bufs.resize(threads);
        for (int t = 0; t < threads; ++t)
            bufs[t].alloc(max_points);
    }

    ~CUDAContext() {
        cudaFree(d_px); cudaFree(d_py);
    }
};

// ============================================================
//  CUDA-ACCELERATED  findClosestPoint
// ============================================================
// Each thread uses its own per-thread buffers.  GPU operations are
// serialized with #pragma omp critical (avoids concurrent-kernel GPU
// races that can occur on some GPU architectures).

static int findClosestPointCUDA(
    CUDAContext& ctx, int tid,
    const std::vector<int>&  cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    double threshold, int point_count)
{
    auto& b = ctx.bufs[tid];

    // build candidate list
    std::vector<int> cands;
    for (int i = 0; i < point_count; ++i)
        if (!clustered[i] && !in_cluster[i])
            cands.push_back(i);
    if (cands.empty()) return -1;

    int nc = static_cast<int>(cands.size());
    int nm = static_cast<int>(cluster_members.size());

    // copy into per-thread pinned staging buffers
    memcpy(b.h_cand, cands.data(), nc * sizeof(int));
    memcpy(b.h_mem, cluster_members.data(), nm * sizeof(int));

    // serialize GPU operations to avoid concurrent-kernel GPU races
    #pragma omp critical(cuda_op)
    {
        cudaMemcpy(b.d_cand, b.h_cand, nc * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(b.d_mem, b.h_mem, nm * sizeof(int),
                   cudaMemcpyHostToDevice);

        computeMaxDistancesKernel<<<nc, 256>>>(
            ctx.d_px, ctx.d_py,
            b.d_cand, b.d_mem,
            b.d_mdist, nc, nm);

        cudaMemcpy(b.h_mdist, b.d_mdist, nc * sizeof(double),
                   cudaMemcpyDeviceToHost);
    }

    // pick closest valid candidate
    int    closest = -1;
    double min_diam = std::numeric_limits<double>::max();
    for (int i = 0; i < nc; ++i) {
        if (b.h_mdist[i] < threshold && b.h_mdist[i] < min_diam) {
            min_diam = b.h_mdist[i];
            closest  = cands[i];
        }
    }
    return closest;
}

// ============================================================
//  CUDA-ACCELERATED  generateCandidateCluster
// ============================================================

static int generateCandidateClusterCUDA(
    CUDAContext& ctx, int tid,
    int seed_point,
    const std::vector<bool>& clustered,
    const std::vector<Point>& /*points*/,
    double threshold, int point_count,
    std::vector<int>* out_members)
{
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int>  members;
    members.reserve(point_count);

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        int closest = findClosestPointCUDA(ctx, tid, members,
                                           clustered, in_cluster,
                                           threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (out_members) *out_members = members;
    return static_cast<int>(members.size());
}

// ============================================================
//  DATA GENERATION
// ============================================================

static void generateSyntheticData(std::vector<Point>& points, int N,
                                  unsigned int seed = 42)
{
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R      = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (group_cnt > (N - count)) group_cnt = N - count;

        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r    = frand() * R;
            const double dx   = (2.0 * frand() - 1.0) * r;
            const double dy   = std::sqrt(r * r - dx * dx) * sign;
            const double x    = cntr_x + dx;
            const double y    = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT)
                continue;
            points[count] = {x, y};
            ++count;
            --group_cnt;
        }
    }
}

// ============================================================
//  DISTANCE  (for validation only)
// ============================================================

static inline double distance(const Point& a, const Point& b)
{
    double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ============================================================
//  HYBRID QT CLUSTERING  (MPI + OpenMP + CUDA)
// ============================================================

static std::vector<Cluster> qtClusteringHybrid(
    CUDAContext& ctx,
    const std::vector<Point>& points,
    double threshold,
    int mpi_rank, int mpi_size)
{
    int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int>  unclustered;
    unclustered.reserve(N);
    for (int i = 0; i < N; ++i) unclustered.push_back(i);

    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        int nu = static_cast<int>(unclustered.size());
        int chunk     = (nu + mpi_size - 1) / mpi_size;
        int lo_start  = mpi_rank * chunk;
        int lo_end    = std::min(lo_start + chunk, nu);

        int nthreads = ctx.num_threads;

        struct TResult { int seed = -1, card = -1; std::vector<int> members; };
        std::vector<TResult> tresults(nthreads);

        // ----- OpenMP: parallel seed evaluation -----
        #pragma omp parallel num_threads(nthreads)
        {
            // ensure each thread uses the correct CUDA device
            cudaSetDevice(ctx.device_id);

            int  tid = omp_get_thread_num();
            TResult thread_best;

            #pragma omp for schedule(dynamic, 4)
            for (int i = lo_start; i < lo_end; ++i) {
                int seed = unclustered[i];
                if (clustered[seed]) continue;

                std::vector<int> cm;
                int card = generateCandidateClusterCUDA(
                    ctx, tid, seed, clustered, points, threshold, N, &cm);

                if (card > thread_best.card ||
                    (card == thread_best.card &&
                     (thread_best.seed < 0 || seed < thread_best.seed))) {
                    thread_best.seed    = seed;
                    thread_best.card    = card;
                    thread_best.members = std::move(cm);
                }
            }
            tresults[tid] = std::move(thread_best);
        }

        // reduce across threads
        int  lo_seed = -1, lo_card = -1;
        std::vector<int> lo_members;
        for (int t = 0; t < nthreads; ++t) {
            if (tresults[t].card > lo_card ||
                (tresults[t].card == lo_card &&
                 (lo_seed < 0 || tresults[t].seed < lo_seed))) {
                lo_card    = tresults[t].card;
                lo_seed    = tresults[t].seed;
                lo_members = std::move(tresults[t].members);
            }
        }

        // ----- MPI: global reduce -----
        // encode (card+1, 0x7FFFFFFF - seed) so MPI_MAX picks
        // highest cardinality, then lowest seed for ties
        uint64_t lo_val = ((uint64_t)(lo_card + 1) << 32) |
                          (uint64_t)(0x7FFFFFFF - lo_seed);
        uint64_t gl_val;
        MPI_Allreduce(&lo_val, &gl_val, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);

        int gl_card = static_cast<int>(gl_val >> 32) - 1;
        int gl_seed = 0x7FFFFFFF - static_cast<int>(gl_val & 0xFFFFFFFF);
        if (gl_card < 1) break;

        // find the rank that owns the winner
        int my_win_rank = (lo_seed == gl_seed) ? mpi_rank : -1;
        int win_rank;
        MPI_Allreduce(&my_win_rank, &win_rank, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);

        // broadcast members from winner
        std::vector<int> best_members;
        if (mpi_rank == win_rank) {
            MPI_Bcast(&gl_card, 1, MPI_INT, win_rank, MPI_COMM_WORLD);
            MPI_Bcast(lo_members.data(), gl_card, MPI_INT,
                      win_rank, MPI_COMM_WORLD);
            best_members = std::move(lo_members);
        } else {
            int card;
            MPI_Bcast(&card, 1, MPI_INT, win_rank, MPI_COMM_WORLD);
            best_members.resize(card);
            MPI_Bcast(best_members.data(), card, MPI_INT,
                      win_rank, MPI_COMM_WORLD);
        }

        // commit cluster
        Cluster cl;
        cl.seed_point = gl_seed;
        cl.members    = std::move(best_members);
        clusters.push_back(cl);

        for (int m : cl.members) clustered[m] = true;

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                [&clustered](int idx){ return clustered[idx]; }),
            unclustered.end());
    }

    return clusters;
}

// ============================================================
//  VALIDATION
// ============================================================

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points,
                             double threshold)
{
    bool valid = true;
    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cl = clusters[c];
        double max_diam = 0.0;
        for (size_t i = 0; i < cl.members.size(); ++i)
            for (size_t j = i + 1; j < cl.members.size(); ++j) {
                double d = distance(points[cl.members[i]],
                                    points[cl.members[j]]);
                max_diam = std::max(max_diam, d);
            }
        if (c < 10)
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cl.members.size(), cl.seed_point, max_diam);
        if (max_diam > threshold * 1.001) {
            printf("ERROR: Cluster %zu diameter %.4f > threshold %.4f\n",
                   c, max_diam, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            int m = clusters[c].members[i];
            if (membership[m] >= 0) {
                printf("ERROR: Point %d in clusters %d and %zu\n",
                       m, membership[m], c);
                valid = false;
            }
            membership[m] = static_cast<int>(c);
        }

    int cc = 0;
    for (size_t i = 0; i < membership.size(); ++i)
        if (membership[i] >= 0) ++cc;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), cc, points.size() - cc);
    return valid;
}

// ============================================================
//  MAIN
// ============================================================

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // select one CUDA device per rank
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    int my_dev = (ndev > 0) ? (mpi_rank % ndev) : 0;
    cudaSetDevice(my_dev);

    // ---- parse args (rank 0) ----
    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0, printResults = 0;

    if (mpi_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                num_points = atoi(argv[++i]);
            else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
                threshold = atof(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)
                validate = 1;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = 1;
            else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Number of points (default: 1000)\n");
                printf("  -t <float>   Distance threshold (default: 2.0)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                return 1;
            }
        }
    }

    // broadcast parameters
    MPI_Bcast(&num_points,   1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold,    1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT,    0, MPI_COMM_WORLD);

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    // ---- generate data (rank 0 -> broadcast) ----
    std::vector<Point> points(num_points);
    if (mpi_rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- init CUDA ----
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;

    CUDAContext ctx;
    ctx.init(points, nthreads, my_dev);

    // ---- print banner (rank 0) ----
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA devices: %d, OpenMP threads: %d\n",
               mpi_size, ndev, nthreads);
    }

    // ---- clustering ----
    double t0 = MPI_Wtime();
    std::vector<Cluster> clusters =
        qtClusteringHybrid(ctx, points, threshold, mpi_rank, mpi_size);
    double t1 = MPI_Wtime();

    // ---- results (rank 0) ----
    if (mpi_rank == 0) {
        double elapsed_ms = (t1 - t0) * 1000.0;
        printf("Clustering time: %.0f ms\n", elapsed_ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0, max_cs = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            int s = static_cast<int>(clusters[i].members.size());
            total_clustered += s;
            max_cs = std::max(max_cs, s);
        }
        double avg_cs = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cs);
        printf("Maximum cluster size: %d\n", max_cs);

        double tsec = t1 - t0;
        if (tsec > 0)
            printf("Performance: %.1f clusters/s, %.1f points/s\n",
                   clusters.size() / tsec, num_points / tsec);

        if (printResults) {
            std::vector<double> mdata;
            mdata.reserve(num_points);
            std::vector<int> mem(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (size_t i = 0; i < clusters[c].members.size(); ++i)
                    mem[clusters[c].members[i]] = static_cast<int>(c);
            for (int m : mem) mdata.push_back(static_cast<double>(m));
            print_results(mdata, "ClusterMembership");
        }

        if (validate) {
            bool ok = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            if (!ok) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
