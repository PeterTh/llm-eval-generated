// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - MPI distributes the set of "candidate seed" evaluations (the
//    expensive O(N) x O(N^2) search for the best cluster in each round)
//    across ranks/nodes.
//  - Each rank drives one CUDA-capable GPU (via the CUDA driver API and
//    NVRTC, so no .cu files or build-system special casing are needed)
//    that evaluates, in parallel, the candidate cluster cardinality for
//    every seed assigned to that rank (one CUDA thread-block per seed).
//  - MPI_Allreduce (MAXLOC-style) combines the per-rank best candidate
//    into the global best candidate for the round, with tie-breaking
//    identical to the original sequential algorithm (lowest seed index
//    wins ties).
//  - The winning cluster's exact membership is then reconstructed
//    independently (and identically) on every rank using an
//    OpenMP-parallelized version of the original CPU algorithm, so the
//    final result is bit-for-bit equivalent to the sequential reference
//    implementation without requiring any extra communication.
//  - OpenMP is additionally used for host-side reductions/statistics.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>
#include <mpi.h>
#include <omp.h>

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

// ---------------------------------------------------------------------
// CUDA error checking helpers
// ---------------------------------------------------------------------
#define CU_CHECK(call)                                                      \
    do {                                                                    \
        CUresult _res = (call);                                             \
        if (_res != CUDA_SUCCESS) {                                         \
            const char* _errStr = nullptr;                                  \
            cuGetErrorString(_res, &_errStr);                               \
            fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__,   \
                    __LINE__, _errStr ? _errStr : "unknown");               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

#define NVRTC_CHECK(call)                                                    \
    do {                                                                     \
        nvrtcResult _res = (call);                                           \
        if (_res != NVRTC_SUCCESS) {                                         \
            fprintf(stderr, "NVRTC error at %s:%d: %s\n", __FILE__, __LINE__,\
                    nvrtcGetErrorString(_res));                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------
// CUDA kernel source, compiled at runtime via NVRTC. One thread block
// evaluates one candidate seed, growing its cluster exactly as the
// original sequential algorithm would, but with the per-round "find
// closest point" scan parallelized across the block's threads.
// ---------------------------------------------------------------------
static const char* kQtKernelSrc = R"CUDA(
extern "C" __global__
void growClusterKernel(const double* __restrict__ x,
                        const double* __restrict__ y,
                        const unsigned char* __restrict__ clustered,
                        const int* __restrict__ seeds,
                        int num_seeds_in_batch,
                        double threshold,
                        int N,
                        unsigned char* __restrict__ scratch_in_cluster,
                        int* __restrict__ scratch_members,
                        int* __restrict__ out_cardinality) {
    int b = blockIdx.x;
    if (b >= num_seeds_in_batch) return;

    int seed = seeds[b];
    unsigned char* in_cluster = scratch_in_cluster + (size_t)b * (size_t)N;
    int* members = scratch_members + (size_t)b * (size_t)N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        in_cluster[i] = 0;
    }
    __syncthreads();

    __shared__ int member_count;
    if (threadIdx.x == 0) {
        in_cluster[seed] = 1;
        members[0] = seed;
        member_count = 1;
    }
    __syncthreads();

    extern __shared__ unsigned char sdata[];
    double* red_val = (double*)sdata;
    int* red_idx = (int*)(red_val + blockDim.x);

    while (member_count < N) {
        double best_val = 1.0e300;
        int best_idx = -1;

        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            if (clustered[c] || in_cluster[c]) continue;

            double max_dist = 0.0;
            for (int m = 0; m < member_count; ++m) {
                int mem = members[m];
                double dx = x[c] - x[mem];
                double dy = y[c] - y[mem];
                double d = sqrt(dx * dx + dy * dy);
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold && max_dist < best_val) {
                best_val = max_dist;
                best_idx = c;
            }
        }

        red_val[threadIdx.x] = best_val;
        red_idx[threadIdx.x] = best_idx;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                unsigned int other = threadIdx.x + stride;
                bool take_other = false;
                if (red_idx[other] >= 0) {
                    if (red_idx[threadIdx.x] < 0) {
                        take_other = true;
                    } else if (red_val[other] < red_val[threadIdx.x]) {
                        take_other = true;
                    } else if (red_val[other] == red_val[threadIdx.x] &&
                               red_idx[other] < red_idx[threadIdx.x]) {
                        take_other = true;
                    }
                }
                if (take_other) {
                    red_val[threadIdx.x] = red_val[other];
                    red_idx[threadIdx.x] = red_idx[other];
                }
            }
            __syncthreads();
        }

        int closest = red_idx[0];
        if (closest < 0) break;

        if (threadIdx.x == 0) {
            in_cluster[closest] = 1;
            members[member_count] = closest;
            member_count++;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_cardinality[b] = member_count;
    }
}
)CUDA";

// ---------------------------------------------------------------------
// Thin wrapper around the CUDA driver context / compiled module used by
// this rank for the lifetime of the program.
// ---------------------------------------------------------------------
struct GpuContext {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    int max_threads_per_block = 256;

    // Persistent device-side state, allocated once and reused across every
    // round of the clustering loop to avoid repeated cuMemAlloc/Free and
    // redundant re-uploads of the (unchanging) point coordinates.
    int alloc_N = 0;
    int batch_size = 0;
    CUdeviceptr d_x = 0, d_y = 0, d_clustered = 0;
    CUdeviceptr d_seeds = 0, d_out = 0;
    CUdeviceptr d_scratch_in_cluster = 0, d_scratch_members = 0;

    void init(int local_rank) {
        CU_CHECK(cuInit(0));

        int device_count = 0;
        CU_CHECK(cuDeviceGetCount(&device_count));
        if (device_count <= 0) {
            fprintf(stderr, "No CUDA-capable devices found.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const int dev_id = local_rank % device_count;
        CU_CHECK(cuDeviceGet(&device, dev_id));
        CU_CHECK(cuCtxCreate(&context, 0, device));

        // Compile the kernel with NVRTC once per rank.
        nvrtcProgram prog;
        NVRTC_CHECK(nvrtcCreateProgram(&prog, kQtKernelSrc, "qtclustering_kernel.cu",
                                        0, nullptr, nullptr));

        int major = 0, minor = 0;
        CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        char arch_opt[64];
        snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", major, minor);
        const char* opts[] = {arch_opt};

        nvrtcResult compile_res = nvrtcCompileProgram(prog, 1, opts);
        if (compile_res != NVRTC_SUCCESS) {
            size_t log_size = 0;
            nvrtcGetProgramLogSize(prog, &log_size);
            std::string log(log_size, '\0');
            nvrtcGetProgramLog(prog, log.data());
            fprintf(stderr, "NVRTC compile failed:\n%s\n", log.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t ptx_size = 0;
        NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
        std::string ptx(ptx_size, '\0');
        NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
        NVRTC_CHECK(nvrtcDestroyProgram(&prog));

        CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr));
        CU_CHECK(cuModuleGetFunction(&kernel, module, "growClusterKernel"));

        int max_threads = 0;
        CU_CHECK(cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
        max_threads_per_block = std::min(256, max_threads);
    }

    // Allocate persistent device buffers sized for up to `N` points/seeds
    // and upload the (fixed) point coordinates once.
    void allocateBuffers(const std::vector<double>& xs, const std::vector<double>& ys, int N) {
        CU_CHECK(cuCtxSetCurrent(context));
        alloc_N = N;

        CU_CHECK(cuMemAlloc(&d_x, sizeof(double) * N));
        CU_CHECK(cuMemAlloc(&d_y, sizeof(double) * N));
        CU_CHECK(cuMemAlloc(&d_clustered, sizeof(unsigned char) * N));
        CU_CHECK(cuMemAlloc(&d_seeds, sizeof(int) * N));
        CU_CHECK(cuMemAlloc(&d_out, sizeof(int) * N));

        CU_CHECK(cuMemcpyHtoD(d_x, xs.data(), sizeof(double) * N));
        CU_CHECK(cuMemcpyHtoD(d_y, ys.data(), sizeof(double) * N));

        size_t free_bytes = 0, total_bytes = 0;
        CU_CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
        const size_t per_seed_bytes = static_cast<size_t>(N) * (sizeof(unsigned char) + sizeof(int));
        size_t budget = free_bytes / 2; // leave headroom
        budget = std::min(budget, static_cast<size_t>(1) << 30); // cap at 1 GiB
        batch_size = static_cast<int>(budget / std::max<size_t>(per_seed_bytes, 1));
        batch_size = std::max(1, std::min(batch_size, N));

        CU_CHECK(cuMemAlloc(&d_scratch_in_cluster, static_cast<size_t>(batch_size) * N * sizeof(unsigned char)));
        CU_CHECK(cuMemAlloc(&d_scratch_members, static_cast<size_t>(batch_size) * N * sizeof(int)));
    }

    void freeBuffers() {
        if (d_x) cuMemFree(d_x);
        if (d_y) cuMemFree(d_y);
        if (d_clustered) cuMemFree(d_clustered);
        if (d_seeds) cuMemFree(d_seeds);
        if (d_out) cuMemFree(d_out);
        if (d_scratch_in_cluster) cuMemFree(d_scratch_in_cluster);
        if (d_scratch_members) cuMemFree(d_scratch_members);
        d_x = d_y = d_clustered = d_seeds = d_out = 0;
        d_scratch_in_cluster = d_scratch_members = 0;
    }

    void destroy() {
        freeBuffers();
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }
};

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists. Parallelized across OpenMP threads with a
// deterministic (value, then lowest-index) reduction so the result exactly
// matches the original sequential scan regardless of thread scheduling.
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        int local_closest = -1;
        double local_min = std::numeric_limits<double>::max();

        #pragma omp for schedule(static) nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            double max_dist = 0.0;
            for (size_t i = 0; i < cluster_members.size(); ++i) {
                const int member = cluster_members[i];
                const double dist = distance(points[candidate], points[member]);
                max_dist = std::max(max_dist, dist);
            }

            if (max_dist < threshold && max_dist < local_min) {
                local_min = max_dist;
                local_closest = candidate;
            }
        }

        #pragma omp critical
        {
            if (local_closest >= 0) {
                if (local_min < min_diameter ||
                    (local_min == min_diameter &&
                     (closest_point < 0 || local_closest < closest_point))) {
                    min_diameter = local_min;
                    closest_point = local_closest;
                }
            }
        }
    }

    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;

    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points,
                                             threshold, point_count);

        if (closest < 0) break; // No more points can be added

        in_cluster[closest] = true;
        members.push_back(closest);
    }

    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// ---------------------------------------------------------------------
// GPU-accelerated evaluation of candidate cluster cardinality for a
// batch of seeds assigned to this rank. Returns, for each seed in
// `seeds`, the resulting cardinality in `out_cardinalities` (same order).
// Uses `gpu`'s persistent device buffers (allocated once via
// allocateBuffers()) and only uploads the small per-round `clustered`
// and `seeds` arrays, batching kernel launches to respect the
// pre-computed GPU memory budget for per-seed scratch space.
// ---------------------------------------------------------------------
void evaluateSeedsOnGpu(GpuContext& gpu,
                         const std::vector<unsigned char>& clustered_flags,
                         const std::vector<int>& seeds,
                         double threshold,
                         int N,
                         std::vector<int>& out_cardinalities) {
    const int num_seeds = static_cast<int>(seeds.size());
    out_cardinalities.assign(num_seeds, 0);
    if (num_seeds == 0 || N == 0) return;

    CU_CHECK(cuCtxSetCurrent(gpu.context));

    CU_CHECK(cuMemcpyHtoD(gpu.d_clustered, clustered_flags.data(), sizeof(unsigned char) * N));
    CU_CHECK(cuMemcpyHtoD(gpu.d_seeds, seeds.data(), sizeof(int) * num_seeds));

    const int threads_per_block = gpu.max_threads_per_block;
    const size_t shared_bytes = static_cast<size_t>(threads_per_block) * (sizeof(double) + sizeof(int));
    const int batch_size = gpu.batch_size;

    for (int offset = 0; offset < num_seeds; offset += batch_size) {
        const int this_batch = std::min(batch_size, num_seeds - offset);
        CUdeviceptr d_seeds_batch = gpu.d_seeds + static_cast<size_t>(offset) * sizeof(int);
        CUdeviceptr d_out_batch = gpu.d_out + static_cast<size_t>(offset) * sizeof(int);

        void* args[] = {
            &gpu.d_x, &gpu.d_y, &gpu.d_clustered, &d_seeds_batch, (void*)&this_batch,
            (void*)&threshold, (void*)&N, &gpu.d_scratch_in_cluster, &gpu.d_scratch_members, &d_out_batch
        };

        CU_CHECK(cuLaunchKernel(gpu.kernel,
                                 this_batch, 1, 1,
                                 threads_per_block, 1, 1,
                                 static_cast<unsigned int>(shared_bytes), nullptr,
                                 args, nullptr));
    }
    CU_CHECK(cuCtxSynchronize());

    CU_CHECK(cuMemcpyDtoH(out_cardinalities.data(), gpu.d_out, sizeof(int) * num_seeds));
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  GpuContext& gpu,
                                  int mpi_rank,
                                  int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<unsigned char> clustered_flags(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    std::vector<double> xs(N), ys(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        xs[i] = points[i].x;
        ys[i] = points[i].y;
    }

    gpu.allocateBuffers(xs, ys, N);

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Encode (cardinality, seed) pairs into a single monotonic score so that
    // a plain MPI_MAXLOC comparison reproduces the sequential tie-breaking
    // rule "highest cardinality; ties broken by lowest seed index".
    const double SCALE = static_cast<double>(N) + 1.0;

    while (!unclustered_indices.empty()) {
        // Partition the (identical, on every rank) unclustered index list
        // round-robin across ranks so the per-round GPU work is balanced.
        std::vector<int> local_seeds;
        for (size_t i = mpi_rank; i < unclustered_indices.size(); i += mpi_size) {
            local_seeds.push_back(unclustered_indices[i]);
        }

        std::vector<int> local_cardinalities;
        evaluateSeedsOnGpu(gpu, clustered_flags, local_seeds, threshold, N,
                           local_cardinalities);

        double local_score = -std::numeric_limits<double>::max();
        int local_best_seed = -1;
        for (size_t i = 0; i < local_seeds.size(); ++i) {
            const int seed = local_seeds[i];
            const int card = local_cardinalities[i];
            const double score = static_cast<double>(card) * SCALE - static_cast<double>(seed);
            if (score > local_score) {
                local_score = score;
                local_best_seed = seed;
            }
        }

        struct { double score; int seed; } local_pair, global_pair;
        local_pair.score = local_score;
        local_pair.seed = local_best_seed;

        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        const int best_seed = global_pair.seed;

        if (best_seed < 0) {
            break; // No more clusters can be formed
        }

        // Reconstruct the exact winning cluster identically (and
        // redundantly, but cheaply) on every rank using the untouched
        // sequential-semantics algorithm, guaranteeing bit-for-bit
        // equivalence with the reference implementation.
        std::vector<int> best_cluster_members;
        const int max_cardinality = generateCandidateCluster(best_seed, clustered, points,
                                                              threshold, N, &best_cluster_members);

        if (max_cardinality <= 0) {
            break;
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_cluster_members;
        clusters.push_back(cluster);

        for (size_t i = 0; i < best_cluster_members.size(); ++i) {
            clustered[best_cluster_members[i]] = true;
            clustered_flags[best_cluster_members[i]] = 1;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
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
    #pragma omp parallel for schedule(static) reduction(+:clustered_count)
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
    int provided = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Avoid oversubscribing the node's CPU cores when multiple MPI ranks
    // share it: unless the user has explicitly pinned OMP_NUM_THREADS,
    // split the available cores evenly across the ranks co-located on
    // this node.
    if (!getenv("OMP_NUM_THREADS")) {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &local_comm);
        int local_size = 1;
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_free(&local_comm);

        const int hw_threads = omp_get_num_procs();
        const int threads_per_rank = std::max(1, hw_threads / std::max(1, local_size));
        omp_set_num_threads(threads_per_rank);
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI + OpenMP + CUDA)\n");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", mpi_size, omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Set up this rank's GPU context (one rank per GPU, round-robin).
    GpuContext gpu;
    gpu.init(mpi_rank);

    // Generate synthetic data identically (deterministically) on every
    // rank, avoiding the need to broadcast the point set.
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, gpu, mpi_rank, mpi_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    gpu.destroy();

    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;

        #pragma omp parallel for schedule(static) reduction(+:total_clustered) reduction(max:max_cluster_size)
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

    // Validation (every rank has an identical `clusters` result, so this
    // is redundant but harmless; only rank 0 prints).
    int exit_code = 0;
    if (validate) {
        const bool valid = mpi_rank == 0 ? validateClusters(clusters, points, threshold) : true;
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            printf(valid_int ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        exit_code = valid_int ? 0 : 1;
    }

    MPI_Finalize();
    return exit_code;
}
