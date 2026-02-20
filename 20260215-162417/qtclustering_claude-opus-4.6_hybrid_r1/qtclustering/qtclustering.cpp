// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// MPI: distributes candidate seed evaluation across ranks
// OpenMP: parallelizes seed processing within each rank
// CUDA: accelerates the distance computation kernel (findClosestPoint)

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

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
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// CUDA kernel: for each non-excluded point, compute max distance to all cluster members.
// Uses shared memory for member coordinates when they fit, otherwise global memory.
__global__ void computeMaxDistsKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ members,
    int num_members,
    const int8_t* __restrict__ excluded,
    double* __restrict__ max_dists,
    int N,
    double threshold,
    int use_smem)
{
    extern __shared__ double smem[];

    if (use_smem) {
        double* sx = smem;
        double* sy = sx + num_members;
        for (int i = threadIdx.x; i < num_members; i += blockDim.x) {
            int m = members[i];
            sx[i] = px[m];
            sy[i] = py[m];
        }
        __syncthreads();
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (excluded[idx]) {
        max_dists[idx] = 1.0e308;
        return;
    }

    double cx = px[idx];
    double cy = py[idx];
    double maxd = 0.0;

    if (use_smem) {
        const double* sx = smem;
        const double* sy = sx + num_members;
        for (int j = 0; j < num_members; j++) {
            double dx = cx - sx[j];
            double dy = cy - sy[j];
            double d = sqrt(dx * dx + dy * dy);
            if (d > maxd) maxd = d;
        }
    } else {
        for (int j = 0; j < num_members; j++) {
            int m = members[j];
            double dx = cx - px[m];
            double dy = cy - py[m];
            double d = sqrt(dx * dx + dy * dy);
            if (d > maxd) maxd = d;
        }
    }

    max_dists[idx] = (maxd < threshold) ? maxd : 1.0e308;
}

// Per-OpenMP-thread GPU resources
struct GPUResources {
    int8_t* d_excluded;
    int* d_members;
    double* d_max_dists;
    double* h_max_dists;  // pinned host memory for async transfer
    cudaStream_t stream;
};

// Main QT clustering algorithm using MPI + OpenMP + CUDA
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  int device_id) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i)
        unclustered_indices.push_back(i);

    // SOA layout for coalesced GPU memory access
    std::vector<double> h_px(N), h_py(N);
    for (int i = 0; i < N; ++i) {
        h_px[i] = points[i].x;
        h_py[i] = points[i].y;
    }

    // Allocate and copy point data to GPU (read-only, shared across threads)
    double *d_px, *d_py;
    CHECK_CUDA(cudaMalloc(&d_px, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_py, N * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Per-thread GPU resources allocated inside parallel region for correct CUDA context
    int max_threads = omp_get_max_threads();
    std::vector<GPUResources> gpu_res(max_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        CHECK_CUDA(cudaSetDevice(device_id));
        CHECK_CUDA(cudaMalloc(&gpu_res[tid].d_excluded, N * sizeof(int8_t)));
        CHECK_CUDA(cudaMalloc(&gpu_res[tid].d_members, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&gpu_res[tid].d_max_dists, N * sizeof(double)));
        CHECK_CUDA(cudaMallocHost(&gpu_res[tid].h_max_dists, N * sizeof(double)));
        CHECK_CUDA(cudaStreamCreate(&gpu_res[tid].stream));
    }

    std::vector<int8_t> h_clustered(N, 0);

    while (!unclustered_indices.empty()) {
        const int num_unc = static_cast<int>(unclustered_indices.size());

        // MPI: distribute seeds round-robin across ranks
        std::vector<int> my_seeds;
        for (int i = rank; i < num_unc; i += nprocs)
            my_seeds.push_back(unclustered_indices[i]);

        int local_best_card = -1;
        int local_best_seed = INT_MAX;
        std::vector<int> local_best_members;

        const int nseeds = static_cast<int>(my_seeds.size());

        // OpenMP: parallelize seed processing within this rank
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            CHECK_CUDA(cudaSetDevice(device_id));
            GPUResources& res = gpu_res[tid];

            int thr_card = -1;
            int thr_seed = INT_MAX;
            std::vector<int> thr_members;

            #pragma omp for schedule(dynamic)
            for (int si = 0; si < nseeds; ++si) {
                int seed = my_seeds[si];

                // Build candidate cluster for this seed using CUDA
                std::vector<int8_t> excl(N);
                std::memcpy(excl.data(), h_clustered.data(), N);

                std::vector<int> members;
                members.reserve(256);
                members.push_back(seed);
                excl[seed] = 1;

                // Upload full excluded state for this seed
                cudaMemcpyAsync(res.d_excluded, excl.data(), N * sizeof(int8_t),
                                cudaMemcpyHostToDevice, res.stream);

                // Iteratively grow cluster
                while (static_cast<int>(members.size()) < N) {
                    int nmem = static_cast<int>(members.size());

                    // Upload current cluster members
                    cudaMemcpyAsync(res.d_members, members.data(),
                                    nmem * sizeof(int),
                                    cudaMemcpyHostToDevice, res.stream);

                    // Launch kernel: each GPU thread evaluates one candidate point
                    int bs = 256;
                    int gs = (N + bs - 1) / bs;
                    size_t smem_bytes = 2 * nmem * sizeof(double);
                    int use_smem = (smem_bytes <= 48u * 1024u) ? 1 : 0;
                    if (!use_smem) smem_bytes = 0;

                    computeMaxDistsKernel<<<gs, bs, smem_bytes, res.stream>>>(
                        d_px, d_py, res.d_members, nmem,
                        res.d_excluded, res.d_max_dists, N, threshold, use_smem);

                    // Download results (pinned host memory for true async)
                    cudaMemcpyAsync(res.h_max_dists, res.d_max_dists,
                                    N * sizeof(double),
                                    cudaMemcpyDeviceToHost, res.stream);
                    cudaStreamSynchronize(res.stream);

                    // CPU reduction: find candidate with minimum max-distance
                    int closest = -1;
                    double min_d = 1.0e308;
                    for (int i = 0; i < N; ++i) {
                        if (res.h_max_dists[i] < min_d) {
                            min_d = res.h_max_dists[i];
                            closest = i;
                        }
                    }

                    if (closest < 0 || min_d >= 1.0e300) break;

                    members.push_back(closest);
                    excl[closest] = 1;
                    // Incrementally update single excluded element on GPU
                    cudaMemsetAsync(res.d_excluded + closest, 1, 1, res.stream);
                }

                int card = static_cast<int>(members.size());
                if (card > thr_card || (card == thr_card && seed < thr_seed)) {
                    thr_card = card;
                    thr_seed = seed;
                    thr_members = std::move(members);
                }
            }

            // Merge thread-local results
            #pragma omp critical
            {
                if (thr_card > local_best_card ||
                    (thr_card == local_best_card && thr_seed < local_best_seed)) {
                    local_best_card = thr_card;
                    local_best_seed = thr_seed;
                    local_best_members = std::move(thr_members);
                }
            }
        }

        // MPI: find global best cluster with deterministic tiebreaking
        int global_max_card;
        MPI_Allreduce(&local_best_card, &global_max_card, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_card <= 0) break;

        // Tiebreak by lowest seed index (matches sequential iteration order)
        int my_seed_val = (local_best_card == global_max_card) ? local_best_seed : INT_MAX;
        int global_min_seed;
        MPI_Allreduce(&my_seed_val, &global_min_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Identify winner rank
        int am_winner = (local_best_card == global_max_card && local_best_seed == global_min_seed)
                        ? rank : nprocs;
        int winner_rank;
        MPI_Allreduce(&am_winner, &winner_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Winner broadcasts cluster members to all ranks
        std::vector<int> best_members(global_max_card);
        if (rank == winner_rank)
            best_members = local_best_members;
        MPI_Bcast(best_members.data(), global_max_card, MPI_INT, winner_rank, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global_min_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        // All ranks update clustered state identically
        for (int m : best_members) {
            clustered[m] = true;
            h_clustered[m] = 1;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    // Cleanup GPU resources
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        CHECK_CUDA(cudaSetDevice(device_id));
        cudaFree(gpu_res[tid].d_excluded);
        cudaFree(gpu_res[tid].d_members);
        cudaFree(gpu_res[tid].d_max_dists);
        cudaFreeHost(gpu_res[tid].h_max_dists);
        cudaStreamDestroy(gpu_res[tid].stream);
    }
    cudaFree(d_px);
    cudaFree(d_py);

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Multi-GPU: each MPI rank uses a different GPU
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    int device_id = 0;
    if (num_devices > 0) {
        device_id = rank % num_devices;
        cudaSetDevice(device_id);
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

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
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // All ranks generate identical data (same seed)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, device_id);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

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

        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
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

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);

            if (valid) {
                printf("Validation: PASSED\n");
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
