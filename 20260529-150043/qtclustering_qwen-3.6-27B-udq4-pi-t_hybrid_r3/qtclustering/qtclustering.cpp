// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   MPI    - Distribute seed evaluation across ranks; collective reduction for best cluster
//   OpenMP - Parallelize seed evaluation loop within each MPI rank
//   CUDA   - Offload distance/max computation in findClosestPoint to GPU
//
// Performance: Pre-allocated device memory buffers to avoid per-call malloc/free overhead.

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

#include "../common/results_output.hpp"

// ── CUDA kernels ──────────────────────────────────────────────────────────────

// Kernel: compute squared distance from each candidate to every cluster member,
//         reduce to max distance per candidate.
//   candidates: flat array of candidate point indices (size: num_candidates)
//   members:    flat array of cluster member point indices (size: num_members)
//   px, py:     point coordinates (size: point_count)
//   maxDist:    output, max distance from candidate[i] to any member (size: num_candidates)
//   threshold2: squared threshold
extern "C" __global__ void computeMaxDistKernel(
    const int* __restrict__ candidates,
    const int* __restrict__ members,
    const double* __restrict__ px,
    const double* __restrict__ py,
    double* __restrict__ maxDist,
    const int num_candidates,
    const int num_members,
    const double threshold2)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_candidates) return;

    int idx = candidates[c];
    double maxd = 0.0;
    for (int m = 0; m < num_members; ++m) {
        int j = members[m];
        double dx = px[idx] - px[j];
        double dy = py[idx] - py[j];
        double d2 = dx * dx + dy * dy;
        if (d2 > maxd) maxd = d2;
    }
    maxDist[c] = maxd;
}

// Kernel: per-block reduction to find best candidate, writes to per-block output array
//   maxDist:      max squared distance per candidate (size: num_candidates)
//   candidates:   candidate point indices (size: num_candidates)
//   threshold2:   squared threshold
//   blockBestIdx: output array, one entry per block: best candidate index
//   blockBestDist: output array, one entry per block: best maxDist
extern "C" __global__ void findBestCandidateKernel(
    const double* __restrict__ maxDist,
    const int* __restrict__ candidates,
    const int num_candidates,
    const double threshold2,
    int* blockBestIdx,
    double* blockBestDist)
{
    extern __shared__ int sIdx[];
    double* sDist = reinterpret_cast<double*>(sIdx + blockDim.x);
    int tid = threadIdx.x;

    // Initialize shared memory
    sIdx[tid] = -1;
    sDist[tid] = 1e300;

    // Block-level scan over candidates
    for (int i = tid; i < num_candidates; i += blockDim.x) {
        if (maxDist[i] < threshold2 && maxDist[i] < sDist[tid]) {
            sDist[tid] = maxDist[i];
            sIdx[tid] = i;
        }
    }
    __syncthreads();

    // Shared memory reduction (min)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sIdx[tid + s] >= 0 && (sIdx[tid] < 0 || sDist[tid + s] < sDist[tid])) {
                sIdx[tid] = sIdx[tid + s];
                sDist[tid] = sDist[tid + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        blockBestIdx[blockIdx.x] = sIdx[0];
        blockBestDist[blockIdx.x] = sDist[0];
    }
}

// ── CUDA helper ───────────────────────────────────────────────────────────────

static cudaError_t cudaCheck(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(e));
        cudaDeviceSynchronize();
    }
    return e;
}
#define CUDA_CHECK(call) cudaCheck(call, __FILE__, __LINE__)

// ── Pre-allocated device memory pool ──────────────────────────────────────────

// Maximum blocks for reduction output (covers up to ~65K candidates with 256 threads/block)
static const int MAX_BLOCKS = 2048;

struct DeviceBuffers {
    int* d_candidates = nullptr;    // max N ints
    int* d_members = nullptr;       // max N ints
    double* d_px = nullptr;         // max N doubles
    double* d_py = nullptr;         // max N doubles
    double* d_maxDist = nullptr;    // max N doubles
    int* d_blockBestIdx = nullptr;  // max MAX_BLOCKS ints
    double* d_blockBestDist = nullptr; // max MAX_BLOCKS doubles

    std::vector<int> h_blockBestIdx;
    std::vector<double> h_blockBestDist;

    void allocate(int N) {
        h_blockBestIdx.resize(MAX_BLOCKS);
        h_blockBestDist.resize(MAX_BLOCKS);
        CUDA_CHECK(cudaMalloc(&d_candidates, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_maxDist, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_blockBestIdx, MAX_BLOCKS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_blockBestDist, MAX_BLOCKS * sizeof(double)));
    }

    void free() {
        CUDA_CHECK(cudaFree(d_candidates));
        CUDA_CHECK(cudaFree(d_members));
        CUDA_CHECK(cudaFree(d_px));
        CUDA_CHECK(cudaFree(d_py));
        CUDA_CHECK(cudaFree(d_maxDist));
        CUDA_CHECK(cudaFree(d_blockBestIdx));
        CUDA_CHECK(cudaFree(d_blockBestDist));
    }
};

// ── Data structures ───────────────────────────────────────────────────────────

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ── Data generation (sequential, deterministic) ───────────────────────────────

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

// ── Distance helper ───────────────────────────────────────────────────────────

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ── CUDA-accelerated findClosestPoint (with pre-allocated buffers) ────────────

// Returns -1 if no candidate satisfies the threshold
static int findClosestPointGPU(
    DeviceBuffers& buf,
    const int* members, int num_members,
    const int* unclustered, int num_unclustered,
    double threshold)
{
    if (num_unclustered <= 0 || num_members <= 0) return -1;

    double threshold2 = threshold * threshold;

    // Upload data (only what changed)
    CUDA_CHECK(cudaMemcpy(buf.d_candidates, unclustered, num_unclustered * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_members, members, num_members * sizeof(int), cudaMemcpyHostToDevice));
    // px/py are uploaded once and reused

    // Kernel 1: compute max distance from each candidate to all members
    int blockSize = 256;
    int numBlocks = (num_unclustered + blockSize - 1) / blockSize;
    computeMaxDistKernel<<<numBlocks, blockSize>>>(
        buf.d_candidates, buf.d_members, buf.d_px, buf.d_py, buf.d_maxDist,
        num_unclustered, num_members, threshold2);

    // Kernel 2: per-block reduction to find best candidate
    size_t sharedMemSize = blockSize * sizeof(int) + blockSize * sizeof(double);
    findBestCandidateKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        buf.d_maxDist, buf.d_candidates,
        num_unclustered, threshold2,
        buf.d_blockBestIdx, buf.d_blockBestDist);

    // Host-side reduction over block results
    CUDA_CHECK(cudaMemcpy(buf.h_blockBestIdx.data(), buf.d_blockBestIdx,
                          numBlocks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buf.h_blockBestDist.data(), buf.d_blockBestDist,
                          numBlocks * sizeof(double), cudaMemcpyDeviceToHost));

    int bestLocalIdx = -1;
    double bestDist = 1e300;
    for (int b = 0; b < numBlocks; ++b) {
        if (buf.h_blockBestIdx[b] >= 0 && buf.h_blockBestDist[b] < bestDist) {
            bestDist = buf.h_blockBestDist[b];
            bestLocalIdx = buf.h_blockBestIdx[b];
        }
    }

    if (bestLocalIdx < 0) return -1;
    return unclustered[bestLocalIdx];
}

// ── Hybrid MPI/OpenMP/CUDA QT Clustering ──────────────────────────────────────

// Evaluate a single seed: returns cardinality and cluster members
static void evaluateSeed(
    DeviceBuffers& buf,
    int seed,
    const int* clustered,
    int point_count,
    double threshold,
    int& out_cardinality,
    std::vector<int>& out_members)
{
    std::vector<int> members;
    members.push_back(seed);

    // Build unclustered list for this seed evaluation
    std::vector<int> unclustered;
    unclustered.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        if (!clustered[i] && i != seed) {
            unclustered.push_back(i);
        }
    }

    while (!unclustered.empty()) {
        int closest = findClosestPointGPU(
            buf,
            members.data(), static_cast<int>(members.size()),
            unclustered.data(), static_cast<int>(unclustered.size()),
            threshold);

        if (closest < 0) break;

        members.push_back(closest);

        auto it = std::find(unclustered.begin(), unclustered.end(), closest);
        if (it != unclustered.end()) {
            unclustered.erase(it);
        }
    }

    out_cardinality = static_cast<int>(members.size());
    out_members = std::move(members);
}

std::vector<Cluster> qtClusteringParallel(
    const std::vector<Point>& points,
    const double threshold,
    int mpi_rank,
    int mpi_size)
{
    const int N = static_cast<int>(points.size());

    // Extract x and y arrays for GPU
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    std::vector<int> clustered(N, 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<Cluster> clusters;

    // Main clustering loop - sequential across iterations
    while (!unclustered_indices.empty()) {
        int num_seeds = static_cast<int>(unclustered_indices.size());

        // Distribute seeds across MPI ranks
        std::vector<int> local_seeds;
        for (int i = mpi_rank; i < num_seeds; i += mpi_size) {
            local_seeds.push_back(unclustered_indices[i]);
        }

        // OpenMP: parallelize seed evaluation within this rank
        int num_local = static_cast<int>(local_seeds.size());
        std::vector<int> local_best_card(num_local, -1);
        std::vector<std::vector<int>> local_best_members(num_local);
        std::vector<int> local_best_seed(num_local, -1);

        #pragma omp parallel
        {
            // Each thread gets its own device buffers
            DeviceBuffers buf;
            buf.allocate(N);
            // Upload px/py once per thread
            CUDA_CHECK(cudaMemcpy(buf.d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(buf.d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

            #pragma omp for schedule(dynamic, 1)
            for (int si = 0; si < num_local; ++si) {
                int seed = local_seeds[si];
                if (clustered[seed]) continue;

                int cardinality = 0;
                std::vector<int> members;
                evaluateSeed(buf, seed, clustered.data(),
                            N, threshold, cardinality, members);

                if (cardinality > local_best_card[si]) {
                    local_best_card[si] = cardinality;
                    local_best_members[si] = std::move(members);
                    local_best_seed[si] = seed;
                }
            }

            buf.free();
        }

        // Find local best on this rank
        int rank_best_card = -1;
        int rank_best_seed = -1;
        std::vector<int> rank_best_members;

        for (int si = 0; si < num_local; ++si) {
            if (local_best_card[si] > rank_best_card) {
                rank_best_card = local_best_card[si];
                rank_best_seed = local_best_seed[si];
                rank_best_members = std::move(local_best_members[si]);
            }
        }

        // MPI: find global best cluster using MPI_2INT + MPI_MAXLOC
        int local_pair[2] = {rank_best_card, rank_best_seed};
        int global_pair[2] = {-1, -1};

        MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        int best_seed = global_pair[1];
        int best_card = global_pair[0];
        int source_rank = -1;

        // Find which MPI rank owns this seed
        for (int r = 0; r < mpi_size; ++r) {
            for (int i = r; i < num_seeds; i += mpi_size) {
                if (unclustered_indices[i] == best_seed) {
                    source_rank = r;
                    break;
                }
            }
            if (source_rank >= 0) break;
        }

        // If this rank is the source, broadcast; otherwise receive
        int bcast_card = 0;
        std::vector<int> bcast_members;

        if (source_rank == mpi_rank) {
            bcast_card = best_card;
            bcast_members = rank_best_members;
        }

        // Broadcast cluster size first
        MPI_Bcast(&bcast_card, 1, MPI_INT, source_rank, MPI_COMM_WORLD);

        if (source_rank != mpi_rank) {
            bcast_members.resize(bcast_card);
        }

        // Broadcast members
        MPI_Bcast(bcast_members.data(), bcast_card, MPI_INT, source_rank, MPI_COMM_WORLD);

        // All ranks now have the best cluster
        if (best_card > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = bcast_members;
            clusters.push_back(cluster);

            // Mark members as clustered
            for (int m : cluster.members) {
                clustered[m] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break; // No more clusters can be formed
        }
    }

    return clusters;
}

// ── Validation ────────────────────────────────────────────────────────────────

bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
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
    
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
    
    // Print configuration (rank 0 only)
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("MPI ranks: %d, OpenMP threads: %d\n", mpi_size, omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        // Print CUDA device info
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
        }
    }

    // Generate synthetic data (same on all ranks for determinism)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform QT clustering with hybrid parallelization
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClusteringParallel(points, threshold, mpi_rank, mpi_size);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time_ms = cluster_time.count();
    long max_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &max_cluster_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Print results (rank 0 only)
    if (mpi_rank == 0) {
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
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
