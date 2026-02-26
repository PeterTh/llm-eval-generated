// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <climits>
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

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static inline void mpiCheck(int err, const char* msg) {
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "MPI error (%s)\n", msg);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
}

__global__ void updateMaxAndBlockBestKernel(const double* __restrict__ x,
                                           const double* __restrict__ y,
                                           int N,
                                           int new_member,
                                           const uint8_t* __restrict__ clustered,
                                           const uint8_t* __restrict__ in_cluster,
                                           double* __restrict__ max_dist2,
                                           double threshold2,
                                           double* __restrict__ block_best_dist2,
                                           int* __restrict__ block_best_idx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double best_d2 = DBL_MAX;
    int best_i = INT_MAX;

    if (idx < N) {
        const bool valid = (clustered[idx] == 0) && (in_cluster[idx] == 0);
        if (valid) {
            const double dx = x[idx] - x[new_member];
            const double dy = y[idx] - y[new_member];
            const double d2 = dx * dx + dy * dy;
            const double prev = max_dist2[idx];
            const double newMax = (prev > d2) ? prev : d2;
            max_dist2[idx] = newMax;
            if (newMax < threshold2) {
                best_d2 = newMax;
                best_i = idx;
            }
        }
    }

    __shared__ double s_d2[256];
    __shared__ int s_i[256];
    s_d2[threadIdx.x] = best_d2;
    s_i[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const double od2 = s_d2[threadIdx.x + stride];
            const int oi = s_i[threadIdx.x + stride];
            const double cd2 = s_d2[threadIdx.x];
            const int ci = s_i[threadIdx.x];
            if (od2 < cd2 || (od2 == cd2 && oi < ci)) {
                s_d2[threadIdx.x] = od2;
                s_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int outIdx = blockIdx.x;
        block_best_dist2[outIdx] = s_d2[0];
        block_best_idx[outIdx] = (s_i[0] == INT_MAX) ? -1 : s_i[0];
    }
}

__global__ void markClusteredKernel(uint8_t* clustered, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        clustered[members[i]] = 1;
    }
}

struct CudaCtx {
    int N = 0;
    int blocks = 0;
    static constexpr int TPB = 256;

    double* d_x = nullptr;
    double* d_y = nullptr;
    uint8_t* d_clustered = nullptr;
    uint8_t* d_in_cluster = nullptr;
    double* d_max_d2 = nullptr;

    double* d_block_best_d2 = nullptr;
    int* d_block_best_i = nullptr;

    double* h_block_best_d2 = nullptr;
    int* h_block_best_i = nullptr;

    int* d_members = nullptr;

    cudaStream_t stream = nullptr;
};

static void initCudaCtx(CudaCtx& ctx, const std::vector<Point>& points, int rank) {
    ctx.N = static_cast<int>(points.size());
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    const int dev = rank % deviceCount;
    cudaCheck(cudaSetDevice(dev), "cudaSetDevice");

    cudaCheck(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "cudaStreamCreate");

    ctx.blocks = (ctx.N + CudaCtx::TPB - 1) / CudaCtx::TPB;

    std::vector<double> hx(ctx.N), hy(ctx.N);
    for (int i = 0; i < ctx.N; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
    }

    cudaCheck(cudaMalloc(&ctx.d_x, ctx.N * sizeof(double)), "malloc d_x");
    cudaCheck(cudaMalloc(&ctx.d_y, ctx.N * sizeof(double)), "malloc d_y");
    cudaCheck(cudaMemcpyAsync(ctx.d_x, hx.data(), ctx.N * sizeof(double), cudaMemcpyHostToDevice, ctx.stream), "cpy x");
    cudaCheck(cudaMemcpyAsync(ctx.d_y, hy.data(), ctx.N * sizeof(double), cudaMemcpyHostToDevice, ctx.stream), "cpy y");

    cudaCheck(cudaMalloc(&ctx.d_clustered, ctx.N * sizeof(uint8_t)), "malloc d_clustered");
    cudaCheck(cudaMemsetAsync(ctx.d_clustered, 0, ctx.N * sizeof(uint8_t), ctx.stream), "memset clustered");

    cudaCheck(cudaMalloc(&ctx.d_in_cluster, ctx.N * sizeof(uint8_t)), "malloc d_in_cluster");
    cudaCheck(cudaMalloc(&ctx.d_max_d2, ctx.N * sizeof(double)), "malloc d_max_d2");

    cudaCheck(cudaMalloc(&ctx.d_block_best_d2, ctx.blocks * sizeof(double)), "malloc d_block_best_d2");
    cudaCheck(cudaMalloc(&ctx.d_block_best_i, ctx.blocks * sizeof(int)), "malloc d_block_best_i");

    cudaCheck(cudaHostAlloc(&ctx.h_block_best_d2, ctx.blocks * sizeof(double), cudaHostAllocPortable), "hostalloc d2");
    cudaCheck(cudaHostAlloc(&ctx.h_block_best_i, ctx.blocks * sizeof(int), cudaHostAllocPortable), "hostalloc i");

    cudaCheck(cudaMalloc(&ctx.d_members, ctx.N * sizeof(int)), "malloc d_members");

    cudaCheck(cudaStreamSynchronize(ctx.stream), "sync init");
}

static void freeCudaCtx(CudaCtx& ctx) {
    if (ctx.stream) cudaStreamSynchronize(ctx.stream);
    if (ctx.d_members) cudaFree(ctx.d_members);
    if (ctx.h_block_best_i) cudaFreeHost(ctx.h_block_best_i);
    if (ctx.h_block_best_d2) cudaFreeHost(ctx.h_block_best_d2);
    if (ctx.d_block_best_i) cudaFree(ctx.d_block_best_i);
    if (ctx.d_block_best_d2) cudaFree(ctx.d_block_best_d2);
    if (ctx.d_max_d2) cudaFree(ctx.d_max_d2);
    if (ctx.d_in_cluster) cudaFree(ctx.d_in_cluster);
    if (ctx.d_clustered) cudaFree(ctx.d_clustered);
    if (ctx.d_y) cudaFree(ctx.d_y);
    if (ctx.d_x) cudaFree(ctx.d_x);
    if (ctx.stream) cudaStreamDestroy(ctx.stream);
    ctx = CudaCtx{};
}

static int pickBestCandidateFromBlocks(CudaCtx& ctx) {
    cudaCheck(cudaMemcpyAsync(ctx.h_block_best_d2, ctx.d_block_best_d2, ctx.blocks * sizeof(double), cudaMemcpyDeviceToHost, ctx.stream), "cpy block d2");
    cudaCheck(cudaMemcpyAsync(ctx.h_block_best_i, ctx.d_block_best_i, ctx.blocks * sizeof(int), cudaMemcpyDeviceToHost, ctx.stream), "cpy block i");
    cudaCheck(cudaStreamSynchronize(ctx.stream), "sync block copy");

    double best_d2 = DBL_MAX;
    int best_i = -1;
    for (int b = 0; b < ctx.blocks; ++b) {
        const double d2 = ctx.h_block_best_d2[b];
        const int i = ctx.h_block_best_i[b];
        if (i < 0) continue;
        if (d2 < best_d2 || (d2 == best_d2 && i < best_i)) {
            best_d2 = d2;
            best_i = i;
        }
    }
    return best_i;
}

static int generateCandidateClusterGPU(const int seed_point,
                                      CudaCtx& ctx,
                                      const double threshold2,
                                      std::vector<int>* cluster_members) {
    const int N = ctx.N;
    cudaCheck(cudaMemsetAsync(ctx.d_in_cluster, 0, N * sizeof(uint8_t), ctx.stream), "memset in_cluster");
    cudaCheck(cudaMemsetAsync(ctx.d_max_d2, 0, N * sizeof(double), ctx.stream), "memset max_d2");

    const uint8_t one = 1;
    cudaCheck(cudaMemcpyAsync(ctx.d_in_cluster + seed_point, &one, sizeof(uint8_t), cudaMemcpyHostToDevice, ctx.stream), "set seed in_cluster");

    int count = 1;
    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    int new_member = seed_point;
    while (count < N) {
        updateMaxAndBlockBestKernel<<<ctx.blocks, CudaCtx::TPB, 0, ctx.stream>>>(
            ctx.d_x, ctx.d_y, N, new_member, ctx.d_clustered, ctx.d_in_cluster, ctx.d_max_d2,
            threshold2, ctx.d_block_best_d2, ctx.d_block_best_i);
        cudaCheck(cudaGetLastError(), "updateMaxAndBlockBestKernel launch");

        const int closest = pickBestCandidateFromBlocks(ctx);
        if (closest < 0) break;

        ++count;
        if (cluster_members) cluster_members->push_back(closest);

        cudaCheck(cudaMemcpyAsync(ctx.d_in_cluster + closest, &one, sizeof(uint8_t), cudaMemcpyHostToDevice, ctx.stream), "set member in_cluster");
        new_member = closest;
    }

    cudaCheck(cudaStreamSynchronize(ctx.stream), "sync candidate");
    return count;
}

static void markClusteredOnDevice(CudaCtx& ctx, const std::vector<int>& members) {
    const int count = static_cast<int>(members.size());
    if (count <= 0) return;
    cudaCheck(cudaMemcpyAsync(ctx.d_members, members.data(), count * sizeof(int), cudaMemcpyHostToDevice, ctx.stream), "cpy members");
    const int blocks = (count + CudaCtx::TPB - 1) / CudaCtx::TPB;
    markClusteredKernel<<<blocks, CudaCtx::TPB, 0, ctx.stream>>>(ctx.d_clustered, ctx.d_members, count);
    cudaCheck(cudaGetLastError(), "markClusteredKernel launch");
    cudaCheck(cudaStreamSynchronize(ctx.stream), "sync mark clustered");
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
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

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    std::vector<Cluster> clusters;
    clusters.reserve(N / 2);

    const double threshold2 = threshold * threshold;
    int rank = 0, world = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &world), "Comm_size");

    CudaCtx ctx;
    initCudaCtx(ctx, points, rank);

    while (!unclustered_indices.empty()) {
        int local_best_card = -1;
        int local_best_pos = INT_MAX;
        int local_best_seed = -1;

        const int U = static_cast<int>(unclustered_indices.size());
        for (int pos = 0; pos < U; ++pos) {
            if ((pos % world) != rank) continue;
            const int seed = unclustered_indices[pos];
            if (clustered[seed]) continue;

            const int card = generateCandidateClusterGPU(seed, ctx, threshold2, nullptr);
            if (card > local_best_card || (card == local_best_card && pos < local_best_pos)) {
                local_best_card = card;
                local_best_pos = pos;
                local_best_seed = seed;
            }
        }

        int sendbuf[3] = {local_best_card, local_best_pos, local_best_seed};
        std::vector<int> recvbuf;
        if (rank == 0) recvbuf.resize(3 * world);
        mpiCheck(MPI_Gather(sendbuf, 3, MPI_INT, rank == 0 ? recvbuf.data() : nullptr, 3, MPI_INT, 0, MPI_COMM_WORLD), "Gather best");

        int best_seed = -1;
        if (rank == 0) {
            int best_card = -1;
            int best_pos = INT_MAX;
            for (int r = 0; r < world; ++r) {
                const int card = recvbuf[3 * r + 0];
                const int pos = recvbuf[3 * r + 1];
                const int seed = recvbuf[3 * r + 2];
                if (seed < 0) continue;
                if (card > best_card || (card == best_card && pos < best_pos)) {
                    best_card = card;
                    best_pos = pos;
                    best_seed = seed;
                }
            }
        }

        mpiCheck(MPI_Bcast(&best_seed, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast best_seed");
        if (best_seed < 0) break;

        std::vector<int> best_members;
        if (rank == 0) {
            generateCandidateClusterGPU(best_seed, ctx, threshold2, &best_members);
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_members;
            clusters.push_back(std::move(cluster));
        }

        int member_count = (rank == 0) ? static_cast<int>(best_members.size()) : 0;
        mpiCheck(MPI_Bcast(&member_count, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast member_count");
        if (rank != 0) best_members.resize(member_count);
        mpiCheck(MPI_Bcast(best_members.data(), member_count, MPI_INT, 0, MPI_COMM_WORLD), "Bcast members");

        for (int i = 0; i < member_count; ++i) {
            clustered[best_members[i]] = 1;
        }
        markClusteredOnDevice(ctx, best_members);

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    freeCudaCtx(ctx);

    if (rank != 0) clusters.clear();
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
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");

    int rank = 0, world = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &world), "Comm_size");

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int printResults = 0;
    int run = 1;
    int exit_code = 0;

    if (rank == 0) {
        // Parse command line arguments (rank 0), then broadcast to all ranks
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                num_points = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                threshold = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                run = 0;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                run = 0;
                exit_code = 1;
                break;
            }
        }

        if (run && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            run = 0;
            exit_code = 1;
        }
    }

    mpiCheck(MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast run");
    mpiCheck(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast exit_code");
    mpiCheck(MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast num_points");
    mpiCheck(MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), "Bcast threshold");
    mpiCheck(MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast validate");
    mpiCheck(MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast printResults");

    if (!run) {
        mpiCheck(MPI_Finalize(), "MPI_Finalize");
        return exit_code;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
    }

    // Generate synthetic data (all ranks; deterministic)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier before clustering");
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier after clustering");
    auto cluster_end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            cluster_end - cluster_start);

        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;

        #pragma omp parallel for reduction(+:total_clustered) reduction(max:max_cluster_size)
        for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
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
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            mpiCheck(MPI_Finalize(), "MPI_Finalize");
            return valid ? 0 : 1;
        }
    }

    mpiCheck(MPI_Finalize(), "MPI_Finalize");
    return 0;
}
