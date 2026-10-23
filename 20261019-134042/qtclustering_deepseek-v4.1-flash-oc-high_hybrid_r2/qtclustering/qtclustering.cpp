// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * MPI      - distributes the candidate seeds over processes.  Every
//                process performs the same deterministic global reduction so
//                that all ranks advance the clustering identically.
//   * OpenMP   - parallelizes the host-side work (candidate selection,
//                cluster bookkeeping, statistics and validation).
//   * CUDA     - evaluates the candidate clusters.  One CUDA block builds one
//                complete candidate cluster, scanning all candidate points in
//                parallel, so the O(N^3) distance work runs on the GPU.

#include <algorithm>
#include <chrono>
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

// Calculate Euclidean distance between two points (host path)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Exactly rounded squared distance used by the device.  The host compiler
// contracts dx*dx + dy*dy into an FMA (the reference build uses -march=native),
// so use the same operation order here to keep every comparison bit-identical.
__device__ __forceinline__ double deviceDistanceSquared(const double2 a,
                                                         const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

// One block builds one complete candidate cluster.  Every thread scans a
// strided subset of the candidate points; the block then reduces the best
// (distance, index) pair.  Ties are resolved in favour of the lower point
// index, matching the sequential reference exactly.
template <int BLOCK>
__global__ void generateCandidateClustersKernel(
        const double2* __restrict__ points,
        const unsigned char* __restrict__ clustered,
        const int* __restrict__ seeds,
        const int point_count,
        const double threshold,
        int* __restrict__ members,
        unsigned char* __restrict__ in_cluster,
        double* __restrict__ max_d2,
        int* __restrict__ counts) {
    const int tid = threadIdx.x;
    const int seed_slot = blockIdx.x;
    const int seed = seeds[seed_slot];

    int* my_members = members + static_cast<size_t>(seed_slot) * point_count;
    unsigned char* my_in_cluster =
        in_cluster + static_cast<size_t>(seed_slot) * point_count;
    double* my_max_d2 =
        max_d2 + static_cast<size_t>(seed_slot) * point_count;

    const double2 seed_point = points[seed];

    // Initialise the running maximum distance of every candidate to the seed.
    for (int i = tid; i < point_count; i += BLOCK) {
        my_in_cluster[i] = 0;
        my_max_d2[i] = clustered[i]
                           ? 0.0
                           : deviceDistanceSquared(points[i], seed_point);
    }
    __syncthreads();

    if (tid == 0) {
        my_in_cluster[seed] = 1;
        my_members[0] = seed;
        counts[seed_slot] = 1;
    }
    __syncthreads();

    __shared__ double warp_dist[BLOCK / 32];
    __shared__ int warp_idx[BLOCK / 32];
    __shared__ int found_shared;

    int k = 1;
    while (k < point_count) {
        double best_dist = 1.7976931348623157e308;
        int best_idx = 2147483647;

        // The running maxima already contain the distance to every member, so
        // selecting the next point is a single O(N) scan.
        for (int candidate = tid; candidate < point_count; candidate += BLOCK) {
            if (clustered[candidate] || my_in_cluster[candidate]) continue;

            // sqrt is monotonic, so max(sqrt(d2_i)) == sqrt(max(d2_i)).
            const double dist = sqrt(my_max_d2[candidate]);
            if (dist < threshold &&
                (dist < best_dist || (dist == best_dist && candidate < best_idx))) {
                best_dist = dist;
                best_idx = candidate;
            }
        }

        // Warp reduction of the (distance, index) pair.
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double od = __shfl_down_sync(0xffffffffu, best_dist, offset);
            const int oi = __shfl_down_sync(0xffffffffu, best_idx, offset);
            if (od < best_dist || (od == best_dist && oi < best_idx)) {
                best_dist = od;
                best_idx = oi;
            }
        }

        const int lane = tid & 31;
        const int warp = tid >> 5;
        if (lane == 0) {
            warp_dist[warp] = best_dist;
            warp_idx[warp] = best_idx;
        }
        __syncthreads();

        if (tid == 0) {
            double bd = warp_dist[0];
            int bi = warp_idx[0];
            for (int w = 1; w < BLOCK / 32; ++w) {
                if (warp_dist[w] < bd ||
                    (warp_dist[w] == bd && warp_idx[w] < bi)) {
                    bd = warp_dist[w];
                    bi = warp_idx[w];
                }
            }
            if (bi != 2147483647) {
                my_members[k] = bi;
                my_in_cluster[bi] = 1;
                counts[seed_slot] = k + 1;
                found_shared = 1;
            } else {
                counts[seed_slot] = k;
                found_shared = 0;
            }
        }
        __syncthreads();

        if (!found_shared) break;

        // Incrementally extend the running maxima with the newly added member.
        const int new_member = my_members[k];
        const double2 pm = points[new_member];
        for (int candidate = tid; candidate < point_count;
             candidate += BLOCK) {
            if (clustered[candidate] || my_in_cluster[candidate]) continue;
            const double d2 =
                deviceDistanceSquared(points[candidate], pm);
            if (d2 > my_max_d2[candidate]) my_max_d2[candidate] = d2;
        }
        __syncthreads();

        ++k;
    }
}

struct GpuContext {
    int rank = 0;
    int size = 1;
    int device = 0;
    int max_chunk = 1;
    cudaStream_t stream = nullptr;
    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_counts = nullptr;
    int* d_members = nullptr;
    unsigned char* d_in_cluster = nullptr;
    double* d_max_d2 = nullptr;
};

static inline uint64_t bestKey(const int cardinality, const int seed) {
    // Larger cardinality wins; for equal cardinality the lower seed wins.
    return (static_cast<uint64_t>(static_cast<uint32_t>(cardinality)) << 32) |
           static_cast<uint32_t>(0xFFFFFFFFu - static_cast<uint32_t>(seed));
}

static inline void decodeKey(const uint64_t key, int& cardinality, int& seed) {
    cardinality = static_cast<int>(key >> 32);
    seed = static_cast<int>(0xFFFFFFFFu -
                            static_cast<uint32_t>(key & 0xFFFFFFFFu));
}

static void cudaCheck(const cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what,
                cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void initGpu(GpuContext& gpu, const std::vector<Point>& points,
                    const int point_count, const int rank, const int size) {
    gpu.rank = rank;
    gpu.size = size;

    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rank = 0;
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);

    gpu.device = local_rank % device_count;
    cudaCheck(cudaSetDevice(gpu.device), "cudaSetDevice");

    // Bound temporary GPU storage to roughly 182 MiB per rank (13 bytes per
    // cell: member index + in-cluster flag + running squared distance).
    const size_t max_cells = 14u * 1024u * 1024u;
    size_t chunk_limit = max_cells / static_cast<size_t>(point_count);
    if (chunk_limit < 1) chunk_limit = 1;
    if (chunk_limit > 1024) chunk_limit = 1024;
    gpu.max_chunk = static_cast<int>(chunk_limit);

    cudaCheck(cudaStreamCreate(&gpu.stream), "cudaStreamCreate");
    cudaCheck(cudaMalloc(&gpu.d_points,
                         static_cast<size_t>(point_count) * sizeof(double2)),
              "cudaMalloc points");
    cudaCheck(cudaMalloc(&gpu.d_clustered,
                         static_cast<size_t>(point_count) * sizeof(unsigned char)),
              "cudaMalloc clustered");
    cudaCheck(cudaMalloc(&gpu.d_seeds,
                         static_cast<size_t>(gpu.max_chunk) * sizeof(int)),
              "cudaMalloc seeds");
    cudaCheck(cudaMalloc(&gpu.d_counts,
                         static_cast<size_t>(gpu.max_chunk) * sizeof(int)),
              "cudaMalloc counts");
    cudaCheck(cudaMalloc(&gpu.d_members,
                         static_cast<size_t>(gpu.max_chunk) *
                             static_cast<size_t>(point_count) * sizeof(int)),
              "cudaMalloc members");
    cudaCheck(cudaMalloc(&gpu.d_in_cluster,
                         static_cast<size_t>(gpu.max_chunk) *
                             static_cast<size_t>(point_count) *
                             sizeof(unsigned char)),
              "cudaMalloc in_cluster");
    cudaCheck(cudaMalloc(&gpu.d_max_d2,
                         static_cast<size_t>(gpu.max_chunk) *
                             static_cast<size_t>(point_count) * sizeof(double)),
              "cudaMalloc max_d2");

    cudaCheck(cudaMemcpyAsync(gpu.d_points, points.data(),
                              static_cast<size_t>(point_count) * sizeof(Point),
                              cudaMemcpyHostToDevice, gpu.stream),
              "copy points");
    cudaCheck(cudaStreamSynchronize(gpu.stream), "point upload");
}

static void freeGpu(GpuContext& gpu) {
    if (gpu.stream) cudaStreamDestroy(gpu.stream);
    cudaFree(gpu.d_points);
    cudaFree(gpu.d_clustered);
    cudaFree(gpu.d_seeds);
    cudaFree(gpu.d_counts);
    cudaFree(gpu.d_members);
    cudaFree(gpu.d_in_cluster);
    cudaFree(gpu.d_max_d2);
    gpu.stream = nullptr;
    gpu.d_points = nullptr;
    gpu.d_clustered = nullptr;
    gpu.d_seeds = nullptr;
    gpu.d_counts = nullptr;
    gpu.d_members = nullptr;
    gpu.d_in_cluster = nullptr;
    gpu.d_max_d2 = nullptr;
}

// Build all candidate clusters assigned to this rank and return this rank's
// best (cardinality, seed) key together with the corresponding member list.
static uint64_t computeLocalBest(GpuContext& gpu,
                                 const std::vector<unsigned char>& clustered,
                                 const std::vector<int>& local_seeds,
                                 const int point_count,
                                 const double threshold,
                                 std::vector<int>& best_members,
                                 int& best_cardinality,
                                 int& best_seed) {
    const int local_count = static_cast<int>(local_seeds.size());
    const int N = point_count;
    const int omp_threads = std::min(omp_get_max_threads(), 8);

    cudaCheck(cudaMemcpyAsync(gpu.d_clustered, clustered.data(),
                              static_cast<size_t>(N) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice, gpu.stream),
              "copy clustered");

    std::vector<int> host_counts(gpu.max_chunk);
    uint64_t rank_key = 0;
    best_cardinality = 0;
    best_seed = -1;

    for (int base = 0; base < local_count; base += gpu.max_chunk) {
        const int count = std::min(gpu.max_chunk, local_count - base);

        cudaCheck(cudaMemcpyAsync(gpu.d_seeds, local_seeds.data() + base,
                                  static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyHostToDevice, gpu.stream),
                  "copy seeds");

        generateCandidateClustersKernel<512>
            <<<count, 512, 0, gpu.stream>>>(
                gpu.d_points, gpu.d_clustered, gpu.d_seeds, N, threshold,
                gpu.d_members, gpu.d_in_cluster, gpu.d_max_d2, gpu.d_counts);
        cudaCheck(cudaGetLastError(), "candidate kernel");

        cudaCheck(cudaMemcpyAsync(host_counts.data(), gpu.d_counts,
                                  static_cast<size_t>(count) * sizeof(int),
                                  cudaMemcpyDeviceToHost, gpu.stream),
                  "copy counts");
        cudaCheck(cudaStreamSynchronize(gpu.stream), "candidate kernel wait");

        uint64_t chunk_key = 0;
#pragma omp parallel for reduction(max : chunk_key) num_threads(omp_threads) if(count >= 64)
        for (int s = 0; s < count; ++s) {
            const uint64_t key =
                bestKey(host_counts[s], local_seeds[base + s]);
            if (key > chunk_key) chunk_key = key;
        }

        if (chunk_key > rank_key) {
            int chunk_best = 0;
            for (int s = 0; s < count; ++s) {
                if (bestKey(host_counts[s], local_seeds[base + s]) == chunk_key) {
                    chunk_best = s;
                    break;
                }
            }

            int card = 0;
            int seed = -1;
            decodeKey(chunk_key, card, seed);
            rank_key = chunk_key;
            best_cardinality = card;
            best_seed = seed;

            cudaCheck(cudaMemcpy(best_members.data(),
                                 gpu.d_members +
                                     static_cast<size_t>(chunk_best) * N,
                                 static_cast<size_t>(N) * sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      "copy best members");
        }
    }

    return rank_key;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  GpuContext& gpu) {
    const int N = static_cast<int>(points.size());
    const int rank = gpu.rank;
    const int size = gpu.size;
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    std::vector<int> best_members(N);

    const int omp_threads = std::min(omp_get_max_threads(), 8);

    // Main clustering loop
    while (true) {
        unclustered_indices.clear();
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) unclustered_indices.push_back(i);
        }
        if (unclustered_indices.empty()) break;

        // Evenly interleaved seed distribution keeps every rank busy.
        std::vector<int> local_seeds;
        local_seeds.reserve(unclustered_indices.size() / size + 1);
        for (size_t i = static_cast<size_t>(rank); i < unclustered_indices.size();
             i += static_cast<size_t>(size)) {
            local_seeds.push_back(unclustered_indices[i]);
        }

        int local_cardinality = 0;
        int local_seed = -1;
        const uint64_t local_key =
            computeLocalBest(gpu, clustered, local_seeds, N, threshold,
                             best_members, local_cardinality, local_seed);

        uint64_t global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);

        int global_cardinality = 0;
        int global_seed = -1;
        decodeKey(global_key, global_cardinality, global_seed);

        // Locate the owner rank of the winning seed and broadcast its members.
        const auto it = std::lower_bound(unclustered_indices.begin(),
                                         unclustered_indices.end(), global_seed);
        const size_t position =
            static_cast<size_t>(it - unclustered_indices.begin());
        const int owner = static_cast<int>(position % static_cast<size_t>(size));

        MPI_Bcast(&global_cardinality, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner) best_members.resize(N);
        MPI_Bcast(best_members.data(), global_cardinality, MPI_INT, owner,
                  MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global_seed;
        cluster.members.assign(best_members.begin(),
                               best_members.begin() + global_cardinality);
        clusters.push_back(cluster);

        // Mark all members as clustered
#pragma omp parallel for num_threads(omp_threads) if(global_cardinality >= 256)
        for (int i = 0; i < global_cardinality; ++i) {
            clustered[cluster.members[i]] = 1;
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
    std::vector<double> diameters(clusters.size(), 0.0);
    const int omp_threads = std::min(omp_get_max_threads(), 16);

    // Check each cluster diameter (max distance between any two points)
#pragma omp parallel for schedule(dynamic) num_threads(omp_threads) if(clusters.size() >= 8)
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
        diameters[c] = max_diameter;
    }

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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    
    // Generate synthetic data on rank 0 and distribute it.  The generator is
    // kept serial so that the random sequence (and therefore the input data)
    // is identical to the reference implementation.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(),
              num_points * static_cast<int>(sizeof(Point)), MPI_BYTE, 0,
              MPI_COMM_WORLD);

    GpuContext gpu;
    initGpu(gpu, points, num_points, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, gpu);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    freeGpu(gpu);
    
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
        const int sz = static_cast<int>(clusters[i].members.size());
        total_clustered += sz;
        max_cluster_size = std::max(max_cluster_size, sz);
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
        const int omp_threads = std::min(omp_get_max_threads(), 8);
#pragma omp parallel for num_threads(omp_threads) if(clusters.size() >= 64)
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
