// QT Clustering Benchmark - MPI / OpenMP / CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Fail collectively rather than leaving other ranks blocked in a collective.
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation,
                cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// CUDA contraction is disabled in CMake: use the same double-precision
// subtraction, multiplication, addition and square root as the reference.
__device__ inline double deviceDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ inline void warpMinimum(double& value, int& index) {
    for (int offset = 16; offset; offset >>= 1) {
        const double other = __shfl_down_sync(0xffffffffu, value, offset);
        const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
        if (other < value || (other == value && other_index < index)) {
            value = other;
            index = other_index;
        }
    }
}

// One block builds one candidate cluster. Only seed neighbors can ever join;
// compact them once, then update their maximum distance to the newest member.
// All ties are resolved by point index, independent of compaction order.
constexpr int BLOCK_THREADS = 128;
__global__ void candidateClusters(const Point* points, const int* active,
                                  int active_count, const int* seeds,
                                  double threshold, int stride,
                                  int* neighbors, double* maxima,
                                  int* members, int* cardinalities) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const size_t base = static_cast<size_t>(blockIdx.x) * stride;
    int* ids = neighbors + base;
    double* distances = maxima + base;
    int* output = members + base;
    const int seed = seeds[blockIdx.x];
    __shared__ int neighbor_count, selected;
    __shared__ double warp_values[BLOCK_THREADS / 32];
    __shared__ int warp_indices[BLOCK_THREADS / 32];
    if (tid == 0) {
        neighbor_count = 0;
        output[0] = seed;
    }
    __syncthreads();

    // Warp-level reservations avoid one atomic operation per neighbor.
    for (int start = 0; start < active_count; start += BLOCK_THREADS) {
        const int pos = start + tid;
        const int point = pos < active_count ? active[pos] : seed;
        const double dist = point != seed ? deviceDistance(points[point], points[seed]) : DBL_MAX;
        const bool keep = pos < active_count && point != seed && dist < threshold;
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        int reservation = 0;
        if (lane == 0 && mask) reservation = atomicAdd(&neighbor_count, __popc(mask));
        reservation = __shfl_sync(0xffffffffu, reservation, 0);
        if (keep) {
            const int slot = reservation + __popc(mask & ((1u << lane) - 1u));
            ids[slot] = point;
            distances[slot] = dist;
        }
    }
    __syncthreads();
    const int candidates = neighbor_count;
    int count = 1;
    while (true) {
        double best_distance = DBL_MAX;
        int best_index = INT_MAX;
        for (int i = tid; i < candidates; i += BLOCK_THREADS) {
            const double d = distances[i];
            const int point = ids[i];
            if (d < best_distance || (d == best_distance && point < best_index)) {
                best_distance = d;
                best_index = point;
            }
        }
        warpMinimum(best_distance, best_index);
        if (lane == 0) {
            warp_values[warp] = best_distance;
            warp_indices[warp] = best_index;
        }
        __syncthreads();
        if (warp == 0) {
            best_distance = lane < BLOCK_THREADS / 32 ? warp_values[lane] : DBL_MAX;
            best_index = lane < BLOCK_THREADS / 32 ? warp_indices[lane] : INT_MAX;
            warpMinimum(best_distance, best_index);
            if (lane == 0) {
                selected = best_distance < threshold && best_distance < DBL_MAX ? best_index : -1;
                if (selected >= 0) output[count] = selected;
            }
        }
        __syncthreads();
        if (selected < 0) break;
        ++count;
        const Point newest = points[selected];
        for (int i = tid; i < candidates; i += BLOCK_THREADS) {
            if (distances[i] == DBL_MAX) continue;
            const int point = ids[i];
            if (point == selected) {
                distances[i] = DBL_MAX;
            } else {
                const double d = deviceDistance(points[point], newest);
                const double maximum = fmax(distances[i], d);
                distances[i] = maximum < threshold ? maximum : DBL_MAX;
            }
        }
        __syncthreads();
    }
    if (tid == 0) cardinalities[blockIdx.x] = count;
}

// Reuse neighbor storage after candidate construction to pack memberships.
// Only live members cross PCIe, rather than the padded batch-by-N matrix.
__global__ void packMembers(const int* members, const int* sizes,
                            const size_t* offsets, int stride, int* packed) {
    const size_t source = static_cast<size_t>(blockIdx.x) * stride;
    const size_t destination = offsets[blockIdx.x];
    for (int i = threadIdx.x; i < sizes[blockIdx.x]; i += blockDim.x)
        packed[destination + i] = members[source + i];
}

struct SeedChoice { int size, seed; };
static SeedChoice betterChoice(SeedChoice a, SeedChoice b) {
    return (b.size > a.size || (b.size == a.size && b.seed < a.seed)) ? b : a;
}
#pragma omp declare reduction(bestSeed : SeedChoice : omp_out = betterChoice(omp_out, omp_in)) initializer(omp_priv = SeedChoice{-1, INT_MAX})

// Device allocations persist across all clustering rounds. Bounded batches
// avoid a quadratic device-memory requirement; no CPU fallback is used.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    if (n == 0) return {};
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t bytes_per_seed = static_cast<size_t>(n) * (2 * sizeof(int) + sizeof(double));
    const size_t budget = std::min<size_t>(free_bytes / 4, 256ull * 1024 * 1024);
    const int local_capacity = (n - 1) / ranks + 1;
    const int batch_capacity = static_cast<int>(std::min<size_t>(
        local_capacity, std::min<size_t>(8 * properties.multiProcessorCount,
                                       budget / bytes_per_seed)));
    if (batch_capacity < 1) {
        fprintf(stderr, "Rank %d: insufficient CUDA memory for a candidate cluster\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    Point* d_points = nullptr;
    int *d_active = nullptr, *d_seeds = nullptr, *d_neighbors = nullptr;
    int *d_members = nullptr, *d_sizes = nullptr;
    size_t* d_offsets = nullptr;
    double* d_maxima = nullptr;
    const size_t slots = static_cast<size_t>(batch_capacity) * n;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_active, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seeds, batch_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, batch_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, (batch_capacity + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_maxima, slots * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> active(n), seeds(local_capacity), sizes(batch_capacity);
    std::vector<size_t> offsets(batch_capacity + 1);
    std::vector<int> packed;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) active[i] = i;

    // A candidate remains exactly the same if no member has been removed:
    // deleting candidates that were never selected cannot change any greedy
    // choice. Cache these clusters on the host, with a bounded memory budget.
    // Static cyclic seed ownership keeps the cache local across rounds.
    const int local_count = n > rank ? (n - 1 - rank) / ranks + 1 : 0;
    std::vector<std::vector<int>> cache(local_count);
    std::vector<unsigned char> dirty(local_count), admit(batch_capacity);
    const size_t cache_budget = 256ull * 1024 * 1024;
    size_t cache_bytes = 0;
    std::vector<Cluster> clusters;
    while (!active.empty()) {
        const int remaining = static_cast<int>(active.size());
        SeedChoice local_best{-1, INT_MAX};
        size_t released_bytes = 0;
        #pragma omp parallel for schedule(static) reduction(bestSeed:local_best) reduction(+:released_bytes)
        for (int i = 0; i < local_count; ++i) {
            const int seed = rank + i * ranks;
            auto& members = cache[i];
            bool affected = clustered[seed] != 0;
            if (!affected) {
                for (int member : members) {
                    if (clustered[member]) { affected = true; break; }
                }
            }
            if (affected && !members.empty()) {
                released_bytes += members.size() * sizeof(int);
                std::vector<int>().swap(members);
            }
            dirty[i] = !clustered[seed] && members.empty();
            if (!members.empty())
                local_best = betterChoice(local_best, SeedChoice{static_cast<int>(members.size()), seed});
        }
        cache_bytes -= released_bytes;
        int dirty_count = 0;
        for (int i = 0; i < local_count; ++i)
            if (dirty[i]) seeds[dirty_count++] = rank + i * ranks;
        std::vector<int> local_members;
        if (local_best.size > 0) local_members = cache[local_best.seed / ranks];
        if (dirty_count)
            CUDA_CHECK(cudaMemcpy(d_active, active.data(), remaining * sizeof(int), cudaMemcpyHostToDevice));
        for (int start = 0; start < dirty_count; start += batch_capacity) {
            const int batch = std::min(batch_capacity, dirty_count - start);
            CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data() + start, batch * sizeof(int), cudaMemcpyHostToDevice));
            candidateClusters<<<batch, BLOCK_THREADS>>>(d_points, d_active, remaining,
                d_seeds, threshold, n, d_neighbors, d_maxima, d_members, d_sizes);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes, batch * sizeof(int), cudaMemcpyDeviceToHost));
            offsets[0] = 0;
            for (int i = 0; i < batch; ++i) {
                offsets[i + 1] = offsets[i] + sizes[i];
                const size_t bytes = static_cast<size_t>(sizes[i]) * sizeof(int);
                admit[i] = bytes <= cache_budget - cache_bytes;
                if (admit[i]) cache_bytes += bytes;
            }
            packed.resize(offsets[batch]);
            CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), (batch + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
            packMembers<<<batch, BLOCK_THREADS>>>(d_members, d_sizes, d_offsets, n, d_neighbors);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(packed.data(), d_neighbors, offsets[batch] * sizeof(int), cudaMemcpyDeviceToHost));
            SeedChoice batch_best{-1, INT_MAX};
            #pragma omp parallel for reduction(bestSeed:batch_best) schedule(static)
            for (int i = 0; i < batch; ++i) {
                const int seed = seeds[start + i];
                batch_best = betterChoice(batch_best, SeedChoice{sizes[i], seed});
                if (admit[i]) cache[seed / ranks].assign(
                    packed.begin() + offsets[i], packed.begin() + offsets[i + 1]);
            }
            const SeedChoice next = betterChoice(local_best, batch_best);
            if (next.seed != local_best.seed) {
                local_best = next;
                const int slot = static_cast<int>(std::lower_bound(seeds.begin() + start,
                    seeds.begin() + start + batch, local_best.seed) - (seeds.begin() + start));
                local_members.assign(packed.begin() + offsets[slot], packed.begin() + offsets[slot + 1]);
            }
        }
        // MPI_MAXLOC maximizes cardinality, then chooses the smallest seed.
        int local_pair[2] = {local_best.size, local_best.seed};
        int global_pair[2];
        MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_size = global_pair[0], best_seed = global_pair[1];
        // If the maximum cardinality is one, all remaining clusters are
        // singletons in ascending seed order: no remaining pair satisfies the
        // threshold, so no further rounds are necessary.
        if (best_size == 1) {
            for (int seed : active) clusters.push_back(Cluster{{seed}, seed});
            active.clear();
            break;
        }
        const int owner = best_seed % ranks;
        Cluster cluster;
        cluster.seed_point = best_seed;
        if (rank == owner) cluster.members = std::move(local_members);
        else cluster.members.resize(best_size);
        MPI_Bcast(cluster.members.data(), best_size, MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static) if(best_size >= 4096)
        for (int i = 0; i < best_size; ++i) clustered[cluster.members[i]] = 1;
        active.erase(std::remove_if(active.begin(), active.end(),
            [&clustered](int i) { return clustered[i] != 0; }), active.end());
        clusters.push_back(std::move(cluster));
    }
    CUDA_CHECK(cudaFree(d_maxima));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_active));
    CUDA_CHECK(cudaFree(d_points));
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& members = clusters[c].members;
        double maximum = 0.0;
        for (size_t i = 0; i < members.size(); ++i) {
            for (size_t j = i + 1; j < members.size(); ++j) {
                maximum = std::max(maximum, distance(points[members[i]], points[members[j]]));
            }
        }
        diameters[c] = maximum;
    }
    // Check each cluster
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // Finalize on every ordinary exit path, including help/argument errors.
    struct MpiLifetime { ~MpiLifetime() { MPI_Finalize(); } } mpi_lifetime;
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                              num_points, threshold);
        return 1;
    }
    
    // Respect explicit OpenMP settings. Otherwise use a modest host team;
    // GPU work dominates, and every MPI rank needs only its own host workers.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(omp_get_max_threads(), std::min(8, std::max(2, num_points / 512))));

    // Respect CUDA_VISIBLE_DEVICES and map node-local MPI ranks to devices.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int local_rank = 0, devices = 0;
    MPI_Comm_rank(node, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (rank == 0) fprintf(stderr, "A CUDA accelerator is required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    CUDA_CHECK(cudaFree(nullptr)); // Initialize the context outside benchmark timing.
    MPI_Comm_free(&node);

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two packed doubles");
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    long local_ms = cluster_time.count();
    long global_ms;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank != 0) return 0;

    printf("Clustering time: %lld ms\n", static_cast<long long>(global_ms));
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
    const double time_sec = static_cast<double>(global_ms) / 1000.0;
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
