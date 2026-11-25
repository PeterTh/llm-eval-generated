// QT Clustering Benchmark - MPI, OpenMP and CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

__host__ __device__ static bool inDomain(Point p) {
    return p.x >= 0.0 && p.x <= MAX_WIDTH && p.y >= 0.0 && p.y <= MAX_HEIGHT;
}

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
        const int sampled_count = static_cast<int>(frand() * (N / 30.0));
        // For these sizes the original sampling cannot reliably produce a point.
        int group_cnt = N <= 30 ? 1 : sampled_count;
        
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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call) checkCuda((call), #call)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count) CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const T* source, size_t count) {
        CUDA_CHECK(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* data = nullptr;
};

struct Neighborhood {
    int start[3];
    int end[3]; // Cumulative length of the rows in this window.

    __device__ int sortedIndex(int slot) const {
        if (slot < end[0]) return start[0] + slot;
        if (slot < end[1]) return start[1] + (slot - end[0]);
        return start[2] + (slot - end[1]);
    }
};

struct SpatialGrid {
    std::vector<int> cells;
    std::vector<int> sorted_ids;
    std::vector<Point> sorted_points;
    std::vector<Neighborhood> neighborhoods;
    int max_neighbors = 0;
};

static SpatialGrid makeGrid(const std::vector<Point>& points, double threshold) {
    const int count = static_cast<int>(points.size());
    // Make each cell slightly wider than the threshold so rounding at a cell
    // boundary cannot hide a neighbor. A qualifying point is at most one cell away.
    int side = 1;
    while (side < 128 && threshold > 0.0 &&
           MAX_WIDTH / (side + 1) > threshold * 1.000000001) {
        ++side;
    }

    SpatialGrid grid;
    grid.cells.resize(count);
    grid.sorted_ids.resize(count);
    grid.sorted_points.resize(count);
    grid.neighborhoods.resize(side * side);

    const double scale = side / MAX_WIDTH;
    const int threads = std::min(omp_get_max_threads(), std::max(2, 1 + (count - 1) / 512));
    int outside_domain = 0;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:outside_domain)
    for (int i = 0; i < count; ++i) {
        const Point p = points[i];
        if (!inDomain(p)) {
            outside_domain = 1;
            continue;
        }
        const int x = std::min(side - 1, static_cast<int>(p.x * scale));
        const int y = std::min(side - 1, static_cast<int>(p.y * scale));
        grid.cells[i] = y * side + x;
    }
    // Keep the original NaN-distance behavior if the generator rounds a radicand below zero.
    if (outside_domain) {
        side = 1;
        grid.neighborhoods.resize(1);
        std::fill(grid.cells.begin(), grid.cells.end(), 0);
    }

    std::vector<int> offsets(side * side + 1, 0);
    for (int cell : grid.cells) ++offsets[cell + 1];
    for (size_t cell = 1; cell < offsets.size(); ++cell) offsets[cell] += offsets[cell - 1];

    auto next = offsets;
    for (int i = 0; i < count; ++i) grid.sorted_ids[next[grid.cells[i]]++] = i;

    int max_neighbors = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < count; ++i) grid.sorted_points[i] = points[grid.sorted_ids[i]];

        #pragma omp for schedule(static) reduction(max:max_neighbors)
        for (int cell = 0; cell < side * side; ++cell) {
            const int x = cell % side;
            const int y = cell / side;
            Neighborhood rows{};
            int total = 0;
            for (int row = 0; row < 3; ++row) {
                const int yy = y + row - 1;
                if (yy >= 0 && yy < side) {
                    const int first = offsets[yy * side + std::max(0, x - 1)];
                    const int last = offsets[yy * side + std::min(side, x + 2)];
                    rows.start[row] = first;
                    total += last - first;
                }
                rows.end[row] = total;
            }
            grid.neighborhoods[cell] = rows;
            if (offsets[cell] != offsets[cell + 1]) max_neighbors = std::max(max_neighbors, total);
        }
    }
    grid.max_neighbors = max_neighbors;
    return grid;
}

constexpr double infinity = std::numeric_limits<double>::infinity();

struct Bounds {
    double min_x, max_x, min_y, max_y;

    __host__ __device__ void include(Point p) {
        if (!inDomain(p)) {
            *this = {-infinity, infinity, -infinity, infinity};
            return;
        }
        min_x = fmin(min_x, p.x);
        max_x = fmax(max_x, p.x);
        min_y = fmin(min_y, p.y);
        max_y = fmax(max_y, p.y);
    }

    __device__ bool overlaps(const Bounds& other) const {
        return min_x <= other.max_x && max_x >= other.min_x &&
               min_y <= other.max_y && max_y >= other.min_y;
    }
};

struct SeedResult {
    Bounds bounds;
    int size;
};

struct DeviceData {
    const Point* points;
    const Point* sorted_points;
    const int* sorted_ids;
    const int* cells;
    const Neighborhood* neighborhoods;
    const unsigned char* clustered;
    double threshold;
    bool fused_distance;
};

constexpr int warp_size = 32;
constexpr int warps_per_block = 4;
constexpr int threads_per_block = warp_size * warps_per_block;
constexpr unsigned int warp_mask = 0xffffffffu;
using SeedKey = unsigned long long;

// MPI and CUDA use the same total order, largest cluster then lowest seed index.
__host__ __device__ static SeedKey seedKey(int size, int seed) {
    return (static_cast<SeedKey>(INT_MAX - size) << 32) | static_cast<unsigned int>(seed);
}

// One warp grows a seed. A candidate only needs its maximum distance so far and
// the distance to the last added member. Once rejected it cannot become eligible.
template <bool SaveMembers>
__device__ static SeedResult growSeed(DeviceData data, int seed, double* diameters,
                                      int* members) {
    const int lane = threadIdx.x % warp_size;
    const Neighborhood rows = data.neighborhoods[data.cells[seed]];
    const int count = rows.end[2];
    for (int slot = lane; slot < count; slot += warp_size) {
        const int id = data.sorted_ids[rows.sortedIndex(slot)];
        diameters[slot] = data.clustered[id] ? infinity : 0.0;
    }

    int newest = seed;
    int size = 0;
    Bounds bounds{infinity, -infinity, infinity, -infinity};
    const double threshold_squared = data.threshold * data.threshold;
    do {
        const Point added = data.points[newest];
        if (lane == 0) {
            if constexpr (SaveMembers) members[size] = newest;
            else bounds.include(added);
        }
        ++size;

        double best_distance = infinity;
        int best_id = INT_MAX;
        for (int slot = lane; slot < count; slot += warp_size) {
            const double previous = diameters[slot];
            if (!(previous < data.threshold)) continue;

            const int sorted = rows.sortedIndex(slot);
            const int id = data.sorted_ids[sorted];
            if (id == newest) {
                diameters[slot] = infinity;
                continue;
            }

            const Point candidate = data.sorted_points[sorted];
            const double dx = candidate.x - added.x;
            const double dy = candidate.y - added.y;
            const double squared = data.fused_distance ? __fma_rn(dx, dx, dy * dy)
                                                       : dx * dx + dy * dy;
            if (squared > threshold_squared) {
                diameters[slot] = infinity;
                continue;
            }
            // Strict squared comparisons can skip the square root away from a
            // boundary. Keep rounded distances for eligibility and tie-breaking.
            const double updated = squared < previous * previous ? previous
                : fmax(previous, __dsqrt_rn(squared));
            diameters[slot] = updated;
            if (updated < data.threshold &&
                (updated < best_distance || (updated == best_distance && id < best_id))) {
                best_distance = updated;
                best_id = id;
            }
        }

        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            const double other_distance = __shfl_down_sync(warp_mask, best_distance, offset);
            const int other_id = __shfl_down_sync(warp_mask, best_id, offset);
            if (other_distance < best_distance ||
                (other_distance == best_distance && other_id < best_id)) {
                best_distance = other_distance;
                best_id = other_id;
            }
        }
        newest = __shfl_sync(warp_mask, best_id, 0);
    } while (newest != INT_MAX);

    return {bounds, size};
}

__global__ static void evaluateSeeds(DeviceData data, int rank, int ranks, int local_count,
                                     Bounds removed, SeedResult* cache, int capacity,
                                     double* workspace, SeedKey* best) {
    extern __shared__ double shared_workspace[];
    __shared__ SeedKey warp_winners[warps_per_block];
    const int warp = threadIdx.x / warp_size;
    const int worker = blockIdx.x * warps_per_block + warp;
    const int stride = gridDim.x * warps_per_block;
    const size_t slice = static_cast<size_t>(workspace ? worker : warp) * capacity;
    double* diameters = (workspace ? workspace : shared_workspace) + slice;
    SeedKey warp_best = ULLONG_MAX;

    for (long long local = worker; local < local_count; local += stride) {
        const int seed = static_cast<int>(local * ranks + rank);
        if (data.clustered[seed]) continue;
        SeedResult result = cache[local];
        // Deleting an unchosen point cannot change a greedy trajectory.
        // Nonoverlapping bounds guarantee that none of its members was deleted.
        if (result.size == 0 || result.bounds.overlaps(removed)) {
            result = growSeed<false>(data, seed, diameters, nullptr);
            if (threadIdx.x % warp_size == 0) cache[local] = result;
        }
        warp_best = min(warp_best, seedKey(result.size, seed));
    }
    if (threadIdx.x % warp_size == 0) warp_winners[warp] = warp_best;
    __syncthreads();
    if (threadIdx.x == 0) {
        SeedKey block_best = warp_winners[0];
        for (int i = 1; i < warps_per_block; ++i) block_best = min(block_best, warp_winners[i]);
        if (block_best != ULLONG_MAX) atomicMin(best, block_best);
    }
}

__global__ static void recoverMembers(DeviceData data, int seed, double* workspace,
                                      int* members) {
    extern __shared__ double shared_workspace[];
    growSeed<true>(data, seed, workspace ? workspace : shared_workspace, members);
}

__global__ static void markClustered(unsigned char* clustered, const int* members, int size) {
    const long long start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (long long i = start; i < size; i += stride) clustered[members[i]] = 1;
}

static int selectDevice() {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int local_rank = 0;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) CUDA_CHECK(cudaErrorNoDevice);
    const int device = local_rank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                         int rank, int ranks, int device) {
    const int count = static_cast<int>(points.size());
    const int local_count = count / ranks + (rank < count % ranks);
    const SpatialGrid grid = makeGrid(points, threshold);
    DeviceBuffer<Point> device_points(count);
    DeviceBuffer<Point> sorted_points(count);
    DeviceBuffer<int> sorted_ids(count);
    DeviceBuffer<int> cells(count);
    DeviceBuffer<Neighborhood> neighborhoods(grid.neighborhoods.size());
    DeviceBuffer<unsigned char> clustered(count);
    DeviceBuffer<SeedResult> cache(local_count);
    DeviceBuffer<SeedKey> best(1);
    DeviceBuffer<int> device_members(count);

    device_points.upload(points.data(), count);
    sorted_points.upload(grid.sorted_points.data(), count);
    sorted_ids.upload(grid.sorted_ids.data(), count);
    cells.upload(grid.cells.data(), count);
    neighborhoods.upload(grid.neighborhoods.data(), grid.neighborhoods.size());
    CUDA_CHECK(cudaMemset(clustered.data, 0, count));
    if (local_count) CUDA_CHECK(cudaMemset(cache.data, 0, local_count * sizeof(SeedResult)));

    int sms = 0;
    int shared_limit = 0;
    int block_shared_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                      device));
    CUDA_CHECK(cudaDeviceGetAttribute(&block_shared_limit, cudaDevAttrMaxSharedMemoryPerBlock,
                                      device));
    const size_t warp_bytes = static_cast<size_t>(grid.max_neighbors) * sizeof(double);
    const size_t block_bytes = warp_bytes * warps_per_block;
    // Prefer shared memory when at least four blocks can reside on an SM.
    const bool use_shared = block_bytes + warps_per_block * sizeof(SeedKey) <=
        static_cast<size_t>(std::min(block_shared_limit, shared_limit / 4));
    const size_t scratch_budget = 128 * 1024 * 1024;
    const int block_limit = use_shared ? sms * 8 :
        static_cast<int>(std::min<size_t>(sms * 8, std::max<size_t>(1, scratch_budget / block_bytes)));
    const int blocks = std::min(block_limit, static_cast<int>(
        (static_cast<long long>(local_count) + warps_per_block - 1) / warps_per_block));
    DeviceBuffer<double> workspace(use_shared ? 0 :
        static_cast<size_t>(blocks) * warps_per_block * grid.max_neighbors);

    // Match contraction in the original native host build. The GPU uses rounded
    // square roots as well to preserve ties between distances that round alike.
#if defined(__FMA__) || defined(__aarch64__)
    constexpr bool fused_distance = true;
#else
    constexpr bool fused_distance = false;
#endif
    const DeviceData data{device_points.data, sorted_points.data, sorted_ids.data,
                          cells.data, neighborhoods.data, clustered.data, threshold, fused_distance};
    const size_t shared_bytes = use_shared ? block_bytes : 0;

    std::vector<int> members(count);
    std::vector<Cluster> clusters;
    std::vector<unsigned char> assigned(rank == 0 ? count : 0, 0);
    Bounds removed{infinity, -infinity, infinity, -infinity};
    // Above the domain's diagonal every seed reaches all points. The first
    // seed wins, and its member order is still generated by the regular kernel.
    const bool all_compatible = count > 1 &&
        threshold > distance({0.0, 0.0}, {MAX_WIDTH, MAX_HEIGHT});
    int remaining = count;
    while (remaining > 0) {
        SeedKey local_best = ULLONG_MAX;
        if (all_compatible) {
            if (rank == 0) local_best = seedKey(count, 0);
        } else if (blocks) {
            CUDA_CHECK(cudaMemsetAsync(best.data, 0xff, sizeof(SeedKey)));
            evaluateSeeds<<<blocks, threads_per_block, shared_bytes>>>(
                data, rank, ranks, local_count, removed, cache.data, grid.max_neighbors,
                workspace.data, best.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&local_best, best.data, sizeof(SeedKey), cudaMemcpyDeviceToHost));
        }

        SeedKey global_best = 0;
        MPI_Allreduce(&local_best, &global_best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        const int size = INT_MAX - static_cast<int>(global_best >> 32);
        const int seed = static_cast<int>(global_best & 0xffffffffULL);
        if (size == 1) {
            // No remaining seed can grow, and deleting points cannot create an edge.
            if (rank == 0) {
                for (int i = 0; i < count; ++i) {
                    if (!assigned[i]) clusters.push_back({{i}, i});
                }
            }
            break;
        }

        const int owner = seed % ranks;
        if (rank == owner) {
            recoverMembers<<<1, warp_size, use_shared ? warp_bytes : 0>>>(
                data, seed, workspace.data, device_members.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(members.data(), device_members.data, size * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank == 0) {
            clusters.push_back({std::vector<int>(members.begin(), members.begin() + size), seed});
            for (int i = 0; i < size; ++i) assigned[members[i]] = 1;
        }

        remaining -= size;
        if (remaining == 0 || !blocks) continue;
        removed = {infinity, -infinity, infinity, -infinity};
        for (int i = 0; i < size; ++i) removed.include(points[members[i]]);

        if (rank != owner) device_members.upload(members.data(), size);
        const int mark_blocks = std::min(sms * 8, 1 + (size - 1) / threads_per_block);
        markClustered<<<mark_blocks, threads_per_block>>>(clustered.data, device_members.data, size);
        CUDA_CHECK(cudaGetLastError());
    }
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute diameters concurrently and report them in clustering order.
    std::vector<double> diameters(clusters.size(), 0.0);
    const int validation_threads = std::min({omp_get_max_threads(),
        1 + static_cast<int>(points.size() / 256),
        static_cast<int>(std::max<size_t>(1, clusters.size()))});
    #pragma omp parallel for schedule(dynamic) num_threads(validation_threads)
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

static int runBenchmark(int argc, char** argv, int rank, int ranks) {
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
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }
    
    const int device = selectDevice();
    std::vector<Point> points(num_points);
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        generateSyntheticData(points, num_points);
    }

    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    const auto cluster_start = std::chrono::steady_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks, device);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - cluster_start).count();
    double time_sec = 0.0;
    MPI_Reduce(&elapsed, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;

    const auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(time_sec));
    
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

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = EXIT_FAILURE;
    try {
        if (provided >= MPI_THREAD_FUNNELED) {
            status = runBenchmark(argc, argv, rank, ranks);
        } else {
            fprintf(stderr, "Rank %d: MPI does not support MPI_THREAD_FUNNELED\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
