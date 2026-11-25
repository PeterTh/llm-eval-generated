// QT Clustering Benchmark - MPI, OpenMP and CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <climits>
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
        // Small inputs would otherwise keep drawing empty groups.
        if (N <= 30) group_cnt = 1;
        
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

// Match the original native CPU build's contraction on FMA-capable hosts.
__host__ __device__ inline double squaredDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
#if defined(__FMA__) || defined(__aarch64__)
    return fma(dx, dx, dy * dy);
#else
    return dx * dx + dy * dy;
#endif
}

inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(squaredDistance(p1, p2));
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data) cudaFree(data);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data = nullptr;
};

struct IndexedPoint {
    Point position;
    int id;
};

struct SpatialIndex {
    std::vector<IndexedPoint> points;
    std::vector<int> slots;
    std::vector<int> cell_offsets;
    int side;
};

int hostThreads(int work_items) {
    return std::min(omp_get_max_threads(), 2 + work_items / 4096);
}

int cellCoordinate(double coordinate, int side, double span) {
    return side == 1 ? 0 : std::min(side - 1, static_cast<int>(coordinate * side / span));
}

SpatialIndex buildSpatialIndex(const std::vector<Point>& points, double threshold) {
    const int count = static_cast<int>(points.size());
    const int threads = hostThreads(count);
    bool nonfinite = false;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(||:nonfinite)
    for (int id = 0; id < count; ++id) {
        nonfinite |= !std::isfinite(points[id].x) || !std::isfinite(points[id].y);
    }
    // Cells wider than the threshold restrict a seed search to its adjacent cells.
    const int target_side = std::clamp(static_cast<int>(std::sqrt(count / 12.0)), 1, 256);
    const double width = std::fmax(1.000001 * threshold, MAX_WIDTH / target_side);
    // The sequential diameter calculation ignores NaN distances even across cells.
    const int side = nonfinite ? 1 : std::clamp(static_cast<int>(MAX_WIDTH / width), 1, target_side);
    const int cells = side * side;
    SpatialIndex index{std::vector<IndexedPoint>(count), std::vector<int>(count),
                       std::vector<int>(cells + 1), side};
    std::vector<int> point_cells(count);
    std::vector<int> offsets(static_cast<size_t>(threads) * cells);

    #pragma omp parallel num_threads(threads)
    {
        const int thread = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const int begin = static_cast<long long>(count) * thread / team;
        const int end = static_cast<long long>(count) * (thread + 1) / team;
        int* const local_offsets = offsets.data() + static_cast<size_t>(thread) * cells;

        for (int id = begin; id < end; ++id) {
            const int x = cellCoordinate(points[id].x, side, MAX_WIDTH);
            const int y = cellCoordinate(points[id].y, side, MAX_HEIGHT);
            const int cell = y * side + x;
            point_cells[id] = cell;
            ++local_offsets[cell];
        }

        #pragma omp barrier
        #pragma omp single
        {
            int offset = 0;
            for (int cell = 0; cell < cells; ++cell) {
                index.cell_offsets[cell] = offset;
                for (int worker = 0; worker < team; ++worker) {
                    int& local = offsets[static_cast<size_t>(worker) * cells + cell];
                    const int entries = local;
                    local = offset;
                    offset += entries;
                }
            }
            index.cell_offsets[cells] = offset;
        }

        for (int id = begin; id < end; ++id) {
            const int slot = local_offsets[point_cells[id]]++;
            index.points[slot] = {points[id], id};
            index.slots[id] = slot;
        }
    }
    return index;
}

struct SeedNeighborhood {
    int slot;
    int begin[3];
    int end[3];
    int extent;
};

std::vector<SeedNeighborhood> buildNeighborhoods(const SpatialIndex& index,
                                                int rank, int ranks, int seed_count,
                                                int& max_extent) {
    std::vector<SeedNeighborhood> neighborhoods(seed_count);
    const int threads = hostThreads(seed_count);
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(max:max_extent)
    for (int local = 0; local < seed_count; ++local) {
        SeedNeighborhood region{};
        region.slot = index.slots[rank + local * ranks];
        const Point point = index.points[region.slot].position;
        const int side = index.side;
        const int x = cellCoordinate(point.x, side, MAX_WIDTH);
        const int y = cellCoordinate(point.y, side, MAX_HEIGHT);
        int row = 0;
        for (int iy = std::max(0, y - 1); iy <= std::min(side - 1, y + 1); ++iy, ++row) {
            region.begin[row] = index.cell_offsets[iy * side + std::max(0, x - 1)];
            region.end[row] = index.cell_offsets[iy * side + std::min(side, x + 2)];
            region.extent += region.end[row] - region.begin[row];
        }
        max_extent = std::max(max_extent, region.extent);
        neighborhoods[local] = region;
    }
    return neighborhoods;
}

// Find the inclusive squared cutoff for the original strict, rounded sqrt test.
double squaredCutoff(double threshold) {
    if (std::isnan(threshold)) return -1.0;
    if (threshold > MAX_WIDTH + MAX_HEIGHT) return DBL_MAX;
    double cutoff = threshold * threshold;
    while (std::sqrt(cutoff) >= threshold) cutoff = std::nextafter(cutoff, 0.0);
    while (std::sqrt(std::nextafter(cutoff, DBL_MAX)) < threshold) {
        cutoff = std::nextafter(cutoff, DBL_MAX);
    }
    return cutoff;
}

constexpr int BLOCK_SIZE = 128;
constexpr int SHARED_NEIGHBORS = 512;

using WinnerKey = unsigned long long;

struct Candidate {
    double squared;
    int id;
    int position;
};

struct ClosestCandidate {
    __device__ Candidate operator()(const Candidate& a, const Candidate& b) const {
        if (a.squared == b.squared) return a.id < b.id ? a : b;
        const bool b_is_closer = b.squared < a.squared;
        const Candidate& nearer = b_is_closer ? b : a;
        const Candidate& farther = b_is_closer ? a : b;
        // Rounded square-root ties only span a few consecutive squared values.
        // Test for those ties when they can change the winner.
        if (farther.id < nearer.id &&
            __double_as_longlong(farther.squared) - __double_as_longlong(nearer.squared) <= 8 &&
            sqrt(farther.squared) == sqrt(nearer.squared)) {
            return farther;
        }
        return nearer;
    }
};

struct SearchData {
    const IndexedPoint* points;
    const SeedNeighborhood* neighborhoods;
    const unsigned char* removed;
    int* sizes;
    int* cached_members;
    int cache_stride;
    double cutoff;
};

__global__ void evaluateSeeds(SearchData data, int first, int last, int shared_limit,
                              int global_stride, double* global_distances, int* global_slots,
                              WinnerKey* winner, int* full_members) {
    using BlockReduce = cub::BlockReduce<Candidate, BLOCK_SIZE>;
    __shared__ BlockReduce::TempStorage reduction;
    __shared__ int neighbor_count;
    __shared__ int next_position;
    __shared__ Point newest;
    extern __shared__ double shared_distances[];

    WinnerKey block_best = 0;
    const int lane = threadIdx.x % 32;
    for (long long local = static_cast<long long>(first) + blockIdx.x;
         local < last; local += gridDim.x) {
        const SeedNeighborhood region = data.neighborhoods[local];
        const IndexedPoint seed = data.points[region.slot];
        if (data.removed[seed.id]) continue;

        int* const cached = data.cached_members + local * data.cache_stride;
        const int previous_size = full_members ? 0 : data.sizes[local];
        bool dirty = previous_size == 0 || previous_size > data.cache_stride;
        if (!dirty) {
            for (int i = threadIdx.x + 1; i < previous_size; i += BLOCK_SIZE) {
                dirty |= data.removed[cached[i]] != 0;
            }
        }

        int size = previous_size;
        // Removing points outside the cached sequence leaves every greedy choice intact.
        if (__syncthreads_or(dirty)) {
            const bool use_shared = region.extent <= shared_limit;
            const size_t scratch_offset = static_cast<size_t>(blockIdx.x) * global_stride;
            double* distances = use_shared ? shared_distances : global_distances + scratch_offset;
            int* slots = use_shared ? reinterpret_cast<int*>(shared_distances + shared_limit) :
                global_slots + scratch_offset;
            int* members = full_members ? full_members : cached;
            const int capacity = full_members ? region.extent : data.cache_stride;

            if (threadIdx.x == 0) {
                neighbor_count = 0;
                members[0] = seed.id;
            }
            __syncthreads();

            for (int row = 0; row < 3; ++row) {
                for (long long base = region.begin[row]; base < region.end[row]; base += BLOCK_SIZE) {
                    const long long slot = base + threadIdx.x;
                    bool eligible = false;
                    double squared = 0.0;
                    if (slot < region.end[row] && slot != region.slot) {
                        const IndexedPoint point = data.points[slot];
                        if (!data.removed[point.id]) {
                            squared = fmax(0.0, squaredDistance(seed.position, point.position));
                            eligible = squared <= data.cutoff;
                        }
                    }

                    // Compact qualifying neighbors with one shared atomic per nonempty warp.
                    constexpr unsigned warp = 0xffffffffu;
                    const unsigned mask = __ballot_sync(warp, eligible);
                    if (mask != 0) {
                        int offset = 0;
                        if (lane == 0) offset = atomicAdd(&neighbor_count, __popc(mask));
                        offset = __shfl_sync(warp, offset, 0);
                        if (eligible) {
                            const int i = offset + __popc(mask & ((1u << lane) - 1));
                            distances[i] = squared;
                            slots[i] = static_cast<int>(slot);
                        }
                    }
                }
            }
            __syncthreads();

            size = 1;
            const int count = neighbor_count;
            while (size <= count) {
                Candidate best{DBL_MAX, INT_MAX, -1};
                for (long long i = threadIdx.x; i < count; i += BLOCK_SIZE) {
                    double squared = distances[i];
                    if (squared > data.cutoff) continue;
                    const IndexedPoint point = data.points[slots[i]];
                    if (size > 1) {
                        squared = fmax(squared, squaredDistance(newest, point.position));
                        distances[i] = squared;
                    }
                    if (squared <= data.cutoff) {
                        best = ClosestCandidate{}(best, {squared, point.id, static_cast<int>(i)});
                    }
                }

                best = BlockReduce(reduction).Reduce(best, ClosestCandidate{});
                if (threadIdx.x == 0) {
                    next_position = best.position;
                    if (best.position >= 0) {
                        distances[best.position] = CUDART_INF;
                        newest = data.points[slots[best.position]].position;
                        if (size < capacity) members[size] = best.id;
                    }
                }
                __syncthreads();
                if (next_position < 0) break;
                ++size;
            }
            if (threadIdx.x == 0 && !full_members) data.sizes[local] = size;
        }

        if (threadIdx.x == 0) {
            const WinnerKey key = (static_cast<WinnerKey>(size) << 32) |
                                  (UINT_MAX - static_cast<unsigned>(seed.id));
            block_best = block_best < key ? key : block_best;
        }
        // Make sure every lane has finished using shared state before the next seed.
        __syncthreads();
    }
    if (threadIdx.x == 0 && !full_members) atomicMax(winner, block_best);
}

__global__ void markRemoved(unsigned char* removed, const int* members, int size, WinnerKey* winner) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned>(size)) removed[members[i]] = 1;
    if (i == 0) *winner = 0;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks, int multiprocessors) {
    const int count = static_cast<int>(points.size());
    // When the entire input is eligible, seed zero is certain to win.
    const int total_seeds = threshold > std::hypot(MAX_WIDTH, MAX_HEIGHT) ? 1 : count;
    const int local_seeds = rank < total_seeds ? 1 + (total_seeds - 1 - rank) / ranks : 0;
    const SpatialIndex index = local_seeds ? buildSpatialIndex(points, threshold) : SpatialIndex{};
    int max_extent = 0;
    const auto neighborhoods = local_seeds ?
        buildNeighborhoods(index, rank, ranks, local_seeds, max_extent) :
        std::vector<SeedNeighborhood>{};

    // Cap cache memory per rank while retaining full candidates for typical workloads.
    constexpr int cache_budget = 256 * 1024 * 1024;
    const int cache_slots = cache_budget / sizeof(int);
    const int cache_stride = std::min(max_extent, std::max(1, cache_slots / std::max(1, local_seeds)));
    const int shared_limit = std::min(max_extent, SHARED_NEIGHBORS);
    const int global_stride = max_extent > shared_limit ? max_extent : 0;
    const size_t shared_bytes = shared_limit * (sizeof(double) + sizeof(int));
    int blocks = std::min(local_seeds, multiprocessors * 8);
    if (global_stride != 0) {
        // Bound workspace independently of the number of seeds on this rank.
        constexpr size_t scratch_budget = 128 * 1024 * 1024;
        const size_t bytes_per_block = static_cast<size_t>(global_stride) *
                                       (sizeof(double) + sizeof(int));
        blocks = static_cast<int>(std::min<size_t>(blocks, std::max<size_t>(1, scratch_budget / bytes_per_block)));
    }

    DeviceBuffer<IndexedPoint> gpu_points(local_seeds ? count : 0);
    DeviceBuffer<SeedNeighborhood> gpu_neighborhoods(local_seeds);
    DeviceBuffer<int> gpu_sizes(local_seeds);
    DeviceBuffer<int> cache(static_cast<size_t>(local_seeds) * cache_stride);
    DeviceBuffer<double> distances(static_cast<size_t>(blocks) * global_stride);
    DeviceBuffer<int> slots(static_cast<size_t>(blocks) * global_stride);
    DeviceBuffer<unsigned char> removed(local_seeds ? count : 0);
    DeviceBuffer<int> gpu_members(local_seeds ? count : 0);
    DeviceBuffer<WinnerKey> gpu_winner(local_seeds ? 1 : 0);
    if (local_seeds != 0) {
        CUDA_CHECK(cudaMemcpy(gpu_points.data, index.points.data(), count * sizeof(IndexedPoint),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_neighborhoods.data, neighborhoods.data(),
                              local_seeds * sizeof(SeedNeighborhood), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(gpu_sizes.data, 0, local_seeds * sizeof(int)));
        CUDA_CHECK(cudaMemset(removed.data, 0, count));
        CUDA_CHECK(cudaMemset(gpu_winner.data, 0, sizeof(WinnerKey)));
    }

    const SearchData search{gpu_points.data, gpu_neighborhoods.data, removed.data, gpu_sizes.data,
                            cache.data, cache_stride, squaredCutoff(threshold)};
    std::vector<Cluster> clusters;
    std::vector<unsigned char> assigned(rank == 0 ? count : 0);
    std::vector<int> members;
    int remaining = count;
    while (remaining != 0) {
        WinnerKey winner = 0;
        if (local_seeds != 0) {
            evaluateSeeds<<<blocks, BLOCK_SIZE, shared_bytes>>>(
                search, 0, local_seeds, shared_limit, global_stride, distances.data, slots.data,
                gpu_winner.data, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&winner, gpu_winner.data, sizeof(winner), cudaMemcpyDeviceToHost));
        }
        MPI_Allreduce(MPI_IN_PLACE, &winner, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int size = static_cast<int>(winner >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned>(winner));

        // Removing singleton clusters cannot change the remaining singleton candidates.
        if (size == 1) {
            if (rank == 0) {
                for (int id = 0; id < count; ++id) {
                    if (!assigned[id]) clusters.push_back(Cluster{{id}, id});
                }
            }
            break;
        }

        members.resize(size);
        const int owner = seed % ranks;
        const int* result = gpu_members.data;
        if (rank == owner) {
            const int local = seed / ranks;
            if (size <= cache_stride) {
                result = cache.data + static_cast<size_t>(local) * cache_stride;
            } else {
                evaluateSeeds<<<1, BLOCK_SIZE, shared_bytes>>>(
                    search, local, local + 1, shared_limit, global_stride, distances.data, slots.data,
                    nullptr, gpu_members.data);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaMemcpy(members.data(), result, size * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        remaining -= size;
        if (remaining != 0 && local_seeds != 0) {
            if (rank != owner) {
                CUDA_CHECK(cudaMemcpy(gpu_members.data, members.data(), size * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }
            markRemoved<<<1 + (size - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                removed.data, result, size, gpu_winner.data);
            CUDA_CHECK(cudaGetLastError());
        }
        if (rank == 0) {
            for (int id : members) assigned[id] = 1;
            clusters.push_back(Cluster{members, seed});
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
    
    std::vector<double> diameters(clusters.size());
    const int threads = hostThreads(static_cast<int>(points.size()));
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
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
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int local_rank;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) CUDA_CHECK(cudaErrorNoDevice);
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    int multiprocessors;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype mpi_point;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_point);
    MPI_Type_commit(&mpi_point);
    MPI_Bcast(points.data(), num_points, mpi_point, 0, MPI_COMM_WORLD);
    MPI_Type_free(&mpi_point);
    
    // Report elapsed clustering time on the slowest rank.
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks, multiprocessors);
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;

    printf("Clustering time: %lld ms\n", static_cast<long long>(time_sec * 1000.0));
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
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int status = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
