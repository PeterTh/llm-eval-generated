// QT Clustering Benchmark
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <exception>
#include <limits>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
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
        // Use a nonzero group size for small inputs.
        const double group_scale = N <= 30 ? 2.0 : N / 30.0;
        int group_cnt = static_cast<int>(frand() * group_scale);
        
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

// Use the same rounded distance on the host and device, including at a tie
// or a strict threshold boundary.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(fma(dx, dx, dy * dy));
}

namespace {

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) cudaCheck(cudaMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data_) cudaCheck(cudaFree(data_));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

    void upload(const T* source, size_t count) const {
        cudaCheck(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* destination, size_t count) const {
        cudaCheck(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* data_ = nullptr;
};

struct SpatialGrid {
    std::vector<int> cells;
    std::vector<int> indices;
    std::vector<int> offsets;
    std::vector<int> neighborhood_sizes;
    int side;
    int max_neighborhood;
    bool all_neighbors;
};

SpatialGrid makeGrid(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    // Use cells wider than the threshold so a seed can only reach adjacent cells.
    // Also cap the grid size to keep the counting sort linear and the bins useful.
    const int limit = std::max(1, static_cast<int>(std::sqrt(n / 8.0)));
    const double width = std::max(MAX_WIDTH, MAX_HEIGHT);
    const int side = threshold < width
        ? std::max(1, static_cast<int>(std::min(width / threshold, double(limit)))) : 1;
    // A threshold equal to the cell width can lie on a rounded bin boundary.
    const double margin = 16 * std::numeric_limits<double>::epsilon() * width;
    const int grid_side = side > 1 && width / side <= threshold + margin ? side - 1 : side;
    const int cell_count = grid_side * grid_side;
    SpatialGrid grid{std::vector<int>(n), std::vector<int>(n),
                     std::vector<int>(cell_count + 1), std::vector<int>(cell_count),
                     grid_side, 0, false};
    const int threads = std::min({omp_get_max_threads(), 1 + (n - 1) / 512, n / cell_count});
    if (grid_side == 1) {
        double min_x = MAX_WIDTH, min_y = MAX_HEIGHT, max_x = 0.0, max_y = 0.0;
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:min_x,min_y) reduction(max:max_x,max_y)
        for (int i = 0; i < n; ++i) {
            grid.indices[i] = i;
            min_x = std::min(min_x, points[i].x);
            min_y = std::min(min_y, points[i].y);
            max_x = std::max(max_x, points[i].x);
            max_y = std::max(max_y, points[i].y);
        }
        grid.all_neighbors = distance({min_x, min_y}, {max_x, max_y}) < threshold;
        grid.offsets[1] = grid.neighborhood_sizes[0] = grid.max_neighborhood = n;
        return grid;
    }

    // Each thread counts and fills its own slice of each bin.
    std::vector<int> counts(static_cast<size_t>(threads) * cell_count);
    const double cell_width = width / grid_side;
    #pragma omp parallel num_threads(threads)
    {
        const int thread = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const int first = static_cast<long long>(n) * thread / team;
        const int last = static_cast<long long>(n) * (thread + 1) / team;
        int* bins = counts.data() + static_cast<size_t>(thread) * cell_count;
        for (int i = first; i < last; ++i) {
            const int x = std::min(grid_side - 1, static_cast<int>(points[i].x / cell_width));
            const int y = std::min(grid_side - 1, static_cast<int>(points[i].y / cell_width));
            const int cell = y * grid_side + x;
            grid.cells[i] = cell;
            ++bins[cell];
        }
        #pragma omp barrier
        #pragma omp single
        {
            int offset = 0;
            for (int cell = 0; cell < cell_count; ++cell) {
                grid.offsets[cell] = offset;
                for (int t = 0; t < team; ++t) {
                    int& count = counts[static_cast<size_t>(t) * cell_count + cell];
                    const int next = offset + count;
                    count = offset;
                    offset = next;
                }
            }
            grid.offsets[cell_count] = offset;
        }
        for (int i = first; i < last; ++i) grid.indices[bins[grid.cells[i]]++] = i;
    }

    int max_neighborhood = 0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(max:max_neighborhood)
    for (int cell = 0; cell < cell_count; ++cell) {
        if (grid.offsets[cell] == grid.offsets[cell + 1]) continue;
        const int x = cell % grid_side;
        const int y = cell / grid_side;
        const int left = std::max(0, x - 1);
        const int right = std::min(grid_side - 1, x + 1);
        int count = 0;
        for (int row = std::max(0, y - 1); row <= std::min(grid_side - 1, y + 1); ++row) {
            count += grid.offsets[row * grid_side + right + 1]
                   - grid.offsets[row * grid_side + left];
        }
        grid.neighborhood_sizes[cell] = count;
        max_neighborhood = std::max(max_neighborhood, count);
    }
    grid.max_neighborhood = max_neighborhood;
    return grid;
}

constexpr int warp_size = 32;
constexpr int warps_per_block = 4;
constexpr int block_size = warp_size * warps_per_block;
constexpr int small_neighborhood = 128;
constexpr int max_cached_members = 1024;
constexpr unsigned full_warp = 0xffffffffu;

struct SearchData {
    const Point* points;
    const int* cells;
    const int* indices;
    const int* offsets;
    const int* neighborhood_sizes;
    const unsigned char* clustered;
    double threshold;
    int grid_side;
};

struct Workspace {
    int* indices;
    double* diameters;
};

struct CellBounds {
    int left, right, bottom, top;

    __device__ bool contains(int cell, int side) const {
        const int x = cell % side;
        const int y = cell / side;
        return left <= x && x <= right && bottom <= y && y <= top;
    }
};

__device__ void consider(int index, double diameter, int& best, double& best_diameter) {
    if (diameter < best_diameter || (diameter == best_diameter && index < best)) {
        best = index;
        best_diameter = diameter;
    }
}

__device__ int warpBest(int best, double best_diameter) {
    for (int shift = warp_size / 2; shift > 0; shift /= 2) {
        const int other = __shfl_down_sync(full_warp, best, shift);
        const double other_diameter = __shfl_down_sync(full_warp, best_diameter, shift);
        consider(other, other_diameter, best, best_diameter);
    }
    return __shfl_sync(full_warp, best, 0);
}

__device__ void compactPoint(Workspace work, int& size, int index, double diameter,
                             bool keep, int lane) {
    const unsigned mask = __ballot_sync(full_warp, keep);
    if (keep) {
        const int position = size + __popc(mask & ((1u << lane) - 1));
        work.indices[position] = index;
        work.diameters[position] = diameter;
    }
    size += __popc(mask);
}

__device__ int searchSeed(SearchData data, int seed, Workspace work,
                          int* members, int capacity, int lane) {
    const int cell = data.cells[seed];
    const int x = cell % data.grid_side;
    const int y = cell / data.grid_side;
    const int left = max(0, x - 1);
    const int right = min(data.grid_side - 1, x + 1);
    const Point seed_point = data.points[seed];
    int size = 0;
    int best = INT_MAX;
    double best_diameter = data.threshold;

    // Only points strictly inside the seed's threshold can enter the cluster.
    for (int row = max(0, y - 1); row <= min(data.grid_side - 1, y + 1); ++row) {
        const int begin = data.offsets[row * data.grid_side + left];
        const int end = data.offsets[row * data.grid_side + right + 1];
        for (int base = begin; base < end; base += warp_size) {
            int index = INT_MAX;
            double diameter = data.threshold;
            bool keep = false;
            if (lane < end - base) {
                index = data.indices[base + lane];
                if (index != seed && !data.clustered[index]) {
                    diameter = distance(seed_point, data.points[index]);
                    keep = diameter < data.threshold;
                }
            }
            if (keep) consider(index, diameter, best, best_diameter);
            compactPoint(work, size, index, diameter, keep, lane);
        }
    }

    if (lane == 0) members[0] = seed;
    int cardinality = 1;
    int selected = warpBest(best, best_diameter);
    while (selected != INT_MAX) {
        if (lane == 0 && cardinality < capacity) members[cardinality] = selected;
        ++cardinality;
        if (size == 1) break;

        const Point added = data.points[selected];
        best = INT_MAX;
        best_diameter = data.threshold;
        int kept = 0;
        __syncwarp();
        for (int base = 0; base < size; base += warp_size) {
            int index = INT_MAX;
            double diameter = data.threshold;
            bool keep = false;
            if (lane < size - base) {
                index = work.indices[base + lane];
                if (index != selected) {
                    diameter = fmax(work.diameters[base + lane], distance(added, data.points[index]));
                    keep = diameter < data.threshold;
                }
            }
            if (keep) consider(index, diameter, best, best_diameter);
            // Compaction writes into this chunk or earlier ones. Finish its
            // reads before any lane can overwrite an entry.
            __syncwarp();
            compactPoint(work, kept, index, diameter, keep, lane);
        }
        size = kept;
        selected = warpBest(best, best_diameter);
    }
    // The warp can immediately reuse its workspace for a different seed.
    __syncwarp();
    return cardinality;
}

using Winner = unsigned long long;

__device__ Winner winnerKey(int cardinality, int seed) {
    return (static_cast<Winner>(cardinality) << 32) | (0xffffffffu - seed);
}

__global__ void evaluateSeeds(SearchData data, int rank, int ranks, int local_count,
                              int* sizes, int* members, int capacity, Workspace large_work,
                              int stride, const int* removed, int removed_count,
                              CellBounds affected, Winner* winner) {
    __shared__ int small_indices[warps_per_block][small_neighborhood];
    __shared__ double small_diameters[warps_per_block][small_neighborhood];
    __shared__ Winner block_best[warps_per_block];
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const size_t slot = blockIdx.x * warps_per_block + warp;
    Winner best = 0;

    for (size_t local = slot; local < local_count; local += gridDim.x * warps_per_block) {
        const int seed = rank + local * ranks;
        if (data.clustered[seed]) continue;
        const int cell = data.cells[seed];
        int cardinality = sizes[local];
        int* cached = members + local * capacity;
        bool dirty = cardinality == 0;
        if (!dirty && affected.contains(cell, data.grid_side)) {
            if (cardinality <= capacity) {
                // Removing an unselected point cannot change any greedy choice.
                for (int i = lane; i < cardinality; i += warp_size) {
                    dirty |= data.clustered[cached[i]] != 0;
                }
            } else {
                // The full list is not cached. Invalidate if any removed point
                // could have been a member of this seed's cluster.
                const Point point = data.points[seed];
                for (size_t i = lane; i < removed_count; i += warp_size) {
                    dirty |= distance(point, data.points[removed[i]]) < data.threshold;
                }
            }
        }

        if (__any_sync(full_warp, dirty)) {
            Workspace work{small_indices[warp], small_diameters[warp]};
            if (data.neighborhood_sizes[cell] > small_neighborhood) {
                work = {large_work.indices + slot * stride, large_work.diameters + slot * stride};
            }
            cardinality = searchSeed(data, seed, work, cached, capacity, lane);
            if (lane == 0) sizes[local] = cardinality;
        }
        if (lane == 0) best = max(best, winnerKey(cardinality, seed));
    }

    if (lane == 0) block_best[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < warps_per_block; ++w) best = max(best, block_best[w]);
        if (best != 0) atomicMax(winner, best);
    }
}

__global__ void replayWinner(SearchData data, int seed, Workspace work,
                             int* members, int cardinality) {
    searchSeed(data, seed, work, members, cardinality, threadIdx.x);
}

__global__ void markClustered(unsigned char* clustered, const int* removed, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[removed[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks, int multiprocessors) {
    const int n = static_cast<int>(points.size());
    const SpatialGrid grid = makeGrid(points, threshold);
    // If the bounding box fits, every seed reaches every point. Only the
    // lowest seed needs a search, which still produces the greedy member order.
    const int seeds = grid.all_neighbors ? 1 : n;
    const int local_count = rank < seeds ? 1 + (seeds - 1 - rank) / ranks : 0;
    constexpr size_t cache_budget = 128 * 1024 * 1024;
    const int cache_limit = static_cast<int>(cache_budget / (sizeof(int) * std::max(1, local_count)));
    const int capacity = grid.all_neighbors ? n :
        std::min({max_cached_members, grid.max_neighborhood, std::max(1, cache_limit)});
    const int stride = grid.max_neighborhood > small_neighborhood ? grid.max_neighborhood : 0;
    // A bounded number of resident warps reuse scratch rows even when there are
    // many seeds or a dense neighborhood. Small rows use shared memory.
    constexpr size_t scratch_budget = 128 * 1024 * 1024;
    int blocks = std::min(local_count == 0 ? 0 : 1 + (local_count - 1) / warps_per_block,
                          multiprocessors * 8);
    if (stride > 0) {
        const size_t bytes_per_block = size_t(stride) * (sizeof(int) + sizeof(double)) * warps_per_block;
        blocks = std::min(blocks, static_cast<int>(std::max(size_t(1), scratch_budget / bytes_per_block)));
    }

    DeviceBuffer<Point> device_points(n);
    DeviceBuffer<int> cells(n), indices(n), offsets(grid.offsets.size());
    DeviceBuffer<int> neighborhood_sizes(grid.neighborhood_sizes.size());
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<int> sizes(local_count), members(size_t(local_count) * capacity), removed(n);
    const size_t large_entries = size_t(blocks) * warps_per_block * stride;
    DeviceBuffer<int> large_indices(large_entries);
    DeviceBuffer<double> large_diameters(large_entries);
    DeviceBuffer<Winner> device_winner(1);
    const SearchData data{device_points.get(), cells.get(), indices.get(), offsets.get(),
                          neighborhood_sizes.get(), clustered.get(), threshold, grid.side};
    const Workspace large_work{large_indices.get(), large_diameters.get()};
    device_points.upload(points.data(), points.size());
    cells.upload(grid.cells.data(), grid.cells.size());
    indices.upload(grid.indices.data(), grid.indices.size());
    offsets.upload(grid.offsets.data(), grid.offsets.size());
    neighborhood_sizes.upload(grid.neighborhood_sizes.data(), grid.neighborhood_sizes.size());
    cudaCheck(cudaMemset(clustered.get(), 0, n));
    if (local_count > 0) cudaCheck(cudaMemset(sizes.get(), 0, size_t(local_count) * sizeof(int)));

    std::vector<Cluster> clusters;
    std::vector<unsigned char> host_clustered(rank == 0 ? n : 0);
    std::vector<int> selected;
    int remaining = n;
    int removed_count = 0;
    CellBounds affected{};
    while (remaining > 0) {
        Winner local_winner = 0;
        if (blocks > 0) {
            cudaCheck(cudaMemset(device_winner.get(), 0, sizeof(Winner)));
            evaluateSeeds<<<blocks, block_size>>>(data, rank, ranks, local_count, sizes.get(), members.get(),
                                                  capacity, large_work, stride, removed.get(), removed_count,
                                                  affected, device_winner.get());
            cudaCheck(cudaGetLastError());
            device_winner.download(&local_winner, 1);
        }

        // Larger clusters win, with lower original seed indices winning ties.
        Winner winner = 0;
        MPI_Allreduce(&local_winner, &winner, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int cardinality = static_cast<int>(winner >> 32);
        const int seed = static_cast<int>(0xffffffffu - static_cast<unsigned>(winner));
        if (cardinality <= 1) {
            // No pair of remaining points can cluster, so the rest are singletons.
            if (rank == 0) {
                for (int i = 0; i < n; ++i) {
                    if (!host_clustered[i]) clusters.push_back({{i}, i});
                }
            }
            break;
        }

        const int owner = seed % ranks;
        selected.resize(cardinality);
        if (rank == owner) {
            if (cardinality <= capacity) {
                const size_t row = static_cast<size_t>(seed / ranks) * capacity;
                cudaCheck(cudaMemcpy(selected.data(), members.get() + row,
                                     size_t(cardinality) * sizeof(int), cudaMemcpyDeviceToHost));
            } else {
                // An oversized winner is reconstructed once, before any removals.
                replayWinner<<<1, warp_size>>>(data, seed, large_work, removed.get(), cardinality);
                cudaCheck(cudaGetLastError());
                removed.download(selected.data(), cardinality);
            }
        }
        MPI_Bcast(selected.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank == 0) {
            clusters.push_back({selected, seed});
            for (int index : selected) host_clustered[index] = 1;
        }
        remaining -= cardinality;
        if (remaining > 0 && blocks > 0) {
            // Only seeds in or adjacent to the removed points' grid cells
            // can have selected any of them.
            affected = {grid.side, -1, grid.side, -1};
            for (int index : selected) {
                const int x = grid.cells[index] % grid.side;
                const int y = grid.cells[index] / grid.side;
                affected.left = std::min(affected.left, x - 1);
                affected.right = std::max(affected.right, x + 1);
                affected.bottom = std::min(affected.bottom, y - 1);
                affected.top = std::max(affected.top, y + 1);
            }
            removed.upload(selected.data(), cardinality);
            markClustered<<<1 + (cardinality - 1) / 256, 256>>>(clustered.get(), removed.get(), cardinality);
            cudaCheck(cudaGetLastError());
            removed_count = cardinality;
        }
    }
    return clusters;
}

int selectDevice() {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) cudaCheck(cudaErrorNoDevice);
    const int device = local_rank % devices;
    cudaCheck(cudaSetDevice(device));
    cudaCheck(cudaFree(nullptr)); // Initialize the context before benchmark timing.
    int multiprocessors = 0;
    cudaCheck(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    return multiprocessors;
}

} // namespace

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    std::vector<double> diameters(clusters.size());
    const int threads = std::min(omp_get_max_threads(), 1 + (static_cast<int>(points.size()) - 1) / 512);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
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
    
    const int multiprocessors = selectDevice();
    std::vector<Point> points(num_points);
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        generateSyntheticData(points, num_points);
    }
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype mpi_point;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_point);
    MPI_Type_commit(&mpi_point);
    MPI_Bcast(points.data(), num_points, mpi_point, 0, MPI_COMM_WORLD);
    MPI_Type_free(&mpi_point);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks, multiprocessors);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const double local_time_sec = std::chrono::duration<double>(cluster_end - cluster_start).count();
    double time_sec = 0.0;
    MPI_Reduce(&local_time_sec, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
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
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int result;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
