// QT Clustering Benchmark - CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <climits>
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
        
        // Below 31 points the original group-size range rounds down to zero.
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

// Calculate Euclidean distance between two points
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
#ifdef __CUDA_ARCH__
    return sqrt(fma(dx, dx, __dmul_rn(dy, dy)));
#else
    return std::sqrt(dx * dx + dy * dy);
#endif
}

namespace {

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t n) {
        if (n) checkCuda(cudaMalloc(&data, n * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr unsigned fullWarp = 0xffffffffu;
constexpr int threads = 128;

// Points are generated in [0,20]^2. Power-of-two cell widths make
// coordinate-to-cell conversion exact at grid boundaries.
struct Grid {
    int side = 1;
    double width = 32.0;

    __host__ __device__ int cell(double coord) const {
        return int(fmin(double(side - 1), fmax(0.0, coord / width)));
    }
};

// Two passes build compact rows of threshold neighbors. Include the seed
// itself so that each row also has room for its complete membership sequence.
template<bool fill>
__global__ void neighborhoods(const Point* points, int first, int end, double threshold,
                              Grid grid, const int* cells, const int* indices,
                              int* degrees, const size_t* offsets, int* neighbors) {
    size_t row = first + (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= size_t(end)) return;
    int seed = int(row);
    int lane = threadIdx.x % 32;
    Point origin = points[seed];
    int count = 0;
    int x = grid.cell(origin.x), y = grid.cell(origin.y);
    // All threshold neighbors lie in the seed cell or an adjacent cell.
    for (int rowY = max(y - 1, 0); rowY <= min(y + 1, grid.side - 1); ++rowY) {
        int rowStart = rowY * grid.side;
        int left = cells[rowStart + max(x - 1, 0)];
        int right = cells[rowStart + min(x + 2, grid.side)];
        for (size_t base = left; base < size_t(right); base += 32) {
            size_t pos = base + lane;
            int id = pos < size_t(right) ? indices[pos] : -1;
            bool near = id >= 0 && (id == seed || distance(points[id], origin) < threshold);
            unsigned mask = __ballot_sync(fullWarp, near);
            if (fill && near) {
                int rank = __popc(mask & ((1u << lane) - 1));
                neighbors[offsets[seed] - offsets[first] + count + rank] = id;
            }
            count += __popc(mask);
        }
    }
    if (!fill && lane == 0) degrees[seed] = count;
}

// A warp maintains the maximum distance to the current cluster for each
// eligible neighbor. Small rows stay in registers. Long rows use global
// scratch and follow the same reduction and tie rules.
template<int slots>
__device__ int grow(int seed, const Point* points, double threshold,
                    const int* neighbors, int degree, const int* used,
                    int* members, double* scratch) {
    int lane = threadIdx.x % 32;
    int ids[slots ? slots : 1];
    double maxima[slots ? slots : 1];
    if (slots) {
        #pragma unroll
        for (int j = 0; j < slots; ++j) {
            int i = 32 * j + lane;
            int id = i < degree ? neighbors[i] : -1;
            ids[j] = id;
            maxima[j] = id >= 0 && !used[id] ? 0.0 : CUDART_INF;
        }
    }
    int latest = seed;
    int size = 0;
    do {
        if (lane == 0) members[size] = latest;
        ++size;
        Point center = points[latest];
        double best = CUDART_INF;
        int bestId = INT_MAX;
        for (int j = 0; j < (slots ? slots : (degree - 1) / 32 + 1); ++j) {
            int i = 32 * j + lane;
            if (!slots && i >= degree) break;
            int id = slots ? ids[j] : neighbors[i];
            double d = slots ? maxima[j] :
                       (size == 1 ? (used[id] ? CUDART_INF : 0.0) : scratch[i]);
            if (d < threshold) {
                d = id == latest ? CUDART_INF : fmax(d, distance(points[id], center));
                if (slots) maxima[j] = d;
                else scratch[i] = d;
                if (d < threshold && (d < best || (d == best && id < bestId))) {
                    best = d;
                    bestId = id;
                }
            } else if (!slots && size == 1) {
                scratch[i] = d;
            }
        }
        // Reduce (diameter, point index) to preserve the serial scan order.
        #pragma unroll
        for (int delta = 16; delta; delta /= 2) {
            double other = __shfl_down_sync(fullWarp, best, delta);
            int otherId = __shfl_down_sync(fullWarp, bestId, delta);
            if (other < best || (other == best && otherId < bestId)) {
                best = other;
                bestId = otherId;
            }
        }
        latest = __shfl_sync(fullWarp, bestId, 0);
    } while (latest != INT_MAX);
    return size;
}

__global__ void evaluate(const Point* points, int first, int end, double threshold,
                         const size_t* offsets, const int* neighbors,
                         const int* used, int* sizes, bool cached, int* members,
                         double* scratch, unsigned long long* winner) {
    size_t row = first + (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= size_t(end) || used[row]) return;
    int seed = int(row);
    int lane = threadIdx.x % 32;
    size_t start = offsets[seed] - offsets[first];
    int degree = int(offsets[seed + 1] - offsets[seed]);
    int size = cached ? sizes[seed] : 0;
    bool dirty = size == 0;
    // Removing an unselected point cannot alter any greedy choice.
    for (int base = 0; base < size; base += 32) {
        int i = base + lane;
        if (__any_sync(fullWarp, i < size && used[members[start + i]])) {
            dirty = true;
            break;
        }
    }
    if (dirty) {
        if (degree <= 32)
            size = grow<1>(seed, points, threshold, neighbors + start, degree, used,
                           members + start, nullptr);
        else if (degree <= 128)
            size = grow<4>(seed, points, threshold, neighbors + start, degree, used,
                           members + start, nullptr);
        else if (degree <= 256)
            size = grow<8>(seed, points, threshold, neighbors + start, degree, used,
                           members + start, nullptr);
        else
            size = grow<0>(seed, points, threshold, neighbors + start, degree, used,
                           members + start, scratch + start);
        if (lane == 0) sizes[seed] = size;
    }
    if (lane == 0) {
        unsigned long long key = (static_cast<unsigned long long>(size) << 32) |
                                 (UINT_MAX - unsigned(seed));
        atomicMax(winner, key);
    }
}

__global__ void markUsed(int* used, const int* members, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) used[members[i]] = 1;
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    int blocks = (n - 1) / (threads / 32) + 1;
    DeviceBuffer<Point> coords(n);
    DeviceBuffer<int> sizes(n), used(n);
    DeviceBuffer<size_t> offsets(size_t(n) + 1);
    DeviceBuffer<unsigned long long> winner(1);
    checkCuda(cudaMemcpy(coords.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    Grid grid;
    while (grid.width * 0.5 >= threshold && 4LL * grid.side * grid.side <= n) {
        grid.side *= 2;
        grid.width *= 0.5;
    }
    std::vector<int> cells(grid.side * grid.side + 1), indices(n);
    auto bin = [&](Point p) { return grid.cell(p.y) * grid.side + grid.cell(p.x); };
    for (Point p : points) ++cells[bin(p) + 1];
    for (size_t i = 1; i < cells.size(); ++i) cells[i] += cells[i - 1];
    auto cursor = cells;
    for (int i = 0; i < n; ++i) indices[cursor[bin(points[i])]++] = i;
    DeviceBuffer<int> deviceCells(cells.size()), deviceIndices(n);
    checkCuda(cudaMemcpy(deviceCells.data, cells.data(), cells.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceIndices.data, indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    neighborhoods<false><<<blocks, threads>>>(coords.data, 0, n, threshold, grid, deviceCells.data,
                                              deviceIndices.data, sizes.data, nullptr, nullptr);
    checkCuda(cudaGetLastError());
    std::vector<int> degrees(n);
    checkCuda(cudaMemcpy(degrees.data(), sizes.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> starts(size_t(n) + 1);
    for (int i = 0; i < n; ++i) starts[i + 1] = starts[i] + degrees[i];
    checkCuda(cudaMemcpy(offsets.data, starts.data(), starts.size() * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    // If every pair fits, all seeds attain n and seed zero wins.
    const int seeds = *std::min_element(degrees.begin(), degrees.end()) == n ? 1 : n;
    const bool needsScratch = *std::max_element(degrees.begin(), degrees.end()) > 256;
    size_t freeBytes, totalBytes;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t bytesPerEntry = 2 * sizeof(int) + (needsScratch ? sizeof(double) : 0);
    const size_t capacity = (freeBytes - freeBytes / 8) / bytesPerEntry;

    // Cache the full set of rows if it fits. Otherwise reuse the buffers
    // across batches while comparing their winners with the same global key.
    std::vector<int> boundaries{0};
    size_t largestBatch = 0;
    for (int first = 0; first < seeds;) {
        auto endIt = std::upper_bound(starts.begin() + first + 1, starts.begin() + seeds + 1,
                                      starts[first] + capacity);
        int end = std::max(first + 1, int(endIt - starts.begin() - 1));
        largestBatch = std::max(largestBatch, starts[end] - starts[first]);
        boundaries.push_back(end);
        first = end;
    }
    const bool cached = boundaries.size() == 2;
    DeviceBuffer<int> neighbors(largestBatch), members(largestBatch), saved(cached ? 0 : n);
    DeviceBuffer<double> scratch(needsScratch ? largestBatch : 0);
    checkCuda(cudaMemset(used.data, 0, n * sizeof(int)));
    checkCuda(cudaMemset(sizes.data, 0, n * sizeof(int)));

    std::vector<Cluster> clusters;
    std::vector<bool> assigned(n, false);
    int remaining = n;
    while (remaining) {
        unsigned long long best = 0;
        const int* selected = nullptr;
        for (size_t batch = 1; batch < boundaries.size(); ++batch) {
            int first = boundaries[batch - 1], end = boundaries[batch];
            int batchBlocks = (end - first - 1) / (threads / 32) + 1;
            if (!cached || remaining == n) {
                neighborhoods<true><<<batchBlocks, threads>>>(coords.data, first, end, threshold, grid,
                                                              deviceCells.data, deviceIndices.data,
                                                              nullptr, offsets.data, neighbors.data);
                checkCuda(cudaGetLastError());
            }
            checkCuda(cudaMemset(winner.data, 0, sizeof(unsigned long long)));
            evaluate<<<batchBlocks, threads>>>(coords.data, first, end, threshold, offsets.data,
                                               neighbors.data, used.data, sizes.data, cached,
                                               members.data, scratch.data, winner.data);
            checkCuda(cudaGetLastError());
            unsigned long long key;
            checkCuda(cudaMemcpy(&key, winner.data, sizeof(key), cudaMemcpyDeviceToHost));
            if (key > best) {
                best = key;
                int seed = int(UINT_MAX - static_cast<unsigned>(key));
                selected = members.data + (starts[seed] - starts[first]);
                if (!cached) {
                    checkCuda(cudaMemcpy(saved.data, selected, (key >> 32) * sizeof(int),
                                         cudaMemcpyDeviceToDevice));
                    selected = saved.data;
                }
            }
        }
        int size = int(best >> 32);
        int seed = int(UINT_MAX - static_cast<unsigned>(best));
        if (size == 1) {
            // Every remaining seed is isolated. Emit them in seed order.
            for (int i = 0; i < n; ++i)
                if (!assigned[i]) clusters.push_back({{i}, i});
            break;
        }
        clusters.push_back({std::vector<int>(size), seed});
        auto& group = clusters.back().members;
        checkCuda(cudaMemcpy(group.data(), selected, size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i : group) assigned[i] = true;
        markUsed<<<(size - 1) / threads + 1, threads>>>(used.data, selected, size);
        checkCuda(cudaGetLastError());
        remaining -= size;
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
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
