// QT Clustering Benchmark

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>

#include <cuda_runtime.h>
#include <math_constants.h>
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
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        // For small inputs the original group-size formula cannot reliably advance.
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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

namespace {

constexpr int warpSizeQT = 32;
constexpr int warpsPerBlock = 4;
constexpr int sharedWindow = 256;
constexpr int cachedMembers = 64;
constexpr unsigned fullWarp = 0xffffffffu;

struct IndexedPoint {
    Point point;
    int index;
};

struct Window {
    int begin, end;
};

struct Candidate {
    double squared;
    int index;
};

struct Bounds {
    Point low, high;
};

struct SeedState {
    int cardinality;
    Bounds bounds;
};

// Also the layout of MPI_2INT, ordered by largest cardinality and then smallest seed.
struct Winner {
    int cardinality;
    int seed;
};
static_assert(sizeof(Winner) == 2 * sizeof(int));
static_assert(sizeof(Point) == 2 * sizeof(double));

void checkCuda(cudaError_t status) {
    if (status == cudaSuccess) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) checkCuda(cudaMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_) checkCuda(cudaFree(data_));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

void selectDevice() {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) checkCuda(cudaErrorNoDevice);
    checkCuda(cudaSetDevice(localRank % devices));
    checkCuda(cudaFree(nullptr));
}

int hostThreads(int work) {
    return std::min(omp_get_max_threads(), 1 + (work - 1) / 256);
}

// Convert the strict comparison on rounded square roots into an inclusive comparison
// on squared distances. Positive IEEE doubles have the same order as their bit patterns.
double squaredCutoff(double threshold) {
    if (!(threshold > 0.0)) return -1.0;
    std::uint64_t low = 0, high = 0x7fefffffffffffffULL;
    while (low < high) {
        const std::uint64_t mid = low + (high - low + 1) / 2;
        double squared;
        std::memcpy(&squared, &mid, sizeof(squared));
        if (std::sqrt(squared) < threshold) low = mid;
        else high = mid - 1;
    }
    double cutoff;
    std::memcpy(&cutoff, &low, sizeof(cutoff));
    return cutoff;
}

int prepareWindows(const std::vector<Point>& points, double threshold, int rank, int ranks,
                   std::vector<IndexedPoint>& ordered, std::vector<Window>& windows,
                   bool& wholeCloudFits) {
    const int count = static_cast<int>(points.size());
    const int localSeeds = static_cast<int>(windows.size());
    std::vector<int> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return points[a].x < points[b].x; });

    // Expand the x window to cover rounding and squared distances that underflow.
    const double radius = std::max(std::sqrt(std::numeric_limits<double>::min()),
                                   threshold * (1.0 + 4.0 * std::numeric_limits<double>::epsilon()));
    const double minX = points[order.front()].x;
    const double maxX = points[order.back()].x;
    const bool checkExtent = maxX - minX < threshold;
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -minY;
    int nonfinite = 0;
    int widest = 0;
    const int threads = hostThreads(count);
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for reduction(|:nonfinite) reduction(min:minY) reduction(max:maxY)
        for (int i = 0; i < count; ++i) {
            ordered[i] = {points[order[i]], order[i]};
            const double y = ordered[i].point.y;
            nonfinite |= !std::isfinite(y);
            if (checkExtent) {
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }
        }

        #pragma omp for reduction(max:widest)
        for (int i = 0; i < localSeeds; ++i) {
            Window window{0, 0};
            if (threshold > 0.0) {
                if (nonfinite) {
                    window = {0, count};
                } else {
                    const double x = points[rank + i * ranks].x;
                    const auto begin = std::partition_point(ordered.begin(), ordered.end(),
                        [&](const IndexedPoint& p) { return x - p.point.x > radius; });
                    const auto end = std::partition_point(begin, ordered.end(),
                        [&](const IndexedPoint& p) { return p.point.x - x <= radius; });
                    window = {static_cast<int>(begin - ordered.begin()),
                              static_cast<int>(end - ordered.begin())};
                }
            }
            windows[i] = window;
            widest = std::max(widest, window.end - window.begin);
        }
    }
    // A bounding rectangle strictly inside the threshold guarantees seed zero wins.
    wholeCloudFits = checkExtent && !nonfinite &&
        distance({minX, minY}, {maxX, maxY}) <
            threshold * (1.0 - 4.0 * std::numeric_limits<double>::epsilon());
    return widest;
}

__device__ double squaredDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
#if QT_HOST_FMA
    return __fma_rn(dx, dx, dy * dy);
#else
    return dx * dx + dy * dy;
#endif
}

__device__ bool better(double distance, int index, double bestDistance, int bestIndex) {
    return distance < bestDistance || (distance == bestDistance && index < bestIndex);
}

__device__ Candidate& candidateSlot(Candidate* scratch, Candidate* spill, int position) {
    return position < sharedWindow ? scratch[position] : spill[position - sharedWindow];
}

// Each warp grows one seed. A point's maximum distance only needs to be updated
// against the last member, and a point that fails the threshold can never return.
__device__ int growCluster(int seed, Window window, const Point* points, const IndexedPoint* ordered,
                           const unsigned char* removed, double cutoff, Candidate* scratch,
                           Candidate* spill, int* members, int capacity, Bounds* bounds) {
    const int lane = threadIdx.x % warpSizeQT;
    const int width = window.end - window.begin;
    Point memberPoint = points[seed];
    Bounds extent{memberPoint, memberPoint};
    int cardinality = 1;
    if (lane == 0) members[0] = seed;

    // Pack candidates that can join the seed so later iterations scan only this subset.
    int candidateCount = 0;
    for (int base = 0; base < width; base += warpSizeQT) {
        Candidate candidate{CUDART_INF, INT_MAX};
        if (lane < width - base) {
            const IndexedPoint point = ordered[window.begin + base + lane];
            if (point.index != seed && !removed[point.index]) {
                candidate = {fmax(0.0, squaredDistance(point.point, memberPoint)), point.index};
            }
        }
        const bool eligible = candidate.squared <= cutoff;
        const unsigned lanes = __ballot_sync(fullWarp, eligible);
        if (eligible) {
            const int position = candidateCount + __popc(lanes & ((1u << lane) - 1));
            candidateSlot(scratch, spill, position) = candidate;
        }
        candidateCount += __popc(lanes);
    }
    __syncwarp(fullWarp);

    int member = seed;
    while (true) {
        double bestDistance = CUDART_INF;
        double bestSquared = CUDART_INF;
        int bestIndex = INT_MAX;
        for (int offset = lane; offset < candidateCount; offset += warpSizeQT) {
            Candidate& candidate = candidateSlot(scratch, spill, offset);
            double squared = candidate.squared;
            if (squared > cutoff) continue;
            if (cardinality != 1) {
                if (candidate.index == member) {
                    candidate.squared = CUDART_INF;
                    continue;
                }
                squared = fmax(squared, squaredDistance(points[candidate.index], memberPoint));
                candidate.squared = squared;
            }
            if (squared <= cutoff && (squared <= bestSquared || candidate.index < bestIndex)) {
                const double distance = __dsqrt_rn(squared);
                if (better(distance, candidate.index, bestDistance, bestIndex)) {
                    bestSquared = squared;
                    bestDistance = distance;
                    bestIndex = candidate.index;
                }
            }
        }

        for (int stride = warpSizeQT / 2; stride > 0; stride /= 2) {
            const double distance = __shfl_down_sync(fullWarp, bestDistance, stride);
            const int index = __shfl_down_sync(fullWarp, bestIndex, stride);
            if (better(distance, index, bestDistance, bestIndex)) {
                bestDistance = distance;
                bestIndex = index;
            }
        }
        member = __shfl_sync(fullWarp, bestIndex, 0);
        if (member == INT_MAX) break;
        memberPoint = points[member];
        if (lane == 0) {
            if (cardinality < capacity) members[cardinality] = member;
            extent.low.x = fmin(extent.low.x, memberPoint.x);
            extent.low.y = fmin(extent.low.y, memberPoint.y);
            extent.high.x = fmax(extent.high.x, memberPoint.x);
            extent.high.y = fmax(extent.high.y, memberPoint.y);
        }
        ++cardinality;
    }
    if (lane == 0 && bounds) *bounds = extent;
    return cardinality;
}

__global__ void evaluateSeeds(const Point* points, const IndexedPoint* ordered, const Window* windows,
                              const unsigned char* removed, const int* lastMembers, int lastCount,
                              double cutoff, int rank, int ranks, int localSeeds, SeedState* states,
                              int* traces, Candidate* spill, int spillWidth,
                              unsigned long long* best) {
    __shared__ Candidate scratch[warpsPerBlock][sharedWindow];
    const int warp = threadIdx.x / warpSizeQT;
    const int lane = threadIdx.x % warpSizeQT;
    const int worker = blockIdx.x * warpsPerBlock + warp;
    const int workers = gridDim.x * warpsPerBlock;
    Candidate* warpSpill = spillWidth ? spill + static_cast<size_t>(worker) * spillWidth : nullptr;
    unsigned long long warpBest = 0;

    for (int local = worker; local < localSeeds; local += workers) {
        const int seed = rank + local * ranks;
        if (removed[seed]) continue;
        SeedState& state = states[local];
        int* trace = traces + static_cast<size_t>(local) * cachedMembers;
        int cardinality = state.cardinality;
        bool dirty = false;

        // Deleting points outside a greedy trace leaves every decision intact.
        if (cardinality > 0) {
            if (cardinality <= cachedMembers) {
                for (int i = lane; i < cardinality; i += warpSizeQT) dirty |= removed[trace[i]];
            } else {
                // Long traces only store their bounding box for invalidation.
                const Bounds bounds = state.bounds;
                for (int i = lane; i < lastCount; i += warpSizeQT) {
                    const Point p = points[lastMembers[i]];
                    dirty |= !(p.x < bounds.low.x || p.x > bounds.high.x ||
                               p.y < bounds.low.y || p.y > bounds.high.y);
                }
            }
        }
        const bool anyDirty = __any_sync(fullWarp, dirty);
        if (cardinality == 0 || anyDirty) {
            cardinality = growCluster(seed, windows[local], points, ordered, removed, cutoff,
                                      scratch[warp], warpSpill, trace, cachedMembers, &state.bounds);
            if (lane == 0) state.cardinality = cardinality;
        }

        const auto key = (static_cast<unsigned long long>(cardinality) << 32) |
                         static_cast<unsigned>(INT_MAX - seed);
        if (key > warpBest) warpBest = key;
    }
    if (lane == 0 && warpBest != 0) atomicMax(best, warpBest);
}

__global__ void recoverMembers(int seed, const Window* window, const Point* points,
                               const IndexedPoint* ordered, const unsigned char* removed,
                               double cutoff, Candidate* spill, int* members, int count) {
    __shared__ Candidate scratch[sharedWindow];
    growCluster(seed, *window, points, ordered, removed, cutoff, scratch, spill, members, count,
                nullptr);
}

__global__ void removeMembers(unsigned char* removed, const int* members, int count,
                              unsigned long long* best) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) removed[members[i]] = 1;
    if (i == 0) *best = 0;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold, int rank,
                                  int ranks) {
    const int count = static_cast<int>(points.size());
    int localSeeds = count / ranks + (rank < count % ranks);
    std::vector<Window> windows(localSeeds);
    std::vector<IndexedPoint> ordered(localSeeds ? count : 0);
    bool wholeCloudFits = false;
    int widest = localSeeds ? prepareWindows(points, threshold, rank, ranks, ordered, windows,
                                             wholeCloudFits) : 0;
    if (wholeCloudFits) {
        localSeeds = rank == 0 ? 1 : 0;
        widest = localSeeds ? windows[0].end - windows[0].begin : 0;
        windows.resize(localSeeds);
        if (!localSeeds) ordered.clear();
    }
    const int spillWidth = std::max(0, widest - sharedWindow);
    const double cutoff = squaredCutoff(threshold);

    int device, multiprocessors;
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    int blocks = std::min(localSeeds / warpsPerBlock + (localSeeds % warpsPerBlock != 0),
                          multiprocessors * 8);
    if (spillWidth) {
        // Reuse each resident warp's scratch across seeds and bound total spill storage.
        constexpr size_t spillBudget = 64 * 1024 * 1024;
        const size_t perBlock = static_cast<size_t>(spillWidth) * warpsPerBlock * sizeof(Candidate);
        blocks = std::min(blocks, static_cast<int>(std::max(size_t{1}, spillBudget / perBlock)));
    }

    DeviceBuffer<Point> devicePoints(localSeeds ? count : 0);
    DeviceBuffer<IndexedPoint> deviceOrdered(ordered.size());
    DeviceBuffer<Window> deviceWindows(localSeeds);
    DeviceBuffer<unsigned char> removed(localSeeds ? count : 0);
    DeviceBuffer<SeedState> states(localSeeds);
    DeviceBuffer<int> traces(static_cast<size_t>(localSeeds) * cachedMembers);
    DeviceBuffer<Candidate> spill(static_cast<size_t>(blocks) * warpsPerBlock * spillWidth);
    DeviceBuffer<int> deviceMembers(localSeeds ? count : 0);
    DeviceBuffer<unsigned long long> deviceBest(localSeeds ? 1 : 0);

    if (localSeeds) {
        checkCuda(cudaMemcpy(devicePoints.get(), points.data(), count * sizeof(Point),
                             cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(deviceOrdered.get(), ordered.data(), count * sizeof(IndexedPoint),
                             cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(deviceWindows.get(), windows.data(), localSeeds * sizeof(Window),
                             cudaMemcpyHostToDevice));
        checkCuda(cudaMemset(removed.get(), 0, count * sizeof(unsigned char)));
        checkCuda(cudaMemset(states.get(), 0, localSeeds * sizeof(SeedState)));
        checkCuda(cudaMemset(deviceBest.get(), 0, sizeof(unsigned long long)));
    }

    std::vector<Cluster> clusters;
    std::vector<unsigned char> assigned(rank == 0 ? count : 0);
    std::vector<int> members(count);
    int lastCount = 0;
    int remaining = count;
    while (remaining > 0) {
        Winner local{0, INT_MAX};
        if (localSeeds) {
            evaluateSeeds<<<blocks, warpsPerBlock * warpSizeQT>>>(
                devicePoints.get(), deviceOrdered.get(), deviceWindows.get(), removed.get(),
                deviceMembers.get(), lastCount, cutoff, rank, ranks, localSeeds, states.get(),
                traces.get(), spill.get(), spillWidth, deviceBest.get());
            checkCuda(cudaGetLastError());
            unsigned long long key;
            checkCuda(cudaMemcpy(&key, deviceBest.get(), sizeof(key), cudaMemcpyDeviceToHost));
            local = {static_cast<int>(key >> 32), INT_MAX - static_cast<int>(key & 0xffffffffULL)};
        }
        Winner best;
        MPI_Allreduce(&local, &best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // Once all remaining seeds yield singletons, future removals cannot change that.
        if (best.cardinality == 1) {
            if (rank == 0) {
                for (int seed = 0; seed < count; ++seed) {
                    if (!assigned[seed]) clusters.push_back({{seed}, seed});
                }
            }
            break;
        }

        const int owner = best.seed % ranks;
        if (rank == owner) {
            const int position = best.seed / ranks;
            const int* source = traces.get() + static_cast<size_t>(position) * cachedMembers;
            if (best.cardinality > cachedMembers) {
                recoverMembers<<<1, warpSizeQT>>>(
                    best.seed, deviceWindows.get() + position, devicePoints.get(), deviceOrdered.get(),
                    removed.get(), cutoff, spill.get(), deviceMembers.get(), best.cardinality);
                checkCuda(cudaGetLastError());
                source = deviceMembers.get();
            }
            checkCuda(cudaMemcpy(members.data(), source, best.cardinality * sizeof(int),
                                 cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), best.cardinality, MPI_INT, owner, MPI_COMM_WORLD);

        remaining -= best.cardinality;
        if (remaining > 0 && localSeeds) {
            lastCount = best.cardinality;
            if (rank != owner || lastCount <= cachedMembers) {
                checkCuda(cudaMemcpy(deviceMembers.get(), members.data(), lastCount * sizeof(int),
                                     cudaMemcpyHostToDevice));
            }
            removeMembers<<<1 + (lastCount - 1) / 256, 256>>>(removed.get(), deviceMembers.get(),
                                                             lastCount, deviceBest.get());
            checkCuda(cudaGetLastError());
        }
        if (rank == 0) {
            clusters.push_back({std::vector<int>(members.begin(), members.begin() + best.cardinality),
                                best.seed});
            for (int member : clusters.back().members) assigned[member] = 1;
        }
    }
    return clusters;
}

}  // namespace

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Calculate diameters in parallel, then report in cluster order.
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
    
    selectDevice();

    std::vector<Point> points(num_points);
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        generateSyntheticData(points, num_points);
    }
    MPI_Datatype pointType;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pointType);
    MPI_Type_commit(&pointType);
    MPI_Bcast(points.data(), num_points, pointType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pointType);

    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    if (rank != 0) return 0;

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
    const double time_sec = std::chrono::duration<double>(cluster_end - cluster_start).count();
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = runBenchmark(argc, argv, rank, ranks);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
