// QT Clustering Benchmark
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
        // The original group-size formula cannot make practical progress at N <= 30.
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
    return std::sqrt(std::fma(dx, dx, dy * dy));
}

namespace {

constexpr int warpSizeQt = 32;
constexpr int warpsPerBlock = 4;
constexpr int sharedLimit = 128;
constexpr unsigned fullWarp = 0xffffffffu;
using Score = unsigned long long;

void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        // Only the initializing thread may call MPI under MPI_THREAD_FUNNELED.
        std::abort();
    }
}

#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

struct Window {
    int first, last, original;
};

struct Problem {
    const Point* points;
    const Window* windows;
    const unsigned char* active;
    const int* removed;
    int removedCount;
    double cutoff;
};

struct Scratch {
    double* sharedMax;
    int* sharedPos;
    double* globalMax;
    int* globalPos;
    int sharedCapacity;

    __device__ double& maximum(int i) const {
        return i < sharedCapacity ? sharedMax[i] : globalMax[i - sharedCapacity];
    }

    __device__ int& position(int i) const {
        return i < sharedCapacity ? sharedPos[i] : globalPos[i - sharedCapacity];
    }
};

struct Choice {
    double squared;
    int original;
    int position;

    __device__ Choice() : squared(DBL_MAX), original(INT_MAX), position(-1) {}

    __device__ void consider(double otherSquared, int otherOriginal, int otherPosition) {
        const bool earlier = otherOriginal < original;
        bool better = otherSquared < squared;
        if (earlier != better) {
            const long long gap = __double_as_longlong(otherSquared) - __double_as_longlong(squared);
            // Rounding sqrt to the same double can span only a few input ulps.
            // Each normal root has relative rounding error <= 2^-53, so two
            // such squared inputs differ by less than 5 * 2^-53 of the smaller.
            if (gap == 0 || (gap >= -8 && gap <= 8
                && __dsqrt_rn(otherSquared) == __dsqrt_rn(squared))) {
                better = earlier;
            }
        }
        if (better) {
            squared = otherSquared;
            original = otherOriginal;
            position = otherPosition;
        }
    }

    __device__ int winner() {
        for (int offset = warpSizeQt / 2; offset > 0; offset /= 2) {
            const double otherSquared = __shfl_down_sync(fullWarp, squared, offset);
            const int otherOriginal = __shfl_down_sync(fullWarp, original, offset);
            const int otherPosition = __shfl_down_sync(fullWarp, position, offset);
            consider(otherSquared, otherOriginal, otherPosition);
        }
        return __shfl_sync(fullWarp, position, 0);
    }
};

__device__ double squaredDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // The sequential std::max ignores NaN distances.
    return fmax(0.0, __fma_rn(dx, dx, __dmul_rn(dy, dy)));
}

__device__ int appendCandidates(Scratch scratch, int count, int lane,
                                int position, double maximum, bool keep) {
    const unsigned mask = __ballot_sync(fullWarp, keep);
    if (keep) {
        const int dest = count + __popc(mask & ((1u << lane) - 1));
        scratch.position(dest) = position;
        scratch.maximum(dest) = maximum;
    }
    __syncwarp();
    return count + __popc(mask);
}

// One warp grows a seed's candidate cluster. Removing an infeasible point is
// permanent within this candidate because its running maximum can only grow.
__device__ int growCluster(Problem problem, Scratch scratch, int seed, int lane,
                           int* members) {
    const Window window = problem.windows[seed];
    const Point seedPoint = problem.points[seed];
    if (members && lane == 0) members[0] = seed;

    int count = 0;
    Choice closest;
    for (int first = window.first; first < window.last;) {
        const int width = min(warpSizeQt, window.last - first);
        const int pos = lane < width ? first + lane : seed;
        double maximum = 0.0;
        bool keep = lane < width && pos != seed && problem.active[pos];
        if (keep) {
            maximum = squaredDistance(problem.points[pos], seedPoint);
            keep = maximum <= problem.cutoff;
            if (keep) closest.consider(maximum, problem.windows[pos].original, pos);
        }
        count = appendCandidates(scratch, count, lane, pos, maximum, keep);
        first += width;
    }

    int size = 1;
    for (int selected = closest.winner(); selected >= 0; selected = closest.winner()) {
        if (members && lane == 0) members[size] = selected;
        ++size;
        if (count == 1) break;

        const Point last = problem.points[selected];
        closest = Choice{};
        const int previousCount = count;
        count = 0;
        for (int first = 0; first < previousCount;) {
            const int width = min(warpSizeQt, previousCount - first);
            int pos = -1;
            double maximum = 0.0;
            bool keep = false;
            if (lane < width) {
                const int slot = first + lane;
                pos = scratch.position(slot);
                if (pos != selected) {
                    maximum = fmax(scratch.maximum(slot), squaredDistance(problem.points[pos], last));
                    keep = maximum <= problem.cutoff;
                    if (keep) closest.consider(maximum, problem.windows[pos].original, pos);
                }
            }
            // The compacted output can overlap this warp's current input.
            __syncwarp();
            count = appendCandidates(scratch, count, lane, pos, maximum, keep);
            first += width;
        }
    }
    return size;
}

__global__ void removeMembers(unsigned char* active, const int* removed, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        active[removed[i]] = 0;
    }
}

__device__ bool needsRescore(Problem problem, int seed, int lane) {
    if (problem.removedCount == 0) return true;

    const Point point = problem.points[seed];
    const Window window = problem.windows[seed];
    for (int first = 0; first < problem.removedCount;) {
        const int width = min(warpSizeQt, problem.removedCount - first);
        bool affected = false;
        if (lane < width) {
            const int pos = problem.removed[first + lane];
            affected = pos >= window.first && pos < window.last
                && squaredDistance(problem.points[pos], point) <= problem.cutoff;
        }
        if (__any_sync(fullWarp, affected)) return true;
        first += width;
    }
    // No removed point could be added to this seed's cluster.
    return false;
}

__device__ Scratch warpScratch(double* shared, double* globalMax, int* globalPos,
                               int sharedCapacity, int globalCapacity) {
    const int warp = threadIdx.x / warpSizeQt;
    const size_t globalOffset = (static_cast<size_t>(blockIdx.x) * warpsPerBlock + warp)
                                * globalCapacity;
    return {shared + warp * sharedCapacity,
            reinterpret_cast<int*>(shared + blockDim.x / warpSizeQt * sharedCapacity)
                + warp * sharedCapacity,
            globalMax ? globalMax + globalOffset : nullptr,
            globalPos ? globalPos + globalOffset : nullptr,
            sharedCapacity};
}

__global__ void scoreSeeds(Problem problem, int firstSeed, int seedCount, int stride,
                           int* sizes, int sharedCapacity, int globalCapacity,
                           double* globalMax, int* globalPos, Score* best) {
    extern __shared__ double shared[];
    const Scratch scratch = warpScratch(shared, globalMax, globalPos, sharedCapacity, globalCapacity);
    const int lane = threadIdx.x % warpSizeQt;
    Score warpBest = 0;
    for (int64_t i = blockIdx.x * warpsPerBlock + threadIdx.x / warpSizeQt; i < seedCount;
         i += gridDim.x * warpsPerBlock) {
        const int seed = firstSeed + static_cast<int>(i) * stride;
        if (!problem.active[seed]) continue;

        int size;
        if (needsRescore(problem, seed, lane)) {
            size = growCluster(problem, scratch, seed, lane, nullptr);
            if (lane == 0) sizes[i] = size;
            __syncwarp();
        } else {
            size = sizes[i];
        }

        if (lane == 0) {
            // MPI_MAX can use the same ordering as the GPU: largest size, then first seed.
            const Score score = (static_cast<Score>(size) << 32)
                                | static_cast<unsigned>(INT_MAX - problem.windows[seed].original);
            if (score > warpBest) warpBest = score;
        }
    }
    if (lane == 0 && warpBest) atomicMax(best, warpBest);
}

__global__ void getMembers(Problem problem, int seed, int sharedCapacity, int globalCapacity,
                           double* globalMax, int* globalPos, int* members) {
    extern __shared__ double shared[];
    const Scratch scratch = warpScratch(shared, globalMax, globalPos, sharedCapacity, globalCapacity);
    growCluster(problem, scratch, seed, threadIdx.x, members);
}

// Find the last squared distance whose correctly rounded square root is
// strictly below the threshold. Comparing to threshold*threshold can differ
// at the boundary because both multiplication and square root round.
double squaredCutoff(double threshold) {
    if (!(threshold > 0.0)) return -1.0;
    uint64_t low = 0, high = UINT64_C(0x7ff0000000000000); // IEEE positive infinity
    while (high - low > 1) {
        const uint64_t mid = low + (high - low) / 2;
        double value;
        std::memcpy(&value, &mid, sizeof(value));
        if (std::sqrt(value) < threshold) low = mid;
        else high = mid;
    }
    double cutoff;
    std::memcpy(&cutoff, &low, sizeof(cutoff));
    return cutoff;
}

// Assign each visible physical GPU once per node, including when the launcher
// masks visibility differently for individual ranks.
std::vector<int> assignedDevices() {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int rank, ranks;
    MPI_Comm_rank(node, &rank);
    MPI_Comm_size(node, &ranks);

    int visible = 0;
    const cudaError_t status = cudaGetDeviceCount(&visible);
    if (status != cudaErrorNoDevice) CUDA_CHECK(status);
    std::vector<cudaUUID_t> ids(visible);
    for (int device = 0; device < visible; ++device) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        ids[device] = prop.uuid;
    }

    std::vector<int> counts(ranks), offsets(ranks);
    const int bytes = visible * sizeof(cudaUUID_t);
    MPI_Allgather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, node);
    int totalBytes = 0;
    for (int i = 0; i < ranks; ++i) {
        offsets[i] = totalBytes;
        totalBytes += counts[i];
    }

    std::vector<char> allIds(totalBytes);
    MPI_Allgatherv(ids.data(), bytes, MPI_BYTE, allIds.data(), counts.data(), offsets.data(),
                   MPI_BYTE, node);
    MPI_Comm_free(&node);

    std::vector<int> result;
    std::vector<int> loads(ranks);
    for (int i = 0; i < totalBytes; i += sizeof(cudaUUID_t)) {
        const char* id = allIds.data() + i;
        bool seen = false;
        for (int j = 0; j < i; j += sizeof(cudaUUID_t)) {
            if (std::memcmp(id, allIds.data() + j, sizeof(cudaUUID_t)) == 0) {
                seen = true;
                break;
            }
        }
        if (seen) continue;

        int owner = -1;
        int ordinal = -1;
        for (int r = 0; r < ranks; ++r) {
            if (owner >= 0 && loads[r] >= loads[owner]) continue;
            for (int j = 0; j < counts[r]; j += sizeof(cudaUUID_t)) {
                if (std::memcmp(id, allIds.data() + offsets[r] + j, sizeof(cudaUUID_t)) == 0) {
                    owner = r;
                    ordinal = j / sizeof(cudaUUID_t);
                    break;
                }
            }
        }
        ++loads[owner];
        if (owner == rank) result.push_back(ordinal);
    }
    return result;
}

template<class T>
void allocateDevice(T*& pointer, size_t count) {
    if (count) CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
}

template<class T>
void upload(T* destination, const T* source, size_t count, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template<class T>
void download(T* destination, const T* source, size_t count, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

struct DeviceWork {
    int id = -1;
    int firstSeed = 0, seedCount = 0, stride = 0;
    int blocks = 0, sharedCapacity = 0, globalCapacity = 0;
    cudaStream_t stream = nullptr;
    Point* points = nullptr;
    Window* windows = nullptr;
    unsigned char* active = nullptr;
    int* sizes = nullptr;
    int* members = nullptr;
    int* removed = nullptr;
    int* hostMembers = nullptr;
    Score* best = nullptr;
    Score* hostBest = nullptr;
    double* globalMax = nullptr;
    int* globalPos = nullptr;
    double cutoff = 0.0;

    void initialize(int device, int first, int workers, const std::vector<Point>& ordered,
                    const std::vector<Window>& bounds, int maxWindow, double limit, int completeSeed) {
        id = device;
        firstSeed = first;
        stride = workers;
        cutoff = limit;
        const int n = static_cast<int>(ordered.size());
        seedCount = first < n ? 1 + (n - 1 - first) / stride : 0;
        if (!seedCount || (completeSeed >= 0 && completeSeed % stride != firstSeed)) return;

        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int multiprocessors;
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, id));
        blocks = std::min(1 + (seedCount - 1) / warpsPerBlock, 8 * multiprocessors);
        if (completeSeed >= 0) blocks = 1;
        sharedCapacity = std::min(maxWindow, sharedLimit);
        globalCapacity = maxWindow - sharedCapacity;
        if (globalCapacity) {
            size_t availableBytes, totalBytes;
            CUDA_CHECK(cudaMemGetInfo(&availableBytes, &totalBytes));
            constexpr size_t scratchLimit = 512ull * 1024 * 1024;
            const size_t scratchBudget = std::min(scratchLimit, availableBytes / 16);
            const size_t perBlock = static_cast<size_t>(globalCapacity) * warpsPerBlock
                                    * (sizeof(double) + sizeof(int));
            blocks = static_cast<int>(std::clamp(scratchBudget / perBlock, size_t{1},
                                                 static_cast<size_t>(blocks)));
        }
        const size_t slots = static_cast<size_t>(blocks) * warpsPerBlock * globalCapacity;
        allocateDevice(globalMax, slots);
        allocateDevice(globalPos, slots);
        allocateDevice(points, n);
        allocateDevice(windows, n);
        allocateDevice(active, n);
        allocateDevice(sizes, seedCount);
        allocateDevice(best, 1);
        allocateDevice(members, maxWindow);
        allocateDevice(removed, maxWindow);
        CUDA_CHECK(cudaMallocHost(&hostMembers, static_cast<size_t>(maxWindow) * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&hostBest, sizeof(Score)));
        upload(points, ordered.data(), n, stream);
        upload(windows, bounds.data(), n, stream);
        CUDA_CHECK(cudaMemsetAsync(active, 1, static_cast<size_t>(n), stream));
    }

    size_t sharedBytes(int warps) const {
        return static_cast<size_t>(sharedCapacity) * warps * (sizeof(double) + sizeof(int));
    }

    void submit(const std::vector<int>& latest) const {
        if (!seedCount) return;
        CUDA_CHECK(cudaSetDevice(id));
        const int count = static_cast<int>(latest.size());
        if (count) {
            std::copy(latest.begin(), latest.end(), hostMembers);
            upload(removed, hostMembers, count, stream);
            removeMembers<<<std::min(1 + (count - 1) / 256, 1024), 256, 0, stream>>>(active, removed, count);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemsetAsync(best, 0, sizeof(Score), stream));
        scoreSeeds<<<blocks, warpsPerBlock * warpSizeQt, sharedBytes(warpsPerBlock), stream>>>(
            {points, windows, active, removed, count, cutoff}, firstSeed, seedCount, stride, sizes,
            sharedCapacity, globalCapacity, globalMax, globalPos, best);
        CUDA_CHECK(cudaGetLastError());
        download(hostBest, best, 1, stream);
    }

    Score collect() const {
        if (!seedCount) return 0;
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return *hostBest;
    }

    void readMembers(int seed, std::vector<int>& result) const {
        CUDA_CHECK(cudaSetDevice(id));
        getMembers<<<1, warpSizeQt, sharedBytes(1), stream>>>(
            {points, windows, active, nullptr, 0, cutoff}, seed,
            sharedCapacity, globalCapacity, globalMax, globalPos, members);
        CUDA_CHECK(cudaGetLastError());
        download(hostMembers, members, result.size(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::copy_n(hostMembers, result.size(), result.begin());
    }

    ~DeviceWork() {
        if (!stream) return;
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(points));
        CUDA_CHECK(cudaFree(windows));
        CUDA_CHECK(cudaFree(active));
        CUDA_CHECK(cudaFree(sizes));
        CUDA_CHECK(cudaFree(members));
        CUDA_CHECK(cudaFree(removed));
        CUDA_CHECK(cudaFree(best));
        CUDA_CHECK(cudaFree(globalMax));
        CUDA_CHECK(cudaFree(globalPos));
        CUDA_CHECK(cudaFreeHost(hostMembers));
        CUDA_CHECK(cudaFreeHost(hostBest));
    }
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks, const std::vector<int>& devices) {
    const int n = static_cast<int>(points.size());
    const int localWorkers = static_cast<int>(devices.size());
    std::vector<int> counts(ranks), offsets(ranks + 1);
    MPI_Allgather(&localWorkers, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
    const int workers = offsets.back();
    if (workers == 0) {
        if (rank == 0) fprintf(stderr, "QT clustering requires a visible CUDA device.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Sorting by x bounds each seed's search; its original index still breaks ties.
    const bool finite = std::all_of(points.begin(), points.end(),
                                   [](Point p) { return std::isfinite(p.x) && std::isfinite(p.y); });
    std::vector<int> order(n), positions(n);
    std::iota(order.begin(), order.end(), 0);
    if (finite) {
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return points[a].x < points[b].x || (points[a].x == points[b].x && a < b);
        });
    }
    std::vector<Point> ordered(n);
    std::vector<Window> windows(n);
    const int availableThreads = omp_get_max_threads();
    const int deviceThreads = std::min(availableThreads, std::max(1, localWorkers));
    const int prepThreads = std::min(availableThreads, std::max(deviceThreads, 2 + (n - 1) / 4096));
    #pragma omp parallel for num_threads(prepThreads)
    for (int i = 0; i < n; ++i) {
        ordered[i] = points[order[i]];
        positions[order[i]] = i;
    }

    const double cutoff = squaredCutoff(threshold);
    // A conservative outward-rounded x radius also covers squared underflow.
    const double radius = std::max(0x1p-500, std::nextafter(std::nextafter(
        threshold, std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity()));
    // If the bounding box fits, every seed would return the entire set.
    // Only the first seed's ordered members need to be constructed.
    int completeSeed = -1;
    if (finite && cutoff >= 0.0) {
        const auto ys = std::minmax_element(ordered.begin(), ordered.end(),
                                           [](Point a, Point b) { return a.y < b.y; });
        const double dx = ordered.back().x - ordered.front().x;
        const double dy = ys.second->y - ys.first->y;
        if (std::fma(dx, dx, dy * dy) <= cutoff) completeSeed = positions[0];
    }

    int maxWindow = 1;
    #pragma omp parallel for num_threads(prepThreads) reduction(max:maxWindow)
    for (int i = 0; i < n; ++i) {
        int first = 0;
        int last = n;
        if (cutoff < 0.0) {
            first = i;
            last = i + 1;
        } else if (finite) {
            const double x = ordered[i].x;
            const double inf = std::numeric_limits<double>::infinity();
            const double lower = std::nextafter(x - radius, -inf);
            const double upper = std::nextafter(x + radius, inf);
            first = static_cast<int>(std::lower_bound(ordered.begin(), ordered.end(), lower,
                [](Point p, double key) { return p.x < key; }) - ordered.begin());
            last = static_cast<int>(std::upper_bound(ordered.begin(), ordered.end(), upper,
                [](double key, Point p) { return key < p.x; }) - ordered.begin());
        }
        windows[i] = {first, last, order[i]};
        maxWindow = std::max(maxWindow, last - first);
    }

    std::vector<DeviceWork> work(localWorkers);
    #pragma omp parallel for num_threads(deviceThreads)
    for (int i = 0; i < localWorkers; ++i) {
        work[i].initialize(devices[i], offsets[rank] + i, workers, ordered, windows,
                           maxWindow, cutoff, completeSeed);
    }

    std::vector<Cluster> clusters;
    std::vector<unsigned char> clustered(rank == 0 ? n : 0);
    std::vector<int> latest;
    for (int remaining = n; remaining > 0;) {
        Score best = (static_cast<Score>(n) << 32) | INT_MAX;
        if (completeSeed < 0) {
            #pragma omp parallel for num_threads(deviceThreads)
            for (int i = 0; i < localWorkers; ++i) work[i].submit(latest);

            Score localBest = 0;
            #pragma omp parallel for num_threads(deviceThreads) reduction(max:localBest)
            for (int i = 0; i < localWorkers; ++i) localBest = std::max(localBest, work[i].collect());

            MPI_Allreduce(&localBest, &best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        }
        const int size = static_cast<int>(best >> 32);
        if (size == 1 && completeSeed < 0) {
            // A compatible pair would have made a cluster of at least two.
            if (rank == 0) {
                for (int i = 0; i < n; ++i) {
                    if (!clustered[i]) clusters.push_back({{i}, i});
                }
            }
            break;
        }

        const int seed = INT_MAX - static_cast<int>(static_cast<unsigned>(best));
        const int position = positions[seed];
        const int worker = position % workers;
        const int owner = static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), worker)
                                           - offsets.begin()) - 1;
        latest.resize(size);
        if (rank == owner) work[worker - offsets[owner]].readMembers(position, latest);
        if (size < remaining) MPI_Bcast(latest.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        else if (owner != 0) {
            if (rank == owner) MPI_Send(latest.data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD);
            if (rank == 0) MPI_Recv(latest.data(), size, MPI_INT, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = seed;
            cluster.members.reserve(size);
            for (int pos : latest) {
                const int member = order[pos];
                cluster.members.push_back(member);
                clustered[member] = true;
            }
            clusters.push_back(std::move(cluster));
        }
        remaining -= size;
    }
    return clusters;
}

} // namespace

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
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                              num_points, threshold);
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const auto devices = assignedDevices();
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    // Use an MPI point datatype so the collective count is in points, not bytes.
    MPI_Datatype pointType;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pointType);
    MPI_Type_commit(&pointType);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Bcast(points.data(), num_points, pointType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pointType);

    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks, devices);
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

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "QT clustering requires MPI_THREAD_FUNNELED support.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int result = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
