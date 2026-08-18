#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Phase 1: close the diagonal tile. Padding is represented by INF so arbitrary
// matrix sizes use exactly the same kernel as multiples of TILE.
__global__ void diagonalKernel(unsigned int* __restrict__ dist, int n, int round) {
    __shared__ unsigned int pivot[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int base = round * TILE;

#pragma unroll
    for (int dy = 0; dy < TILE; dy += BLOCK_ROWS) {
        const int y = threadIdx.y + dy;
        const int row = base + y;
        const int col = base + x;
        pivot[y][x] = (row < n && col < n) ? dist[static_cast<size_t>(row) * n + col]
                                           : INF;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int dy = 0; dy < TILE; dy += BLOCK_ROWS) {
            const int y = threadIdx.y + dy;
            const unsigned int candidate = pivot[y][k] + pivot[k][x];
            if (candidate < pivot[y][x]) pivot[y][x] = candidate;
        }
        __syncthreads();
    }

#pragma unroll
    for (int dy = 0; dy < TILE; dy += BLOCK_ROWS) {
        const int y = threadIdx.y + dy;
        const int row = base + y;
        const int col = base + x;
        if (row < n && col < n) dist[static_cast<size_t>(row) * n + col] = pivot[y][x];
    }
}

// Phase 2: update every tile in the pivot row and pivot column. Results remain
// in registers until all 32 intermediates have been inspected, making the
// shared source tile read-only and eliminating synchronization inside the loop.
__global__ void pivotRowColumnKernel(unsigned int* __restrict__ dist, int n,
                                     int round, int tileCount) {
    __shared__ unsigned int pivot[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int otherTile = blockIdx.x + (blockIdx.x >= round);
    const bool columnTile = blockIdx.y != 0;
    const int tileRow = columnTile ? otherTile : round;
    const int tileCol = columnTile ? round : otherTile;
    const int pivotBase = round * TILE;

    if (otherTile >= tileCount) return;

#pragma unroll
    for (int dy = 0; dy < TILE; dy += BLOCK_ROWS) {
        const int y = threadIdx.y + dy;
        const int pr = pivotBase + y;
        const int pc = pivotBase + x;
        pivot[y][x] = (pr < n && pc < n) ? dist[static_cast<size_t>(pr) * n + pc] : INF;

        const int row = tileRow * TILE + y;
        const int col = tileCol * TILE + x;
        tile[y][x] = (row < n && col < n) ? dist[static_cast<size_t>(row) * n + col] : INF;
    }
    __syncthreads();

#pragma unroll
    for (int dy = 0; dy < TILE; dy += BLOCK_ROWS) {
        const int y = threadIdx.y + dy;
        unsigned int best = tile[y][x];
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int candidate = columnTile ? tile[y][k] + pivot[k][x]
                                                       : pivot[y][k] + tile[k][x];
            best = min(best, candidate);
        }
        const int row = tileRow * TILE + y;
        const int col = tileCol * TILE + x;
        if (row < n && col < n) dist[static_cast<size_t>(row) * n + col] = best;
    }
}

// Phase 3: all non-pivot tiles are independent. Each 256-thread block updates
// a 32x32 output tile, reusing 8 KiB of shared row/column data.
__global__ void remainderKernel(unsigned int* __restrict__ dist, int n,
                                int round, int tileCount) {
    __shared__ unsigned int left[TILE][TILE + 1];
    __shared__ unsigned int top[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int tileRow = blockIdx.y + (blockIdx.y >= round);
    const int tileCol = blockIdx.x + (blockIdx.x >= round);

    if (tileRow >= tileCount || tileCol >= tileCount) return;

#pragma unroll
    for (int dy = 0; dy < TILE; dy += BLOCK_ROWS) {
        const int y = threadIdx.y + dy;
        const int row = tileRow * TILE + y;
        const int col = tileCol * TILE + x;
        const int pivotRow = round * TILE + y;
        const int pivotCol = round * TILE + x;
        left[y][x] = (row < n && pivotCol < n)
                         ? dist[static_cast<size_t>(row) * n + pivotCol]
                         : INF;
        top[y][x] = (pivotRow < n && col < n)
                        ? dist[static_cast<size_t>(pivotRow) * n + col]
                        : INF;
    }
    __syncthreads();

#pragma unroll
    for (int dy = 0; dy < TILE; dy += BLOCK_ROWS) {
        const int y = threadIdx.y + dy;
        const int row = tileRow * TILE + y;
        const int col = tileCol * TILE + x;
        if (row < n && col < n) {
            unsigned int best = dist[static_cast<size_t>(row) * n + col];
#pragma unroll
            for (int k = 0; k < TILE; ++k) {
                best = min(best, left[y][k] + top[k][x]);
            }
            dist[static_cast<size_t>(row) * n + col] = best;
        }
    }
}

class DeviceMatrix {
public:
    explicit DeviceMatrix(const std::vector<unsigned int>& host) : count_(host.size()) {
        checkCuda(cudaMalloc(&data_, count_ * sizeof(unsigned int)), "device allocation");
        checkCuda(cudaMemcpy(data_, host.data(), count_ * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "host-to-device copy");
    }

    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    ~DeviceMatrix() { cudaFree(data_); }

    unsigned int* data() { return data_; }

    void copyTo(std::vector<unsigned int>& host) const {
        checkCuda(cudaMemcpy(host.data(), data_, count_ * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "device-to-host copy");
    }

private:
    unsigned int* data_ = nullptr;
    size_t count_ = 0;
};

double floydWarshall(DeviceMatrix& device, const size_t numNodes) {
    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Number of nodes exceeds the CUDA kernel index range\n");
        std::exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const int tileCount = (n + TILE - 1) / TILE;
    const dim3 threads(TILE, BLOCK_ROWS);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "start event creation");
    checkCuda(cudaEventCreate(&stop), "stop event creation");
    checkCuda(cudaEventRecord(start), "start event recording");

    for (int round = 0; round < tileCount; ++round) {
        diagonalKernel<<<1, threads>>>(device.data(), n, round);
        if (tileCount > 1) {
            pivotRowColumnKernel<<<dim3(tileCount - 1, 2), threads>>>(
                device.data(), n, round, tileCount);
            remainderKernel<<<dim3(tileCount - 1, tileCount - 1), threads>>>(
                device.data(), n, round, tileCount);
        }
    }

    checkCuda(cudaGetLastError(), "Floyd-Warshall kernel launch");
    checkCuda(cudaEventRecord(stop), "stop event recording");
    checkCuda(cudaEventSynchronize(stop), "Floyd-Warshall execution");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "elapsed time measurement");
    checkCuda(cudaEventDestroy(start), "start event destruction");
    checkCuda(cudaEventDestroy(stop), "stop event destruction");
    return elapsedMs;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long long parsed = std::atoll(argv[++i]);
            if (parsed <= 0) {
                std::fprintf(stderr, "Number of nodes must be positive\n");
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        std::fprintf(stderr, "Distance matrix size overflows address space\n");
        return 1;
    }

    std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    std::printf("Number of nodes: %zu\n", numNodes);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<unsigned int> dist(numNodes * numNodes);
    std::printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);

    // Allocation and transfer are setup costs; the reported computation time is
    // measured by GPU events around the three blocked kernel phases only.
    DeviceMatrix device(dist);
    std::printf("Computing shortest paths...\n");
    const double durationMs = floydWarshall(device, numNodes);
    device.copyTo(dist);

    std::printf("Computation time: %.3f ms\n", durationMs);
    const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
    const double gops = ops / (durationMs * 1.0e6);
    std::printf("Performance: %.3f GOPS\n", gops);

    if (printResults) print_results_int(dist, "DistanceMatrix");

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(dist, numNodes)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
