#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cstdint>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

constexpr int TILE = 32;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// glibc rand_r advances this LCG three times per output. Jumping to each
// row's seed preserves the original stream without serial initialization.
unsigned int rowSeed(uint64_t outputs) {
    uint32_t a = 1103515245u, c = 12345u, seed = 42;
    for (uint64_t steps = outputs * 3; steps; steps >>= 1) {
        if (steps & 1) seed = a * seed + c;
        c = c * (a + 1);
        a *= a;
    }
    return seed;
}

// Save the pivot-column coefficients at precisely their original k step.
__global__ void diagonalSnapshots(const unsigned int* dist, unsigned int* left,
                                  size_t pitch, size_t localPivot, size_t pivot) {
    __shared__ unsigned int d[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    d[y][x] = dist[(localPivot + y) * pitch + pivot + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        if (x == k) left[y * TILE + k] = d[y][k];
        const unsigned int candidate = d[y][k] + d[k][x];
        // Pivot row/column cannot improve (nonnegative weights, zero diagonal).
        if (y != k && x != k) d[y][x] = min(d[y][x], candidate);
        __syncthreads();
    }
}

// Each output row is a snapshot BEFORE its own pivot step, rather than a
// fully closed panel. This preserves even the original strict-tie path values.
__global__ void rowSnapshots(const unsigned int* dist, const unsigned int* left,
                            unsigned int* panel, size_t pitch, size_t localPivot) {
    __shared__ unsigned int d[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t col = size_t(blockIdx.x) * TILE + x;
    d[y][x] = dist[(localPivot + y) * pitch + col];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        if (y == k) panel[size_t(k) * pitch + col] = d[k][x];
        const unsigned int candidate = left[y * TILE + k] + d[k][x];
        if (y != k) d[y][x] = min(d[y][x], candidate);
        __syncthreads();
    }
}

__global__ void columnSnapshots(const unsigned int* dist, const unsigned int* panel,
                               unsigned int* left, size_t pitch, size_t pivot) {
    __shared__ unsigned int d[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t row = size_t(blockIdx.x) * TILE + y;
    d[y][x] = dist[row * pitch + pivot + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        if (x == k) left[row * TILE + k] = d[y][k];
        const unsigned int candidate = d[y][k] + panel[size_t(k) * pitch + pivot + x];
        if (x != k) d[y][x] = min(d[y][x], candidate);
        __syncthreads();
    }
}

// Read-only snapshots allow the main O(n^3) work to run without barriers in
// the inner loop. A warp accesses adjacent columns; each thread owns 4 cells.
__global__ void updateTiles(unsigned int* dist, unsigned int* path,
                            const unsigned int* left, const unsigned int* panel,
                            size_t pitch, size_t pivot) {
    __shared__ unsigned int l[TILE][TILE], r[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t rowBase = size_t(blockIdx.y) * TILE;
    const size_t col = size_t(blockIdx.x) * TILE + x;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int row = y + t * 8;
        l[row][x] = left[(rowBase + row) * TILE + x];
        r[row][x] = panel[size_t(row) * pitch + col];
    }
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int row = y + t * 8;
        const size_t index = (rowBase + row) * pitch + col;
        unsigned int value = dist[index], via = path[index];
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int candidate = l[row][k] + r[k][x];
            if (candidate < value) { value = candidate; via = pivot + k; }
        }
        dist[index] = value;
        path[index] = via;
    }
}

void floydWarshall(unsigned int* dist, unsigned int* path, size_t pitch,
                   size_t firstTile, size_t localTiles, int rank, int ranks) {
    unsigned int *panel, *left, *diagonal, *hostPanel;
    const size_t panelSize = TILE * pitch;
    CUDA(cudaMalloc(&panel, panelSize * sizeof(unsigned int)));
    CUDA(cudaMalloc(&left, std::max(size_t(1), localTiles * TILE * TILE) * sizeof(unsigned int)));
    CUDA(cudaMalloc(&diagonal, TILE * TILE * sizeof(unsigned int)));
    CUDA(cudaMallocHost(&hostPanel, panelSize * sizeof(unsigned int)));
    const size_t tiles = pitch / TILE;
    for (size_t tile = 0; tile < tiles; ++tile) {
        const int owner = ((tile + 1) * ranks - 1) / tiles;
        if (rank == owner) {
            const size_t localPivot = (tile - firstTile) * TILE;
            diagonalSnapshots<<<1, dim3(TILE, TILE)>>>(dist, diagonal, pitch, localPivot, tile * TILE);
            rowSnapshots<<<tiles, dim3(TILE, TILE)>>>(dist, diagonal, panel, pitch, localPivot);
            CUDA(cudaGetLastError());
            if (ranks > 1)
                CUDA(cudaMemcpy(hostPanel, panel, panelSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) {
            // Pinned staging works on MPI installations without CUDA-aware MPI.
            MPI_Bcast(hostPanel, static_cast<int>(panelSize), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            if (rank != owner)
                CUDA(cudaMemcpy(panel, hostPanel, panelSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        if (localTiles) {
            columnSnapshots<<<localTiles, dim3(TILE, TILE)>>>(dist, panel, left, pitch, tile * TILE);
            updateTiles<<<dim3(tiles, localTiles), dim3(TILE, 8)>>>(dist, path, left, panel, pitch, tile * TILE);
            CUDA(cudaGetLastError());
        }
    }
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaFreeHost(hostPanel));
    CUDA(cudaFree(diagonal));
    CUDA(cudaFree(left));
    CUDA(cudaFree(panel));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* argument = argv[++i];
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (*argument == '-' || end == argument || *end ||
                parsed > static_cast<unsigned long long>(INT_MAX / TILE - TILE)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argument);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        if (rank == 0) fprintf(stderr, "A CUDA device is required on every MPI node.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const size_t pitch = tiles * TILE;
    const size_t firstTile = tiles * rank / ranks;
    const size_t localTiles = tiles * (rank + 1) / ranks - firstTile;
    const size_t rows = localTiles * TILE;
    const size_t elements = rows * pitch;
    if (localTiles > 65535 || (pitch && rows > std::numeric_limits<size_t>::max() / pitch / sizeof(unsigned int))) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds supported allocation or CUDA grid dimensions.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(elements), path(elements);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        const size_t globalRow = firstTile * TILE + row;
        unsigned int seed = rowSeed(uint64_t(globalRow) * numNodes);
        for (size_t col = 0; col < pitch; ++col) {
            unsigned int value = INF;
            if (globalRow < numNodes && col < numNodes)
                value = 1 + static_cast<unsigned int>(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
            dist[row * pitch + col] = globalRow == col ? 0 : value;
            path[row * pitch + col] = globalRow;
        }
    }
    unsigned int *deviceDist, *devicePath;
    CUDA(cudaMalloc(&deviceDist, std::max(size_t(1), elements) * sizeof(unsigned int)));
    CUDA(cudaMalloc(&devicePath, std::max(size_t(1), elements) * sizeof(unsigned int)));
    if (elements) {
        CUDA(cudaMemcpy(deviceDist, dist.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(devicePath, path.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    // Matrices stay distributed on the GPUs throughout computation. Only the
    // externally visible distances need copying back, and only for -r or -v.
    std::vector<unsigned int>().swap(path);
    std::vector<unsigned int>().swap(dist);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (tiles) floydWarshall(deviceDist, devicePath, pitch, firstTile, localTiles, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        // Floyd-Warshall has O(n³) complexity.
        const double ops = double(numNodes) * numNodes * numNodes;
        const double gflops = seconds > 0 ? ops / seconds / 1e9 : 0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> result;
    if (validate || printResults) {
        dist.resize(elements);
        if (elements) CUDA(cudaMemcpy(dist.data(), deviceDist, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        const size_t firstRow = firstTile * TILE;
        const size_t realRows = firstRow < numNodes ? std::min(rows, numNodes - firstRow) : 0;
        std::vector<unsigned int> packed(realRows * numNodes);
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < realRows; ++row)
            std::copy_n(dist.data() + row * pitch, numNodes, packed.data() + row * numNodes);
        if (rank == 0) result.resize(numNodes * numNodes);
        // Chunked point-to-point gather avoids MPI's signed-int displacement
        // limit and never replicates the global matrix on worker ranks.
        for (int source = 0; source < ranks; ++source) {
            const size_t begin = std::min(numNodes, (tiles * source / ranks) * TILE);
            const size_t end = std::min(numNodes, (tiles * (source + 1) / ranks) * TILE);
            const size_t count = (end - begin) * numNodes;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(size_t(INT_MAX), count - offset));
                if (rank == 0) {
                    if (source == 0) std::copy_n(packed.data() + offset, chunk, result.data() + begin * numNodes + offset);
                    else MPI_Recv(result.data() + begin * numNodes + offset, chunk, MPI_UNSIGNED, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank == source) {
                    MPI_Send(packed.data() + offset, chunk, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
                }
                offset += chunk;
            }
        }
    }
    CUDA(cudaFree(deviceDist));
    CUDA(cudaFree(devicePath));
    
    // Print results for external validation (integer hash-based)
    if (rank == 0 && printResults) {
        print_results_int(result, "DistanceMatrix");
    }
    
    // Validation
    int status = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
