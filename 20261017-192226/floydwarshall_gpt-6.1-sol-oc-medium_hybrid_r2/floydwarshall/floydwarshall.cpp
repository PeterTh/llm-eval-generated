#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <cerrno>
#include <limits>
#include <exception>
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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// A panel contains the pivot rows as they stood at their individual k steps,
// NOT their values at the end of the block. This preserves even the original
// strict-improvement path/tie semantics, unlike an ordinary blocked algorithm.
constexpr int TILE = 32;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char text[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(error, text, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, text);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunking avoids MPI's int count limit for large distributed matrices.
void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) mpiCheck(MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD));
        else mpiCheck(MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk;
        count -= chunk;
    }
}

__global__ void makePanel(const unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ panel, size_t width,
                          size_t localPivot, size_t pivot) {
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int rows[TILE][TILE];
    const int x = threadIdx.x;
    const size_t column = static_cast<size_t>(blockIdx.x) * TILE + x;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
        diagonal[y][x] = dist[(localPivot + y) * width + pivot + x];
        rows[y][x] = dist[(localPivot + y) * width + column];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        if (threadIdx.y == 0) panel[static_cast<size_t>(k) * width + column] = rows[k][x];
        for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
            const unsigned int via = diagonal[y][k];
            // The pivot row and column cannot improve (positive weights, zero
            // diagonal). Not writing them also avoids shared-memory races.
            if (y != k) rows[y][x] = min(rows[y][x], via + rows[k][x]);
            if (y != k && x != k)
                diagonal[y][x] = min(diagonal[y][x], via + diagonal[k][x]);
        }
        __syncthreads();
    }
}

// Compute the matching column snapshots once per local block row. Reusing them
// across all output column tiles removes loop barriers from the O(n^3) kernel.
__global__ void makeColumns(const unsigned int* __restrict__ dist,
                            const unsigned int* __restrict__ panel,
                            unsigned int* __restrict__ columns,
                            size_t width, size_t pivot) {
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int diagonal[TILE][TILE];
    int x = threadIdx.x;
    size_t rowBase = static_cast<size_t>(blockIdx.x) * TILE;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
        left[y][x] = dist[(rowBase + y) * width + pivot + x];
        diagonal[y][x] = panel[static_cast<size_t>(y) * width + pivot + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
            unsigned int via = left[y][k];
            if (x == k) columns[(rowBase + y) * TILE + k] = via;
            else left[y][x] = min(left[y][x], via + diagonal[k][x]);
        }
        __syncthreads();
    }
}

__global__ void applyPanel(unsigned int* __restrict__ dist,
                           unsigned int* __restrict__ path,
                           const unsigned int* __restrict__ panel,
                           const unsigned int* __restrict__ pivotColumns,
                           size_t width, size_t pivot) {
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    const int x = threadIdx.x;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE;
    const size_t column = static_cast<size_t>(blockIdx.x) * TILE + x;
    unsigned int value[4], intermediate[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int y = threadIdx.y + t * 8;
        left[y][x] = pivotColumns[(rowBase + y) * TILE + x];
        right[y][x] = panel[static_cast<size_t>(y) * width + column];
        value[t] = dist[(rowBase + y) * width + column];
        intermediate[t] = path[(rowBase + y) * width + column];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int y = threadIdx.y + t * 8;
            unsigned int via = left[y][k];
            unsigned int candidate = via + right[k][x];
            if (candidate < value[t]) {
                value[t] = candidate;
                intermediate[t] = static_cast<unsigned int>(pivot + k);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        size_t offset = (rowBase + threadIdx.y + t * 8) * width + column;
        dist[offset] = value[t];
        path[offset] = intermediate[t];
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // For each source node i
        for (size_t i = 0; i < numNodes; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
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

int main(int argc, char** argv) try {
    int provided, rank, ranks;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* argument = argv[++i];
            unsigned long long parsed = strtoull(argument, &end, 10);
            if (errno || argument[0] == '-' || !*argument || *end ||
                parsed > UINT_MAX - TILE ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) /
                         std::max(1ULL, parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argument);
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
    
    // Node-local rank selects a GPU, including when CUDA_VISIBLE_DEVICES is set
    // separately for each process by the job scheduler.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &localComm));
    int localRank, devices;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&localComm));

    const size_t blocks = (numNodes + TILE - 1) / TILE;
    const size_t width = blocks * TILE;
    auto firstBlock = [=](int r) {
        return blocks / ranks * r + std::min(static_cast<size_t>(r), blocks % ranks);
    };
    const size_t firstRow = firstBlock(rank) * TILE;
    const size_t rows = (firstBlock(rank + 1) - firstBlock(rank)) * TILE;
    const size_t elements = rows * width;
    std::vector<unsigned int> local(elements), paths(elements);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < elements; ++i) paths[i] = static_cast<unsigned int>(firstRow + i / width);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\nValidation: %s\n", numNodes, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA tile: %d\n",
               ranks, omp_get_max_threads(), TILE);
        printf("Initializing graph...\n");
        // Stream the original rand_r sequence once, retaining only a rank's
        // portion on the root. Padding never consumes random numbers.
        unsigned int seed = 42;
        for (int r = 0; r < ranks; ++r) {
            size_t begin = firstBlock(r) * TILE;
            size_t countRows = (firstBlock(r + 1) - firstBlock(r)) * TILE;
            std::vector<unsigned int> buffer(r == 0 ? 0 : countRows * width);
            unsigned int* data = r == 0 ? local.data() : buffer.data();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < countRows * width; ++i) data[i] = INF;
            for (size_t y = 0; y < countRows; ++y) {
                if (begin + y < numNodes) {
                    for (size_t x = 0; x < numNodes; ++x)
                        data[y * width + x] = 1 + static_cast<unsigned int>(
                            static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
                }
                data[y * width + begin + y] = 0;
            }
            if (r != 0) transfer(data, countRows * width, r, true);
        }
    } else transfer(local.data(), elements, 0, false);

    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    unsigned int *devicePanel = nullptr, *deviceColumns = nullptr, *hostPanel = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (elements) {
        CUDA_CHECK(cudaMalloc(&deviceDist, elements * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&devicePath, elements * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&deviceColumns, rows * TILE * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpyAsync(deviceDist, local.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(devicePath, paths.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    }
    if (width) {
        CUDA_CHECK(cudaMalloc(&devicePanel, TILE * width * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&hostPanel, TILE * width * sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Loading the kernels is setup, rather than part of the timed computation.
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, makePanel));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, makeColumns));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, applyPanel));
    if (rank == 0) printf("Computing shortest paths...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    int owner = 0;
    for (size_t b = 0; b < blocks; ++b) {
        while (firstBlock(owner + 1) <= b) ++owner;
        size_t pivot = b * TILE;
        if (rank == owner) {
            makePanel<<<static_cast<unsigned int>(blocks), dim3(TILE, 8), 0, stream>>>(
                deviceDist, devicePanel, width, pivot - firstRow, pivot);
            CUDA_CHECK(cudaGetLastError());
            if (ranks > 1)
                CUDA_CHECK(cudaMemcpyAsync(hostPanel, devicePanel, TILE * width * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        }
        // Also waits for the previous local update before consuming a new panel.
        if (ranks > 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (size_t offset = 0; offset < TILE * width;) {
                int count = static_cast<int>(std::min(TILE * width - offset, static_cast<size_t>(INT_MAX)));
                mpiCheck(MPI_Bcast(hostPanel + offset, count, MPI_UNSIGNED, owner, MPI_COMM_WORLD));
                offset += count;
            }
        }
        if (elements) {
            if (rank != owner)
                CUDA_CHECK(cudaMemcpyAsync(devicePanel, hostPanel, TILE * width * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
            // Snapshot the evolving pivot columns before output overwrites them.
            makeColumns<<<static_cast<unsigned int>(rows / TILE), dim3(TILE, 8), 0, stream>>>(
                deviceDist, devicePanel, deviceColumns, width, pivot);
            CUDA_CHECK(cudaGetLastError());
            // Grid-stride batches respect CUDA's y-dimension limit.
            for (size_t rb = 0; rb < rows / TILE; rb += 65535) {
                unsigned int batch = static_cast<unsigned int>(std::min(rows / TILE - rb, size_t(65535)));
                applyPanel<<<dim3(static_cast<unsigned int>(blocks), batch), dim3(TILE, 8), 0, stream>>>(
                    deviceDist + rb * TILE * width, devicePath + rb * TILE * width,
                    devicePanel, deviceColumns + rb * TILE * TILE, width, pivot);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double elapsed = MPI_Wtime() - start, seconds;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GOPS\n", seconds > 0 ?
               static_cast<double>(numNodes) * numNodes * numNodes / seconds / 1e9 : 0.0);
    }

    int status = 0;
    if (printResults || validate) {
        bool exact = validate && numNodes <= 256;
        if (elements) {
            CUDA_CHECK(cudaMemcpy(local.data(), deviceDist, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (exact) CUDA_CHECK(cudaMemcpy(paths.data(), devicePath, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        size_t realRows = firstRow < numNodes ? std::min(rows, numNodes - firstRow) : 0;
        std::vector<unsigned int> packed(realRows * numNodes), packedPaths(exact ? realRows * numNodes : 0);
        #pragma omp parallel for schedule(static)
        for (size_t y = 0; y < realRows; ++y) {
            std::copy_n(local.data() + y * width, numNodes, packed.data() + y * numNodes);
            if (exact) std::copy_n(paths.data() + y * width, numNodes, packedPaths.data() + y * numNodes);
        }
        if (rank == 0) {
            std::vector<unsigned int> dist(numNodes * numNodes), path(exact ? numNodes * numNodes : 0);
            std::copy(packed.begin(), packed.end(), dist.begin());
            if (exact) std::copy(packedPaths.begin(), packedPaths.end(), path.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t begin = std::min(numNodes, firstBlock(r) * TILE);
                size_t end = std::min(numNodes, firstBlock(r + 1) * TILE);
                if (end > begin) {
                    transfer(dist.data() + begin * numNodes, (end - begin) * numNodes, r, false);
                    if (exact) transfer(path.data() + begin * numNodes, (end - begin) * numNodes, r, false);
                }
            }
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(dist, numNodes);
                if (exact) {
                    std::vector<unsigned int> reference(numNodes * numNodes), referencePath(numNodes * numNodes);
                    initializeDistanceMatrix(reference, numNodes, 1, MAX_DISTANCE);
                    initializePathMatrix(referencePath, numNodes);
                    floydWarshall(reference, referencePath, numNodes);
                    valid = valid && reference == dist && referencePath == path;
                    printf("Exact serial distance/path comparison: %s\n", valid ? "PASSED" : "FAILED");
                }
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        } else if (!packed.empty()) {
            transfer(packed.data(), packed.size(), 0, true);
            if (exact) transfer(packedPaths.data(), packedPaths.size(), 0, true);
        }
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(devicePanel));
    CUDA_CHECK(cudaFree(deviceColumns));
    if (hostPanel) CUDA_CHECK(cudaFreeHost(hostPanel));
    CUDA_CHECK(cudaStreamDestroy(stream));
    mpiCheck(MPI_Finalize());
    return status;
} catch (const std::exception& error) {
    fprintf(stderr, "Floyd-Warshall failed: %s\n", error.what());
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
}
