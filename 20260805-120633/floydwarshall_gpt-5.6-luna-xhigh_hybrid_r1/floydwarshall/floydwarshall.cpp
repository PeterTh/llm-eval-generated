#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int CUDA_BLOCK_X = 32;
constexpr int CUDA_BLOCK_Y = 8;

int g_mpiRank = 0;

// The benchmark stores the conceptual row i, column j at i*n+j.  Keeping this
// helper preserves the original API and its column/row argument convention.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void abortCuda(const cudaError_t error, const char* expression,
                            const char* file, const int line) {
    std::fprintf(stderr, "MPI rank %d: CUDA error in %s (%s:%d): %s\n",
                 g_mpiRank, expression, file, line, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cudaStatus = (expression); \
        if (cudaStatus != cudaSuccess) { \
            abortCuda(cudaStatus, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

// One block covers a 32-column by 8-row tile.  Threads in a warp access a
// contiguous part of each row, and the source-to-k distance is loaded once per
// row into shared memory instead of being reread by every column thread.
__global__ void floydWarshallKernel(unsigned int* const dist,
                                     unsigned int* const path,
                                     const unsigned int* const pivot,
                                     const int localRows,
                                     const int numNodes,
                                     const int k) {
    __shared__ unsigned int sourceDistance[CUDA_BLOCK_Y];

    const int column = static_cast<int>(blockIdx.x) * CUDA_BLOCK_X +
                       static_cast<int>(threadIdx.x);
    const int localRow = static_cast<int>(blockIdx.y) * CUDA_BLOCK_Y +
                         static_cast<int>(threadIdx.y);

    if (threadIdx.x == 0 && localRow < localRows) {
        sourceDistance[threadIdx.y] = dist[static_cast<size_t>(localRow) * numNodes + k];
    }
    __syncthreads();

    if (localRow < localRows && column < numNodes) {
        const size_t element = static_cast<size_t>(localRow) * numNodes + column;
        const unsigned int newDistance = sourceDistance[threadIdx.y] + pivot[column];
        if (newDistance < dist[element]) {
            dist[element] = newDistance;
            path[element] = static_cast<unsigned int>(k);
        }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    // rand_r's sequence is part of the benchmark's input contract.  Generate
    // it in its original order, then use OpenMP for the independent diagonal
    // fix-up.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    // Set diagonal to zero (distance from node to itself is 0).
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(numNodes); ++i) {
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                               const size_t firstRow, const size_t localRows) {
    // The original two assignments reduce to path[row][column] == row for
    // every element, including the diagonal.  Initialize only this rank's
    // rows and retain the same flattened layout.
#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t localRow = 0;
         localRow < static_cast<std::ptrdiff_t>(localRows); ++localRow) {
        for (std::ptrdiff_t column = 0;
             column < static_cast<std::ptrdiff_t>(numNodes); ++column) {
            path[static_cast<size_t>(localRow) * numNodes + static_cast<size_t>(column)] =
                static_cast<unsigned int>(firstRow + static_cast<size_t>(localRow));
        }
    }
}

void makeRowPartition(const size_t numNodes, const int worldSize,
                      std::vector<int>& rowCounts,
                      std::vector<int>& rowOffsets,
                      std::vector<int>& elementCounts,
                      std::vector<int>& elementOffsets,
                      std::vector<int>& ownerForNode) {
    rowCounts.resize(static_cast<size_t>(worldSize));
    rowOffsets.resize(static_cast<size_t>(worldSize));
    elementCounts.resize(static_cast<size_t>(worldSize));
    elementOffsets.resize(static_cast<size_t>(worldSize));
    ownerForNode.resize(numNodes);

    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);
    size_t firstRow = 0;
    size_t firstElement = 0;
    for (int rank = 0; rank < worldSize; ++rank) {
        const size_t rows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
        const size_t elements = rows * numNodes;
        rowCounts[static_cast<size_t>(rank)] = static_cast<int>(rows);
        rowOffsets[static_cast<size_t>(rank)] = static_cast<int>(firstRow);
        elementCounts[static_cast<size_t>(rank)] = static_cast<int>(elements);
        elementOffsets[static_cast<size_t>(rank)] = static_cast<int>(firstElement);
        for (size_t row = firstRow; row < firstRow + rows; ++row) {
            ownerForNode[row] = rank;
        }
        firstRow += rows;
        firstElement += elements;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    int valid = 1;

    // Diagonal should be zero.
#pragma omp parallel for reduction(&:valid) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(numNodes); ++i) {
        if (dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] != 0) {
            valid = 0;
        }
    }
    if (valid == 0) {
        for (size_t i = 0; i < numNodes; ++i) {
            if (dist[idx2(i, i, numNodes)] != 0) {
                std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                            i, i);
                return false;
            }
        }
    }

    // Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j].
    // Check a sample of paths to avoid O(n^3) validation time.
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
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpiRank);

    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (g_mpiRank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0') {
                if (g_mpiRank == 0) {
                    std::fprintf(stderr, "Invalid node count: %s\n", argv[i]);
                }
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (g_mpiRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (g_mpiRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // MPI counts/displacements and CUDA kernel indices are int-sized.  This
    // also prevents an allocation that cannot be represented by MPI_Scatterv.
    const size_t maxRowsForMpi = static_cast<size_t>(INT_MAX) / std::max<size_t>(numNodes, 1);
    if (numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes > maxRowsForMpi || numNodes * numNodes > static_cast<size_t>(INT_MAX)) {
        if (g_mpiRank == 0) {
            std::fprintf(stderr, "Node count %zu is too large (maximum supported is %zu)\n",
                         numNodes, std::min(static_cast<size_t>(INT_MAX), maxRowsForMpi));
        }
        MPI_Finalize();
        return 1;
    }

    // MPI_Comm_split_type gives a stable local rank on each node, so separate
    // MPI ranks on a multi-GPU node select different accelerators.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_mpiRank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (g_mpiRank == 0) {
            std::fprintf(stderr, "No CUDA device is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    std::vector<int> rowCounts;
    std::vector<int> rowOffsets;
    std::vector<int> elementCounts;
    std::vector<int> elementOffsets;
    std::vector<int> ownerForNode;
    makeRowPartition(numNodes, worldSize, rowCounts, rowOffsets,
                     elementCounts, elementOffsets, ownerForNode);

    const int localRows = rowCounts[static_cast<size_t>(g_mpiRank)];
    const int firstRow = rowOffsets[static_cast<size_t>(g_mpiRank)];
    const int localElements = elementCounts[static_cast<size_t>(g_mpiRank)];
    const size_t localElementCount = static_cast<size_t>(localElements);

    if (g_mpiRank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA device/rank: %d\n",
                    worldSize, omp_get_max_threads(), localRank % deviceCount);
    }

    // Rank zero creates the exact original graph sequence.  It is distributed
    // by rows so each rank can keep its working set on its local accelerator.
    std::vector<unsigned int> dist;
    if (g_mpiRank == 0) {
        dist.resize(numNodes * numNodes);
        std::printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localElementCount);
    std::vector<unsigned int> localPath(localElementCount);
    initializeLocalPathMatrix(localPath, numNodes, static_cast<size_t>(firstRow),
                              static_cast<size_t>(localRows));

    MPI_Scatterv(g_mpiRank == 0 ? dist.data() : nullptr,
                 elementCounts.data(), elementOffsets.data(), MPI_UNSIGNED,
                 localDist.empty() ? nullptr : localDist.data(), localElements,
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivot = nullptr;
    unsigned int* hostPivot = nullptr;
    cudaStream_t stream = nullptr;

    const size_t allocationElements = std::max<size_t>(localElementCount, 1);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                          allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                          allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePivot),
                          numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostPivot),
                              numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (localElementCount != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceDist, localDist.data(),
                                   localElementCount * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(devicePath, localPath.data(),
                                   localElementCount * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (g_mpiRank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    const dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y, 1);
    const dim3 grid((static_cast<unsigned int>(numNodes) + CUDA_BLOCK_X - 1) /
                        CUDA_BLOCK_X,
                    (static_cast<unsigned int>(localRows) + CUDA_BLOCK_Y - 1) /
                        CUDA_BLOCK_Y,
                    1);

    for (size_t k = 0; k < numNodes; ++k) {
        const size_t localPivotOffset =
            localRows > 0 && k >= static_cast<size_t>(firstRow)
                ? (k - static_cast<size_t>(firstRow)) * numNodes
                : 0;

        // With one rank, the pivot is already on the GPU and stream ordering
        // provides the phase dependency; avoid unnecessary PCIe traffic.
        if (worldSize == 1) {
            if (localRows > 0) {
                floydWarshallKernel<<<grid, block, 0, stream>>>(
                    deviceDist, devicePath, deviceDist + localPivotOffset,
                    localRows, static_cast<int>(numNodes), static_cast<int>(k));
                CUDA_CHECK(cudaGetLastError());
            }
            continue;
        }

        const int pivotOwner = ownerForNode[k];
        const unsigned int* kernelPivot = devicePivot;

        // hostPivot is reused by MPI_Bcast.  All ranks must finish consuming
        // the previous phase's pinned buffer before the next collective can
        // overwrite it.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // The owner copies the just-completed pivot row from its GPU.
        if (g_mpiRank == pivotOwner) {
            CUDA_CHECK(cudaMemcpyAsync(hostPivot, deviceDist + localPivotOffset,
                                       numNodes * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // This rank already has the broadcast row in deviceDist.  The
            // generated graph has nonnegative weights, so row k cannot be
            // improved during its own phase (dist[k][k] is zero).
            kernelPivot = deviceDist + localPivotOffset;
        }
        MPI_Bcast(hostPivot, static_cast<int>(numNodes), MPI_UNSIGNED,
                  pivotOwner, MPI_COMM_WORLD);

        if (localRows > 0 && g_mpiRank != pivotOwner) {
            CUDA_CHECK(cudaMemcpyAsync(devicePivot, hostPivot,
                                       numNodes * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream));
        }
        if (localRows > 0) {
            floydWarshallKernel<<<grid, block, 0, stream>>>(
                deviceDist, devicePath, kernelPivot, localRows,
                static_cast<int>(numNodes), static_cast<int>(k));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto localDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDurationMilliseconds = localDuration.count();
    long globalDurationMilliseconds = 0;
    MPI_Reduce(&localDurationMilliseconds, &globalDurationMilliseconds, 1,
               MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (g_mpiRank == 0) {
        std::printf("Computation time: %ld ms\n", globalDurationMilliseconds);
        const double seconds =
            std::max(static_cast<double>(globalDurationMilliseconds) / 1000.0, 1.0e-9);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        std::printf("Performance: %.3f GOPS\n", ops / seconds / 1.0e9);
    }

    if (printResults || validate) {
        if (localElementCount != 0) {
            CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist,
                                  localElementCount * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Gatherv(localDist.empty() ? nullptr : localDist.data(), localElements,
                    MPI_UNSIGNED, g_mpiRank == 0 ? dist.data() : nullptr,
                    elementCounts.data(), elementOffsets.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    if (g_mpiRank == 0 && printResults) {
        print_results_int(dist, "DistanceMatrix");
    }

    int valid = 1;
    if (g_mpiRank == 0 && validate) {
        std::printf("Validating result...\n");
        valid = validateResult(dist, numNodes) ? 1 : 0;
        std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
    }
    if (validate) {
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFreeHost(hostPivot));
    CUDA_CHECK(cudaFree(devicePivot));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return validate && valid == 0 ? 1 : 0;
}
