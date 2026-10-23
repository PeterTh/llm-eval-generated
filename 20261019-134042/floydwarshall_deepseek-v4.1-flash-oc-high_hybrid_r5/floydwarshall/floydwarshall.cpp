#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t K_BLOCK = 64;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

inline void rowRange(const size_t n, const int rank, const int size,
                     size_t& begin, size_t& end) noexcept {
    const size_t s = static_cast<size_t>(size);
    const size_t base = n / s;
    const size_t rem = n % s;
    const size_t r = static_cast<size_t>(rank);
    begin = r * base + std::min(r, rem);
    end = begin + base + (r < rem ? 1 : 0);
}

__global__ void floydWarshallKernelScalar(unsigned int* __restrict__ dist,
                                          unsigned int* __restrict__ path,
                                          const size_t rows,
                                          unsigned int* __restrict__ pivot,
                                          unsigned int* __restrict__ pivotPath,
                                          const size_t pivotRows,
                                          const unsigned int* __restrict__ rowk,
                                          const unsigned int k,
                                          const size_t n) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= n) {
        return;
    }
    const size_t totalRows = rows + pivotRows;
    for (size_t t = blockIdx.y; t < totalRows; t += gridDim.y) {
        unsigned int* d;
        unsigned int* p;
        size_t base;
        if (t < rows) {
            base = t * n;
            d = dist;
            p = path;
        } else {
            base = (t - rows) * n;
            d = pivot;
            p = pivotPath;
        }
        const unsigned int dik = d[base + k];
        const unsigned int newDist = dik + rowk[j];
        const size_t id = base + j;
        if (newDist < d[id]) {
            d[id] = newDist;
            p[id] = static_cast<unsigned int>(k);
        }
    }
}

__global__ void floydWarshallKernelVec4(unsigned int* __restrict__ dist,
                                        unsigned int* __restrict__ path,
                                        const size_t rows,
                                        unsigned int* __restrict__ pivot,
                                        unsigned int* __restrict__ pivotPath,
                                        const size_t pivotRows,
                                        const unsigned int* __restrict__ rowk,
                                        const unsigned int k,
                                        const size_t n) {
    const size_t j = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (j >= n) {
        return;
    }
    const size_t totalRows = rows + pivotRows;
    for (size_t t = blockIdx.y; t < totalRows; t += gridDim.y) {
        unsigned int* d;
        unsigned int* p;
        size_t base;
        if (t < rows) {
            base = t * n;
            d = dist;
            p = path;
        } else {
            base = (t - rows) * n;
            d = pivot;
            p = pivotPath;
        }
        const unsigned int dik = d[base + k];

        uint4 dv = *reinterpret_cast<uint4*>(d + base + j);
        const uint4 r = *reinterpret_cast<const uint4*>(rowk + j);

        unsigned int mask = 0;
        unsigned int t2;

        t2 = dik + r.x;
        if (t2 < dv.x) { dv.x = t2; mask |= 1u; }
        t2 = dik + r.y;
        if (t2 < dv.y) { dv.y = t2; mask |= 2u; }
        t2 = dik + r.z;
        if (t2 < dv.z) { dv.z = t2; mask |= 4u; }
        t2 = dik + r.w;
        if (t2 < dv.w) { dv.w = t2; mask |= 8u; }

        if (mask) {
            uint4 pv = *reinterpret_cast<uint4*>(p + base + j);
            if (mask & 1u) { pv.x = k; }
            if (mask & 2u) { pv.y = k; }
            if (mask & 4u) { pv.z = k; }
            if (mask & 8u) { pv.w = k; }
            *reinterpret_cast<uint4*>(d + base + j) = dv;
            *reinterpret_cast<uint4*>(p + base + j) = pv;
        }
    }
}

void initializeDistanceMatrixLocal(std::vector<unsigned int>& dist,
                                   const size_t numNodes,
                                   const size_t rowStart,
                                   const size_t rows,
                                   const unsigned int rangeMin,
                                   const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t skip = rowStart * numNodes;
    for (size_t i = 0; i < skip; ++i) {
        rand_r(&seed);
    }

    for (size_t r = 0; r < rows; ++r) {
        unsigned int* row = dist.data() + r * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
        }
        row[rowStart + r] = 0;
    }
}

void initializePathMatrixLocal(std::vector<unsigned int>& path,
                               const size_t numNodes,
                               const size_t rowStart,
                               const size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        unsigned int* row = path.data() + r * numNodes;
        const unsigned int value = static_cast<unsigned int>(rowStart + r);
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = value;
        }
    }
}

void floydWarshall(unsigned int* d_dist,
                   unsigned int* d_path,
                   const size_t numNodes,
                   const size_t rows,
                   const size_t rowStart,
                   const int rank,
                   const int size,
                   unsigned int* d_pivot,
                   unsigned int* d_pivotPath,
                   unsigned int* h_pivot,
                   unsigned int* h_send) {
    const bool vec4 = (numNodes % 4 == 0);
    const unsigned int scalarBlock = 256;
    const unsigned int vecBlock = 128;
    const size_t vecGroups = (numNodes + 3) / 4;

    auto launch = [&](const unsigned int* rowk, const size_t k, const size_t pivotRows) {
        const size_t totalRows = rows + pivotRows;
        if (totalRows == 0) {
            return;
        }
        const unsigned int gridY = static_cast<unsigned int>(
            std::min<size_t>(totalRows, 65535));
        if (vec4) {
            const unsigned int gridX = static_cast<unsigned int>(
                (vecGroups + vecBlock - 1) / vecBlock);
            dim3 grid(gridX, gridY);
            floydWarshallKernelVec4<<<grid, vecBlock>>>(
                d_dist, d_path, rows, d_pivot, d_pivotPath, pivotRows,
                rowk, static_cast<unsigned int>(k), numNodes);
        } else {
            const unsigned int gridX = static_cast<unsigned int>(
                (numNodes + scalarBlock - 1) / scalarBlock);
            dim3 grid(gridX, gridY);
            floydWarshallKernelScalar<<<grid, scalarBlock>>>(
                d_dist, d_path, rows, d_pivot, d_pivotPath, pivotRows,
                rowk, static_cast<unsigned int>(k), numNodes);
        }
    };

    size_t myBegin = 0;
    size_t myEnd = 0;
    rowRange(numNodes, rank, size, myBegin, myEnd);

    std::vector<int> counts(size);
    std::vector<int> displacements(size);

    for (size_t kStart = 0; kStart < numNodes; kStart += K_BLOCK) {
        const size_t kEnd = std::min(kStart + K_BLOCK, numNodes);
        const size_t pivotRows = kEnd - kStart;

        const size_t interStart = std::max(myBegin, kStart);
        const size_t interEnd = std::min(myEnd, kEnd);
        const size_t sendCount = interStart < interEnd
            ? (interEnd - interStart) * numNodes
            : 0;

        if (sendCount > 0) {
            cudaCheck(cudaMemcpy(h_send,
                                 d_dist + (interStart - rowStart) * numNodes,
                                 sendCount * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost),
                      "gather pivot rows to host");
        }

        for (int p = 0; p < size; ++p) {
            size_t begin = 0;
            size_t finish = 0;
            rowRange(numNodes, p, size, begin, finish);
            const size_t start = std::max(begin, kStart);
            const size_t stop = std::min(finish, kEnd);
            counts[p] = static_cast<int>(start < stop ? (stop - start) * numNodes : 0);
            displacements[p] = static_cast<int>((start - kStart) * numNodes);
        }

        MPI_Allgatherv(h_send, static_cast<int>(sendCount), MPI_UNSIGNED,
                       h_pivot, counts.data(), displacements.data(), MPI_UNSIGNED,
                       MPI_COMM_WORLD);

        cudaCheck(cudaMemcpy(d_pivot, h_pivot,
                             pivotRows * numNodes * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "send pivot rows to device");

        for (size_t k = kStart; k < kEnd; ++k) {
            const unsigned int* rowk = d_pivot + (k - kStart) * numNodes;
            launch(rowk, k, pivotRows);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badOption = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            badOption = true;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (badOption) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "device count");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "set device");

    size_t rowStart = 0;
    size_t rowEnd = 0;
    rowRange(numNodes, rank, size, rowStart, rowEnd);
    const size_t localRows = rowEnd - rowStart;
    const size_t localCount = std::max<size_t>(localRows * numNodes, 1);

    const size_t pivotCount = std::max<size_t>(K_BLOCK * numNodes, 1);

    std::vector<unsigned int> distLocal(localCount);
    std::vector<unsigned int> pathLocal(localCount);
    std::vector<unsigned int> pivotHost(pivotCount);
    std::vector<unsigned int> pivotSend(pivotCount);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrixLocal(distLocal, numNodes, rowStart, localRows, 1, MAX_DISTANCE);
    initializePathMatrixLocal(pathLocal, numNodes, rowStart, localRows);

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_pivot = nullptr;
    unsigned int* d_pivotPath = nullptr;
    cudaCheck(cudaMalloc(&d_dist, localCount * sizeof(unsigned int)), "alloc dist");
    cudaCheck(cudaMalloc(&d_path, localCount * sizeof(unsigned int)), "alloc path");
    cudaCheck(cudaMalloc(&d_pivot, pivotCount * sizeof(unsigned int)), "alloc pivot");
    cudaCheck(cudaMalloc(&d_pivotPath, pivotCount * sizeof(unsigned int)), "alloc pivot path");
    cudaCheck(cudaMemset(d_pivotPath, 0, pivotCount * sizeof(unsigned int)), "init pivot path");

    const size_t localBytes = localRows * numNodes * sizeof(unsigned int);
    if (localBytes > 0) {
        cudaCheck(cudaMemcpy(d_dist, distLocal.data(), localBytes, cudaMemcpyHostToDevice), "copy dist");
        cudaCheck(cudaMemcpy(d_path, pathLocal.data(), localBytes, cudaMemcpyHostToDevice), "copy path");
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(d_dist, d_path, numNodes, localRows, rowStart, rank, size,
                  d_pivot, d_pivotPath, pivotHost.data(), pivotSend.data());

    cudaCheck(cudaDeviceSynchronize(), "final synchronize");
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    const bool needFull = printResults || validate;
    std::vector<unsigned int> fullDist;

    if (needFull) {
        if (localBytes > 0) {
            cudaCheck(cudaMemcpy(distLocal.data(), d_dist, localBytes,
                                 cudaMemcpyDeviceToHost),
                      "copy result");
        }

        std::vector<int> counts(size);
        std::vector<int> displacements(size);
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            size_t begin = 0;
            size_t finish = 0;
            rowRange(numNodes, p, size, begin, finish);
            const int count = static_cast<int>((finish - begin) * numNodes);
            counts[p] = count;
            displacements[p] = offset;
            offset += count;
        }

        if (rank == 0) {
            fullDist.assign(numNodes * numNodes, 0);
        }

        if (numNodes > 0) {
            MPI_Gatherv(distLocal.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                        rank == 0 ? fullDist.data() : nullptr,
                        counts.data(), displacements.data(), MPI_UNSIGNED,
                        0, MPI_COMM_WORLD);
        }
    }

    // Print results for external validation (integer hash-based)
    if (printResults) {
        if (rank == 0) {
            print_results_int(fullDist, "DistanceMatrix");
        }
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(fullDist, numNodes) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    cudaFree(d_dist);
    cudaFree(d_path);
    cudaFree(d_pivot);
    cudaFree(d_pivotPath);
    MPI_Comm_free(&localComm);
    MPI_Finalize();

    return valid ? 0 : 1;
}
