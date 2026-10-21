#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

constexpr int FW_BLOCK_X = 128;
constexpr int FW_BLOCK_Y = 4;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void fwUpdateKernel(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const unsigned int* __restrict__ rowK,
                               const size_t rows,
                               const size_t n,
                               const unsigned int k) {
    __shared__ unsigned int colShared[FW_BLOCK_Y];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t li = static_cast<size_t>(blockIdx.y) * blockDim.y + ty;

    if (tx == 0 && li < rows) {
        colShared[ty] = dist[li * n + k];
    }
    __syncthreads();

    if (li >= rows) {
        return;
    }

    const unsigned int distIK = colShared[ty];
    const size_t rowOffset = li * n;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + tx; j < n; j += stride) {
        const size_t pos = rowOffset + j;
        const unsigned int distIJ = dist[pos];
        const unsigned int newDist = distIK + rowK[j];
        if (newDist < distIJ) {
            dist[pos] = newDist;
            path[pos] = k;
        }
    }
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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t rows,
                          const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(rows); ++li) {
        unsigned int* row = path.data() + static_cast<size_t>(li) * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = static_cast<unsigned int>(j);
        }
    }
}

struct DeviceState {
    int id;
    size_t rowStart;
    size_t rows;
    unsigned int* d_dist;
    unsigned int* d_path;
    unsigned int* d_row;
    cudaStream_t stream;
};

void floydWarshallHybridGPU(std::vector<unsigned int>& localDist,
                            std::vector<unsigned int>& localPath,
                            const size_t numNodes,
                            const size_t globalRowStart,
                            const int worldSize,
                            const std::vector<int>& devices,
                            double& timeStart,
                            double& timeEnd) {
    const size_t localRows = localDist.size() / numNodes;
    const size_t deviceCount = std::min<size_t>(
        devices.size(), std::max<size_t>(localRows, static_cast<size_t>(1)));

    std::vector<DeviceState> states(deviceCount);
    for (size_t t = 0; t < deviceCount; ++t) {
        states[t].id = devices[t];
        states[t].rowStart = (localRows * t) / deviceCount;
        states[t].rows = (localRows * (t + 1)) / deviceCount - states[t].rowStart;
        states[t].d_dist = nullptr;
        states[t].d_path = nullptr;
        states[t].d_row = nullptr;
        states[t].stream = nullptr;
    }

    const size_t rowBytes = numNodes * sizeof(unsigned int);
    unsigned int* rowHost = nullptr;
    cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&rowHost), rowBytes,
                            cudaHostAllocDefault),
              "cudaHostAlloc row");

    for (DeviceState& st : states) {
        cudaCheck(cudaSetDevice(st.id), "cudaSetDevice");
        cudaCheck(cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking),
                  "cudaStreamCreate");

        const size_t matrixBytes =
            std::max<size_t>(1, st.rows * numNodes) * sizeof(unsigned int);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&st.d_dist), matrixBytes),
                  "cudaMalloc dist");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&st.d_path), matrixBytes),
                  "cudaMalloc path");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&st.d_row), rowBytes),
                  "cudaMalloc row");

        if (st.rows > 0) {
            const size_t bytes = st.rows * numNodes * sizeof(unsigned int);
            cudaCheck(cudaMemcpyAsync(st.d_dist,
                                      localDist.data() + st.rowStart * numNodes,
                                      bytes, cudaMemcpyHostToDevice, st.stream),
                      "H2D dist");
            cudaCheck(cudaMemcpyAsync(st.d_path,
                                      localPath.data() + st.rowStart * numNodes,
                                      bytes, cudaMemcpyHostToDevice, st.stream),
                      "H2D path");
            cudaCheck(cudaStreamSynchronize(st.stream), "H2D sync");
        }
    }

    if (worldSize > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    timeStart = MPI_Wtime();

    for (size_t k = 0; k < numNodes; ++k) {
        if (k >= globalRowStart && k < globalRowStart + localRows && localRows > 0) {
            const size_t lk = k - globalRowStart;
            const size_t owner =
                ((lk + 1) * deviceCount - 1) / localRows;
            DeviceState& ost = states[owner];
            cudaCheck(cudaSetDevice(ost.id), "cudaSetDevice owner");
            cudaCheck(cudaMemcpyAsync(rowHost,
                                      ost.d_dist + (lk - ost.rowStart) * numNodes,
                                      rowBytes, cudaMemcpyDeviceToHost,
                                      ost.stream),
                      "D2H row");
            cudaCheck(cudaStreamSynchronize(ost.stream), "D2H row sync");
        }

        if (worldSize > 1) {
            const int owner = static_cast<int>(
                (static_cast<unsigned long long>(k + 1) * worldSize - 1) /
                numNodes);
            MPI_Bcast(rowHost, static_cast<int>(numNodes), MPI_UNSIGNED,
                      owner, MPI_COMM_WORLD);
        }

        for (DeviceState& st : states) {
            if (st.rows == 0) {
                continue;
            }
            cudaCheck(cudaSetDevice(st.id), "cudaSetDevice update");
            cudaCheck(cudaMemcpyAsync(st.d_row, rowHost, rowBytes,
                                      cudaMemcpyHostToDevice, st.stream),
                      "H2D row");

            const dim3 block(FW_BLOCK_X, FW_BLOCK_Y);
            const dim3 grid(
                static_cast<unsigned int>((numNodes + FW_BLOCK_X - 1) /
                                          FW_BLOCK_X),
                static_cast<unsigned int>((st.rows + FW_BLOCK_Y - 1) /
                                          FW_BLOCK_Y));
            fwUpdateKernel<<<grid, block, 0, st.stream>>>(
                st.d_dist, st.d_path, st.d_row, st.rows, numNodes,
                static_cast<unsigned int>(k));
            cudaCheck(cudaGetLastError(), "fwUpdateKernel launch");
        }

        for (DeviceState& st : states) {
            if (st.rows > 0) {
                cudaCheck(cudaSetDevice(st.id), "cudaSetDevice sync");
                cudaCheck(cudaStreamSynchronize(st.stream), "kernel sync");
            }
        }
    }

    if (worldSize > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    timeEnd = MPI_Wtime();

    for (DeviceState& st : states) {
        cudaCheck(cudaSetDevice(st.id), "cudaSetDevice finish");
        if (st.rows > 0) {
            const size_t bytes = st.rows * numNodes * sizeof(unsigned int);
            cudaCheck(cudaMemcpyAsync(localDist.data() + st.rowStart * numNodes,
                                      st.d_dist, bytes, cudaMemcpyDeviceToHost,
                                      st.stream),
                      "D2H dist");
            cudaCheck(cudaMemcpyAsync(localPath.data() + st.rowStart * numNodes,
                                      st.d_path, bytes, cudaMemcpyDeviceToHost,
                                      st.stream),
                      "D2H path");
            cudaCheck(cudaStreamSynchronize(st.stream), "D2H result sync");
        }

        cudaCheck(cudaFree(st.d_dist), "cudaFree dist");
        cudaCheck(cudaFree(st.d_path), "cudaFree path");
        cudaCheck(cudaFree(st.d_row), "cudaFree row");
        cudaCheck(cudaStreamDestroy(st.stream), "cudaStreamDestroy");
    }

    cudaCheck(cudaFreeHost(rowHost), "cudaFreeHost row");
}

void floydWarshallHybridCPU(std::vector<unsigned int>& localDist,
                            std::vector<unsigned int>& localPath,
                            const size_t numNodes,
                            const size_t globalRowStart,
                            const int worldSize,
                            double& timeStart,
                            double& timeEnd) {
    const size_t localRows = localDist.size() / numNodes;
    std::vector<unsigned int> rowHost(numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    timeStart = MPI_Wtime();

    for (size_t k = 0; k < numNodes; ++k) {
        if (k >= globalRowStart && k < globalRowStart + localRows) {
            std::memcpy(rowHost.data(),
                        localDist.data() + (k - globalRowStart) * numNodes,
                        numNodes * sizeof(unsigned int));
        }

        if (worldSize > 1) {
            const int owner = static_cast<int>(
                (static_cast<unsigned long long>(k + 1) * worldSize - 1) /
                numNodes);
            MPI_Bcast(rowHost.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                      owner, MPI_COMM_WORLD);
        }

        #pragma omp parallel for schedule(static)
        for (long long li = 0; li < static_cast<long long>(localRows); ++li) {
            const size_t row = static_cast<size_t>(li);
            const unsigned int distIK = localDist[row * numNodes + k];
            unsigned int* dRow = localDist.data() + row * numNodes;
            unsigned int* pRow = localPath.data() + row * numNodes;

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rowHost[j];
                if (newDist < dRow[j]) {
                    dRow[j] = newDist;
                    pRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timeEnd = MPI_Wtime();
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badOption = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            badOption = true;
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
            printf("Unknown command line option\n");
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

    const size_t globalRowStart =
        static_cast<size_t>((static_cast<unsigned long long>(numNodes) * rank) /
                            worldSize);
    const size_t globalRowEnd =
        static_cast<size_t>((static_cast<unsigned long long>(numNodes) *
                             (rank + 1)) /
                            worldSize);
    const size_t localRows = globalRowEnd - globalRowStart;

    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t start =
            static_cast<size_t>((static_cast<unsigned long long>(numNodes) * r) /
                                worldSize);
        const size_t end =
            static_cast<size_t>((static_cast<unsigned long long>(numNodes) *
                                 (r + 1)) /
                                worldSize);
        counts[r] = static_cast<int>((end - start) * numNodes);
        displs[r] = static_cast<int>(start * numNodes);
    }

    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }

    if (worldSize == 1) {
        initializeDistanceMatrix(localDist, numNodes, 1, MAX_DISTANCE);
    } else {
        std::vector<unsigned int> rootDist;
        if (rank == 0) {
            rootDist.resize(numNodes * numNodes);
            initializeDistanceMatrix(rootDist, numNodes, 1, MAX_DISTANCE);
        }
        if (numNodes > 0) {
            MPI_Scatterv(rootDist.data(), counts.data(), displs.data(),
                         MPI_UNSIGNED, localDist.data(),
                         static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                         0, MPI_COMM_WORLD);
        }
    }
    initializePathMatrix(localPath, localRows, numNodes);

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        deviceCount = 0;
    }

    std::vector<int> devices;
    if (deviceCount > 0) {
        int localRank = 0;
        int localSize = 1;
        MPI_Comm nodeComm = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &nodeComm);
        if (nodeComm != MPI_COMM_NULL) {
            MPI_Comm_rank(nodeComm, &localRank);
            MPI_Comm_size(nodeComm, &localSize);
        }

        int deviceStart = 0;
        int deviceEnd = 1;
        if (localSize >= deviceCount) {
            deviceStart = localRank % deviceCount;
            deviceEnd = deviceStart + 1;
        } else {
            const int base = deviceCount / localSize;
            const int rem = deviceCount % localSize;
            const int extra = (localRank < rem) ? 1 : 0;
            const int priorExtra = std::min(localRank, rem);
            deviceStart = localRank * base + priorExtra;
            deviceEnd = deviceStart + base + extra;
        }
        for (int d = deviceStart; d < deviceEnd; ++d) {
            devices.push_back(d);
        }

        if (nodeComm != MPI_COMM_NULL) {
            MPI_Comm_free(&nodeComm);
        }
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    double timeStart = MPI_Wtime();
    double timeEnd = timeStart;

    if (numNodes > 0) {
        if (!devices.empty()) {
            floydWarshallHybridGPU(localDist, localPath, numNodes,
                                   globalRowStart, worldSize, devices,
                                   timeStart, timeEnd);
        } else {
            floydWarshallHybridCPU(localDist, localPath, numNodes,
                                   globalRowStart, worldSize, timeStart,
                                   timeEnd);
        }
    }

    const double elapsedSeconds = timeEnd - timeStart;
    const double elapsedMilliseconds = elapsedSeconds * 1000.0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n",
               static_cast<long>(std::llround(elapsedMilliseconds)));

        // Floyd-Warshall has O(n^3) complexity
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops =
            (elapsedSeconds > 0.0) ? ops / elapsedSeconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gops);
    }

    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
    }
    if (numNodes > 0) {
        MPI_Gatherv(localDist.data(), static_cast<int>(localRows * numNodes),
                    MPI_UNSIGNED, rank == 0 ? fullDist.data() : nullptr,
                    counts.data(), displs.data(), MPI_UNSIGNED, 0,
                    MPI_COMM_WORLD);
    }

    int status = 0;
    if (rank == 0) {
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

