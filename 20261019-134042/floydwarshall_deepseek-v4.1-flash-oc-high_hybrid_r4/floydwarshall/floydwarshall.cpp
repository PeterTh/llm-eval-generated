#include <algorithm>
#include <chrono>
#include <climits>
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

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int err__ = (call);                                                     \
        if (err__ != MPI_SUCCESS) {                                             \
            char msg__[MPI_MAX_ERROR_STRING];                                   \
            int len__ = 0;                                                      \
            MPI_Error_string(err__, msg__, &len__);                             \
            fprintf(stderr, "MPI error %.*s at %s:%d\n",                        \
                    len__, msg__, __FILE__, __LINE__);                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowStart, const size_t localRows) {
#pragma omp parallel for schedule(static)
    for (size_t li = 0; li < localRows; ++li) {
        unsigned int* row = path.data() + li * numNodes;
        const unsigned int value = static_cast<unsigned int>(rowStart + li);
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = value;
        }
    }
}

__global__ void fwDiagonalKernel(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 const int n, const int S, const int B,
                                 const int kk, const unsigned int kGlobal) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int ii = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii >= B || c >= B) {
        return;
    }

    const size_t row = static_cast<size_t>(ii) * n;
    const size_t pos = row + S + c;
    const unsigned int dik = dist[row + S + kk];
    const unsigned int dkj = dist[static_cast<size_t>(kk) * n + S + c];
    const unsigned int dij = dist[pos];
    const unsigned int nd = dik + dkj;

    if (nd < dij) {
        dist[pos] = nd;
        path[pos] = kGlobal;
    }
}

__global__ void fwPivotColumnKernel(const unsigned int* __restrict__ dist,
                                    const unsigned int* __restrict__ panelD,
                                    unsigned int* __restrict__ tmp,
                                    unsigned int* __restrict__ path,
                                    const int n, const int localRows,
                                    const int S, const int B) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= localRows || c >= B) {
        return;
    }

    const size_t row = static_cast<size_t>(i) * n;
    unsigned int best = dist[row + S + c];
    int bestT = -1;

    for (int t = 0; t < B; ++t) {
        const unsigned int nd = dist[row + S + t] + panelD[static_cast<size_t>(t) * B + c];
        if (nd < best) {
            best = nd;
            bestT = t;
        }
    }

    tmp[static_cast<size_t>(i) * B + c] = best;
    if (bestT >= 0) {
        path[row + S + c] = static_cast<unsigned int>(S + bestT);
    }
}

__global__ void fwPivotRowKernel(const unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ tmp,
                                 unsigned int* __restrict__ path,
                                 const int n, const int S, const int B) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int ii = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii >= B || j >= n) {
        return;
    }

    const size_t row = static_cast<size_t>(ii) * n;
    unsigned int best = dist[row + j];
    int bestT = -1;

    if (j < S || j >= S + B) {
        for (int t = 0; t < B; ++t) {
            const unsigned int nd = dist[row + S + t] + dist[static_cast<size_t>(t) * n + j];
            if (nd < best) {
                best = nd;
                bestT = t;
            }
        }
        if (bestT >= 0) {
            path[row + j] = static_cast<unsigned int>(S + bestT);
        }
    }

    tmp[row + j] = best;
}

__global__ void fwRemainingKernel(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ panelRows,
                                  const int n, const int localRows,
                                  const int S, const int B, const int outside) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= localRows || q >= outside) {
        return;
    }

    const int j = (q < S) ? q : q + B;
    const size_t row = static_cast<size_t>(i) * n;
    unsigned int best = dist[row + j];
    int bestT = -1;

    for (int t = 0; t < B; ++t) {
        const unsigned int nd = dist[row + S + t] + panelRows[static_cast<size_t>(t) * n + j];
        if (nd < best) {
            best = nd;
            bestT = t;
        }
    }

    if (bestT >= 0) {
        dist[row + j] = best;
        path[row + j] = static_cast<unsigned int>(S + bestT);
    }
}

static void bcastUnsigned(unsigned int* data, size_t count, int root, MPI_Comm comm) {
    while (count > 0) {
        const int chunk = static_cast<int>(std::min<size_t>(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_UNSIGNED, root, comm));
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static void runDiagonalUpdate(unsigned int* d_dist, unsigned int* d_path,
                              const size_t n, const size_t S, const size_t B) {
    if (B == 0) {
        return;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((B + 15) / 16),
                    static_cast<unsigned int>((B + 15) / 16));
    for (size_t kk = 0; kk < B; ++kk) {
        fwDiagonalKernel<<<grid, block>>>(d_dist, d_path,
                                          static_cast<int>(n), static_cast<int>(S),
                                          static_cast<int>(B), static_cast<int>(kk),
                                          static_cast<unsigned int>(S + kk));
    }
}

void floydWarshall(unsigned int* d_dist, unsigned int* d_path,
                   const size_t numNodes, const int rank, const int numRanks,
                   const size_t localRows, unsigned int* d_panelD,
                   unsigned int* d_panelRows, unsigned int* d_tmp,
                   unsigned int* h_panelD, unsigned int* h_panelRows) {
    const bool multi = (numRanks > 1);
    const dim3 block(16, 16);

    for (int b = 0; b < numRanks; ++b) {
        const size_t S = static_cast<size_t>(b) * numNodes / static_cast<size_t>(numRanks);
        const size_t E = static_cast<size_t>(b + 1) * numNodes / static_cast<size_t>(numRanks);
        const size_t B = E - S;
        if (B == 0) {
            continue;
        }

        if (B == numNodes) {
            if (rank == 0) {
                runDiagonalUpdate(d_dist, d_path, numNodes, S, B);
            }
            continue;
        }

        if (!multi) {
            continue;
        }

        if (rank == b) {
            runDiagonalUpdate(d_dist, d_path, numNodes, S, B);
            CUDA_CHECK(cudaMemcpy2D(h_panelD, B * sizeof(unsigned int),
                                    d_dist + S, numNodes * sizeof(unsigned int),
                                    B * sizeof(unsigned int), B,
                                    cudaMemcpyDeviceToHost));
        }

        bcastUnsigned(h_panelD, B * B, b, MPI_COMM_WORLD);

        if (rank != b) {
            CUDA_CHECK(cudaMemcpy(d_panelD, h_panelD,
                                  B * B * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            if (localRows > 0) {
                const dim3 gridCols(static_cast<unsigned int>((B + 15) / 16),
                                    static_cast<unsigned int>((localRows + 15) / 16));
                fwPivotColumnKernel<<<gridCols, block>>>(d_dist, d_panelD, d_tmp,
                                                         d_path, static_cast<int>(numNodes),
                                                         static_cast<int>(localRows),
                                                         static_cast<int>(S),
                                                         static_cast<int>(B));
                CUDA_CHECK(cudaMemcpy2D(d_dist + S, numNodes * sizeof(unsigned int),
                                        d_tmp, B * sizeof(unsigned int),
                                        B * sizeof(unsigned int), localRows,
                                        cudaMemcpyDeviceToDevice));
            }
        } else {
            const dim3 gridRows(static_cast<unsigned int>((numNodes + 15) / 16),
                                static_cast<unsigned int>((B + 15) / 16));
            fwPivotRowKernel<<<gridRows, block>>>(d_dist, d_tmp, d_path,
                                                  static_cast<int>(numNodes),
                                                  static_cast<int>(S),
                                                  static_cast<int>(B));
            CUDA_CHECK(cudaMemcpy(d_dist, d_tmp, B * numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(h_panelRows, d_tmp, B * numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        bcastUnsigned(h_panelRows, B * numNodes, b, MPI_COMM_WORLD);

        if (rank != b) {
            CUDA_CHECK(cudaMemcpy(d_panelRows, h_panelRows,
                                  B * numNodes * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            const size_t outside = numNodes - B;
            if (outside > 0 && localRows > 0) {
                const dim3 gridRem(static_cast<unsigned int>((outside + 15) / 16),
                                   static_cast<unsigned int>((localRows + 15) / 16));
                fwRemainingKernel<<<gridRem, block>>>(d_dist, d_path, d_panelRows,
                                                      static_cast<int>(numNodes),
                                                      static_cast<int>(localRows),
                                                      static_cast<int>(S),
                                                      static_cast<int>(B),
                                                      static_cast<int>(outside));
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    bool valid = true;
#pragma omp parallel for schedule(static) reduction(&& : valid)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        bool rowValid = true;
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        rowValid = false;
                    }
                }
            }
        }
        valid = valid && rowValid;
    }

    if (!valid) {
        printf("Validation failed: triangle inequality violated\n");
        return false;
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
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0;
    int numRanks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t localStart = static_cast<size_t>(rank) * numNodes / static_cast<size_t>(numRanks);
    const size_t localEnd = static_cast<size_t>(rank + 1) * numNodes / static_cast<size_t>(numRanks);
    const size_t localRows = localEnd - localStart;
    const size_t localElements = localRows * numNodes;
    const size_t maxBlock = (numNodes + static_cast<size_t>(numRanks) - 1) /
                            static_cast<size_t>(numRanks);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    }

    const size_t hostElems = std::max<size_t>(localElements, 1);
    unsigned int* h_localDist = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_localDist),
                              hostElems * sizeof(unsigned int)));

    std::vector<int> counts(static_cast<size_t>(numRanks));
    std::vector<int> displs(static_cast<size_t>(numRanks));
    for (int b = 0; b < numRanks; ++b) {
        const size_t s = static_cast<size_t>(b) * numNodes / static_cast<size_t>(numRanks);
        const size_t e = static_cast<size_t>(b + 1) * numNodes / static_cast<size_t>(numRanks);
        const size_t elems = (e - s) * numNodes;
        counts[static_cast<size_t>(b)] = static_cast<int>(elems);
        displs[static_cast<size_t>(b)] = static_cast<int>(s * numNodes);
    }

    MPI_CHECK(MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr,
                           counts.data(), displs.data(), MPI_UNSIGNED,
                           h_localDist, static_cast<int>(localElements), MPI_UNSIGNED,
                           0, MPI_COMM_WORLD));
    fullDist.clear();
    fullDist.shrink_to_fit();

    std::vector<unsigned int> localPath(localElements);
    initializePathMatrix(localPath, numNodes, localStart, localRows);

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_panelD = nullptr;
    unsigned int* d_panelRows = nullptr;
    unsigned int* d_tmp = nullptr;
    unsigned int* h_panelD = nullptr;
    unsigned int* h_panelRows = nullptr;

    const size_t gpuElems = std::max<size_t>(localElements, 1);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dist), gpuElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_path), gpuElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_dist, h_localDist, localElements * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, localPath.data(), localElements * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    localPath.clear();
    localPath.shrink_to_fit();

    if (numRanks > 1) {
        const size_t panelElems = std::max<size_t>(maxBlock * numNodes, 1);
        const size_t dElems = std::max<size_t>(maxBlock * maxBlock, 1);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_panelD), dElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_panelRows), panelElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp), panelElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_panelD), dElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_panelRows), panelElems * sizeof(unsigned int)));
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(d_dist, d_path, numNodes, rank, numRanks, localRows,
                  d_panelD, d_panelRows, d_tmp, h_panelD, h_panelRows);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double seconds = static_cast<double>(duration.count()) / 1000.0;
        const double gflops = (seconds > 0.0) ? ops / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    CUDA_CHECK(cudaMemcpy(h_localDist, d_dist, localElements * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());

    std::vector<unsigned int> gathered;
    if (rank == 0) {
        gathered.resize(numNodes * numNodes);
    }

    MPI_CHECK(MPI_Gatherv(h_localDist, static_cast<int>(localElements), MPI_UNSIGNED,
                          rank == 0 ? gathered.data() : nullptr,
                          counts.data(), displs.data(), MPI_UNSIGNED,
                          0, MPI_COMM_WORLD));

    int valid = 1;
    if (rank == 0) {
        if (printResults) {
            print_results_int(gathered, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(gathered, numNodes);
            valid = ok ? 1 : 0;
            if (ok) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    if (numRanks > 1) {
        CUDA_CHECK(cudaFree(d_panelD));
        CUDA_CHECK(cudaFree(d_panelRows));
        CUDA_CHECK(cudaFree(d_tmp));
        CUDA_CHECK(cudaFreeHost(h_panelD));
        CUDA_CHECK(cudaFreeHost(h_panelRows));
    }
    CUDA_CHECK(cudaFreeHost(h_localDist));

    if (nodeComm != MPI_COMM_NULL) {
        MPI_CHECK(MPI_Comm_free(&nodeComm));
    }
    MPI_CHECK(MPI_Finalize());

    return valid ? 0 : 1;
}
