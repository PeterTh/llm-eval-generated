#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Each rank owns a contiguous block of rows i of the distance matrix.  For a
// fixed k the classic in-place update is
//
//     D[i][j] = min(D[i][j], D[i][k] + D[k][j])
//
// The rank that owns row k broadcasts it once, then every rank applies the
// update to all of its own rows in parallel on its GPU.  D[i][k] is local to
// the rank, so no other communication is required.
__global__ void fwIterationKernel(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ rowk,
                                  const int n, const int localN,
                                  const int k) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int li = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || li >= localN) {
        return;
    }

    const size_t rowOffset = static_cast<size_t>(li) * static_cast<size_t>(n);
    const unsigned int distIK = dist[rowOffset + static_cast<size_t>(k)];
    const unsigned int newDist = distIK + rowk[j];
    const size_t idx = rowOffset + static_cast<size_t>(j);
    if (newDist < dist[idx]) {
        dist[idx] = newDist;
        path[idx] = static_cast<unsigned int>(k);
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
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
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

int ownerOf(const std::vector<int>& starts, const int k) {
    // starts[r] is the first row owned by rank r.
    const auto it = std::upper_bound(starts.begin(), starts.end(), k);
    return static_cast<int>(it - starts.begin()) - 1;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badOption = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            badOption = true;
        }
    }

    if (badOption) {
        MPI_Finalize();
        return 1;
    }
    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    // Bind one MPI rank to a local GPU.  On nodes with more ranks than GPUs,
    // ranks share GPUs (the distribution still remains correct).
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device found; this build requires an accelerator.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(nodeRank % deviceCount), "cudaSetDevice");

    const int n = static_cast<int>(std::min<size_t>(numNodes, INT_MAX));

    // Balanced contiguous partition of the rows of the matrix.
    std::vector<int> starts(nranks, 0);
    std::vector<int> counts(nranks, 0);
    std::vector<int> displs(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        const long long s = static_cast<long long>(n) * r / nranks;
        const long long e = static_cast<long long>(n) * (r + 1) / nranks;
        starts[r] = static_cast<int>(s);
        counts[r] = static_cast<int>(e - s);
        displs[r] = static_cast<int>(s);
    }
    const int iStart = starts[rank];
    const int localN = counts[rank];

    std::vector<int> countsElems(nranks, 0);
    std::vector<int> displsElems(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        countsElems[r] = counts[r] * n;
        displsElems[r] = displs[r] * n;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        fflush(stdout);
    }

    // Rank 0 generates the reference matrix in the same order as the original
    // code (this is a sequential rand_r stream), then scatters row blocks.
    std::vector<unsigned int> sendBuf;
    if (rank == 0) {
        const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
        sendBuf.resize(total ? total : 1);
        if (total > 0) {
            initializeDistanceMatrix(sendBuf, static_cast<size_t>(n), 1, MAX_DISTANCE);
        }
    }

    const size_t localElems = static_cast<size_t>(localN) * static_cast<size_t>(n);
    std::vector<unsigned int> localDist(localElems ? localElems : 1);
    MPI_Scatterv(rank == 0 ? sendBuf.data() : nullptr,
                 countsElems.data(), displsElems.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localElems), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    sendBuf.clear();
    sendBuf.shrink_to_fit();

    // Initial path matrix: path[i*n + j] = i.
    std::vector<unsigned int> localPath(localElems ? localElems : 1);
    if (localN > 0 && n > 0) {
#pragma omp parallel for schedule(static)
        for (int li = 0; li < localN; ++li) {
            const unsigned int value = static_cast<unsigned int>(iStart + li);
            unsigned int* dst = localPath.data() + static_cast<size_t>(li) * n;
            for (int j = 0; j < n; ++j) {
                dst[j] = value;
            }
        }
    }

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dRow = nullptr;
    const size_t localBytes = (localElems ? localElems : 1) * sizeof(unsigned int);
    checkCuda(cudaMalloc(&dDist, localBytes), "cudaMalloc dist");
    checkCuda(cudaMalloc(&dPath, localBytes), "cudaMalloc path");
    checkCuda(cudaMalloc(&dRow, static_cast<size_t>(n ? n : 1) * sizeof(unsigned int)),
              "cudaMalloc row");

    checkCuda(cudaMemcpy(dDist, localDist.data(), localElems * sizeof(unsigned int),
                         cudaMemcpyHostToDevice), "copy initial dist");
    checkCuda(cudaMemcpy(dPath, localPath.data(), localElems * sizeof(unsigned int),
                         cudaMemcpyHostToDevice), "copy initial path");
    localDist.clear();
    localDist.shrink_to_fit();
    localPath.clear();
    localPath.shrink_to_fit();

    unsigned int* hRow = nullptr;
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&hRow),
                            static_cast<size_t>(n ? n : 1) * sizeof(unsigned int),
                            cudaHostAllocDefault), "cudaHostAlloc row");

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    if (rank == 0) {
        printf("Computing shortest paths...\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    const dim3 block(128, 4, 1);
    for (int k = 0; k < n; ++k) {
        const int owner = ownerOf(starts, k);

        // The owner of row k packs the full row; every rank needs it.
        if (rank == owner && n > 0) {
            checkCuda(cudaMemcpyAsync(hRow,
                                      dDist + static_cast<size_t>(k - iStart) * n,
                                      static_cast<size_t>(n) * sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost, stream),
                      "pack row");
        }

        // All ranks must finish the previous kernel (and the owner its D2H
        // copy) before the pinned receive buffer is overwritten by the
        // broadcast and before the next row is uploaded.
        checkCuda(cudaStreamSynchronize(stream), "stream sync before broadcast");

        MPI_Bcast(hRow, n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (n > 0) {
            checkCuda(cudaMemcpyAsync(dRow, hRow,
                                      static_cast<size_t>(n) * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice, stream),
                      "upload row");
        }

        if (localN > 0 && n > 0) {
            const dim3 grid((static_cast<unsigned int>(n) + block.x - 1) / block.x,
                            (static_cast<unsigned int>(localN) + block.y - 1) / block.y,
                            1);
            fwIterationKernel<<<grid, block, 0, stream>>>(dDist, dPath, dRow,
                                                          n, localN, k);
        }
    }

    checkCuda(cudaStreamSynchronize(stream), "final stream sync");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) *
                     static_cast<double>(numNodes);
        double gflops = ops / (static_cast<double>(duration.count()) / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        fflush(stdout);
    }

    // Gather the completed row blocks back to rank 0.
    std::vector<unsigned int> hostLocal(localElems ? localElems : 1);
    if (localElems > 0) {
        checkCuda(cudaMemcpy(hostLocal.data(), dDist, localElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "copy final dist");
    }

    std::vector<unsigned int> result;
    if (rank == 0) {
        result.resize(static_cast<size_t>(n) * static_cast<size_t>(n));
    }
    MPI_Gatherv(localElems > 0 ? hostLocal.data() : nullptr,
                static_cast<int>(localElems), MPI_UNSIGNED,
                (rank == 0 && !result.empty()) ? result.data() : nullptr,
                countsElems.data(), displsElems.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    checkCuda(cudaStreamDestroy(stream), "stream destroy");
    checkCuda(cudaFreeHost(hRow), "free row");
    checkCuda(cudaFree(dRow), "free dRow");
    checkCuda(cudaFree(dPath), "free dPath");
    checkCuda(cudaFree(dDist), "free dDist");

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(result, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(result, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        fflush(stdout);
    }

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
