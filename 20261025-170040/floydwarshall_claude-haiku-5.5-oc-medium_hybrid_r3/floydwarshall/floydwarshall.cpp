#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Each rank splits its rows between its GPU and OpenMP host threads, which
// update their parts concurrently. The split is calibrated at runtime from the
// measured throughput of each side during the first kCalibSteps steps.
constexpr size_t kCalibSteps = 16;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// First row owned by `rank` in a block-row distribution of n rows over `size` ranks.
static size_t rowStart(const size_t n, const int rank, const int size) {
    return n * static_cast<size_t>(rank) / static_cast<size_t>(size);
}

// Generates the full random matrix in the same sequence as the original code,
// but stores only the rows [r0, r1) owned by this rank.
void initializeDistanceRows(std::vector<unsigned int>& local, const size_t numNodes,
                            const size_t r0, const size_t r1,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes; ++i) {
        const bool owned = (i >= r0 && i < r1);
        unsigned int* row = owned ? local.data() + (i - r0) * numNodes : nullptr;
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int v =
                rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (owned) {
                row[j] = v;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = r0; i < r1; ++i) {
        local[(i - r0) * numNodes + i] = 0;
    }
}

// Equivalent to initializePathMatrix: path[s][d] = s for every s, d.
void initializePathRows(std::vector<unsigned int>& local, const size_t numNodes,
                        const size_t r0, const size_t r1) {
    for (size_t i = r0; i < r1; ++i) {
        std::fill_n(local.data() + (i - r0) * numNodes, numNodes, static_cast<unsigned int>(i));
    }
}

// GPU rank-1 update for pivot k over `rows` local rows. Row k and column k are
// unchanged during step k (diagonal is zero), so this matches the sequential order.
__global__ void fwStepGpu(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ pivot,
                          const size_t n, const size_t rows, const unsigned int k) {
    const size_t j = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t i = blockIdx.y * static_cast<size_t>(blockDim.y) + threadIdx.y;
    if (i >= rows || j >= n) {
        return;
    }
    unsigned int* const drow = dist + i * n;
    const unsigned int dik = drow[k];
    const unsigned int newDist = dik + pivot[j];
    if (newDist < drow[j]) {
        drow[j] = newDist;
        path[i * n + j] = k;
    }
}

// OpenMP host update for pivot k over local rows [begin, end).
static void fwStepHost(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                       const unsigned int* __restrict__ pivot, const size_t n,
                       const size_t begin, const size_t end, const unsigned int k) {
#pragma omp parallel for schedule(static)
    for (size_t li = begin; li < end; ++li) {
        unsigned int* __restrict__ drow = dist + li * n;
        unsigned int* __restrict__ prow = path + li * n;
        const unsigned int dik = drow[k];
        for (size_t j = 0; j < n; ++j) {
            const unsigned int newDist = dik + pivot[j];
            const unsigned int oldDist = drow[j];
            const bool better = newDist < oldDist;
            drow[j] = better ? newDist : oldDist;
            prow[j] = better ? k : prow[j];
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // One GPU per rank: ranks on the same node are bound to distinct devices.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Split the hardware threads of the node among the ranks sharing it.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    const size_t n = numNodes;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Block-row partition: rank r owns rows [starts[r], starts[r+1]).
    std::vector<size_t> starts(static_cast<size_t>(size) + 1);
    for (int r = 0; r < size; ++r) {
        starts[r] = rowStart(n, r, size);
    }
    starts[size] = n;

    const size_t r0 = starts[rank];
    const size_t r1 = starts[rank + 1];
    const size_t localRows = r1 - r0;
    // Local rows [0, gpuRows) live on the GPU; [gpuRows, localRows) live on the host.
    // The GPU always gets at least one row when the rank owns rows.
    size_t gpuRows = localRows == 0 ? 0 : std::max<size_t>(1, (localRows + 1) / 2);

    std::vector<unsigned int> hostDist(localRows * n);
    std::vector<unsigned int> hostPath(localRows * n);
    initializeDistanceRows(hostDist, n, r0, r1, 1, MAX_DISTANCE);
    initializePathRows(hostPath, n, r0, r1);

    // Device buffers have room for all local rows so the split can be rebalanced.
    unsigned int* devDist = nullptr;
    unsigned int* devPath = nullptr;
    unsigned int* devPivot = nullptr;
    unsigned int* pivotHost = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t evStart = nullptr;
    cudaEvent_t evStop = nullptr;
    CUDA_CHECK(cudaMallocHost(&pivotHost, std::max<size_t>(n, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    if (localRows > 0) {
        const size_t capBytes = localRows * n * sizeof(unsigned int);
        CUDA_CHECK(cudaMalloc(&devDist, capBytes));
        CUDA_CHECK(cudaMalloc(&devPath, capBytes));
        CUDA_CHECK(cudaMalloc(&devPivot, n * sizeof(unsigned int)));
        const size_t bytes = gpuRows * n * sizeof(unsigned int);
        CUDA_CHECK(cudaMemcpy(devDist, hostDist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devPath, hostPath.data(), bytes, cudaMemcpyHostToDevice));
    }

    // Moves the GPU/host boundary to `newGpuRows`, copying the rows that change side.
    auto setGpuRows = [&](const size_t newGpuRows) {
        if (newGpuRows > gpuRows) {
            const size_t off = gpuRows * n;
            const size_t cnt = (newGpuRows - gpuRows) * n * sizeof(unsigned int);
            CUDA_CHECK(cudaMemcpy(devDist + off, hostDist.data() + off, cnt, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(devPath + off, hostPath.data() + off, cnt, cudaMemcpyHostToDevice));
        } else if (newGpuRows < gpuRows) {
            const size_t off = newGpuRows * n;
            const size_t cnt = (gpuRows - newGpuRows) * n * sizeof(unsigned int);
            CUDA_CHECK(cudaMemcpy(hostDist.data() + off, devDist + off, cnt, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostPath.data() + off, devPath + off, cnt, cudaMemcpyDeviceToHost));
        }
        gpuRows = newGpuRows;
    };

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    auto ownerOf = [&](const size_t k) {
        return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), k) - starts.begin()) - 1;
    };

    const dim3 block(32, 8);

    // Calibration accumulators (GPU time measured with events, host time with a clock).
    float calibGpuMs = 0.0f;
    double calibCpuMs = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < n; ++k) {
        const unsigned int kk = static_cast<unsigned int>(k);
        const int owner = ownerOf(k);
        const bool calibrating = (k >= 1 && k <= kCalibSteps);

        // The owner provides pivot row k (row k is stable during step k).
        if (owner == rank) {
            const size_t lr = k - r0;
            if (lr < gpuRows) {
                CUDA_CHECK(cudaMemcpyAsync(pivotHost, devDist + lr * n, n * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                std::memcpy(pivotHost, hostDist.data() + lr * n, n * sizeof(unsigned int));
            }
        }
        MPI_Bcast(pivotHost, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // GPU rows (async) and host rows (OpenMP) are updated concurrently.
        if (gpuRows > 0) {
            const dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x),
                            static_cast<unsigned int>((gpuRows + block.y - 1) / block.y));
            CUDA_CHECK(cudaMemcpyAsync(devPivot, pivotHost, n * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream));
            if (calibrating) CUDA_CHECK(cudaEventRecord(evStart, stream));
            fwStepGpu<<<grid, block, 0, stream>>>(devDist, devPath, devPivot, n, gpuRows, kk);
            if (calibrating) CUDA_CHECK(cudaEventRecord(evStop, stream));
        }
        const auto hostBegin = std::chrono::steady_clock::now();
        fwStepHost(hostDist.data(), hostPath.data(), pivotHost, n, gpuRows, localRows, kk);
        const auto hostEnd = std::chrono::steady_clock::now();
        if (gpuRows > 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (calibrating) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
                calibGpuMs += ms;
            }
        }
        if (calibrating && localRows > gpuRows) {
            calibCpuMs += std::chrono::duration<double, std::milli>(hostEnd - hostBegin).count();
        }

        // After the calibration window, move the GPU/host boundary so both sides
        // finish a step in about the same time (rows/time as measured above).
        if (k == kCalibSteps && gpuRows > 0 && localRows > gpuRows &&
            calibGpuMs > 0.0f && calibCpuMs > 0.0) {
            const double gpuRate = static_cast<double>(gpuRows) / calibGpuMs;
            const double cpuRate = static_cast<double>(localRows - gpuRows) / calibCpuMs;
            const double ideal = static_cast<double>(localRows) * gpuRate / (gpuRate + cpuRate);
            const size_t target = std::clamp<size_t>(
                static_cast<size_t>(std::llround(ideal)), 1, localRows);
            setGpuRows(target);
        }
    }

    if (gpuRows > 0) {
        const size_t bytes = gpuRows * n * sizeof(unsigned int);
        CUDA_CHECK(cudaMemcpy(hostDist.data(), devDist, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostPath.data(), devPath, bytes, cudaMemcpyDeviceToHost));
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full result matrices on rank 0.
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(n), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(starts[r + 1] - starts[r]);
        displs[r] = static_cast<int>(starts[r]);
    }
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (rank == 0) {
        dist.resize(n * n);
        path.resize(n * n);
    }
    MPI_Gatherv(hostDist.data(), static_cast<int>(localRows), rowType,
                dist.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
    MPI_Gatherv(hostPath.data(), static_cast<int>(localRows), rowType,
                path.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&rowType);

    if (devDist) CUDA_CHECK(cudaFree(devDist));
    if (devPath) CUDA_CHECK(cudaFree(devPath));
    if (devPivot) CUDA_CHECK(cudaFree(devPivot));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(pivotHost));

    int exitCode = 0;
    if (rank == 0) {
        const long durationMs = static_cast<long>(maxMs);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
