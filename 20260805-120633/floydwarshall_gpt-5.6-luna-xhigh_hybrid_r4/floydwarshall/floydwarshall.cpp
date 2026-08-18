#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The benchmark stores a matrix in column-major order.  Keeping this helper
// available to host and device code makes the indexing convention explicit.
#if defined(__CUDACC__)
#define FW_HOST_DEVICE __host__ __device__
#else
#define FW_HOST_DEVICE
#endif

FW_HOST_DEVICE inline constexpr std::size_t idx2(const std::size_t i,
                                                  const std::size_t j,
                                                  const std::size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void abortMpi(const char* message, const int rank) {
    std::fprintf(stderr, "Floyd-Warshall rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int error, const char* operation, const int rank) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char errorString[MPI_MAX_ERROR_STRING]{};
    int errorLength = 0;
    MPI_Error_string(error, errorString, &errorLength);
    std::fprintf(stderr, "Floyd-Warshall rank %d: MPI %s failed: %.*s\n",
                 rank, operation, errorLength, errorString);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation, const int rank) {
    if (error == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "Floyd-Warshall rank %d: CUDA %s failed: %s\n",
                 rank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

#define MPI_CHECK(call) checkMpi((call), #call, rank)
#define CUDA_CHECK(call, ...) checkCuda((call), #call, rank)

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const std::size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    // rand_r and its single seed are deliberately kept in the original order
    // so every MPI run receives exactly the same graph as the serial version.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (std::size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) /
                                            static_cast<double>(RAND_MAX));
    }

    // Set diagonal to zero (distance from node to itself is 0).
    for (std::size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path,
                          const std::size_t numNodes) {
    // For flat index j*n+i, the original two symmetric assignments both
    // produce j.  This is a deterministic, embarrassingly parallel setup.
    const auto elementCount = static_cast<std::int64_t>(path.size());
#pragma omp parallel for schedule(static)
    for (std::int64_t flat = 0; flat < elementCount; ++flat) {
        path[static_cast<std::size_t>(flat)] =
            static_cast<unsigned int>(static_cast<std::size_t>(flat) / numNodes);
    }
}

__global__ void floydWarshallKernel(unsigned int* dist,
                                    unsigned int* path,
                                    const unsigned int* pivotColumn,
                                    const std::size_t numNodes,
                                    const std::size_t localColumns,
                                    const std::size_t pivot) {
    // x walks down a column, giving coalesced loads/stores in the column-major
    // representation; y selects one of the columns owned by this MPI rank.
    const std::size_t destination =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t localColumn =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (destination >= numNodes || localColumn >= localColumns) {
        return;
    }

    const std::size_t columnOffset = localColumn * numNodes;
    const std::size_t element = columnOffset + destination;
    const unsigned int candidate =
        dist[columnOffset + pivot] + pivotColumn[destination];

    if (candidate < dist[element]) {
        dist[element] = candidate;
        path[element] = static_cast<unsigned int>(pivot);
    }
}

struct CudaBuffers {
    unsigned int* deviceDistance = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivot = nullptr;
    unsigned int* hostPivot[2] = {nullptr, nullptr};
    cudaStream_t stream = nullptr;
};

void allocateCudaBuffers(CudaBuffers& buffers,
                         const std::size_t localColumns,
                         const std::size_t numNodes,
                         const int rank) {
    const std::size_t localElements = localColumns * numNodes;
    const std::size_t matrixAllocation =
        std::max<std::size_t>(localElements, 1) * sizeof(unsigned int);
    const std::size_t pivotAllocation =
        std::max<std::size_t>(numNodes, 1) * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.deviceDistance),
                          matrixAllocation), rank);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePath),
                          matrixAllocation), rank);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePivot),
                          pivotAllocation), rank);
    for (unsigned int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaMallocHost(
                       reinterpret_cast<void**>(&buffers.hostPivot[buffer]),
                       pivotAllocation),
                   rank);
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking),
               rank);
}

void releaseCudaBuffers(CudaBuffers& buffers, const int rank) {
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream), rank);
    CUDA_CHECK(cudaStreamDestroy(buffers.stream), rank);
    for (unsigned int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaFreeHost(buffers.hostPivot[buffer]), rank);
    }
    CUDA_CHECK(cudaFree(buffers.devicePivot), rank);
    CUDA_CHECK(cudaFree(buffers.devicePath), rank);
    CUDA_CHECK(cudaFree(buffers.deviceDistance), rank);
}

void selectCudaDevice(const int rank) {
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localCommunicator),
             "MPI_Comm_split_type", rank);

    int localRank = 0;
    checkMpi(MPI_Comm_rank(localCommunicator, &localRank), "MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_free(&localCommunicator), "MPI_Comm_free", rank);

    int deviceCount = 0;
    const cudaError_t deviceQuery = cudaGetDeviceCount(&deviceCount);
    if (deviceQuery != cudaSuccess || deviceCount == 0) {
        std::fprintf(stderr,
                     "Floyd-Warshall rank %d: no usable CUDA device (%s)\n",
                     rank, cudaGetErrorString(deviceQuery));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // MPI ranks on one node are mapped round-robin over the node's visible
    // devices.  CUDA_VISIBLE_DEVICES can therefore control cluster placement.
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
}

void copyHostToDevice(const std::vector<unsigned int>& localDistance,
                      const std::vector<unsigned int>& localPath,
                      CudaBuffers& buffers,
                      const int rank) {
    if (!localDistance.empty()) {
        CUDA_CHECK(cudaMemcpy(buffers.deviceDistance, localDistance.data(),
                              localDistance.size() * sizeof(unsigned int),
                              cudaMemcpyHostToDevice), rank);
    }
    if (!localPath.empty()) {
        CUDA_CHECK(cudaMemcpy(buffers.devicePath, localPath.data(),
                              localPath.size() * sizeof(unsigned int),
                              cudaMemcpyHostToDevice), rank);
    }
}

void copyDeviceToHost(std::vector<unsigned int>& localDistance,
                      std::vector<unsigned int>& localPath,
                      CudaBuffers& buffers,
                      const int rank) {
    if (!localDistance.empty()) {
        CUDA_CHECK(cudaMemcpy(localDistance.data(), buffers.deviceDistance,
                              localDistance.size() * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost), rank);
    }
    if (!localPath.empty()) {
        CUDA_CHECK(cudaMemcpy(localPath.data(), buffers.devicePath,
                              localPath.size() * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost), rank);
    }
}

void floydWarshall(const std::size_t numNodes,
                   const std::size_t localColumns,
                   const std::vector<std::size_t>& columnDisplacements,
                   const std::vector<int>& pivotOwners,
                   const int rank,
                   CudaBuffers& buffers) {
    constexpr unsigned int threadsX = 32;
    constexpr unsigned int threadsY = 8;
    const dim3 block(threadsX, threadsY, 1);
    const dim3 grid(
        static_cast<unsigned int>((numNodes + threadsX - 1) / threadsX),
        static_cast<unsigned int>((localColumns + threadsY - 1) / threadsY),
        1);

    for (std::size_t pivot = 0; pivot < numNodes; ++pivot) {
        const int pivotOwner = pivotOwners[pivot];
        const unsigned int pivotBuffer = static_cast<unsigned int>(pivot & 1U);

        // The owner exposes the just-computed pivot column.  The stream is
        // synchronized only here, allowing the other ranks to continue their
        // previous update while the owner performs its D2H transfer.  The two
        // host buffers ensure MPI never overwrites a buffer still used by the
        // preceding asynchronous H2D transfer.
        if (pivotOwner == rank) {
            const std::size_t localPivot =
                pivot - columnDisplacements[static_cast<std::size_t>(rank)];
            CUDA_CHECK(cudaMemcpyAsync(
                           buffers.hostPivot[pivotBuffer],
                           buffers.deviceDistance + localPivot * numNodes,
                           numNodes * sizeof(unsigned int),
                           cudaMemcpyDeviceToHost, buffers.stream),
                       rank);
            CUDA_CHECK(cudaStreamSynchronize(buffers.stream), rank);
        }

        MPI_CHECK(MPI_Bcast(buffers.hostPivot[pivotBuffer],
                            static_cast<int>(numNodes), MPI_UNSIGNED,
                            pivotOwner, MPI_COMM_WORLD));

        CUDA_CHECK(cudaMemcpyAsync(buffers.devicePivot,
                                   buffers.hostPivot[pivotBuffer],
                                   numNodes * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, buffers.stream),
                   rank);

        if (localColumns != 0) {
            floydWarshallKernel<<<grid, block, 0, buffers.stream>>>(
                buffers.deviceDistance, buffers.devicePath,
                buffers.devicePivot, numNodes, localColumns, pivot);
            CUDA_CHECK(cudaGetLastError(), rank);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(buffers.stream), rank);
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const std::size_t numNodes) {
    int invalidDiagonal = 0;
#pragma omp parallel for reduction(| : invalidDiagonal) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(numNodes); ++i) {
        if (dist[idx2(static_cast<std::size_t>(i), static_cast<std::size_t>(i),
                      numNodes)] != 0) {
            invalidDiagonal = 1;
        }
    }
    if (invalidDiagonal != 0) {
        for (std::size_t i = 0; i < numNodes; ++i) {
            if (dist[idx2(i, i, numNodes)] != 0) {
                std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                            i, i);
                return false;
            }
        }
    }

    // Check a sample of paths to avoid O(n^3) validation time.  The sampled
    // source/destination pairs are independent and are checked by OpenMP.
    const std::size_t sample = std::min(numNodes, static_cast<std::size_t>(10));
    int invalidTriangle = 0;
#pragma omp parallel for collapse(2) reduction(| : invalidTriangle) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(sample); ++i) {
        for (std::int64_t j = 0; j < static_cast<std::int64_t>(sample); ++j) {
            const std::size_t source = static_cast<std::size_t>(i);
            const std::size_t destination = static_cast<std::size_t>(j);
            const unsigned int distIJ = dist[idx2(destination, source, numNodes)];
            for (std::size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIK = dist[idx2(k, source, numNodes)];
                const unsigned int distKJ = dist[idx2(destination, k, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    invalidTriangle = 1;
                }
            }
        }
    }
    if (invalidTriangle != 0) {
        std::printf("Validation failed: triangle inequality violated\n");
        return false;
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

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortMpi("MPI implementation does not provide MPI_THREAD_FUNNELED", rank);
    }

    omp_set_dynamic(0);
    selectCudaDevice(rank);

    std::size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortMpi("number of nodes exceeds the MPI count limit", rank);
    }
    if (numNodes != 0 &&
        numNodes > std::numeric_limits<std::size_t>::max() / numNodes) {
        abortMpi("matrix size overflows size_t", rank);
    }

    const std::size_t matrixElements = numNodes * numNodes;
    if (matrixElements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortMpi("matrix exceeds the MPI_Scatterv count limit", rank);
    }

    std::vector<std::size_t> columnCounts(static_cast<std::size_t>(worldSize));
    std::vector<std::size_t> columnDisplacements(
        static_cast<std::size_t>(worldSize));
    const std::size_t baseColumns = numNodes / static_cast<std::size_t>(worldSize);
    const std::size_t remainder = numNodes % static_cast<std::size_t>(worldSize);
    std::size_t displacement = 0;
    for (int r = 0; r < worldSize; ++r) {
        const std::size_t rankIndex = static_cast<std::size_t>(r);
        columnCounts[rankIndex] = baseColumns + (rankIndex < remainder ? 1 : 0);
        columnDisplacements[rankIndex] = displacement;
        displacement += columnCounts[rankIndex];
    }

    std::vector<int> mpiCounts(static_cast<std::size_t>(worldSize));
    std::vector<int> mpiDisplacements(static_cast<std::size_t>(worldSize));
    for (int r = 0; r < worldSize; ++r) {
        const std::size_t rankIndex = static_cast<std::size_t>(r);
        const std::size_t elementCount = columnCounts[rankIndex] * numNodes;
        const std::size_t elementDisplacement =
            columnDisplacements[rankIndex] * numNodes;
        if (elementCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            elementDisplacement > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            abortMpi("distributed matrix exceeds MPI integer displacement limits", rank);
        }
        mpiCounts[rankIndex] = static_cast<int>(elementCount);
        mpiDisplacements[rankIndex] = static_cast<int>(elementDisplacement);
    }

    std::vector<int> pivotOwners(numNodes, 0);
    for (int r = 0; r < worldSize; ++r) {
        const std::size_t rankIndex = static_cast<std::size_t>(r);
        for (std::size_t k = columnDisplacements[rankIndex];
             k < columnDisplacements[rankIndex] + columnCounts[rankIndex]; ++k) {
            pivotOwners[k] = r;
        }
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA: enabled\n");
    }

    std::vector<unsigned int> globalDistance;
    std::vector<unsigned int> globalPath;
    if (rank == 0) {
        globalDistance.resize(matrixElements);
        globalPath.resize(matrixElements);
        std::printf("Initializing graph...\n");
        initializeDistanceMatrix(globalDistance, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(globalPath, numNodes);
    }

    const std::size_t localColumns = columnCounts[static_cast<std::size_t>(rank)];
    const std::size_t localElements = localColumns * numNodes;
    std::vector<unsigned int> localDistance(localElements);
    std::vector<unsigned int> localPath(localElements);

    MPI_CHECK(MPI_Scatterv(rank == 0 ? globalDistance.data() : nullptr,
                           mpiCounts.data(), mpiDisplacements.data(), MPI_UNSIGNED,
                           localDistance.data(), mpiCounts[static_cast<std::size_t>(rank)],
                           MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(rank == 0 ? globalPath.data() : nullptr,
                           mpiCounts.data(), mpiDisplacements.data(), MPI_UNSIGNED,
                           localPath.data(), mpiCounts[static_cast<std::size_t>(rank)],
                           MPI_UNSIGNED, 0, MPI_COMM_WORLD));

    CudaBuffers buffers;
    allocateCudaBuffers(buffers, localColumns, numNodes, rank);

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }

    // Keep the measured interval focused on the distributed Floyd–Warshall
    // iterations.  The initial and final full-matrix transfers are setup and
    // collection work, not part of the O(n^3) computation.
    copyHostToDevice(localDistance, localPath, buffers, rank);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(numNodes, localColumns, columnDisplacements, pivotOwners,
                  rank, buffers);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    copyDeviceToHost(localDistance, localPath, buffers, rank);
    MPI_CHECK(MPI_Gatherv(localDistance.data(),
                          mpiCounts[static_cast<std::size_t>(rank)], MPI_UNSIGNED,
                          rank == 0 ? globalDistance.data() : nullptr,
                          mpiCounts.data(), mpiDisplacements.data(), MPI_UNSIGNED,
                          0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Gatherv(localPath.data(),
                          mpiCounts[static_cast<std::size_t>(rank)], MPI_UNSIGNED,
                          rank == 0 ? globalPath.data() : nullptr,
                          mpiCounts.data(), mpiDisplacements.data(), MPI_UNSIGNED,
                          0, MPI_COMM_WORLD));

    releaseCudaBuffers(buffers, rank);

    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", duration.count());

        const double seconds = duration.count() / 1000.0;
        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        const double gflops = seconds > 0.0 ? operations / seconds / 1e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(globalDistance, "DistanceMatrix");
        }

        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(globalDistance, numNodes);
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                MPI_CHECK(MPI_Finalize());
                return 1;
            }
        }
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
