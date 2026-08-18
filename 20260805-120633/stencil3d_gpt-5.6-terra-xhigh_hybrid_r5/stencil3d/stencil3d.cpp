#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 65535;
constexpr int kTagFirstPlane = 1101;
constexpr int kTagLastPlane = 1102;

[[noreturn]] void abortWithMessage(const int rank, const char* const message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* const expression, const int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank, expression,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

void checkMpi(const int error, const char* const expression, const int rank) {
    if (error != MPI_SUCCESS) {
        char errorString[MPI_MAX_ERROR_STRING]{};
        int errorLength = 0;
        MPI_Error_string(error, errorString, &errorLength);
        std::fprintf(stderr, "Rank %d: MPI failure in %s: %.*s\n", rank, expression,
                     errorLength, errorString);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, rank)
#define MPI_CHECK(expression) checkMpi((expression), #expression, rank)

inline int blocksFor(const size_t elements) {
    if (elements == 0) {
        return 0;
    }
    const size_t requested = (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return static_cast<int>(std::min(requested, static_cast<size_t>(kMaxBlocks)));
}

bool parseSize(const char* const text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* const text, int& iterations) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    iterations = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* const progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Each local allocation has a leading and a trailing z halo.  The logical local
// planes are therefore [1, localNz], while planes 0 and localNz + 1 are halos.
__global__ void initializeKernel(Real* const grid, const size_t planeElements,
                                 const size_t localNz, const size_t globalStart) {
    const size_t localElements = planeElements * localNz;
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < localElements;
         linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t localZ = linear / planeElements;
        const size_t inPlane = linear - localZ * planeElements;
        const size_t globalIndex = (globalStart + localZ) * planeElements + inPlane;
        grid[(localZ + 1) * planeElements + inPlane] = static_cast<Real>(globalIndex % 19);
    }
}

// Update planes that only reference locally owned z planes.  Keeping these in a
// separate launch lets their execution overlap the host-staged MPI halo transfer.
__global__ void stencilCoreKernel(const Real* const input, Real* const output,
                                  const size_t nx, const size_t ny,
                                  const size_t planeElements, const size_t corePlanes) {
    const size_t workItems = planeElements * corePlanes;
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < workItems;
         linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t coreZ = linear / planeElements;
        const size_t inPlane = linear - coreZ * planeElements;
        const size_t localZ = coreZ + 2;
        const size_t y = inPlane / nx;
        const size_t x = inPlane - y * nx;
        const size_t index = localZ * planeElements + inPlane;

        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny) {
            output[index] = input[index];
        } else {
            output[index] = (input[index] + input[index - 1] + input[index + 1] +
                             input[index - nx] + input[index + nx] +
                             input[index - planeElements] + input[index + planeElements]) /
                            7.0;
        }
    }
}

// The first and last local planes may need an MPI halo.  Physical global
// boundaries are copied verbatim, just as in the original implementation.
__global__ void stencilEdgeKernel(const Real* const input, Real* const output,
                                  const size_t nx, const size_t ny,
                                  const size_t globalNz, const size_t planeElements,
                                  const size_t globalStart, const size_t localZ) {
    for (size_t inPlane = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         inPlane < planeElements;
         inPlane += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t y = inPlane / nx;
        const size_t x = inPlane - y * nx;
        const size_t index = localZ * planeElements + inPlane;
        const size_t globalZ = globalStart + localZ - 1;

        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 ||
            globalZ + 1 == globalNz) {
            output[index] = input[index];
        } else {
            output[index] = (input[index] + input[index - 1] + input[index + 1] +
                             input[index - nx] + input[index + nx] +
                             input[index - planeElements] + input[index + planeElements]) /
                            7.0;
        }
    }
}

bool validateResult(const std::vector<Real>& grid) {
    int invalid = 0;
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -std::numeric_limits<Real>::infinity();

#pragma omp parallel for reduction(| : invalid) reduction(min : minValue) reduction(max : maxValue) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        if (!std::isfinite(value)) {
            invalid = 1;
        } else {
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
    }

    if (invalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 1e6 || minValue < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI does not provide the requested MPI_THREAD_FUNNELED support");
    }

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool validArguments = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            validArguments = parseSize(argv[++i], nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            validArguments = parseSize(argv[++i], ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            validArguments = parseSize(argv[++i], nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            validArguments = parseIterations(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            validArguments = false;
        }
        if (!validArguments) {
            break;
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (!validArguments || nx < 3 || ny < 3 || nz < 3) {
        if (rank == 0) {
            std::printf("Grid dimensions must all be at least 3.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    if (nx > std::numeric_limits<size_t>::max() / ny) {
        abortWithMessage(rank, "grid dimensions overflow size_t");
    }
    const size_t planeElements = nx * ny;
    if (nz > std::numeric_limits<size_t>::max() / planeElements) {
        abortWithMessage(rank, "grid dimensions overflow size_t");
    }
    const size_t gridSize = planeElements * nz;
    if (planeElements > static_cast<size_t>(INT_MAX) || gridSize > static_cast<size_t>(INT_MAX)) {
        abortWithMessage(rank, "grid is too large for the MPI count interface");
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithMessage(rank, "no CUDA devices are visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    // Only ranks with a nonempty slab participate in halo exchange.  This also
    // supports a launch with more MPI ranks than z planes without changing data.
    const int activeRanks = std::min(worldSize, static_cast<int>(nz));
    size_t localNz = 0;
    size_t globalStart = 0;
    if (rank < activeRanks) {
        const size_t basePlanes = nz / static_cast<size_t>(activeRanks);
        const size_t remainder = nz % static_cast<size_t>(activeRanks);
        localNz = basePlanes + (rank < static_cast<int>(remainder) ? 1 : 0);
        globalStart = static_cast<size_t>(rank) * basePlanes +
                      std::min(static_cast<size_t>(rank), remainder);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, CUDA devices per node: %d, OpenMP threads/rank: %d\n",
                    worldSize, deviceCount, omp_get_max_threads());
        std::printf("Initializing grid...\n");
    }

    Real* deviceA = nullptr;
    Real* deviceB = nullptr;
    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* receiveLower = nullptr;
    Real* receiveUpper = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    cudaEvent_t halosReady = nullptr;
    cudaEvent_t fieldReady = nullptr;

    if (localNz != 0) {
        if (localNz + 2 > std::numeric_limits<size_t>::max() / planeElements ||
            (localNz + 2) * planeElements > std::numeric_limits<size_t>::max() / sizeof(Real)) {
            abortWithMessage(rank, "local allocation size overflows size_t");
        }
        const size_t fieldBytes = (localNz + 2) * planeElements * sizeof(Real);
        const size_t planeBytes = planeElements * sizeof(Real);

        CUDA_CHECK(cudaMalloc(&deviceA, fieldBytes));
        CUDA_CHECK(cudaMalloc(&deviceB, fieldBytes));
        CUDA_CHECK(cudaMallocHost(&sendLower, planeBytes));
        CUDA_CHECK(cudaMallocHost(&sendUpper, planeBytes));
        CUDA_CHECK(cudaMallocHost(&receiveLower, planeBytes));
        CUDA_CHECK(cudaMallocHost(&receiveUpper, planeBytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&fieldReady, cudaEventDisableTiming));

        initializeKernel<<<blocksFor(localNz * planeElements), kThreadsPerBlock, 0, computeStream>>>(
            deviceA, planeElements, localNz, globalStart);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
        std::printf("Running stencil computation...\n");
    }
    const double startTime = MPI_Wtime();

    const int lowerRank = rank > 0 && rank < activeRanks ? rank - 1 : MPI_PROC_NULL;
    const int upperRank = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
    Real* current = deviceA;
    Real* next = deviceB;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (localNz == 0) {
            continue;
        }

        const size_t planeBytes = planeElements * sizeof(Real);
        // From the second iteration onward, current was produced on the compute
        // stream.  Make the communication stream consume it only after that
        // output is ready; no host-side device synchronization is necessary.
        if (iteration != 0) {
            CUDA_CHECK(cudaStreamWaitEvent(communicationStream, fieldReady, 0));
        }
        if (lowerRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendLower, current + planeElements, planeBytes,
                                       cudaMemcpyDeviceToHost, communicationStream));
        }
        if (upperRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendUpper, current + localNz * planeElements, planeBytes,
                                       cudaMemcpyDeviceToHost, communicationStream));
        }

        // Planes 2 through localNz-1 do not consume remote data.
        if (localNz > 2) {
            const size_t corePlanes = localNz - 2;
            stencilCoreKernel<<<blocksFor(corePlanes * planeElements), kThreadsPerBlock, 0,
                                computeStream>>>(current, next, nx, ny, planeElements, corePlanes);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
        MPI_Request requests[4];
        int requestCount = 0;
        if (lowerRank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(receiveLower, static_cast<int>(planeElements), MPI_DOUBLE, lowerRank,
                                kTagLastPlane, MPI_COMM_WORLD, &requests[requestCount++]));
            MPI_CHECK(MPI_Isend(sendLower, static_cast<int>(planeElements), MPI_DOUBLE, lowerRank,
                                kTagFirstPlane, MPI_COMM_WORLD, &requests[requestCount++]));
        }
        if (upperRank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(receiveUpper, static_cast<int>(planeElements), MPI_DOUBLE, upperRank,
                                kTagFirstPlane, MPI_COMM_WORLD, &requests[requestCount++]));
            MPI_CHECK(MPI_Isend(sendUpper, static_cast<int>(planeElements), MPI_DOUBLE, upperRank,
                                kTagLastPlane, MPI_COMM_WORLD, &requests[requestCount++]));
        }
        MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE));

        if (lowerRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(current, receiveLower, planeBytes, cudaMemcpyHostToDevice,
                                       communicationStream));
        }
        if (upperRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(current + (localNz + 1) * planeElements, receiveUpper,
                                       planeBytes, cudaMemcpyHostToDevice, communicationStream));
        }
        CUDA_CHECK(cudaEventRecord(halosReady, communicationStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, halosReady, 0));

        stencilEdgeKernel<<<blocksFor(planeElements), kThreadsPerBlock, 0, computeStream>>>(
            current, next, nx, ny, nz, planeElements, globalStart, 1);
        CUDA_CHECK(cudaGetLastError());
        if (localNz > 1) {
            stencilEdgeKernel<<<blocksFor(planeElements), kThreadsPerBlock, 0, computeStream>>>(
                current, next, nx, ny, nz, planeElements, globalStart, localNz);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(fieldReady, computeStream));
        std::swap(current, next);
    }

    if (localNz != 0 && iterations != 0) {
        CUDA_CHECK(cudaEventSynchronize(fieldReady));
    }

    const double localElapsed = MPI_Wtime() - startTime;
    double maximumElapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(std::llround(maximumElapsed * 1000.0));
        const double cellUpdates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                                   static_cast<double>(nz - 2) * static_cast<double>(iterations);
        const double mcups = maximumElapsed > 0.0 ? cellUpdates / maximumElapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const bool needFinalGrid = validate || printResults;
    std::vector<Real> finalGrid;
    if (needFinalGrid) {
        std::vector<Real> localFinal(localNz * planeElements);
        if (localNz != 0) {
            CUDA_CHECK(cudaMemcpy(localFinal.data(), current + planeElements,
                                  localFinal.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        }

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            receiveCounts.resize(worldSize);
            displacements.resize(worldSize);
            for (int peer = 0; peer < worldSize; ++peer) {
                if (peer < activeRanks) {
                    const size_t basePlanes = nz / static_cast<size_t>(activeRanks);
                    const size_t remainder = nz % static_cast<size_t>(activeRanks);
                    const size_t peerPlanes = basePlanes + (peer < static_cast<int>(remainder) ? 1 : 0);
                    const size_t peerStart = static_cast<size_t>(peer) * basePlanes +
                                             std::min(static_cast<size_t>(peer), remainder);
                    receiveCounts[peer] = static_cast<int>(peerPlanes * planeElements);
                    displacements[peer] = static_cast<int>(peerStart * planeElements);
                } else {
                    receiveCounts[peer] = 0;
                    displacements[peer] = 0;
                }
            }
            finalGrid.resize(gridSize);
        }

        MPI_CHECK(MPI_Gatherv(localFinal.empty() ? nullptr : localFinal.data(),
                              static_cast<int>(localFinal.size()), MPI_DOUBLE,
                              rank == 0 ? finalGrid.data() : nullptr,
                              rank == 0 ? receiveCounts.data() : nullptr,
                              rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                              MPI_COMM_WORLD));
    }

    int returnCode = 0;
    if (rank == 0 && printResults) {
        print_results(finalGrid, "Grid");
    }
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        if (validateResult(finalGrid)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            returnCode = 1;
        }
    }

    if (localNz != 0) {
        CUDA_CHECK(cudaEventDestroy(fieldReady));
        CUDA_CHECK(cudaEventDestroy(halosReady));
        CUDA_CHECK(cudaStreamDestroy(communicationStream));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
        CUDA_CHECK(cudaFreeHost(receiveUpper));
        CUDA_CHECK(cudaFreeHost(receiveLower));
        CUDA_CHECK(cudaFreeHost(sendUpper));
        CUDA_CHECK(cudaFreeHost(sendLower));
        CUDA_CHECK(cudaFree(deviceB));
        CUDA_CHECK(cudaFree(deviceA));
    }
    MPI_CHECK(MPI_Comm_free(&localComm));

    int globalReturnCode = 0;
    MPI_CHECK(MPI_Allreduce(&returnCode, &globalReturnCode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI_Finalize();
    return globalReturnCode;
}
