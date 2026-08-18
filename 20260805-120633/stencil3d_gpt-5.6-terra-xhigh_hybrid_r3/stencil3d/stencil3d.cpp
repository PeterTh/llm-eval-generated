#include <algorithm>
#include <cerrno>
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

constexpr int HaloFromLowerTag = 0;
constexpr int HaloFromUpperTag = 1;

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
};

struct HaloBuffers {
    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* recvLower = nullptr;
    Real* recvUpper = nullptr;
};

void printUsage(const char* progName) {
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

bool parseSize(const char* text, size_t& value) {
    if (*text == '\0' || *text == '-') {
        return false;
    }

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }

    value = static_cast<size_t>(parsed);
    return true;
}

bool parseInt(const char* text, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || *text == '\0' || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

bool parseOptions(const int argc, char** argv, Options& options, const int worldRank) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nx)) {
                if (worldRank == 0) std::printf("Invalid X dimension\n");
                return false;
            }
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.ny)) {
                if (worldRank == 0) std::printf("Invalid Y dimension\n");
                return false;
            }
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nz)) {
                if (worldRank == 0) std::printf("Invalid Z dimension\n");
                return false;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseInt(argv[++i], options.iterations)) {
                if (worldRank == 0) std::printf("Invalid iteration count\n");
                return false;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            options.showHelp = true;
            return false;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return false;
        }
    }

    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    if (options.nx == 0 || options.ny == 0 || options.nz == 0) {
        if (worldRank == 0) std::printf("Grid dimensions must be positive\n");
        return false;
    }

    return true;
}

[[noreturn]] void cudaCheck(const cudaError_t status, const char* operation, const int worldRank) {
    std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", worldRank, operation,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* operation, const int worldRank) {
    if (status != cudaSuccess) {
        cudaCheck(status, operation, worldRank);
    }
}

bool checkedProduct(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

dim3 launchGrid(const size_t nx, const size_t ny, const size_t layers) {
    constexpr unsigned int blockX = 32;
    constexpr unsigned int blockY = 8;
    const size_t gridX = (nx + blockX - 1) / blockX;
    const size_t gridY = (ny + blockY - 1) / blockY;
    if (gridX > std::numeric_limits<unsigned int>::max() ||
        gridY > std::numeric_limits<unsigned int>::max() ||
        layers > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "CUDA launch grid exceeds the supported dimension\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return dim3(static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY),
                static_cast<unsigned int>(layers));
}

__global__ void stencilLayers(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx,
                              const size_t ny,
                              const size_t globalNz,
                              const size_t globalZStart,
                              const size_t localZFirst,
                              const size_t layerCount) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = localZFirst + blockIdx.z;
    if (x >= nx || y >= ny || blockIdx.z >= layerCount) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t index = localZ * plane + y * nx + x;
    const size_t globalZ = globalZStart + localZ - 1;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 || globalZ + 1 == globalNz) {
        output[index] = input[index];
        return;
    }

    output[index] = (input[index] + input[index - 1] + input[index + 1] +
                     input[index - nx] + input[index + nx] +
                     input[index - plane] + input[index + plane]) /
                    7.0;
}

void launchStencilLayers(const Real* input,
                         Real* output,
                         const size_t nx,
                         const size_t ny,
                         const size_t globalNz,
                         const size_t globalZStart,
                         const size_t localZFirst,
                         const size_t layerCount,
                         const cudaStream_t stream,
                         const int worldRank) {
    if (layerCount == 0) {
        return;
    }

    const dim3 block(32, 8, 1);
    stencilLayers<<<launchGrid(nx, ny, layerCount), block, 0, stream>>>(
        input, output, nx, ny, globalNz, globalZStart, localZFirst, layerCount);
    checkCuda(cudaGetLastError(), "stencil kernel launch", worldRank);
}

void exchangeHalos(Real* current,
                   const size_t planeElements,
                   const size_t localNz,
                   const int lowerRank,
                   const int upperRank,
                   HaloBuffers& buffers,
                   const cudaStream_t transferStream,
                   const cudaEvent_t outboundReady,
                   const cudaEvent_t haloReady,
                   const MPI_Comm communicator,
                   const int worldRank) {
    MPI_Request requests[4];
    int requestCount = 0;
    const int mpiCount = static_cast<int>(planeElements);

    // Receive first so a rank can send as soon as its asynchronous device-to-host copies finish.
    if (lowerRank != MPI_PROC_NULL) {
        MPI_Irecv(buffers.recvLower, mpiCount, MPI_DOUBLE, lowerRank, HaloFromLowerTag,
                  communicator, &requests[requestCount++]);
    }
    if (upperRank != MPI_PROC_NULL) {
        MPI_Irecv(buffers.recvUpper, mpiCount, MPI_DOUBLE, upperRank, HaloFromUpperTag,
                  communicator, &requests[requestCount++]);
    }

    if (lowerRank != MPI_PROC_NULL) {
        checkCuda(cudaMemcpyAsync(buffers.sendLower, current + planeElements,
                                  planeElements * sizeof(Real), cudaMemcpyDeviceToHost,
                                  transferStream),
                  "lower halo device-to-host copy", worldRank);
    }
    if (upperRank != MPI_PROC_NULL) {
        checkCuda(cudaMemcpyAsync(buffers.sendUpper, current + localNz * planeElements,
                                  planeElements * sizeof(Real), cudaMemcpyDeviceToHost,
                                  transferStream),
                  "upper halo device-to-host copy", worldRank);
    }
    checkCuda(cudaEventRecord(outboundReady, transferStream), "outbound halo event", worldRank);

    // The caller has already launched the halo-independent local layers on another stream.
    checkCuda(cudaEventSynchronize(outboundReady), "outbound halo synchronization", worldRank);
    if (lowerRank != MPI_PROC_NULL) {
        MPI_Isend(buffers.sendLower, mpiCount, MPI_DOUBLE, lowerRank, HaloFromUpperTag,
                  communicator, &requests[requestCount++]);
    }
    if (upperRank != MPI_PROC_NULL) {
        MPI_Isend(buffers.sendUpper, mpiCount, MPI_DOUBLE, upperRank, HaloFromLowerTag,
                  communicator, &requests[requestCount++]);
    }
    if (requestCount != 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }

    if (lowerRank != MPI_PROC_NULL) {
        checkCuda(cudaMemcpyAsync(current, buffers.recvLower,
                                  planeElements * sizeof(Real), cudaMemcpyHostToDevice,
                                  transferStream),
                  "lower halo host-to-device copy", worldRank);
    }
    if (upperRank != MPI_PROC_NULL) {
        checkCuda(cudaMemcpyAsync(current + (localNz + 1) * planeElements,
                                  buffers.recvUpper, planeElements * sizeof(Real),
                                  cudaMemcpyHostToDevice, transferStream),
                  "upper halo host-to-device copy", worldRank);
    }
    checkCuda(cudaEventRecord(haloReady, transferStream), "inbound halo event", worldRank);
}

bool validateResult(const std::vector<Real>& grid) {
    int hasInvalidValue = 0;
    Real minValue = std::numeric_limits<Real>::max();
    Real maxValue = std::numeric_limits<Real>::lowest();

#pragma omp parallel for reduction(| : hasInvalidValue) reduction(min : minValue) reduction(max : maxValue) schedule(static)
    for (long long index = 0; index < static_cast<long long>(grid.size()); ++index) {
        const Real value = grid[static_cast<size_t>(index)];
        hasInvalidValue |= static_cast<int>(std::isnan(value) || std::isinf(value));
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    if (hasInvalidValue != 0) {
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
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    Options options;
    if (!parseOptions(argc, argv, options, worldRank)) {
        MPI_Finalize();
        return options.showHelp ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    size_t planeElements = 0;
    size_t globalElements = 0;
    if (!checkedProduct(options.nx, options.ny, planeElements) ||
        !checkedProduct(planeElements, options.nz, globalElements) ||
        planeElements > static_cast<size_t>(INT_MAX) ||
        globalElements > static_cast<size_t>(LLONG_MAX)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Grid size exceeds supported address or MPI message limits\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int activeRanks =
        static_cast<int>(std::min(static_cast<size_t>(worldSize), options.nz));
    const bool isActive = worldRank < activeRanks;
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, isActive ? 1 : MPI_UNDEFINED, worldRank, &activeComm);

    // This is collective on MPI_COMM_WORLD, so even ranks without a z slab
    // participate.  The local rank selects a GPU consistently on every node.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int exitCode = EXIT_SUCCESS;
    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }
    if (isActive) {
        int activeRank = 0;
        int activeSize = 1;
        MPI_Comm_rank(activeComm, &activeRank);
        MPI_Comm_size(activeComm, &activeSize);

        int deviceCount = 0;
        const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
        if (deviceStatus != cudaSuccess || deviceCount == 0) {
            if (worldRank == 0) {
                std::fprintf(stderr, "No CUDA device is available: %s\n", cudaGetErrorString(deviceStatus));
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        checkCuda(cudaSetDevice(localRank % deviceCount), "CUDA device selection", worldRank);

        const size_t basePlanes = options.nz / static_cast<size_t>(activeSize);
        const size_t extraPlanes = options.nz % static_cast<size_t>(activeSize);
        const size_t localNz = basePlanes + (static_cast<size_t>(activeRank) < extraPlanes ? 1 : 0);
        const size_t globalZStart = static_cast<size_t>(activeRank) * basePlanes +
                                    std::min(static_cast<size_t>(activeRank), extraPlanes);
        size_t localWithHalos = 0;
        size_t localElements = 0;
        if (!checkedProduct(localNz + 2, planeElements, localWithHalos) ||
            !checkedProduct(localNz, planeElements, localElements) ||
            localWithHalos > std::numeric_limits<size_t>::max() / sizeof(Real)) {
            std::fprintf(stderr, "Rank %d: local allocation exceeds addressable memory\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        const int lowerRank = activeRank == 0 ? MPI_PROC_NULL : activeRank - 1;
        const int upperRank = activeRank + 1 == activeSize ? MPI_PROC_NULL : activeRank + 1;
        const size_t allocationBytes = localWithHalos * sizeof(Real);

        Real* current = nullptr;
        Real* next = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&current), allocationBytes), "current grid allocation", worldRank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&next), allocationBytes), "next grid allocation", worldRank);

        // The CPU side of the hybrid implementation initializes each local z slab in parallel.
        // The timed stencil itself is entirely CUDA-resident, apart from MPI halo staging.
        {
            std::vector<Real> initial(localElements);
            const size_t globalOffset = globalZStart * planeElements;
#pragma omp parallel for schedule(static)
            for (long long index = 0; index < static_cast<long long>(localElements); ++index) {
                initial[static_cast<size_t>(index)] =
                    static_cast<Real>((globalOffset + static_cast<size_t>(index)) % 19);
            }
            checkCuda(cudaMemcpy(current + planeElements, initial.data(), localElements * sizeof(Real),
                                 cudaMemcpyHostToDevice),
                      "initial grid upload", worldRank);
        }

        if (worldRank == 0) {
            std::printf("Running stencil computation...\n");
        }

        HaloBuffers halos;
        if (activeSize > 1) {
            const size_t haloBytes = planeElements * sizeof(Real);
            checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&halos.sendLower), haloBytes, cudaHostAllocDefault),
                      "lower send halo allocation", worldRank);
            checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&halos.sendUpper), haloBytes, cudaHostAllocDefault),
                      "upper send halo allocation", worldRank);
            checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&halos.recvLower), haloBytes, cudaHostAllocDefault),
                      "lower receive halo allocation", worldRank);
            checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&halos.recvUpper), haloBytes, cudaHostAllocDefault),
                      "upper receive halo allocation", worldRank);
        }

        cudaStream_t computeStream = nullptr;
        cudaStream_t transferStream = nullptr;
        cudaEvent_t outboundReady = nullptr;
        cudaEvent_t haloReady = nullptr;
        checkCuda(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking), "compute stream creation", worldRank);
        checkCuda(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking), "transfer stream creation", worldRank);
        checkCuda(cudaEventCreateWithFlags(&outboundReady, cudaEventDisableTiming), "outbound event creation", worldRank);
        checkCuda(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming), "inbound event creation", worldRank);

        MPI_Barrier(activeComm);
        checkCuda(cudaDeviceSynchronize(), "pre-timing device synchronization", worldRank);
        const double start = MPI_Wtime();

        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            // Layers 2..localNz-1 depend only on local planes and overlap the halo transfer.
            if (localNz > 2) {
                launchStencilLayers(current, next, options.nx, options.ny, options.nz, globalZStart,
                                    2, localNz - 2, computeStream, worldRank);
            }

            exchangeHalos(current, planeElements, localNz, lowerRank, upperRank, halos,
                          transferStream, outboundReady, haloReady, activeComm, worldRank);
            checkCuda(cudaStreamWaitEvent(computeStream, haloReady, 0), "halo dependency", worldRank);

            // Finish the one or two boundary layers after their neighbor halos arrive.
            launchStencilLayers(current, next, options.nx, options.ny, options.nz, globalZStart,
                                1, 1, computeStream, worldRank);
            if (localNz > 1) {
                launchStencilLayers(current, next, options.nx, options.ny, options.nz, globalZStart,
                                    localNz, 1, computeStream, worldRank);
            }
            checkCuda(cudaStreamSynchronize(computeStream), "stencil iteration synchronization", worldRank);
            std::swap(current, next);
        }

        const double elapsed = MPI_Wtime() - start;
        double maximumElapsed = 0.0;
        MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);

        std::vector<Real> finalGrid;
        if (options.validate || options.printResults) {
            std::vector<Real> localResult(localElements);
            checkCuda(cudaMemcpy(localResult.data(), current + planeElements, localElements * sizeof(Real),
                                 cudaMemcpyDeviceToHost),
                      "final grid download", worldRank);

            std::vector<int> receiveCounts;
            std::vector<int> displacements;
            if (activeRank == 0) {
                finalGrid.resize(globalElements);
                receiveCounts.resize(static_cast<size_t>(activeSize));
                displacements.resize(static_cast<size_t>(activeSize));
                for (int rank = 0; rank < activeSize; ++rank) {
                    const size_t rankPlanes = basePlanes +
                                              (static_cast<size_t>(rank) < extraPlanes ? 1 : 0);
                    const size_t rankStart = static_cast<size_t>(rank) * basePlanes +
                                             std::min(static_cast<size_t>(rank), extraPlanes);
                    receiveCounts[static_cast<size_t>(rank)] =
                        static_cast<int>(rankPlanes * planeElements);
                    displacements[static_cast<size_t>(rank)] =
                        static_cast<int>(rankStart * planeElements);
                }
            }
            MPI_Gatherv(localResult.data(), static_cast<int>(localElements), MPI_DOUBLE,
                        activeRank == 0 ? finalGrid.data() : nullptr,
                        activeRank == 0 ? receiveCounts.data() : nullptr,
                        activeRank == 0 ? displacements.data() : nullptr,
                        MPI_DOUBLE, 0, activeComm);
        }

        if (activeRank == 0) {
            const long long elapsedMilliseconds =
                static_cast<long long>(std::llround(maximumElapsed * 1000.0));
            std::printf("Computation time: %lld ms\n", elapsedMilliseconds);
            const size_t interiorX = options.nx > 2 ? options.nx - 2 : 0;
            const size_t interiorY = options.ny > 2 ? options.ny - 2 : 0;
            const size_t interiorZ = options.nz > 2 ? options.nz - 2 : 0;
            const double cellUpdates = static_cast<double>(interiorX) * static_cast<double>(interiorY) *
                                       static_cast<double>(interiorZ) * options.iterations;
            const double mcups = cellUpdates / std::max(maximumElapsed, 1.0e-9) / 1.0e6;
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

            if (options.printResults) {
                print_results(finalGrid, "Grid");
            }
            if (options.validate) {
                std::printf("Validating result...\n");
                if (validateResult(finalGrid)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    exitCode = EXIT_FAILURE;
                }
            }
        }

        checkCuda(cudaEventDestroy(outboundReady), "outbound event destruction", worldRank);
        checkCuda(cudaEventDestroy(haloReady), "inbound event destruction", worldRank);
        checkCuda(cudaStreamDestroy(computeStream), "compute stream destruction", worldRank);
        checkCuda(cudaStreamDestroy(transferStream), "transfer stream destruction", worldRank);
        if (halos.sendLower != nullptr) checkCuda(cudaFreeHost(halos.sendLower), "lower send halo release", worldRank);
        if (halos.sendUpper != nullptr) checkCuda(cudaFreeHost(halos.sendUpper), "upper send halo release", worldRank);
        if (halos.recvLower != nullptr) checkCuda(cudaFreeHost(halos.recvLower), "lower receive halo release", worldRank);
        if (halos.recvUpper != nullptr) checkCuda(cudaFreeHost(halos.recvUpper), "upper receive halo release", worldRank);
        checkCuda(cudaFree(current), "current grid release", worldRank);
        checkCuda(cudaFree(next), "next grid release", worldRank);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (activeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&activeComm);
    }
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
