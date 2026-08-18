#include <mpi.h>

#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kLowerTag = 100;
constexpr int kUpperTag = 101;

int worldRank = 0;

[[noreturn]] void abortRun(const char* message, const char* file, int line) {
    std::fprintf(stderr, "Rank %d: %s (%s:%d)\n", worldRank, message, file, line);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        char message[1024];
        std::snprintf(message, sizeof(message), "CUDA call '%s' failed: %s", expression,
                      cudaGetErrorString(error));
        abortRun(message, file, line);
    }
}

void checkMpi(int error, const char* expression, const char* file, int line) {
    if (error != MPI_SUCCESS) {
        char errorString[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, errorString, &length);
        char message[1200];
        std::snprintf(message, sizeof(message), "MPI call '%s' failed: %.*s", expression,
                      length, errorString);
        abortRun(message, file, line);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

bool checkedMultiply(size_t lhs, size_t rhs, size_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool parseSize(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || text[0] == '-' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(INT_MAX)) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void initializeOwnedGrid(std::vector<Real>& grid, size_t nx, size_t ny,
                         size_t localPlanes, size_t globalStartZ) {
    const size_t planeElements = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t localZ = 0; localZ < localPlanes; ++localZ) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t globalBase = (globalStartZ + localZ) * planeElements + y * nx;
            const size_t localBase = localZ * planeElements + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                grid[localBase + x] = static_cast<Real>((globalBase + x) % 19);
            }
        }
    }
}

// The three-dimensional tile includes a one-cell halo in every direction.
// It cuts global loads substantially compared with a direct seven-load kernel
// while retaining the exact source expression's addition order.
template <int BlockZ>
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output, size_t planeElements,
                              int nx, int ny, int zBegin, int zEnd) {
    constexpr int tileX = kBlockX + 2;
    constexpr int tileY = kBlockY + 2;
    constexpr int tileZ = BlockZ + 2;
    constexpr int tileElements = tileX * tileY * tileZ;
    constexpr int blockThreads = kBlockX * kBlockY * BlockZ;

    __shared__ Real tile[tileElements];

    const int threadLinear = (threadIdx.z * kBlockY + threadIdx.y) * kBlockX +
                             threadIdx.x;
    const int xStart = 1 + static_cast<int>(blockIdx.x) * kBlockX;
    const int yStart = 1 + static_cast<int>(blockIdx.y) * kBlockY;
    const int zStart = zBegin + static_cast<int>(blockIdx.z) * BlockZ;

    for (int linear = threadLinear; linear < tileElements; linear += blockThreads) {
        const int tileLocalZ = linear / (tileX * tileY);
        const int remainder = linear - tileLocalZ * tileX * tileY;
        const int tileLocalY = remainder / tileX;
        const int tileLocalX = remainder - tileLocalY * tileX;

        const int x = min(max(xStart + tileLocalX - 1, 0), nx - 1);
        const int y = min(max(yStart + tileLocalY - 1, 0), ny - 1);
        const int z = min(max(zStart + tileLocalZ - 1, zBegin - 1), zEnd);
        tile[linear] = input[static_cast<size_t>(z) * planeElements +
                             static_cast<size_t>(y) * nx + x];
    }
    __syncthreads();

    const int x = xStart + threadIdx.x;
    const int y = yStart + threadIdx.y;
    const int z = zStart + threadIdx.z;
    if (x < nx - 1 && y < ny - 1 && z < zEnd) {
        const int centerIndex = ((threadIdx.z + 1) * tileY + threadIdx.y + 1) *
                                    tileX +
                                threadIdx.x + 1;
        const Real center = tile[centerIndex];
        const Real left = tile[centerIndex - 1];
        const Real right = tile[centerIndex + 1];
        const Real front = tile[centerIndex - tileX];
        const Real back = tile[centerIndex + tileX];
        const Real bottom = tile[centerIndex - tileX * tileY];
        const Real top = tile[centerIndex + tileX * tileY];
        const size_t outputIndex = static_cast<size_t>(z) * planeElements +
                                   static_cast<size_t>(y) * nx + x;
        output[outputIndex] =
            (center + left + right + front + back + bottom + top) / 7.0;
    }
}

template <int BlockZ>
void launchStencilRange(const Real* input, Real* output, size_t planeElements,
                        int nx, int ny, int zBegin, int zEnd, int maxGridZ,
                        cudaStream_t stream) {
    if (zBegin >= zEnd || nx <= 2 || ny <= 2) {
        return;
    }

    const dim3 block(kBlockX, kBlockY, BlockZ);
    const unsigned int gridX = static_cast<unsigned int>((nx - 2 + kBlockX - 1) /
                                                          kBlockX);
    const unsigned int gridY = static_cast<unsigned int>((ny - 2 + kBlockY - 1) /
                                                          kBlockY);
    const int planesPerLaunch = maxGridZ * BlockZ;
    for (int firstZ = zBegin; firstZ < zEnd;) {
        const int remaining = zEnd - firstZ;
        const int thisLaunchPlanes = std::min(remaining, planesPerLaunch);
        const int lastZ = firstZ + thisLaunchPlanes;
        const unsigned int gridZ = static_cast<unsigned int>(
            (thisLaunchPlanes + BlockZ - 1) / BlockZ);
        const dim3 grid(gridX, gridY, gridZ);
        stencilKernel<BlockZ><<<grid, block, 0, stream>>>(
            input, output, planeElements, nx, ny, firstZ, lastZ);
        CUDA_CHECK(cudaPeekAtLastError());
        firstZ = lastZ;
    }
}

bool environmentFlag(const char* value, bool& parsed) {
    if (value == nullptr) {
        return false;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0) {
        parsed = true;
        return true;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 || std::strcmp(value, "no") == 0) {
        parsed = false;
        return true;
    }
    return false;
}

bool cudaAwareMpiAvailable() {
    bool forced = false;
    if (environmentFlag(std::getenv("STENCIL3D_CUDA_AWARE_MPI"), forced)) {
        return forced;
    }
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_CUDA_AWARE_SUPPORT != 0;
#else
    return false;
#endif
}

bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) {
        std::printf("Validation failed: empty grid\n");
        return false;
    }

    int invalid = 0;
    Real minValue = grid[0];
    Real maxValue = grid[0];
#pragma omp parallel for reduction(| : invalid) reduction(min : minValue) \
    reduction(max : maxValue) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        invalid |= !std::isfinite(value);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
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

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
    std::printf("Environment:\n");
    std::printf("  STENCIL3D_CUDA_AWARE_MPI=0|1  Override CUDA-aware MPI detection\n");
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));

    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortRun("MPI does not provide MPI_THREAD_FUNNELED", __FILE__, __LINE__);
    }

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            argumentsValid &= parseSize(argv[++i], nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            argumentsValid &= parseSize(argv[++i], ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            argumentsValid &= parseSize(argv[++i], nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentsValid &= parseIterations(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (worldRank == 0) {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    if (!argumentsValid) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Grid dimensions must be positive integers and iterations "
                                 "must be non-negative.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    size_t planeElements = 0;
    size_t gridElements = 0;
    size_t planeBytes = 0;
    if (!checkedMultiply(nx, ny, planeElements) ||
        !checkedMultiply(planeElements, nz, gridElements) ||
        !checkedMultiply(planeElements, sizeof(Real), planeBytes) ||
        planeElements > static_cast<size_t>(INT_MAX)) {
        abortRun("grid dimensions overflow or an XY halo exceeds MPI's count limit", __FILE__,
                 __LINE__);
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &nodeCommunicator));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeCommunicator, &localSize));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortRun("no CUDA device is visible", __FILE__, __LINE__);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    const int activeRanks = std::min(worldSize, static_cast<int>(nz));
    const bool active = worldRank < activeRanks;
    size_t localPlanes = 0;
    size_t globalStartZ = nz;
    if (active) {
        const size_t basePlanes = nz / static_cast<size_t>(activeRanks);
        const size_t remainder = nz % static_cast<size_t>(activeRanks);
        localPlanes = basePlanes + (static_cast<size_t>(worldRank) < remainder ? 1 : 0);
        globalStartZ = static_cast<size_t>(worldRank) * basePlanes +
                       std::min(static_cast<size_t>(worldRank), remainder);
    }

    const int previousRank = active && worldRank > 0 ? worldRank - 1 : MPI_PROC_NULL;
    const int nextRank = active && worldRank + 1 < activeRanks ? worldRank + 1 : MPI_PROC_NULL;

    int cudaAwareLocal = cudaAwareMpiAvailable() ? 1 : 0;
    int cudaAwareAll = 0;
    MPI_CHECK(MPI_Allreduce(&cudaAwareLocal, &cudaAwareAll, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
    const bool cudaAware = cudaAwareAll != 0;

    const int oversubscribedLocal = localSize > deviceCount ? 1 : 0;
    int anyOversubscribed = 0;
    MPI_CHECK(MPI_Allreduce(&oversubscribedLocal, &anyOversubscribed, 1, MPI_INT,
                            MPI_MAX, MPI_COMM_WORLD));

    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), %d active slab(s), up to %d "
                    "OpenMP thread(s)/rank, CUDA\n",
                    worldSize, activeRanks, omp_get_max_threads());
        std::printf("Halo transport: %s\n",
                    cudaAware ? "CUDA-aware MPI (device direct)"
                              : "MPI with pinned host staging");
        if (anyOversubscribed != 0) {
            std::printf("Warning: at least one node oversubscribes its visible CUDA devices; "
                        "one MPI rank per GPU is recommended.\n");
        }
        std::printf("Initializing grid...\n");
    }

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    Real* pinnedHalos = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    cudaEvent_t haloReady = nullptr;
    std::vector<Real> hostOwned;

    size_t localElements = 0;
    size_t allocationElements = 0;
    size_t allocationBytes = 0;
    if (active) {
        if (!checkedMultiply(localPlanes, planeElements, localElements) ||
            !checkedMultiply(localPlanes + 2, planeElements, allocationElements) ||
            !checkedMultiply(allocationElements, sizeof(Real), allocationBytes)) {
            abortRun("local grid allocation overflows size_t", __FILE__, __LINE__);
        }

        hostOwned.resize(localElements);
        initializeOwnedGrid(hostOwned, nx, ny, localPlanes, globalStartZ);

        CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), allocationBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), allocationBytes));
        CUDA_CHECK(cudaMemsetAsync(deviceGrid1, 0, allocationBytes, computeStream));
        CUDA_CHECK(cudaMemsetAsync(deviceGrid2, 0, allocationBytes, computeStream));
        CUDA_CHECK(cudaMemcpyAsync(deviceGrid1 + planeElements, hostOwned.data(),
                                   localElements * sizeof(Real), cudaMemcpyHostToDevice,
                                   computeStream));
        CUDA_CHECK(cudaMemcpyAsync(deviceGrid2 + planeElements, hostOwned.data(),
                                   localElements * sizeof(Real), cudaMemcpyHostToDevice,
                                   computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        // Force both template variants through CUDA's lazy module loader before
        // the timed region. The second grid is restored immediately afterward.
        stencilKernel<1><<<dim3(1, 1, 1), dim3(kBlockX, kBlockY, 1), 0,
                           computeStream>>>(deviceGrid1, deviceGrid2, planeElements,
                                            static_cast<int>(nx), static_cast<int>(ny),
                                            1, 2);
        stencilKernel<4><<<dim3(1, 1, 1), dim3(kBlockX, kBlockY, 4), 0,
                           computeStream>>>(deviceGrid1, deviceGrid2, planeElements,
                                            static_cast<int>(nx), static_cast<int>(ny),
                                            1, 2);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaMemsetAsync(deviceGrid2, 0, allocationBytes, computeStream));
        CUDA_CHECK(cudaMemcpyAsync(deviceGrid2 + planeElements, hostOwned.data(),
                                   localElements * sizeof(Real), cudaMemcpyHostToDevice,
                                   computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        if (!cudaAware && activeRanks > 1) {
            size_t haloBytes = 0;
            if (!checkedMultiply(planeBytes, static_cast<size_t>(4), haloBytes)) {
                abortRun("pinned halo allocation overflows size_t", __FILE__, __LINE__);
            }
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinnedHalos), haloBytes));
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (worldRank == 0) {
        std::printf("Running stencil computation...\n");
    }
    const double startTime = MPI_Wtime();

    Real* currentGrid = deviceGrid1;
    Real* nextGrid = deviceGrid2;
    const int nxInt = static_cast<int>(nx);
    const int nyInt = static_cast<int>(ny);
    const int localPlanesInt = static_cast<int>(localPlanes);
    const int haloCount = static_cast<int>(planeElements);

    int updateBegin = 0;
    int updateEnd = 0;
    if (active) {
        const size_t globalUpdateBegin = std::max(globalStartZ, static_cast<size_t>(1));
        const size_t globalUpdateEnd = std::min(globalStartZ + localPlanes, nz - 1);
        if (globalUpdateBegin < globalUpdateEnd) {
            updateBegin = static_cast<int>(globalUpdateBegin - globalStartZ + 1);
            updateEnd = static_cast<int>(globalUpdateEnd - globalStartZ + 1);
        }
    }

    for (int iteration = 0; iteration < iterations && active; ++iteration) {
        if (iteration != 0) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        }

        Real* sendLower = nullptr;
        Real* sendUpper = nullptr;
        Real* receiveLower = nullptr;
        Real* receiveUpper = nullptr;
        if (!cudaAware && activeRanks > 1) {
            sendLower = pinnedHalos;
            sendUpper = pinnedHalos + planeElements;
            receiveLower = pinnedHalos + 2 * planeElements;
            receiveUpper = pinnedHalos + 3 * planeElements;
            if (previousRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sendLower, currentGrid + planeElements, planeBytes,
                                           cudaMemcpyDeviceToHost, communicationStream));
            }
            if (nextRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sendUpper,
                                           currentGrid + localPlanes * planeElements,
                                           planeBytes, cudaMemcpyDeviceToHost,
                                           communicationStream));
            }
            CUDA_CHECK(cudaStreamSynchronize(communicationStream));
        }

        MPI_Request requests[4];
        int requestCount = 0;
        if (previousRank != MPI_PROC_NULL) {
            Real* lowerGhost = cudaAware ? currentGrid : receiveLower;
            Real* lowerOwned = cudaAware ? currentGrid + planeElements : sendLower;
            MPI_CHECK(MPI_Irecv(lowerGhost, haloCount, MPI_DOUBLE, previousRank, kUpperTag,
                                MPI_COMM_WORLD, &requests[requestCount++]));
            MPI_CHECK(MPI_Isend(lowerOwned, haloCount, MPI_DOUBLE, previousRank, kLowerTag,
                                MPI_COMM_WORLD, &requests[requestCount++]));
        }
        if (nextRank != MPI_PROC_NULL) {
            Real* upperGhost = cudaAware
                                   ? currentGrid + (localPlanes + 1) * planeElements
                                   : receiveUpper;
            Real* upperOwned = cudaAware
                                   ? currentGrid + localPlanes * planeElements
                                   : sendUpper;
            MPI_CHECK(MPI_Irecv(upperGhost, haloCount, MPI_DOUBLE, nextRank, kLowerTag,
                                MPI_COMM_WORLD, &requests[requestCount++]));
            MPI_CHECK(MPI_Isend(upperOwned, haloCount, MPI_DOUBLE, nextRank, kUpperTag,
                                MPI_COMM_WORLD, &requests[requestCount++]));
        }

        // Owned planes 2..N-1 do not consume MPI ghosts, so their computation
        // overlaps the nonblocking halo exchange.
        const int bulkBegin = std::max(updateBegin, 2);
        const int bulkEnd = std::min(updateEnd, localPlanesInt);
        launchStencilRange<4>(currentGrid, nextGrid, planeElements, nxInt, nyInt,
                              bulkBegin, bulkEnd, deviceProperties.maxGridSize[2],
                              computeStream);

        if (requestCount != 0) {
            MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE));
        }
        if (!cudaAware && activeRanks > 1) {
            bool copiedReceive = false;
            if (previousRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(currentGrid, receiveLower, planeBytes,
                                           cudaMemcpyHostToDevice, communicationStream));
                copiedReceive = true;
            }
            if (nextRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(
                    currentGrid + (localPlanes + 1) * planeElements, receiveUpper,
                    planeBytes, cudaMemcpyHostToDevice, communicationStream));
                copiedReceive = true;
            }
            if (copiedReceive) {
                CUDA_CHECK(cudaEventRecord(haloReady, communicationStream));
                CUDA_CHECK(cudaStreamWaitEvent(computeStream, haloReady, 0));
            }
        }

        const bool updateLower = updateBegin <= 1 && 1 < updateEnd;
        const bool updateUpper = updateBegin <= localPlanesInt &&
                                 localPlanesInt < updateEnd;
        if (updateLower) {
            launchStencilRange<1>(currentGrid, nextGrid, planeElements, nxInt, nyInt,
                                  1, 2, deviceProperties.maxGridSize[2], computeStream);
        }
        if (updateUpper && localPlanesInt != 1) {
            launchStencilRange<1>(currentGrid, nextGrid, planeElements, nxInt, nyInt,
                                  localPlanesInt, localPlanesInt + 1,
                                  deviceProperties.maxGridSize[2], computeStream);
        }

        std::swap(currentGrid, nextGrid);
    }

    if (active) {
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (worldRank == 0) {
        const long long elapsedMilliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", elapsedMilliseconds);
        const double xInterior = nx > 2 ? static_cast<double>(nx - 2) : 0.0;
        const double yInterior = ny > 2 ? static_cast<double>(ny - 2) : 0.0;
        const double zInterior = nz > 2 ? static_cast<double>(nz - 2) : 0.0;
        const double cellUpdates =
            xInterior * yInterior * zInterior * static_cast<double>(iterations);
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        if (gridElements > static_cast<size_t>(INT_MAX)) {
            abortRun("-r and -v currently require at most INT_MAX global elements for "
                     "MPI_Gatherv",
                     __FILE__, __LINE__);
        }
        if (active) {
            CUDA_CHECK(cudaMemcpy(hostOwned.data(), currentGrid + planeElements,
                                  localElements * sizeof(Real), cudaMemcpyDeviceToHost));
        }

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (worldRank == 0) {
            finalGrid.resize(gridElements);
            receiveCounts.resize(worldSize, 0);
            displacements.resize(worldSize, static_cast<int>(gridElements));
            for (int rank = 0; rank < activeRanks; ++rank) {
                const size_t basePlanes = nz / static_cast<size_t>(activeRanks);
                const size_t remainder = nz % static_cast<size_t>(activeRanks);
                const size_t rankPlanes =
                    basePlanes + (static_cast<size_t>(rank) < remainder ? 1 : 0);
                const size_t rankStart = static_cast<size_t>(rank) * basePlanes +
                                         std::min(static_cast<size_t>(rank), remainder);
                receiveCounts[rank] = static_cast<int>(rankPlanes * planeElements);
                displacements[rank] = static_cast<int>(rankStart * planeElements);
            }
        }
        const int sendCount = active ? static_cast<int>(localElements) : 0;
        MPI_CHECK(MPI_Gatherv(active ? hostOwned.data() : nullptr, sendCount, MPI_DOUBLE,
                              worldRank == 0 ? finalGrid.data() : nullptr,
                              worldRank == 0 ? receiveCounts.data() : nullptr,
                              worldRank == 0 ? displacements.data() : nullptr, MPI_DOUBLE,
                              0, MPI_COMM_WORLD));
    }

    if (pinnedHalos != nullptr) {
        CUDA_CHECK(cudaFreeHost(pinnedHalos));
    }
    if (deviceGrid1 != nullptr) {
        CUDA_CHECK(cudaFree(deviceGrid1));
        CUDA_CHECK(cudaFree(deviceGrid2));
        CUDA_CHECK(cudaEventDestroy(haloReady));
        CUDA_CHECK(cudaStreamDestroy(communicationStream));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
    }

    int exitCode = EXIT_SUCCESS;
    if (worldRank == 0 && printResults) {
        print_results(finalGrid, "Grid");
    }
    if (worldRank == 0 && validate) {
        std::printf("Validating result...\n");
        if (validateResult(finalGrid)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = EXIT_FAILURE;
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_free(&nodeCommunicator));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
