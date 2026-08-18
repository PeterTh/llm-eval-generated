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

using Real = double;

__host__ __device__ inline constexpr std::size_t idx3(const std::size_t x,
                                                       const std::size_t y,
                                                       const std::size_t z,
                                                       const std::size_t nx,
                                                       const std::size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

[[noreturn]] void abortWithMessage(const int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line,
               const int rank) {
    if (error != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA error in %s (%s:%d): %s", expression, file,
                      line, cudaGetErrorString(error));
        abortWithMessage(rank, message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__, worldRank)

std::size_t parseSize(const char* value, const char* option, const int rank) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (value[0] == '\0' || end == value || *end != '\0' ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        char message[256];
        std::snprintf(message, sizeof(message), "invalid value for %s: %s", option, value);
        abortWithMessage(rank, message);
    }
    return static_cast<std::size_t>(parsed);
}

int parseIterations(const char* value, const int rank) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (value[0] == '\0' || end == value || *end != '\0' || parsed < 0 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        char message[256];
        std::snprintf(message, sizeof(message), "invalid iteration count: %s", value);
        abortWithMessage(rank, message);
    }
    return static_cast<int>(parsed);
}

std::size_t checkedProduct(const std::size_t a, const std::size_t b, const std::size_t c,
                           const int rank) {
    if ((b != 0 && a > std::numeric_limits<std::size_t>::max() / b) ||
        (c != 0 && a * b > std::numeric_limits<std::size_t>::max() / c)) {
        abortWithMessage(rank, "grid dimensions overflow size_t");
    }
    return a * b * c;
}

// The host-side initialization is parallelized over independent planes and rows. The value is
// based on the global index, so the distributed initialization is bit-for-bit identical to the
// original global initialization.
void initializeGrid(Real* grid, const std::size_t nx, const std::size_t ny,
                    const std::size_t globalZBegin, const std::size_t localNz) {
    const std::size_t plane = nx * ny;

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t localZ = 1; localZ <= localNz; ++localZ) {
        for (std::size_t y = 0; y < ny; ++y) {
            const std::size_t globalZ = globalZBegin + localZ - 1;
            const std::size_t globalBase = globalZ * plane + y * nx;
            const std::size_t localBase = localZ * plane + y * nx;
#pragma omp simd
            for (std::size_t x = 0; x < nx; ++x) {
                grid[localBase + x] = static_cast<Real>((globalBase + x) % 19);
            }
        }
    }
}

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const std::size_t nx, const std::size_t ny,
                              const std::size_t nz, const std::size_t globalZBegin,
                              const std::size_t localNz) {
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t localZ = static_cast<std::size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || localZ > localNz) {
        return;
    }

    const std::size_t globalZ = globalZBegin + localZ - 1;
    const std::size_t plane = nx * ny;
    const std::size_t index = localZ * plane + y * nx + x;
    const bool boundary = x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 ||
                          globalZ + 1 == nz;

    if (boundary) {
        output[index] = input[index];
        return;
    }

    // Keep the operation order explicit to retain the original stencil's floating-point
    // semantics across MPI slab boundaries and CUDA execution.
    Real sum = input[index];
    sum += input[index - 1];
    sum += input[index + 1];
    sum += input[index - nx];
    sum += input[index + nx];
    sum += input[index - plane];
    sum += input[index + plane];
    output[index] = sum / static_cast<Real>(7.0);
}

void launchStencil(const Real* input, Real* output, const std::size_t nx, const std::size_t ny,
                   const std::size_t nz, const std::size_t globalZBegin,
                   const std::size_t localNz, cudaStream_t stream, const int worldRank) {
    constexpr unsigned int blockX = 32;
    constexpr unsigned int blockY = 4;
    constexpr unsigned int blockZ = 2;
    const dim3 block(blockX, blockY, blockZ);
    const dim3 grid(static_cast<unsigned int>((nx + blockX - 1) / blockX),
                    static_cast<unsigned int>((ny + blockY - 1) / blockY),
                    static_cast<unsigned int>((localNz + blockZ - 1) / blockZ));

    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, nz, globalZBegin, localNz);
    checkCuda(cudaGetLastError(), "stencilKernel launch", __FILE__, __LINE__, worldRank);
}

Real* allocatePinned(const std::size_t elements, const int worldRank) {
    Real* buffer = nullptr;
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&buffer), elements * sizeof(Real),
                            cudaHostAllocPortable),
              "cudaHostAlloc", __FILE__, __LINE__, worldRank);
    return buffer;
}

void exchangeHalos(Real* grid, const std::size_t plane, const std::size_t localNz,
                   const int rank, const int size, MPI_Comm communicator) {
    if (size == 1) {
        return;
    }

    if (plane > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(rank, "MPI halo plane exceeds the MPI int count limit");
    }
    const int count = static_cast<int>(plane);
    MPI_Request requests[4];
    int requestCount = 0;

    // Tags are directional: tag 0 carries a plane toward increasing Z, tag 1 carries a plane
    // toward decreasing Z. This makes simultaneous exchanges safe for every rank pair.
    if (rank > 0) {
        if (MPI_Irecv(grid, count, MPI_DOUBLE, rank - 1, 0, communicator,
                      &requests[requestCount++]) != MPI_SUCCESS ||
            MPI_Isend(grid + plane, count, MPI_DOUBLE, rank - 1, 1, communicator,
                      &requests[requestCount++]) != MPI_SUCCESS) {
            abortWithMessage(rank, "MPI lower halo exchange failed");
        }
    }
    if (rank + 1 < size) {
        if (MPI_Irecv(grid + (localNz + 1) * plane, count, MPI_DOUBLE, rank + 1, 1, communicator,
                      &requests[requestCount++]) != MPI_SUCCESS ||
            MPI_Isend(grid + localNz * plane, count, MPI_DOUBLE, rank + 1, 0, communicator,
                      &requests[requestCount++]) != MPI_SUCCESS) {
            abortWithMessage(rank, "MPI upper halo exchange failed");
        }
    }

    if (requestCount != 0 && MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        abortWithMessage(rank, "MPI halo wait failed");
    }
}

bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) {
        std::printf("Validation failed: empty grid\n");
        return false;
    }

    int invalid = 0;
    Real minValue = std::numeric_limits<Real>::max();
    Real maxValue = std::numeric_limits<Real>::lowest();

#pragma omp parallel for reduction(| : invalid) reduction(min : minValue) reduction(max : maxValue) schedule(static)
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        if (std::isnan(value) || std::isinf(value)) {
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

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(worldRank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    std::size_t nx = 128;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = parseSize(argv[++i], "-x", worldRank);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = parseSize(argv[++i], "-y", worldRank);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = parseSize(argv[++i], "-z", worldRank);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseIterations(argv[++i], worldRank);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (nx == 0 || ny == 0 || nz == 0) {
        abortWithMessage(worldRank, "grid dimensions must be positive");
    }

    const int activeSize = static_cast<int>(std::min<std::size_t>(nz, static_cast<std::size_t>(worldSize)));
    const bool active = worldRank < activeSize;
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &communicator);

    if (!active) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);

    MPI_Comm sharedCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedCommunicator);
    int localRank = 0;
    MPI_Comm_rank(sharedCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortWithMessage(worldRank, "no CUDA device is visible to this MPI rank");
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const std::size_t baseNz = nz / static_cast<std::size_t>(size);
    const std::size_t remainder = nz % static_cast<std::size_t>(size);
    const std::size_t localNz = baseNz + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
    const std::size_t globalZBegin = static_cast<std::size_t>(rank) * baseNz +
                                     std::min<std::size_t>(static_cast<std::size_t>(rank), remainder);
    const std::size_t plane = checkedProduct(nx, ny, 1, worldRank);
    const std::size_t localElements = checkedProduct(localNz + 2, plane, 1, worldRank);
    const std::size_t localOwnedElements = checkedProduct(localNz, plane, 1, worldRank);

    if (plane > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        localOwnedElements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(worldRank, "local MPI message exceeds the MPI int count limit");
    }

    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n", size,
                    omp_get_max_threads(), deviceCount);
    }

    Real* hostGrid1 = allocatePinned(localElements, worldRank);
    Real* hostGrid2 = allocatePinned(localElements, worldRank);
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), localElements * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), localElements * sizeof(Real)));

    if (worldRank == 0) {
        std::printf("Initializing grid...\n");
    }
    initializeGrid(hostGrid1, nx, ny, globalZBegin, localNz);
    exchangeHalos(hostGrid1, plane, localNz, rank, size, communicator);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(deviceGrid1, hostGrid1, localElements * sizeof(Real),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (worldRank == 0) {
        std::printf("Running stencil computation...\n");
    }
    MPI_Barrier(communicator);
    const auto start = std::chrono::steady_clock::now();

    Real* currentHost = hostGrid1;
    Real* nextHost = hostGrid2;
    Real* currentDevice = deviceGrid1;
    Real* nextDevice = deviceGrid2;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        launchStencil(currentDevice, nextDevice, nx, ny, nz, globalZBegin, localNz, stream, worldRank);

        const bool hasLowerNeighbor = rank > 0;
        const bool hasUpperNeighbor = rank + 1 < size;
        if (hasLowerNeighbor || hasUpperNeighbor) {
            // The pinned buffers make these asynchronous copies safe and allow MPI to consume
            // the data without requiring CUDA-aware MPI support.
            CUDA_CHECK(cudaMemcpyAsync(nextHost + plane, nextDevice + plane, plane * sizeof(Real),
                                       cudaMemcpyDeviceToHost, stream));
            if (hasUpperNeighbor && localNz > 1) {
                CUDA_CHECK(cudaMemcpyAsync(nextHost + localNz * plane,
                                           nextDevice + localNz * plane, plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            exchangeHalos(nextHost, plane, localNz, rank, size, communicator);

            if (hasLowerNeighbor) {
                CUDA_CHECK(cudaMemcpyAsync(nextDevice, nextHost, plane * sizeof(Real),
                                           cudaMemcpyHostToDevice, stream));
            }
            if (hasUpperNeighbor) {
                CUDA_CHECK(cudaMemcpyAsync(nextDevice + (localNz + 1) * plane,
                                           nextHost + (localNz + 1) * plane, plane * sizeof(Real),
                                           cudaMemcpyHostToDevice, stream));
            }
        }

        std::swap(currentHost, nextHost);
        std::swap(currentDevice, nextDevice);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);

    // Only copy/gather the final owned slabs when a caller asks for output or validation. Halo
    // planes are communication scratch space and are excluded from the global result, preserving
    // the original nx*ny*nz layout without adding an unnecessary end-of-run transfer.
    const bool needFinalGrid = printResults || validate;
    if (needFinalGrid) {
        CUDA_CHECK(cudaMemcpyAsync(currentHost + plane, currentDevice + plane,
                                   localOwnedElements * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (worldRank == 0) {
        const long long elapsedMilliseconds =
            std::max<long long>(1, static_cast<long long>(elapsedSeconds * 1000.0));
        std::printf("Computation time: %lld ms\n", elapsedMilliseconds);
        const double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                                   static_cast<double>(ny > 2 ? ny - 2 : 0) *
                                   static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        const double mcups = elapsedSeconds > 0.0 ? cellUpdates / elapsedSeconds / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> finalGrid;
    std::vector<int> counts;
    std::vector<int> displacements;
    if (needFinalGrid && rank == 0) {
        finalGrid.resize(checkedProduct(nx, ny, nz, worldRank));
        counts.resize(static_cast<std::size_t>(size));
        displacements.resize(static_cast<std::size_t>(size));
        for (int sourceRank = 0; sourceRank < size; ++sourceRank) {
            const std::size_t sourceLocalNz = baseNz +
                                               (static_cast<std::size_t>(sourceRank) < remainder ? 1 : 0);
            const std::size_t sourceZBegin = static_cast<std::size_t>(sourceRank) * baseNz +
                                             std::min<std::size_t>(static_cast<std::size_t>(sourceRank), remainder);
            const std::size_t sourceCount = checkedProduct(sourceLocalNz, plane, 1, worldRank);
            const std::size_t sourceDisplacement = checkedProduct(sourceZBegin, plane, 1, worldRank);
            if (sourceCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                sourceDisplacement > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                abortWithMessage(worldRank, "MPI gather exceeds the MPI int count limit");
            }
            counts[static_cast<std::size_t>(sourceRank)] = static_cast<int>(sourceCount);
            displacements[static_cast<std::size_t>(sourceRank)] = static_cast<int>(sourceDisplacement);
        }
    }

    if (needFinalGrid) {
        MPI_Gatherv(currentHost + plane, static_cast<int>(localOwnedElements), MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, communicator);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(finalGrid)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));
    CUDA_CHECK(cudaFreeHost(hostGrid1));
    CUDA_CHECK(cudaFreeHost(hostGrid2));
    MPI_Comm_free(&sharedCommunicator);
    MPI_Comm_free(&communicator);
    MPI_Finalize();
    return exitCode;
}
