#include <algorithm>
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

// The source is compiled by nvcc. MPI decomposes the domain in Z, while each
// rank computes its local slab on one GPU. OpenMP is used for the host-side
// work that prepares and checks each rank's data.

inline constexpr std::size_t idx3(const std::size_t x,
                                  const std::size_t y,
                                  const std::size_t z,
                                  const std::size_t nx,
                                  const std::size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

[[noreturn]] void abortWithMessage(const int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t result, const char* expression, const int rank) {
    if (result != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed: %s", expression,
                      cudaGetErrorString(result));
        abortWithMessage(rank, message);
    }
}

void checkMpi(const int result, const char* expression, const int rank) {
    if (result != MPI_SUCCESS) {
        char errorString[MPI_MAX_ERROR_STRING];
        int errorLength = 0;
        MPI_Error_string(result, errorString, &errorLength);
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed: %.*s", expression,
                      errorLength, errorString);
        abortWithMessage(rank, message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, rank)
#define MPI_CHECK(call) checkMpi((call), #call, rank)

// Each CUDA thread handles consecutive X/Y elements. The grid-stride loop
// also avoids a CUDA grid-dimension limit for very deep local slabs.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const std::size_t nx,
                              const std::size_t ny,
                              const std::size_t nz,
                              const std::size_t globalZStart,
                              const std::size_t localZFirst,
                              const std::size_t localZLast) {
    const std::size_t plane = nx * ny;
    const std::size_t planeCount = localZLast - localZFirst;
    const std::size_t workItems = planeCount * plane;
    const std::size_t threadId = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                                 threadIdx.x;
    const std::size_t threadCount = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t item = threadId; item < workItems; item += threadCount) {
        const std::size_t localZ = localZFirst + item / plane;
        const std::size_t xy = item % plane;
        const std::size_t x = xy % nx;
        const std::size_t y = xy / nx;
        const std::size_t index = localZ * plane + xy;
        const std::size_t globalZ = globalZStart + localZ - 1;

        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
            globalZ == 0 || globalZ == nz - 1) {
            output[index] = input[index];
            continue;
        }

        // Keep the same left-to-right addition order as the original code.
        Real sum = input[index];
        sum += input[index - 1];
        sum += input[index + 1];
        sum += input[index - nx];
        sum += input[index + nx];
        sum += input[index - plane];
        sum += input[index + plane];
        output[index] = sum / 7.0;
    }
}

void initializeLocalGrid(std::vector<Real>& grid,
                         const std::size_t nx,
                         const std::size_t ny,
                         const std::size_t nz,
                         const std::size_t localNz,
                         const std::size_t globalZStart) {
    const std::size_t plane = nx * ny;
    const long long localPlaneCount = static_cast<long long>(localNz + 2);
    const long long planeElements = static_cast<long long>(plane);

    // The two extra planes are the MPI halos. Their initial values are filled
    // as well, so the first iteration has valid data before the first exchange.
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long localZ = 0; localZ < localPlaneCount; ++localZ) {
        for (long long xy = 0; xy < planeElements; ++xy) {
            const long long globalZ = static_cast<long long>(globalZStart) + localZ - 1;
            const std::size_t localIndex = static_cast<std::size_t>(localZ) * plane +
                                           static_cast<std::size_t>(xy);
            if (globalZ >= 0 && static_cast<std::size_t>(globalZ) < nz) {
                const std::size_t globalIndex = static_cast<std::size_t>(globalZ) * plane +
                                                static_cast<std::size_t>(xy);
                grid[localIndex] = static_cast<Real>(globalIndex % 19);
            } else {
                grid[localIndex] = 0.0;
            }
        }
    }
}

void launchStencil(const Real* input,
                   Real* output,
                   const std::size_t nx,
                   const std::size_t ny,
                   const std::size_t nz,
                   const std::size_t globalZStart,
                   const std::size_t localZFirst,
                   const std::size_t localZLast,
                   cudaStream_t stream,
                   const int rank) {
    if (localZFirst >= localZLast) {
        return;
    }

    const std::size_t workItems = (localZLast - localZFirst) * nx * ny;
    constexpr unsigned int threadsPerBlock = 256;
    const std::size_t requiredBlocks =
        (workItems + threadsPerBlock - 1) / threadsPerBlock;
    constexpr std::size_t maxGridX = 2147483647ULL;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(requiredBlocks, maxGridX));

    stencilKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        input, output, nx, ny, nz, globalZStart, localZFirst, localZLast);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid,
                   [[maybe_unused]] const std::size_t nx,
                   [[maybe_unused]] const std::size_t ny,
                   [[maybe_unused]] const std::size_t nz) {
    if (grid.empty()) {
        std::printf("Validation failed: empty grid\n");
        return false;
    }

    int hasInvalid = 0;
    Real minVal = std::numeric_limits<Real>::max();
    Real maxVal = std::numeric_limits<Real>::lowest();

    #pragma omp parallel for reduction(|:hasInvalid) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(grid.size()); ++i) {
        const Real value = grid[static_cast<std::size_t>(i)];
        if (std::isnan(value) || std::isinf(value)) {
            hasInvalid = 1;
        }
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }

    if (hasInvalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
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
    std::size_t nx = 128;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        abortWithMessage(rank, "Grid dimensions must be at least 3 and iterations non-negative");
    }
    if (static_cast<std::size_t>(worldSize) > nz) {
        abortWithMessage(rank, "MPI world size cannot exceed the Z dimension");
    }

    omp_set_dynamic(0);

    // One rank per GPU is the intended accelerator-cluster layout. The
    // shared-memory rank maps correctly even when MPI rank numbering spans
    // multiple nodes, and CUDA_VISIBLE_DEVICES remains respected.
    MPI_Comm sharedComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &sharedComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(sharedComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortWithMessage(rank, "No CUDA devices are visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    MPI_CHECK(MPI_Comm_free(&sharedComm));

    const std::size_t basePlanes = nz / static_cast<std::size_t>(worldSize);
    const std::size_t extraPlanes = nz % static_cast<std::size_t>(worldSize);
    const std::size_t localNz = basePlanes +
                                (static_cast<std::size_t>(rank) < extraPlanes ? 1 : 0);
    const std::size_t globalZStart =
        static_cast<std::size_t>(rank) * basePlanes +
        std::min(static_cast<std::size_t>(rank), extraPlanes);
    const std::size_t plane = nx * ny;
    const std::size_t localElements = (localNz + 2) * plane;
    const std::size_t planeBytes = plane * sizeof(Real);
    const std::size_t localBytes = localElements * sizeof(Real);

    if (plane > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(rank, "An XY plane is too large for the MPI implementation's count type");
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA device: %s\n", deviceProperties.name);
    }

    std::vector<Real> initialGrid(localElements);
    initializeLocalGrid(initialGrid, nx, ny, nz, localNz, globalZStart);

    Real* gridA = nullptr;
    Real* gridB = nullptr;
    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* receiveLower = nullptr;
    Real* receiveUpper = nullptr;
    cudaStream_t copyStream = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaEvent_t haloReady = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gridA), localBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gridB), localBytes));
    CUDA_CHECK(cudaMemcpy(gridA, initialGrid.data(), localBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendLower), planeBytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendUpper), planeBytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&receiveLower), planeBytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&receiveUpper), planeBytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));

    const int previousRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int nextRank = rank == worldSize - 1 ? MPI_PROC_NULL : rank + 1;
    const int mpiPlaneCount = static_cast<int>(plane);
    Real* input = gridA;
    Real* output = gridB;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Copy only the two faces needed by neighboring ranks. The copies are
        // issued on a separate stream so the interior CUDA work can overlap
        // the MPI exchange below.
        if (previousRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendLower, input + plane, planeBytes,
                                       cudaMemcpyDeviceToHost, copyStream));
        }
        if (nextRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendUpper, input + localNz * plane,
                                       planeBytes, cudaMemcpyDeviceToHost,
                                       copyStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(copyStream));

        MPI_Request requests[4];
        MPI_CHECK(MPI_Irecv(receiveLower, mpiPlaneCount, MPI_DOUBLE, previousRank,
                            100, MPI_COMM_WORLD, &requests[0]));
        MPI_CHECK(MPI_Irecv(receiveUpper, mpiPlaneCount, MPI_DOUBLE, nextRank,
                            101, MPI_COMM_WORLD, &requests[1]));
        MPI_CHECK(MPI_Isend(sendLower, mpiPlaneCount, MPI_DOUBLE, previousRank,
                            101, MPI_COMM_WORLD, &requests[2]));
        MPI_CHECK(MPI_Isend(sendUpper, mpiPlaneCount, MPI_DOUBLE, nextRank,
                            100, MPI_COMM_WORLD, &requests[3]));

        // Local planes strictly between the two slab faces do not depend on
        // MPI data and can run while the nonblocking exchange progresses.
        launchStencil(input, output, nx, ny, nz, globalZStart, 2, localNz,
                      computeStream, rank);

        MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));

        if (previousRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(input, receiveLower, planeBytes,
                                       cudaMemcpyHostToDevice, copyStream));
        }
        if (nextRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(input + (localNz + 1) * plane, receiveUpper,
                                       planeBytes, cudaMemcpyHostToDevice,
                                       copyStream));
        }
        CUDA_CHECK(cudaEventRecord(haloReady, copyStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, haloReady, 0));

        // The two slab faces now have valid neighbor halos. For a one-plane
        // slab, launch only the first face because both faces are identical.
        launchStencil(input, output, nx, ny, nz, globalZStart, 1, 2,
                      computeStream, rank);
        if (localNz > 1) {
            launchStencil(input, output, nx, ny, nz, globalZStart, localNz,
                          localNz + 1, computeStream, rank);
        }
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        std::swap(input, output);
    }

    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD));

    std::vector<Real> localResult(localNz * plane);
    CUDA_CHECK(cudaMemcpy(localResult.data(), input + plane,
                          localResult.size() * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    if (localResult.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(rank, "A local gathered slab exceeds MPI's count range");
    }
    const int localResultCount = static_cast<int>(localResult.size());
    std::vector<int> resultCounts(static_cast<std::size_t>(worldSize), 0);
    std::vector<int> resultDisplacements(static_cast<std::size_t>(worldSize), 0);
    MPI_CHECK(MPI_Gather(&localResultCount, 1, MPI_INT, resultCounts.data(), 1,
                         MPI_INT, 0, MPI_COMM_WORLD));

    std::vector<Real> finalGrid;
    bool validationPassed = true;
    if (rank == 0) {
        finalGrid.resize(nx * ny * nz);
        int displacement = 0;
        for (int r = 0; r < worldSize; ++r) {
            resultDisplacements[static_cast<std::size_t>(r)] = displacement;
            if (resultCounts[static_cast<std::size_t>(r)] >
                std::numeric_limits<int>::max() - displacement) {
                abortWithMessage(rank, "The gathered grid exceeds MPI's count range");
            }
            displacement += resultCounts[static_cast<std::size_t>(r)];
        }
    }

    MPI_CHECK(MPI_Gatherv(localResult.data(), localResultCount, MPI_DOUBLE,
                          rank == 0 ? finalGrid.data() : nullptr,
                          resultCounts.data(), resultDisplacements.data(),
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        const double milliseconds = elapsedSeconds * 1000.0;
        std::printf("Computation time: %.0f ms\n", milliseconds);
        const double cellUpdates = static_cast<double>(nx - 2) *
                                   static_cast<double>(ny - 2) *
                                   static_cast<double>(nz - 2) *
                                   static_cast<double>(iterations);
        const double mcups = elapsedSeconds > 0.0
                                 ? cellUpdates / elapsedSeconds / 1e6
                                 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        if (validate) {
            std::printf("Validating result...\n");
            validationPassed = validateResult(finalGrid, nx, ny, nz);
            std::printf("Validation: %s\n", validationPassed ? "PASSED" : "FAILED");
        }
    }

    // All ranks release the accelerator resources. Validation is performed on
    // rank zero above, but cleanup must happen on every rank.
    CUDA_CHECK(cudaEventDestroy(haloReady));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaFreeHost(receiveUpper));
    CUDA_CHECK(cudaFreeHost(receiveLower));
    CUDA_CHECK(cudaFreeHost(sendUpper));
    CUDA_CHECK(cudaFreeHost(sendLower));
    CUDA_CHECK(cudaFree(gridB));
    CUDA_CHECK(cudaFree(gridA));
    MPI_CHECK(MPI_Finalize());
    return (rank == 0 && validate && !validationPassed) ? 1 : 0;
}
