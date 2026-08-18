#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

#include "../common/results_output.hpp"

using Real = double;

// All MPI ranks own a contiguous slab in Z.  Each device allocation has one
// extra plane on either side for the values received from neighboring ranks.
struct Domain {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t planeSize;
    size_t zBegin;
    size_t localNz;
    size_t localElements;
};

__host__ __device__ inline size_t gridIndex(const size_t x, const size_t y,
                                            const size_t z, const size_t nx,
                                            const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

[[noreturn]] void abortWithMessage(const int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

[[noreturn]] void abortWithCudaError(const int rank, const char* operation,
                                     const cudaError_t error) {
    std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK_RANK(rank, operation)                                      \
    do {                                                                       \
        const cudaError_t cudaCheckError = (operation);                       \
        if (cudaCheckError != cudaSuccess) {                                   \
            abortWithCudaError((rank), #operation, cudaCheckError);            \
        }                                                                      \
    } while (false)

__global__ void initializeGridKernel(Real* __restrict__ grid,
                                     const size_t nx, const size_t ny,
                                     const size_t localNz,
                                     const size_t globalZBegin) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && localZ < localNz) {
        const size_t globalZ = globalZBegin + localZ;
        const size_t globalIndex = (globalZ * ny + y) * nx + x;
        // The +1 accounts for the lower halo plane in the local allocation.
        grid[gridIndex(x, y, localZ + 1, nx, ny)] =
            static_cast<Real>(globalIndex % 19);
    }
}

// A shared-memory tiled 7-point stencil.  The tile includes a one-cell halo
// in X/Y and a one-plane halo in Z.  The Z halo is populated by MPI before
// this kernel is launched; the X/Y halos are read from the local allocation.
__global__ void stencilKernelWithGlobalBounds(const Real* __restrict__ input,
                                              Real* __restrict__ output,
                                              const size_t nx, const size_t ny,
                                              const size_t nz,
                                              const size_t localNz,
                                              const size_t globalZBegin) {
    extern __shared__ Real tile[];

    const size_t tileNx = static_cast<size_t>(blockDim.x) + 2;
    const size_t tileNy = static_cast<size_t>(blockDim.y) + 2;
    const size_t tileNz = static_cast<size_t>(blockDim.z) + 2;
    const size_t tilePlane = tileNx * tileNy;
    const size_t tileVolume = tilePlane * tileNz;
    const size_t threadId =
        (static_cast<size_t>(threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x +
        threadIdx.x;
    const size_t threadsPerBlock = static_cast<size_t>(blockDim.x) * blockDim.y *
                                   blockDim.z;

    for (size_t linear = threadId; linear < tileVolume; linear += threadsPerBlock) {
        const size_t tileZ = linear / tilePlane;
        const size_t inPlane = linear - tileZ * tilePlane;
        const size_t tileY = inPlane / tileNx;
        const size_t tileX = inPlane - tileY * tileNx;

        const long long x = static_cast<long long>(blockIdx.x) * blockDim.x +
                            static_cast<long long>(tileX) - 1;
        const long long y = static_cast<long long>(blockIdx.y) * blockDim.y +
                            static_cast<long long>(tileY) - 1;
        const long long localZ = static_cast<long long>(blockIdx.z) * blockDim.z +
                                 static_cast<long long>(tileZ) - 1;

        if (x >= 0 && y >= 0 && x < static_cast<long long>(nx) &&
            y < static_cast<long long>(ny) && localZ >= -1 &&
            localZ <= static_cast<long long>(localNz)) {
            tile[linear] = input[gridIndex(static_cast<size_t>(x),
                                            static_cast<size_t>(y),
                                            static_cast<size_t>(localZ + 1), nx, ny)];
        } else {
            tile[linear] = 0.0;
        }
    }
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || localZ >= localNz) {
        return;
    }

    const size_t sharedX = static_cast<size_t>(threadIdx.x) + 1;
    const size_t sharedY = static_cast<size_t>(threadIdx.y) + 1;
    const size_t sharedZ = static_cast<size_t>(threadIdx.z) + 1;
    const size_t centerIndex = (sharedZ * tileNy + sharedY) * tileNx + sharedX;
    const Real center = tile[centerIndex];
    const size_t globalZ = globalZBegin + localZ;
    const size_t outputIndex = gridIndex(x, y, localZ + 1, nx, ny);

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 ||
        globalZ + 1 == nz) {
        output[outputIndex] = center;
        return;
    }

    const size_t left = centerIndex - 1;
    const size_t right = centerIndex + 1;
    const size_t front = centerIndex - tileNx;
    const size_t back = centerIndex + tileNx;
    const size_t bottom = centerIndex - tilePlane;
    const size_t top = centerIndex + tilePlane;
    output[outputIndex] =
        (center + tile[left] + tile[right] + tile[front] + tile[back] +
         tile[bottom] + tile[top]) /
        7.0;
}

void exchangeZHalos(Real* const deviceInput, const Domain& domain,
                    const int rank, const int worldSize, MPI_Comm communicator,
                    cudaStream_t stream, Real* const sendLower,
                    Real* const sendUpper, Real* const receiveLower,
                    Real* const receiveUpper) {
    const int lowerRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upperRank = rank + 1 == worldSize ? MPI_PROC_NULL : rank + 1;
    const size_t bytes = domain.planeSize * sizeof(Real);

    if (lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK_RANK(rank, cudaMemcpyAsync(
                              sendLower, deviceInput + domain.planeSize, bytes,
                              cudaMemcpyDeviceToHost, stream));
    }
    if (upperRank != MPI_PROC_NULL) {
        CUDA_CHECK_RANK(rank,
                       cudaMemcpyAsync(sendUpper,
                                       deviceInput + domain.localNz * domain.planeSize,
                                       bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK_RANK(rank, cudaStreamSynchronize(stream));

    // Two Sendrecv operations avoid a rank-ordering dependency and work for
    // both one- and multi-rank decompositions.
    MPI_Sendrecv(sendLower, static_cast<int>(domain.planeSize), MPI_DOUBLE,
                 lowerRank, 700, receiveUpper,
                 static_cast<int>(domain.planeSize), MPI_DOUBLE, upperRank, 700,
                 communicator, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendUpper, static_cast<int>(domain.planeSize), MPI_DOUBLE,
                 upperRank, 701, receiveLower,
                 static_cast<int>(domain.planeSize), MPI_DOUBLE, lowerRank, 701,
                 communicator, MPI_STATUS_IGNORE);

    if (lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK_RANK(rank, cudaMemcpyAsync(
                              deviceInput, receiveLower, bytes,
                              cudaMemcpyHostToDevice, stream));
    }
    if (upperRank != MPI_PROC_NULL) {
        CUDA_CHECK_RANK(
            rank, cudaMemcpyAsync(deviceInput + (domain.localNz + 1) * domain.planeSize,
                                  receiveUpper, bytes, cudaMemcpyHostToDevice, stream));
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t nx,
                    const size_t ny, const size_t nz) {
    if (grid.empty()) {
        std::printf("Validation failed: empty grid\n");
        return false;
    }

    int invalid = 0;
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();

    // Validation is intentionally OpenMP-parallel: CUDA handles the stencil,
    // MPI handles domain decomposition, and OpenMP handles this rank-0 scan.
#pragma omp parallel for reduction(| : invalid) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
            invalid = 1;
        } else {
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
        }
    }

    if (invalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }

    (void)nx;
    (void)ny;
    (void)nz;
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

bool parseSize(const char* text, size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return static_cast<unsigned long long>(value) == parsed;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI implementation did not provide MPI_THREAD_FUNNELED");
    }

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], nx) || parseError;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], ny) || parseError;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], nz) || parseError;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (rank == 0 && (showHelp || parseError)) {
        if (parseError) {
            std::printf("Invalid command line arguments\n");
        }
        printUsage(argv[0]);
    }
    if (showHelp || parseError) {
        MPI_Finalize();
        return parseError ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (iterations < 0 || nx == 0 || ny == 0 || nz == 0 ||
        static_cast<size_t>(worldSize) > nz) {
        if (rank == 0) {
            std::printf("Grid dimensions must be nonzero, iterations nonnegative, "
                        "and MPI ranks must not exceed the Z dimension\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const size_t planeSize = nx * ny;
    const size_t baseNz = nz / static_cast<size_t>(worldSize);
    const size_t remainder = nz % static_cast<size_t>(worldSize);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t zBegin = static_cast<size_t>(rank) * baseNz +
                          std::min(static_cast<size_t>(rank), remainder);
    const Domain domain{nx, ny, nz, planeSize, zBegin, localNz,
                        (localNz + 2) * planeSize};

    // Map ranks to GPUs per node instead of using the global MPI rank.  This
    // is the expected layout on accelerator clusters with one or more GPUs
    // attached to every node.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK_RANK(rank, cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortWithMessage(rank, "No CUDA device is available");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK_RANK(rank, cudaSetDevice(device));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK_RANK(rank, cudaGetDeviceProperties(&deviceProperties, device));
    cudaStream_t stream = nullptr;
    CUDA_CHECK_RANK(rank, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    Real* deviceGrid[2] = {nullptr, nullptr};
    const size_t allocationBytes = domain.localElements * sizeof(Real);
    CUDA_CHECK_RANK(rank, cudaMalloc(reinterpret_cast<void**>(&deviceGrid[0]),
                                     allocationBytes));
    CUDA_CHECK_RANK(rank, cudaMalloc(reinterpret_cast<void**>(&deviceGrid[1]),
                                     allocationBytes));

    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* receiveLower = nullptr;
    Real* receiveUpper = nullptr;
    if (worldSize > 1) {
        CUDA_CHECK_RANK(rank, cudaHostAlloc(reinterpret_cast<void**>(&sendLower),
                                            planeSize * sizeof(Real),
                                            cudaHostAllocPortable));
        CUDA_CHECK_RANK(rank, cudaHostAlloc(reinterpret_cast<void**>(&sendUpper),
                                            planeSize * sizeof(Real),
                                            cudaHostAllocPortable));
        CUDA_CHECK_RANK(rank, cudaHostAlloc(reinterpret_cast<void**>(&receiveLower),
                                            planeSize * sizeof(Real),
                                            cudaHostAllocPortable));
        CUDA_CHECK_RANK(rank, cudaHostAlloc(reinterpret_cast<void**>(&receiveUpper),
                                            planeSize * sizeof(Real),
                                            cudaHostAllocPortable));
    }

    const dim3 block(32, 4, 2);
    const dim3 grid((static_cast<unsigned int>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(ny) + block.y - 1) / block.y,
                    (static_cast<unsigned int>(localNz) + block.z - 1) / block.z);
    const size_t sharedBytes = static_cast<size_t>(block.x + 2) *
                               (block.y + 2) * (block.z + 2) * sizeof(Real);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
                    worldSize, omp_get_max_threads(), deviceCount);
        std::printf("CUDA device: %s\n", deviceProperties.name);
    }

    initializeGridKernel<<<grid, block, 0, stream>>>(
        deviceGrid[0], nx, ny, localNz, zBegin);
    CUDA_CHECK_RANK(rank, cudaGetLastError());

    if (rank == 0) {
        std::printf("Initializing grid...\n");
        std::printf("Running stencil computation...\n");
    }
    CUDA_CHECK_RANK(rank, cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        Real* input = deviceGrid[iteration & 1];
        Real* output = deviceGrid[(iteration + 1) & 1];

        if (worldSize > 1) {
            exchangeZHalos(input, domain, rank, worldSize, MPI_COMM_WORLD, stream,
                           sendLower, sendUpper, receiveLower, receiveUpper);
        }

        stencilKernelWithGlobalBounds<<<grid, block, sharedBytes, stream>>>(
            input, output, nx, ny, nz, localNz, zBegin);
        CUDA_CHECK_RANK(rank, cudaGetLastError());
    }
    CUDA_CHECK_RANK(rank, cudaStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();

    const double localSeconds =
        std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<Real> localResult(localNz * planeSize);
    const Real* finalDeviceGrid = deviceGrid[iterations & 1];
    CUDA_CHECK_RANK(rank, cudaMemcpy(localResult.data(),
                                     finalDeviceGrid + planeSize,
                                     localResult.size() * sizeof(Real),
                                     cudaMemcpyDeviceToHost));

    int localCount = 0;
    if (localResult.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(rank, "A local slab is too large for MPI_Gatherv");
    }
    localCount = static_cast<int>(localResult.size());

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<Real> finalGrid;
    if (rank == 0) {
        receiveCounts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
    }
    MPI_Gather(&localCount, 1, MPI_INT,
               rank == 0 ? receiveCounts.data() : nullptr, 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        finalGrid.resize(nx * ny * nz);
        for (int sourceRank = 0; sourceRank < worldSize; ++sourceRank) {
            const size_t sourceLocalNz = baseNz +
                (static_cast<size_t>(sourceRank) < remainder ? 1 : 0);
            const size_t sourceZBegin = static_cast<size_t>(sourceRank) * baseNz +
                std::min(static_cast<size_t>(sourceRank), remainder);
            const size_t displacement = sourceZBegin * planeSize;
            if (displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
                abortWithMessage(rank, "The global grid is too large for MPI_Gatherv");
            }
            displacements[static_cast<size_t>(sourceRank)] =
                static_cast<int>(displacement);
            if (sourceLocalNz * planeSize !=
                static_cast<size_t>(receiveCounts[static_cast<size_t>(sourceRank)])) {
                abortWithMessage(rank, "Inconsistent MPI slab sizes");
            }
        }
    }

    MPI_Gatherv(localResult.data(), localCount, MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr,
                rank == 0 ? receiveCounts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        const long double interiorCells =
            (nx > 2 && ny > 2 && nz > 2)
                ? static_cast<long double>(nx - 2) * static_cast<long double>(ny - 2) *
                      static_cast<long double>(nz - 2)
                : 0.0L;
        const double cellUpdates = static_cast<double>(interiorCells) * iterations;
        const double mcups = elapsedSeconds > 0.0
                                 ? cellUpdates / elapsedSeconds / 1.0e6
                                 : 0.0;
        std::printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(finalGrid, "Grid");
        }
    }

    int validationPassed = 1;
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
        validationPassed = validateResult(finalGrid, nx, ny, nz) ? 1 : 0;
        std::printf("Validation: %s\n", validationPassed != 0 ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (sendLower != nullptr) {
        CUDA_CHECK_RANK(rank, cudaFreeHost(sendLower));
        CUDA_CHECK_RANK(rank, cudaFreeHost(sendUpper));
        CUDA_CHECK_RANK(rank, cudaFreeHost(receiveLower));
        CUDA_CHECK_RANK(rank, cudaFreeHost(receiveUpper));
    }
    CUDA_CHECK_RANK(rank, cudaFree(deviceGrid[0]));
    CUDA_CHECK_RANK(rank, cudaFree(deviceGrid[1]));
    CUDA_CHECK_RANK(rank, cudaStreamDestroy(stream));
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return validationPassed != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
