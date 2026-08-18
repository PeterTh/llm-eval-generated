#include <algorithm>
#include <cmath>
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

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 2;
constexpr int kThreads = 256;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void checkCuda(const cudaError_t error, const char* expression, const int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank,
                     expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, activeRank)

// The local array has one halo plane below and above the owned Z slab.  The
// host initialization is deliberately OpenMP-parallel: it is also used to
// populate the pinned-free input buffer before its one-time H2D transfer.
void initializeLocalGrid(std::vector<Real>& grid, const size_t nx,
                         const size_t ny, const size_t nz,
                         const size_t localNz, const size_t globalZStart) {
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t localZ = 0; localZ < localNz + 2; ++localZ) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const bool owned = localZ != 0 && localZ != localNz + 1;
                if (!owned && (globalZStart == 0 && localZ == 0)) {
                    grid[idx3(x, y, localZ, nx, ny)] = 0.0;
                    continue;
                }

                const size_t globalZ = globalZStart + localZ - 1;
                if (globalZ >= nz) {
                    grid[idx3(x, y, localZ, nx, ny)] = 0.0;
                } else {
                    const size_t globalIndex = idx3(x, y, globalZ, nx, ny);
                    grid[idx3(x, y, localZ, nx, ny)] =
                        static_cast<Real>(globalIndex % 19);
                }
            }
        }
    }
}

// A shared-memory tiled kernel handles all non-boundary Z planes.  X/Y edge
// cells are copied exactly as in the original implementation.  The tile
// dimensions give coalesced X accesses while reusing the Y/Z neighbors.
__global__ void stencilInteriorKernel(const Real* __restrict__ input,
                                      Real* __restrict__ output,
                                      const size_t nx, const size_t ny,
                                      const size_t localNz,
                                      const size_t planeSize) {
    extern __shared__ Real tile[];
    constexpr int tileX = kBlockX + 2;
    constexpr int tileY = kBlockY + 2;
    constexpr int tileZ = kBlockZ + 2;

    const size_t zBase = 2 + static_cast<size_t>(blockIdx.z) * kBlockZ;
    const size_t xBase = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t yBase = static_cast<size_t>(blockIdx.y) * kBlockY;

    const int threadLinear =
        (static_cast<int>(threadIdx.z) * kBlockY +
         static_cast<int>(threadIdx.y)) * kBlockX +
        static_cast<int>(threadIdx.x);
    constexpr int tileElements = tileX * tileY * tileZ;

    for (int element = threadLinear; element < tileElements;
         element += kThreads) {
        const int sx = element % tileX;
        const int sy = (element / tileX) % tileY;
        const int sz = element / (tileX * tileY);

        const size_t rawX = xBase + static_cast<size_t>(sx);
        const size_t rawY = yBase + static_cast<size_t>(sy);
        // X/Y can be partial tiles.  Clamping is only used by threads outside
        // the output domain; every valid stencil neighbor remains unchanged.
        const size_t x = rawX == 0 ? 0 : (rawX - 1 < nx ? rawX - 1 : nx - 1);
        const size_t y = rawY == 0 ? 0 : (rawY - 1 < ny ? rawY - 1 : ny - 1);
        const size_t z = zBase + static_cast<size_t>(sz) - 1;
        tile[(sz * tileY + sy) * tileX + sx] =
            input[z * planeSize + y * nx + x];
    }
    __syncthreads();

    const size_t x = xBase + static_cast<size_t>(threadIdx.x);
    const size_t y = yBase + static_cast<size_t>(threadIdx.y);
    const size_t z = zBase + static_cast<size_t>(threadIdx.z);
    if (x >= nx || y >= ny || z >= localNz) {
        return;
    }

    const size_t outputIndex = z * planeSize + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny) {
        output[outputIndex] = input[outputIndex];
        return;
    }

    const int sx = static_cast<int>(threadIdx.x) + 1;
    const int sy = static_cast<int>(threadIdx.y) + 1;
    const int sz = static_cast<int>(threadIdx.z) + 1;
    const int center = (sz * tileY + sy) * tileX + sx;
    const Real value = tile[center] + tile[center - 1] + tile[center + 1] +
                       tile[center - tileX] + tile[center + tileX] +
                       tile[center - tileX * tileY] +
                       tile[center + tileX * tileY];
    output[outputIndex] = value / 7.0;
}

// Only the two owned planes adjacent to an MPI halo are handled here.  The
// kernel is launched after the received halos have reached device memory.
__global__ void stencilBoundaryKernel(const Real* __restrict__ input,
                                       Real* __restrict__ output,
                                       const size_t nx, const size_t ny,
                                       const size_t nz,
                                       const size_t localNz,
                                       const size_t globalZStart,
                                       const size_t planeSize,
                                       const size_t planeCount) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x +
                          static_cast<size_t>(threadIdx.x);
    if (linear >= planeCount * planeSize) {
        return;
    }

    const size_t plane = linear / planeSize;
    const size_t planeOffset = linear - plane * planeSize;
    const size_t localZ = plane == 0 ? 1 : localNz;
    const size_t y = planeOffset / nx;
    const size_t x = planeOffset - y * nx;
    const size_t globalZ = globalZStart + localZ - 1;
    const size_t index = localZ * planeSize + planeOffset;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 ||
        globalZ + 1 == nz) {
        output[index] = input[index];
        return;
    }

    const Real value = input[index] + input[index - 1] + input[index + 1] +
                       input[index - nx] + input[index + nx] +
                       input[index - planeSize] + input[index + planeSize];
    output[index] = value / 7.0;
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

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool valid = true;
};

Options parseOptions(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.nz = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            options.valid = false;
            break;
        }
    }
    if (options.ny == 0) {
        options.ny = options.nx;
    }
    if (options.nz == 0) {
        options.nz = options.nx;
    }
    return options;
}

bool validateDistributed(const std::vector<Real>& localResult,
                         const size_t nx, const size_t ny, const size_t nz,
                         MPI_Comm comm, const int rank) {
    int localInvalid = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();

#pragma omp parallel for reduction(| : localInvalid) reduction(min : localMin) \
    reduction(max : localMax) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(localResult.size());
         ++i) {
        const Real value = localResult[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
            localInvalid |= 1;
        } else {
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    }

    int globalInvalid = 0;
    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_BOR, comm);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    int valid = (globalInvalid == 0 && globalMax <= 1e6 &&
                 globalMin >= -1e6)
                    ? 1
                    : 0;
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    (void)nx;
    (void)ny;
    (void)nz;
    return valid != 0;
}

bool gatherResults(const std::vector<Real>& localResult, const size_t nx,
                   const size_t ny, const size_t nz, MPI_Comm comm,
                   const int rank, const int size, std::vector<Real>& result) {
    const size_t planeSize = nx * ny;
    std::vector<int> counts;
    std::vector<int> displacements;
    int gatherable = 1;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(size));
        displacements.resize(static_cast<size_t>(size));
        const size_t base = nz / static_cast<size_t>(size);
        const size_t remainder = nz % static_cast<size_t>(size);
        for (int r = 0; r < size; ++r) {
            const size_t localNz = base +
                                   (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t globalZStart =
                static_cast<size_t>(r) * base +
                std::min(static_cast<size_t>(r), remainder);
            const size_t count = localNz * planeSize;
            const size_t displacement = globalZStart * planeSize;
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                displacement >
                    static_cast<size_t>(std::numeric_limits<int>::max())) {
                std::fprintf(stderr,
                             "Result gather exceeds MPI int count limits\n");
                gatherable = 0;
                break;
            }
            counts[static_cast<size_t>(r)] = static_cast<int>(count);
            displacements[static_cast<size_t>(r)] =
                static_cast<int>(displacement);
        }
        if (gatherable != 0) {
            result.resize(nx * ny * nz);
        }
    }

    MPI_Bcast(&gatherable, 1, MPI_INT, 0, comm);
    if (gatherable == 0) {
        return false;
    }

    const int localCount = static_cast<int>(localResult.size());
    MPI_Gatherv(localResult.data(), localCount, MPI_DOUBLE,
                rank == 0 ? result.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                comm);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const Options options = parseOptions(argc, argv);
    if (options.help || !options.valid) {
        if (worldRank == 0) {
            if (!options.valid) {
                std::printf("Unknown or incomplete option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return options.help ? 0 : 1;
    }

    const bool planeFitsSizeT =
        options.nx != 0 &&
        options.ny <= std::numeric_limits<size_t>::max() / options.nx;
    const size_t planeSize =
        planeFitsSizeT ? options.nx * options.ny : size_t{0};
    const bool gridFitsSizeT =
        planeFitsSizeT && planeSize != 0 && options.nz <=
                              std::numeric_limits<size_t>::max() / planeSize;
    if (provided < MPI_THREAD_FUNNELED || options.nx < 3 || options.ny < 3 ||
        options.nz < 3 || options.iterations < 0 ||
        options.nz > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        !gridFitsSizeT ||
        planeSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::printf("Invalid grid dimensions or iteration count\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int activeSize = std::min(
        worldSize, static_cast<int>(options.nz));
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &activeComm);
    if (worldRank >= activeSize) {
        MPI_Finalize();
        return 0;
    }

    int activeRank = 0;
    int activeRanks = 1;
    MPI_Comm_rank(activeComm, &activeRank);
    MPI_Comm_size(activeComm, &activeRanks);

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(activeComm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device is available\n",
                     activeRank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    const size_t baseNz = options.nz / static_cast<size_t>(activeRanks);
    const size_t remainder = options.nz % static_cast<size_t>(activeRanks);
    const size_t localNz = baseNz +
                           (static_cast<size_t>(activeRank) < remainder ? 1 : 0);
    const size_t globalZStart =
        static_cast<size_t>(activeRank) * baseNz +
        std::min(static_cast<size_t>(activeRank), remainder);
    const size_t localCells = (localNz + 2) * planeSize;
    const size_t ownedCells = localNz * planeSize;
    const size_t planeBytes = planeSize * sizeof(Real);
    const size_t localBytes = localCells * sizeof(Real);

    const int previousRank = activeRank == 0 ? MPI_PROC_NULL : activeRank - 1;
    const int nextRank = activeRank + 1 == activeRanks ? MPI_PROC_NULL
                                                        : activeRank + 1;

    if (activeRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny,
                    options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n",
                    options.validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", activeRanks);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA devices per node: %d\n", deviceCount);
    }

    if (activeRank == 0) {
        std::printf("Initializing grid...\n");
    }
    std::vector<Real> hostInitial(localCells);
    initializeLocalGrid(hostInitial, options.nx, options.ny, options.nz,
                        localNz, globalZStart);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), localBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), localBytes));

    cudaStream_t computeStream = nullptr;
    cudaStream_t transferStream = nullptr;
    cudaEvent_t halosReady = nullptr;
    cudaEvent_t computeDone = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream,
                                         cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&computeDone, cudaEventDisableTiming));

    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* recvLower = nullptr;
    Real* recvUpper = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendLower), planeBytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendUpper), planeBytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recvLower), planeBytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recvUpper), planeBytes,
                             cudaHostAllocPortable));

    CUDA_CHECK(cudaMemcpyAsync(deviceGrid1, hostInitial.data(), localBytes,
                               cudaMemcpyHostToDevice, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    if (activeRank == 0) {
        std::printf("Running stencil computation...\n");
    }

    const size_t interiorLayers = localNz > 2 ? localNz - 2 : 0;
    const size_t interiorCells = interiorLayers * planeSize;
    const size_t boundaryPlanes = localNz == 1 ? 1 : 2;
    const size_t boundaryCells = boundaryPlanes * planeSize;
    const size_t sharedBytes = static_cast<size_t>(kBlockX + 2) *
                               static_cast<size_t>(kBlockY + 2) *
                               static_cast<size_t>(kBlockZ + 2) * sizeof(Real);
    const dim3 interiorBlock(kBlockX, kBlockY, kBlockZ);
    const dim3 interiorGrid(
        static_cast<unsigned>((options.nx + kBlockX - 1) / kBlockX),
        static_cast<unsigned>((options.ny + kBlockY - 1) / kBlockY),
        static_cast<unsigned>((interiorLayers + kBlockZ - 1) / kBlockZ));
    const dim3 boundaryGrid(static_cast<unsigned>(
        (boundaryCells + kThreads - 1) / kThreads));

    Real* input = deviceGrid1;
    Real* output = deviceGrid2;
    MPI_Barrier(activeComm);
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        if (interiorCells != 0) {
            stencilInteriorKernel<<<interiorGrid, interiorBlock, sharedBytes,
                                    computeStream>>>(
                input, output, options.nx, options.ny, localNz, planeSize);
            CUDA_CHECK(cudaGetLastError());
        }

        // The previous iteration's output must be complete before its owned
        // boundary planes are copied to the MPI staging buffers.  Interior
        // work for this iteration is already queued in the other stream.
        if (iteration != 0) {
            CUDA_CHECK(cudaStreamWaitEvent(transferStream, computeDone, 0));
        }
        CUDA_CHECK(cudaMemcpyAsync(sendLower, input + planeSize, planeBytes,
                                   cudaMemcpyDeviceToHost, transferStream));
        CUDA_CHECK(cudaMemcpyAsync(sendUpper,
                                   input + localNz * planeSize, planeBytes,
                                   cudaMemcpyDeviceToHost, transferStream));
        CUDA_CHECK(cudaStreamSynchronize(transferStream));

        MPI_Request requests[4];
        MPI_Irecv(recvLower, static_cast<int>(planeSize), MPI_DOUBLE,
                  previousRank, 101, activeComm, &requests[0]);
        MPI_Irecv(recvUpper, static_cast<int>(planeSize), MPI_DOUBLE, nextRank,
                  100, activeComm, &requests[1]);
        MPI_Isend(sendLower, static_cast<int>(planeSize), MPI_DOUBLE,
                  previousRank, 100, activeComm, &requests[2]);
        MPI_Isend(sendUpper, static_cast<int>(planeSize), MPI_DOUBLE, nextRank,
                  101, activeComm, &requests[3]);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        if (previousRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(input, recvLower, planeBytes,
                                       cudaMemcpyHostToDevice,
                                       transferStream));
        }
        if (nextRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(input + (localNz + 1) * planeSize,
                                       recvUpper, planeBytes,
                                       cudaMemcpyHostToDevice, transferStream));
        }
        CUDA_CHECK(cudaEventRecord(halosReady, transferStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, halosReady, 0));

        stencilBoundaryKernel<<<boundaryGrid, kThreads, 0, computeStream>>>(
            input, output, options.nx, options.ny, options.nz, localNz,
            globalZStart, planeSize, boundaryPlanes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(computeDone, computeStream));

        std::swap(input, output);
    }

    if (options.iterations != 0) {
        CUDA_CHECK(cudaEventSynchronize(computeDone));
    }
    MPI_Barrier(activeComm);
    double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               activeComm);

    std::vector<Real> localResult(ownedCells);
    CUDA_CHECK(cudaMemcpy(localResult.data(), input + planeSize,
                          ownedCells * sizeof(Real), cudaMemcpyDeviceToHost));

    if (activeRank == 0) {
        const long long milliseconds = static_cast<long long>(
            std::llround(maximumElapsed * 1000.0));
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double updates = static_cast<double>(options.nx - 2) *
                               static_cast<double>(options.ny - 2) *
                               static_cast<double>(options.nz - 2) *
                               static_cast<double>(options.iterations);
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    maximumElapsed > 0.0 ? updates / maximumElapsed / 1e6
                                         : 0.0);
    }

    std::vector<Real> finalGrid;
    if (options.printResults) {
        const bool gathered = gatherResults(localResult, options.nx,
                                            options.ny, options.nz, activeComm,
                                            activeRank, activeRanks, finalGrid);
        if (!gathered) {
            CUDA_CHECK(cudaFreeHost(sendLower));
            CUDA_CHECK(cudaFreeHost(sendUpper));
            CUDA_CHECK(cudaFreeHost(recvLower));
            CUDA_CHECK(cudaFreeHost(recvUpper));
            CUDA_CHECK(cudaFree(deviceGrid1));
            CUDA_CHECK(cudaFree(deviceGrid2));
            MPI_Comm_free(&nodeComm);
            MPI_Comm_free(&activeComm);
            MPI_Finalize();
            return 1;
        }
        if (activeRank == 0) {
            print_results(finalGrid, "Grid");
        }
    }

    bool valid = true;
    if (options.validate) {
        if (activeRank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateDistributed(localResult, options.nx, options.ny,
                                    options.nz, activeComm, activeRank);
        if (activeRank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaFreeHost(sendLower));
    CUDA_CHECK(cudaFreeHost(sendUpper));
    CUDA_CHECK(cudaFreeHost(recvLower));
    CUDA_CHECK(cudaFreeHost(recvUpper));
    CUDA_CHECK(cudaEventDestroy(halosReady));
    CUDA_CHECK(cudaEventDestroy(computeDone));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(transferStream));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));
    MPI_Comm_free(&nodeComm);
    MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
