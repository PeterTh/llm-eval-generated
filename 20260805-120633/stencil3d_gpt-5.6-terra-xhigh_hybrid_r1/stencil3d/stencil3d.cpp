#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kTagToLower = 0;
constexpr int kTagToUpper = 1;

struct Slab {
    size_t globalBegin;
    size_t localDepth;
    int lowerRank;
    int upperRank;
};

struct DeviceBuffers {
    Real* current = nullptr;
    Real* next = nullptr;
    Real* localHost = nullptr;
    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* recvLower = nullptr;
    Real* recvUpper = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaStream_t transferStream = nullptr;
    cudaEvent_t halosReady = nullptr;
};

[[noreturn]] void abortRun(const char* message, const char* file, const int line) {
    std::fprintf(stderr, "stencil3d error at %s:%d: %s\n", file, line, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed: %s", expression, cudaGetErrorString(status));
        abortRun(message, file, line);
    }
}

void checkMpi(const int status, const char* expression, const char* file, const int line) {
    if (status != MPI_SUCCESS) {
        char mpiMessage[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, mpiMessage, &length);
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed: %.*s", expression, length, mpiMessage);
        abortRun(message, file, line);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)
#define MPI_CHECK(expression) checkMpi((expression), #expression, __FILE__, __LINE__)

size_t checkedProduct(const size_t first, const size_t second, const char* description) {
    if (first != 0 && second > std::numeric_limits<size_t>::max() / first) {
        abortRun(description, __FILE__, __LINE__);
    }
    return first * second;
}

Slab makeSlab(const size_t globalDepth, const int rank, const int worldSize) {
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t baseDepth = globalDepth / ranks;
    const size_t remainder = globalDepth % ranks;
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t localDepth = baseDepth + (rankIndex < remainder ? 1 : 0);
    const size_t globalBegin = rankIndex * baseDepth + std::min(rankIndex, remainder);

    // Non-empty slabs are always the leading contiguous ranks with this partitioning.
    const int activeRanks = static_cast<int>(std::min(globalDepth, ranks));
    const int lowerRank = (rank > 0 && rank < activeRanks) ? rank - 1 : MPI_PROC_NULL;
    const int upperRank = (rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    return {globalBegin, localDepth, lowerRank, upperRank};
}

void initializeGrid(Real* const grid, const size_t localElements, const size_t globalOffset) {
    // Initialization is host work by design; the OpenMP loop keeps startup scalable
    // while CUDA owns all stencil updates.
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(localElements); ++i) {
        const size_t globalIndex = globalOffset + static_cast<size_t>(i);
        grid[i] = static_cast<Real>(globalIndex % 19);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx,
                              const size_t ny,
                              const size_t globalBegin,
                              const size_t globalDepth,
                              const size_t firstPlane,
                              const size_t lastPlane) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }

    const size_t planeSize = nx * ny;
    for (size_t localZ = firstPlane + blockIdx.z; localZ <= lastPlane; localZ += gridDim.z) {
        const size_t globalZ = globalBegin + localZ - 1;  // Device plane 1 is the local first plane.
        const size_t index = localZ * planeSize + y * nx + x;

        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 || globalZ + 1 == globalDepth) {
            output[index] = input[index];
        } else {
            const Real center = input[index];
            const Real left = input[index - 1];
            const Real right = input[index + 1];
            const Real front = input[index - nx];
            const Real back = input[index + nx];
            const Real bottom = input[index - planeSize];
            const Real top = input[index + planeSize];
            output[index] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

void launchPlanes(const Real* const input,
                  Real* const output,
                  const size_t nx,
                  const size_t ny,
                  const size_t globalDepth,
                  const Slab& slab,
                  const size_t firstPlane,
                  const size_t lastPlane,
                  const cudaStream_t stream) {
    if (firstPlane > lastPlane) {
        return;
    }

    constexpr unsigned int blockX = 32;
    constexpr unsigned int blockY = 8;
    constexpr unsigned int maxGridZ = 65535;
    const size_t planeCount = lastPlane - firstPlane + 1;
    const dim3 block(blockX, blockY, 1);
    const dim3 grid(static_cast<unsigned int>((nx + blockX - 1) / blockX),
                    static_cast<unsigned int>((ny + blockY - 1) / blockY),
                    static_cast<unsigned int>(std::min(planeCount, static_cast<size_t>(maxGridZ))));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, slab.globalBegin, globalDepth, firstPlane,
                                              lastPlane);
    CUDA_CHECK(cudaGetLastError());
}

void runIteration(DeviceBuffers& buffers,
                  const size_t nx,
                  const size_t ny,
                  const size_t globalDepth,
                  const Slab& slab,
                  MPI_Comm communicator) {
    if (slab.localDepth == 0) {
        return;
    }

    const size_t planeSize = nx * ny;
    const size_t planeBytes = planeSize * sizeof(Real);

    // Planes that do not touch an MPI halo can execute while the boundary planes
    // are staged and exchanged.  This overlaps the bulk GPU work with MPI traffic.
    if (slab.localDepth > 2) {
        launchPlanes(buffers.current, buffers.next, nx, ny, globalDepth, slab, 2, slab.localDepth - 1,
                     buffers.computeStream);
    }

    if (slab.lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendLower, buffers.current + planeSize, planeBytes,
                                   cudaMemcpyDeviceToHost, buffers.transferStream));
    }
    if (slab.upperRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendUpper, buffers.current + slab.localDepth * planeSize, planeBytes,
                                   cudaMemcpyDeviceToHost, buffers.transferStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(buffers.transferStream));

    MPI_Request requests[4];
    int requestCount = 0;
    const int mpiPlaneSize = static_cast<int>(planeSize);
    if (slab.lowerRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(buffers.recvLower, mpiPlaneSize, MPI_DOUBLE, slab.lowerRank, kTagToUpper,
                            communicator, &requests[requestCount++]));
        MPI_CHECK(MPI_Isend(buffers.sendLower, mpiPlaneSize, MPI_DOUBLE, slab.lowerRank, kTagToLower,
                            communicator, &requests[requestCount++]));
    }
    if (slab.upperRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(buffers.recvUpper, mpiPlaneSize, MPI_DOUBLE, slab.upperRank, kTagToLower,
                            communicator, &requests[requestCount++]));
        MPI_CHECK(MPI_Isend(buffers.sendUpper, mpiPlaneSize, MPI_DOUBLE, slab.upperRank, kTagToUpper,
                            communicator, &requests[requestCount++]));
    }
    if (requestCount != 0) {
        MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE));
    }

    if (slab.lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.current, buffers.recvLower, planeBytes, cudaMemcpyHostToDevice,
                                   buffers.transferStream));
    }
    if (slab.upperRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.current + (slab.localDepth + 1) * planeSize, buffers.recvUpper,
                                   planeBytes, cudaMemcpyHostToDevice, buffers.transferStream));
    }
    CUDA_CHECK(cudaEventRecord(buffers.halosReady, buffers.transferStream));
    CUDA_CHECK(cudaStreamWaitEvent(buffers.computeStream, buffers.halosReady, 0));

    // The first and last local planes may depend on a received neighbor halo.
    launchPlanes(buffers.current, buffers.next, nx, ny, globalDepth, slab, 1, 1, buffers.computeStream);
    if (slab.localDepth > 1) {
        launchPlanes(buffers.current, buffers.next, nx, ny, globalDepth, slab, slab.localDepth, slab.localDepth,
                     buffers.computeStream);
    }
    CUDA_CHECK(cudaStreamSynchronize(buffers.computeStream));

    std::swap(buffers.current, buffers.next);
}

bool validateResult(const std::vector<Real>& grid) {
    bool finite = true;
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -std::numeric_limits<Real>::infinity();

#pragma omp parallel for reduction(&: finite) reduction(min : minValue) reduction(max : maxValue) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        finite = finite && std::isfinite(value);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    if (!finite) {
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

void destroyBuffers(DeviceBuffers& buffers) {
    if (buffers.halosReady != nullptr) CUDA_CHECK(cudaEventDestroy(buffers.halosReady));
    if (buffers.computeStream != nullptr) CUDA_CHECK(cudaStreamDestroy(buffers.computeStream));
    if (buffers.transferStream != nullptr) CUDA_CHECK(cudaStreamDestroy(buffers.transferStream));
    if (buffers.current != nullptr) CUDA_CHECK(cudaFree(buffers.current));
    if (buffers.next != nullptr) CUDA_CHECK(cudaFree(buffers.next));
    if (buffers.localHost != nullptr) CUDA_CHECK(cudaFreeHost(buffers.localHost));
    if (buffers.sendLower != nullptr) CUDA_CHECK(cudaFreeHost(buffers.sendLower));
    if (buffers.sendUpper != nullptr) CUDA_CHECK(cudaFreeHost(buffers.sendUpper));
    if (buffers.recvLower != nullptr) CUDA_CHECK(cudaFreeHost(buffers.recvLower));
    if (buffers.recvUpper != nullptr) CUDA_CHECK(cudaFreeHost(buffers.recvUpper));
    buffers = {};
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

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortRun("MPI implementation does not provide MPI_THREAD_FUNNELED", __FILE__, __LINE__);
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be positive and iterations must be non-negative.\n");
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    const size_t planeSize = checkedProduct(nx, ny, "Grid plane is too large");
    const size_t globalElements = checkedProduct(planeSize, nz, "Grid is too large");
    if (planeSize > static_cast<size_t>(INT_MAX)) {
        abortRun("A halo plane exceeds MPI's count limit", __FILE__, __LINE__);
    }

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortRun("No CUDA device is visible to this MPI rank", __FILE__, __LINE__);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    const Slab slab = makeSlab(nz, rank, worldSize);
    const size_t localElements = checkedProduct(slab.localDepth, planeSize, "Local grid is too large");
    if (localElements > static_cast<size_t>(INT_MAX)) {
        abortRun("A local grid exceeds MPI's gather count limit", __FILE__, __LINE__);
    }
    if (slab.localDepth > std::numeric_limits<size_t>::max() - 2) {
        abortRun("Padded local grid is too large", __FILE__, __LINE__);
    }
    const size_t paddedElements = checkedProduct(slab.localDepth + 2, planeSize, "Padded local grid is too large");
    if (paddedElements > std::numeric_limits<size_t>::max() / sizeof(Real) ||
        localElements > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        abortRun("Grid allocation size overflows", __FILE__, __LINE__);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }

    DeviceBuffers buffers;
    CUDA_CHECK(cudaStreamCreateWithFlags(&buffers.computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&buffers.transferStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&buffers.halosReady, cudaEventDisableTiming));

    if (slab.localDepth != 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.localHost), localElements * sizeof(Real),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.sendLower), planeSize * sizeof(Real),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.sendUpper), planeSize * sizeof(Real),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.recvLower), planeSize * sizeof(Real),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.recvUpper), planeSize * sizeof(Real),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.current), paddedElements * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.next), paddedElements * sizeof(Real)));

        initializeGrid(buffers.localHost, localElements, slab.globalBegin * planeSize);
        CUDA_CHECK(cudaMemcpyAsync(buffers.current + planeSize, buffers.localHost, localElements * sizeof(Real),
                                   cudaMemcpyHostToDevice, buffers.transferStream));
        CUDA_CHECK(cudaStreamSynchronize(buffers.transferStream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) std::printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        runIteration(buffers, nx, ny, nz, slab, MPI_COMM_WORLD);
    }
    const double localSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        const long milliseconds = static_cast<long>(std::llround(elapsedSeconds * 1000.0));
        const size_t interiorX = nx > 2 ? nx - 2 : 0;
        const size_t interiorY = ny > 2 ? ny - 2 : 0;
        const size_t interiorZ = nz > 2 ? nz - 2 : 0;
        const double cellUpdates = static_cast<double>(interiorX) * static_cast<double>(interiorY) *
                                   static_cast<double>(interiorZ) * static_cast<double>(iterations);
        const double mcups = elapsedSeconds > 0.0 ? cellUpdates / elapsedSeconds / 1.0e6 : 0.0;
        std::printf("Computation time: %ld ms\n", milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    bool passed = true;
    if (printResults || validate) {
        if (slab.localDepth != 0) {
            CUDA_CHECK(cudaMemcpyAsync(buffers.localHost, buffers.current + planeSize, localElements * sizeof(Real),
                                       cudaMemcpyDeviceToHost, buffers.transferStream));
            CUDA_CHECK(cudaStreamSynchronize(buffers.transferStream));
        }

        const int localCount = static_cast<int>(localElements);
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<Real> finalGrid;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(worldSize));
        }
        MPI_CHECK(MPI_Gather(&localCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
                             MPI_COMM_WORLD));
        if (rank == 0) {
            displacements.resize(static_cast<size_t>(worldSize));
            int displacement = 0;
            for (int process = 0; process < worldSize; ++process) {
                displacements[static_cast<size_t>(process)] = displacement;
                if (counts[static_cast<size_t>(process)] > INT_MAX - displacement) {
                    abortRun("Global grid exceeds MPI_Gatherv's displacement limit", __FILE__, __LINE__);
                }
                displacement += counts[static_cast<size_t>(process)];
            }
            if (static_cast<size_t>(displacement) != globalElements) {
                abortRun("MPI slab decomposition produced an invalid global size", __FILE__, __LINE__);
            }
            finalGrid.resize(globalElements);
        }
        MPI_CHECK(MPI_Gatherv(buffers.localHost, localCount, MPI_DOUBLE,
                              rank == 0 ? finalGrid.data() : nullptr,
                              rank == 0 ? counts.data() : nullptr,
                              rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD));

        if (rank == 0) {
            if (printResults) print_results(finalGrid, "Grid");
            if (validate) passed = validateResult(finalGrid);
        }
        int passedValue = passed ? 1 : 0;
        MPI_CHECK(MPI_Bcast(&passedValue, 1, MPI_INT, 0, MPI_COMM_WORLD));
        passed = passedValue != 0;
    }

    destroyBuffers(buffers);
    MPI_CHECK(MPI_Finalize());
    return passed ? 0 : 1;
}
