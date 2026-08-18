#include <algorithm>
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

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kTagToLower = 101;
constexpr int kTagToUpper = 102;

[[noreturn]] void mpiFailure(const int error, const char* expression, const char* file, const int line) {
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "MPI error at %s:%d while executing %s: %.*s\n", file, line,
                 expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression, const char* file,
                              const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define MPI_CHECK(call) do { \
    const int mpiStatus_ = (call); \
    if (mpiStatus_ != MPI_SUCCESS) mpiFailure(mpiStatus_, #call, __FILE__, __LINE__); \
} while (false)

#define CUDA_CHECK(call) do { \
    const cudaError_t cudaStatus_ = (call); \
    if (cudaStatus_ != cudaSuccess) cudaFailure(cudaStatus_, #call, __FILE__, __LINE__); \
} while (false)

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t globalNz,
                              const size_t globalZStart, const size_t localZBegin,
                              const size_t zCount) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = localZBegin + blockIdx.z;
    if (x >= nx || y >= ny || blockIdx.z >= zCount) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t index = localZ * plane + y * nx + x;
    const size_t globalZ = globalZStart + localZ - 1;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 || globalZ + 1 == globalNz) {
        output[index] = input[index];
        return;
    }

    output[index] = (input[index] + input[index - 1] + input[index + 1]
                   + input[index - nx] + input[index + nx]
                   + input[index - plane] + input[index + plane]) / 7.0;
}

void launchStencilRange(const Real* input, Real* output, const size_t nx, const size_t ny,
                        const size_t globalNz, const size_t globalZStart,
                        size_t localZBegin, size_t zCount, cudaStream_t stream) {
    const dim3 block(kBlockX, kBlockY, 1);
    const unsigned int gridX = static_cast<unsigned int>((nx + kBlockX - 1) / kBlockX);
    const unsigned int gridY = static_cast<unsigned int>((ny + kBlockY - 1) / kBlockY);

    // CUDA's z grid dimension is finite; chunk exceptionally deep local slabs.
    while (zCount != 0) {
        const size_t chunk = std::min(zCount, static_cast<size_t>(65535));
        const dim3 grid(gridX, gridY, static_cast<unsigned int>(chunk));
        stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, globalNz,
                                                   globalZStart, localZBegin, chunk);
        CUDA_CHECK(cudaGetLastError());
        localZBegin += chunk;
        zCount -= chunk;
    }
}

void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t localNz,
                    const size_t globalZStart) {
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static)
    for (long long localZ = 0; localZ < static_cast<long long>(localNz); ++localZ) {
        const size_t z = static_cast<size_t>(localZ);
        const size_t base = (z + 1) * plane;
        const size_t globalBase = (globalZStart + z) * plane;
        for (size_t offset = 0; offset < plane; ++offset) {
            grid[base + offset] = static_cast<Real>((globalBase + offset) % 19);
        }
    }
}

bool validateResult(const std::vector<Real>& grid, Real& minimum, Real& maximum) {
    int invalid = 0;
    Real localMinimum = std::numeric_limits<Real>::max();
    Real localMaximum = std::numeric_limits<Real>::lowest();

    #pragma omp parallel for reduction(| : invalid) reduction(min : localMinimum) reduction(max : localMaximum) schedule(static)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
            invalid = 1;
        } else {
            localMinimum = std::min(localMinimum, value);
            localMaximum = std::max(localMaximum, value);
        }
    }

    int globalInvalid = 0;
    MPI_CHECK(MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMinimum, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMaximum, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    return globalInvalid == 0 && maximum <= 1e6 && minimum >= -1e6;
}

bool parseSize(const char* text, size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
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

void copyFinalSlab(const Real* deviceGrid, std::vector<Real>& hostGrid, const size_t plane) {
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), deviceGrid + plane, hostGrid.size() * sizeof(Real),
                          cudaMemcpyDeviceToHost));
}

}  // namespace

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel));

    int rank = 0;
    int worldSize = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool parseOk = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parseOk = parseOk && parseSize(argv[++i], nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parseOk = parseOk && parseSize(argv[++i], ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parseOk = parseOk && parseSize(argv[++i], nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parseOk = parseOk && parseIterations(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseOk = false;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const bool dimensionsValid = nx >= 3 && ny >= 3 && nz >= 3 &&
                                 static_cast<size_t>(worldSize) <= nz &&
                                 nx <= std::numeric_limits<size_t>::max() / ny &&
                                 nx * ny <= std::numeric_limits<size_t>::max() / nz &&
                                 nx * ny <= static_cast<size_t>(INT_MAX);
    if (!parseOk || !dimensionsValid) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid dimensions must be at least 3, fit in address space, and provide at least one z-plane per MPI rank.\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    const size_t plane = nx * ny;
    const size_t basePlanes = nz / static_cast<size_t>(worldSize);
    const size_t extraPlanes = nz % static_cast<size_t>(worldSize);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < extraPlanes ? 1 : 0);
    const size_t globalZStart = static_cast<size_t>(rank) * basePlanes +
                                std::min(static_cast<size_t>(rank), extraPlanes);
    const size_t localElements = localNz * plane;

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Real> hostInput((localNz + 2) * plane, 0.0);
    initializeGrid(hostInput.data(), nx, ny, localNz, globalZStart);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* receiveLower = nullptr;
    Real* receiveUpper = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaStream_t transferStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&deviceGrid1, (localNz + 2) * plane * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, (localNz + 2) * plane * sizeof(Real)));
    CUDA_CHECK(cudaMemsetAsync(deviceGrid1, 0, (localNz + 2) * plane * sizeof(Real), transferStream));
    CUDA_CHECK(cudaMemsetAsync(deviceGrid2, 0, (localNz + 2) * plane * sizeof(Real), transferStream));
    CUDA_CHECK(cudaMemcpyAsync(deviceGrid1 + plane, hostInput.data() + plane, localElements * sizeof(Real),
                               cudaMemcpyHostToDevice, transferStream));
    CUDA_CHECK(cudaStreamSynchronize(transferStream));

    // MPI implementations are not universally CUDA-aware. Pinned staging keeps the exchange
    // portable while retaining asynchronous copies and overlap with the interior CUDA kernel.
    CUDA_CHECK(cudaMallocHost(&sendLower, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendUpper, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&receiveLower, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&receiveUpper, plane * sizeof(Real)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double startTime = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        Real* input = (iteration & 1) == 0 ? deviceGrid1 : deviceGrid2;
        Real* output = (iteration & 1) == 0 ? deviceGrid2 : deviceGrid1;

        // Only the first and last local planes require MPI halo data.  Compute all other
        // planes while those two planes are staged and exchanged.
        if (localNz > 2) {
            launchStencilRange(input, output, nx, ny, nz, globalZStart, 2, localNz - 2, computeStream);
        }

        if (rank > 0) {
            CUDA_CHECK(cudaMemcpyAsync(sendLower, input + plane, plane * sizeof(Real),
                                       cudaMemcpyDeviceToHost, transferStream));
        }
        if (rank + 1 < worldSize) {
            CUDA_CHECK(cudaMemcpyAsync(sendUpper, input + localNz * plane, plane * sizeof(Real),
                                       cudaMemcpyDeviceToHost, transferStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(transferStream));

        MPI_Request requests[4];
        int requestCount = 0;
        if (rank > 0) {
            MPI_CHECK(MPI_Irecv(receiveLower, static_cast<int>(plane), MPI_DOUBLE, rank - 1,
                                kTagToUpper, MPI_COMM_WORLD, &requests[requestCount++]));
        }
        if (rank + 1 < worldSize) {
            MPI_CHECK(MPI_Irecv(receiveUpper, static_cast<int>(plane), MPI_DOUBLE, rank + 1,
                                kTagToLower, MPI_COMM_WORLD, &requests[requestCount++]));
        }
        if (rank > 0) {
            MPI_CHECK(MPI_Isend(sendLower, static_cast<int>(plane), MPI_DOUBLE, rank - 1,
                                kTagToLower, MPI_COMM_WORLD, &requests[requestCount++]));
        }
        if (rank + 1 < worldSize) {
            MPI_CHECK(MPI_Isend(sendUpper, static_cast<int>(plane), MPI_DOUBLE, rank + 1,
                                kTagToUpper, MPI_COMM_WORLD, &requests[requestCount++]));
        }
        MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE));

        if (rank > 0) {
            CUDA_CHECK(cudaMemcpyAsync(input, receiveLower, plane * sizeof(Real),
                                       cudaMemcpyHostToDevice, transferStream));
        }
        if (rank + 1 < worldSize) {
            CUDA_CHECK(cudaMemcpyAsync(input + (localNz + 1) * plane, receiveUpper,
                                       plane * sizeof(Real), cudaMemcpyHostToDevice, transferStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(transferStream));

        launchStencilRange(input, output, nx, ny, nz, globalZStart, 1, 1, computeStream);
        if (localNz > 1) {
            launchStencilRange(input, output, nx, ny, nz, globalZStart, localNz, 1, computeStream);
        }
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double localTime = MPI_Wtime() - startTime;
    double elapsedSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localTime, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        const double mcups = elapsedSeconds > 0.0 ? cellUpdates / elapsedSeconds / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const Real* finalDeviceGrid = (iterations & 1) == 0 ? deviceGrid1 : deviceGrid2;
    std::vector<Real> finalLocalGrid(localElements);
    if (validate || printResults) {
        copyFinalSlab(finalDeviceGrid, finalLocalGrid, plane);
    }

    int resultCode = 0;
    if (printResults) {
        const size_t globalElements = nx * ny * nz;
        if (globalElements > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) std::fprintf(stderr, "Grid is too large for MPI_Gatherv result output.\n");
            resultCode = 1;
        } else {
            std::vector<int> receiveCounts;
            std::vector<int> displacements;
            std::vector<Real> globalGrid;
            if (rank == 0) {
                receiveCounts.resize(worldSize);
                displacements.resize(worldSize);
                for (int peer = 0; peer < worldSize; ++peer) {
                    const size_t peerNz = basePlanes + (static_cast<size_t>(peer) < extraPlanes ? 1 : 0);
                    const size_t peerStart = static_cast<size_t>(peer) * basePlanes +
                                             std::min(static_cast<size_t>(peer), extraPlanes);
                    receiveCounts[peer] = static_cast<int>(peerNz * plane);
                    displacements[peer] = static_cast<int>(peerStart * plane);
                }
                globalGrid.resize(nx * ny * nz);
            }
            MPI_CHECK(MPI_Gatherv(finalLocalGrid.data(), static_cast<int>(localElements), MPI_DOUBLE,
                                  rank == 0 ? globalGrid.data() : nullptr,
                                  rank == 0 ? receiveCounts.data() : nullptr,
                                  rank == 0 ? displacements.data() : nullptr,
                                  MPI_DOUBLE, 0, MPI_COMM_WORLD));
            if (rank == 0) print_results(globalGrid, "Grid");
        }
    }

    if (validate) {
        Real minimum = 0.0;
        Real maximum = 0.0;
        const bool valid = validateResult(finalLocalGrid, minimum, maximum);
        if (rank == 0) {
            if (!valid && (!std::isfinite(minimum) || !std::isfinite(maximum))) {
                std::printf("Validation failed: found NaN or Inf value\n");
            }
            std::printf("Value range: [%.6f, %.6f]\n", minimum, maximum);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (!valid) resultCode = 1;
    }

    CUDA_CHECK(cudaFreeHost(sendLower));
    CUDA_CHECK(cudaFreeHost(sendUpper));
    CUDA_CHECK(cudaFreeHost(receiveLower));
    CUDA_CHECK(cudaFreeHost(receiveUpper));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(transferStream));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());
    return resultCode;
}
