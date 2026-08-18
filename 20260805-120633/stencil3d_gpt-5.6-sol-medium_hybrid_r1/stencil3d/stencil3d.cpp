#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

[[noreturn]] static void fatal(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s: %s", operation,
                      cudaGetErrorString(status));
        fatal(message, rank);
    }
}

static void mpiCheck(int status, const char* operation, int rank) {
    if (status != MPI_SUCCESS) {
        char detail[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, detail, &length);
        char message[MPI_MAX_ERROR_STRING + 128];
        std::snprintf(message, sizeof(message), "%s: %.*s", operation, length, detail);
        fatal(message, rank);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny,
                              size_t zFirst, size_t zLast) {
    const size_t x = 1 + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t rowsPerPlane = ny - 2;
    const size_t planeNumber = row / rowsPerPlane;
    const size_t z = zFirst + planeNumber;
    const size_t y = 1 + row - planeNumber * rowsPerPlane;

    if (x >= nx - 1 || z > zLast) return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const Real sum = input[i] + input[i - 1] + input[i + 1]
                   + input[i - nx] + input[i + nx]
                   + input[i - plane] + input[i + plane];
    output[i] = sum / Real(7.0);
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t zFirst, size_t zLast, cudaStream_t stream,
                          int rank) {
    if (zFirst > zLast) return;
    constexpr unsigned blockX = 32;
    constexpr unsigned blockY = 8;
    const size_t rows = (zLast - zFirst + 1) * (ny - 2);
    const size_t gridY = (rows + blockY - 1) / blockY;
    if (gridY > static_cast<size_t>(std::numeric_limits<unsigned>::max()))
        fatal("local CUDA launch grid is too large", rank);
    const dim3 block(blockX, blockY);
    const dim3 grid(static_cast<unsigned>((nx - 2 + blockX - 1) / blockX),
                    static_cast<unsigned>(gridY));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, zFirst, zLast);
    cudaCheck(cudaGetLastError(), "launching stencil kernel", rank);
}

static void initializeSlab(std::vector<Real>& grid, size_t nx, size_t ny,
                           size_t localPlanes, size_t firstGlobalZ) {
    const size_t plane = nx * ny;
#pragma omp parallel for schedule(static)
    for (long long lz = 0; lz < static_cast<long long>(localPlanes + 2); ++lz) {
        const size_t gz = firstGlobalZ + static_cast<size_t>(lz) - 1;
        for (size_t i = 0; i < plane; ++i)
            grid[static_cast<size_t>(lz) * plane + i] = Real((gz * plane + i) % 19);
    }
}

static void initializeGlobal(std::vector<Real>& grid, size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
#pragma omp parallel for schedule(static)
    for (long long z = 0; z < static_cast<long long>(nz); ++z) {
        for (size_t i = 0; i < plane; ++i)
            grid[static_cast<size_t>(z) * plane + i] = Real((static_cast<size_t>(z) * plane + i) % 19);
    }
}

static bool validateResult(const std::vector<Real>& grid) {
    int bad = 0;
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
#pragma omp parallel for reduction(| : bad) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        bad |= !std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    if (bad) {
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

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
    std::printf("Environment:\n");
    std::printf("  STENCIL3D_CUDA_AWARE_MPI=1  use direct GPU-buffer MPI transfers\n");
}

static bool parseSize(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed > std::numeric_limits<size_t>::max())
        return false;
    value = static_cast<size_t>(parsed);
    return true;
}

static bool parseIterations(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return EXIT_FAILURE;

    int rank = 0;
    int ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size", rank);
    if (provided < MPI_THREAD_FUNNELED) fatal("MPI does not provide MPI_THREAD_FUNNELED", rank);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, argumentsValid = true;
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
            help = true;
        } else {
            if (rank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }
    if (help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!argumentsValid || nx < 3 || ny < 3 || nz < 3) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid dimensions must be integers of at least 3 and iterations must be nonnegative.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (static_cast<size_t>(ranks) > nz - 2)
        fatal("number of MPI ranks must not exceed the number of interior Z planes", rank);
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz)
        fatal("grid dimensions overflow addressable memory", rank);

    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(INT_MAX))
        fatal("a Z plane exceeds MPI's count limit", rank);
    const size_t interiorZ = nz - 2;
    const size_t base = interiorZ / static_cast<size_t>(ranks);
    const size_t remainder = interiorZ % static_cast<size_t>(ranks);
    const size_t localPlanes = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstGlobalZ = 1 + static_cast<size_t>(rank) * base
                              + std::min(static_cast<size_t>(rank), remainder);
    if (localPlanes > (std::numeric_limits<size_t>::max() / plane) - 2)
        fatal("local slab size overflows addressable memory", rank);
    const size_t localElements = (localPlanes + 2) * plane;
    const size_t localBytes = localElements * sizeof(Real);

    MPI_Comm localComm = MPI_COMM_NULL;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localComm), "MPI_Comm_split_type", rank);
    int localRank = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank), "local MPI_Comm_rank", rank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "querying CUDA devices", rank);
    if (deviceCount == 0) fatal("no CUDA device is available", rank);
    const int device = localRank % deviceCount;
    cudaCheck(cudaSetDevice(device), "selecting CUDA device", rank);

    const char* directEnvironment = std::getenv("STENCIL3D_CUDA_AWARE_MPI");
    const int direct = directEnvironment && std::strcmp(directEnvironment, "0") != 0;
    int directMin = 0, directMax = 0;
    mpiCheck(MPI_Allreduce(&direct, &directMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
             "checking CUDA-aware MPI setting", rank);
    mpiCheck(MPI_Allreduce(&direct, &directMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
             "checking CUDA-aware MPI setting", rank);
    if (directMin != directMax) fatal("STENCIL3D_CUDA_AWARE_MPI must agree on all ranks", rank);
    const bool cudaAwareMPI = directMin != 0;

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI ranks, up to %d OpenMP threads/rank, CUDA GPUs\n",
                    ranks, omp_get_max_threads());
        std::printf("Halo transport: %s\n", cudaAwareMPI ? "CUDA-aware MPI" : "pinned host staging");
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> initial(localElements);
    initializeSlab(initial, nx, ny, localPlanes, firstGlobalZ);

    Real* deviceA = nullptr;
    Real* deviceB = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceA), localBytes), "allocating first device grid", rank);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceB), localBytes), "allocating second device grid", rank);
    cudaCheck(cudaMemcpy(deviceA, initial.data(), localBytes, cudaMemcpyHostToDevice),
              "copying initial grid to device", rank);
    cudaCheck(cudaMemcpy(deviceB, initial.data(), localBytes, cudaMemcpyHostToDevice),
              "copying boundary values to second device grid", rank);
    initial.clear();
    initial.shrink_to_fit();

    Real *sendLower = nullptr, *sendUpper = nullptr, *recvLower = nullptr, *recvUpper = nullptr;
    if (!cudaAwareMPI) {
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&sendLower), plane * sizeof(Real), cudaHostAllocDefault),
                  "allocating lower send buffer", rank);
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&sendUpper), plane * sizeof(Real), cudaHostAllocDefault),
                  "allocating upper send buffer", rank);
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&recvLower), plane * sizeof(Real), cudaHostAllocDefault),
                  "allocating lower receive buffer", rank);
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&recvUpper), plane * sizeof(Real), cudaHostAllocDefault),
                  "allocating upper receive buffer", rank);
    }

    cudaStream_t computeStream = nullptr, haloStream = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking),
              "creating compute stream", rank);
    cudaCheck(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking),
              "creating halo stream", rank);

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "pre-computation barrier", rank);
    if (rank == 0) std::printf("Running stencil computation...\n");
    const auto start = std::chrono::steady_clock::now();
    const int lowerRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upperRank = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;

    Real* input = deviceA;
    Real* output = deviceB;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                   MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        Real* lowerReceive = cudaAwareMPI ? input : recvLower;
        Real* upperReceive = cudaAwareMPI ? input + (localPlanes + 1) * plane : recvUpper;
        Real* lowerSend = cudaAwareMPI ? input + plane : sendLower;
        Real* upperSend = cudaAwareMPI ? input + localPlanes * plane : sendUpper;

        // Interior planes do not depend on incoming halos and overlap the exchange.
        if (localPlanes > 2)
            launchStencil(input, output, nx, ny, 2, localPlanes - 1, computeStream, rank);

        // Receives can make progress while the portable path stages its send planes.
        mpiCheck(MPI_Irecv(lowerReceive, static_cast<int>(plane), MPI_DOUBLE, lowerRank, 1,
                           MPI_COMM_WORLD, &requests[0]), "posting lower halo receive", rank);
        mpiCheck(MPI_Irecv(upperReceive, static_cast<int>(plane), MPI_DOUBLE, upperRank, 0,
                           MPI_COMM_WORLD, &requests[1]), "posting upper halo receive", rank);

        if (!cudaAwareMPI) {
            cudaCheck(cudaMemcpyAsync(sendLower, input + plane, plane * sizeof(Real),
                                      cudaMemcpyDeviceToHost, haloStream),
                      "staging lower halo", rank);
            cudaCheck(cudaMemcpyAsync(sendUpper, input + localPlanes * plane, plane * sizeof(Real),
                                      cudaMemcpyDeviceToHost, haloStream),
                      "staging upper halo", rank);
            cudaCheck(cudaStreamSynchronize(haloStream), "waiting for staged halos", rank);
        }

        mpiCheck(MPI_Isend(lowerSend, static_cast<int>(plane), MPI_DOUBLE, lowerRank, 0,
                           MPI_COMM_WORLD, &requests[2]), "posting lower halo send", rank);
        mpiCheck(MPI_Isend(upperSend, static_cast<int>(plane), MPI_DOUBLE, upperRank, 1,
                           MPI_COMM_WORLD, &requests[3]), "posting upper halo send", rank);
        mpiCheck(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE), "exchanging halos", rank);

        if (!cudaAwareMPI) {
            if (lowerRank != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(input, recvLower, plane * sizeof(Real),
                                          cudaMemcpyHostToDevice, haloStream),
                          "uploading lower halo", rank);
            if (upperRank != MPI_PROC_NULL)
                cudaCheck(cudaMemcpyAsync(input + (localPlanes + 1) * plane, recvUpper,
                                          plane * sizeof(Real), cudaMemcpyHostToDevice, haloStream),
                          "uploading upper halo", rank);
        }

        launchStencil(input, output, nx, ny, 1, 1, haloStream, rank);
        if (localPlanes > 1)
            launchStencil(input, output, nx, ny, localPlanes, localPlanes, haloStream, rank);
        cudaCheck(cudaStreamSynchronize(computeStream), "waiting for interior stencil", rank);
        cudaCheck(cudaStreamSynchronize(haloStream), "waiting for boundary stencil", rank);
        std::swap(input, output);
    }
    cudaCheck(cudaDeviceSynchronize(), "finishing CUDA computation", rank);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    mpiCheck(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "reducing computation time", rank);

    if (rank == 0) {
        const double cellUpdates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2)
                                 * static_cast<double>(nz - 2) * iterations;
        const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> finalGrid;
    std::vector<Real> localResult;
    if (printResults || validate) {
        if (localPlanes * plane > static_cast<size_t>(INT_MAX))
            fatal("local result exceeds MPI_Gatherv count limit", rank);
        localResult.resize(localPlanes * plane);
        cudaCheck(cudaMemcpy(localResult.data(), input + plane, localResult.size() * sizeof(Real),
                             cudaMemcpyDeviceToHost), "downloading local result", rank);

        std::vector<int> counts, displacements;
        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
            initializeGlobal(finalGrid, nx, ny, nz);
            counts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            for (int r = 0; r < ranks; ++r) {
                const size_t rPlanes = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t rStart = static_cast<size_t>(r) * base
                                    + std::min(static_cast<size_t>(r), remainder);
                if (rPlanes * plane > static_cast<size_t>(INT_MAX) ||
                    rStart * plane > static_cast<size_t>(INT_MAX))
                    fatal("global result exceeds MPI_Gatherv displacement limit", rank);
                counts[static_cast<size_t>(r)] = static_cast<int>(rPlanes * plane);
                displacements[static_cast<size_t>(r)] = static_cast<int>(rStart * plane);
            }
        }
        mpiCheck(MPI_Gatherv(localResult.data(), static_cast<int>(localResult.size()), MPI_DOUBLE,
                             rank == 0 ? finalGrid.data() + plane : nullptr,
                             rank == 0 ? counts.data() : nullptr,
                             rank == 0 ? displacements.data() : nullptr,
                             MPI_DOUBLE, 0, MPI_COMM_WORLD), "gathering final grid", rank);
    }

    int valid = 1;
    if (rank == 0 && printResults) print_results(finalGrid, "Grid");
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        valid = validateResult(finalGrid) ? 1 : 0;
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    mpiCheck(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD), "broadcasting validation status", rank);

    cudaCheck(cudaStreamDestroy(computeStream), "destroying compute stream", rank);
    cudaCheck(cudaStreamDestroy(haloStream), "destroying halo stream", rank);
    if (!cudaAwareMPI) {
        cudaCheck(cudaFreeHost(sendLower), "freeing lower send buffer", rank);
        cudaCheck(cudaFreeHost(sendUpper), "freeing upper send buffer", rank);
        cudaCheck(cudaFreeHost(recvLower), "freeing lower receive buffer", rank);
        cudaCheck(cudaFreeHost(recvUpper), "freeing upper receive buffer", rank);
    }
    cudaCheck(cudaFree(deviceA), "freeing first device grid", rank);
    cudaCheck(cudaFree(deviceB), "freeing second device grid", rank);
    mpiCheck(MPI_Comm_free(&localComm), "freeing local communicator", rank);
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
