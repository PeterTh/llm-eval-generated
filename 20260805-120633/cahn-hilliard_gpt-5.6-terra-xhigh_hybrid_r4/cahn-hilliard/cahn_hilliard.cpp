#include <cuda_runtime.h>
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kLowerToUpperTag = 4101;
constexpr int kUpperToLowerTag = 4102;

[[noreturn]] void abortWithMessage(const int rank, const char* message) {
    if (rank == 0) {
        std::fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* call, const char* file, const int line, const int rank) {
    if (status != cudaSuccess) {
        char message[1024];
        std::snprintf(message, sizeof(message), "CUDA failure at %s:%d in %s: %s", file, line, call,
                      cudaGetErrorString(status));
        abortWithMessage(rank, message);
    }
}

void checkMpi(const int status, const char* call, const int rank) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, error, &length);
        char message[1024];
        std::snprintf(message, sizeof(message), "MPI failure in %s: %.*s", call, length, error);
        abortWithMessage(rank, message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__, rank)
#define MPI_CHECK(call) checkMpi((call), #call, rank)

bool checkedMultiply(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool parseSize(const char* text, size_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// The arrays contain one leading and one trailing Z halo.  X and Y boundaries
// remain local and use the original clamped-boundary rule.
__device__ __forceinline__ double laplacian(const double* __restrict__ field, const size_t index,
                                             const size_t plane, const size_t nx, const size_t ny,
                                             const double dx, const double dy, const double dz) {
    const size_t inPlane = index % plane;
    const size_t x = inPlane % nx;
    const size_t y = inPlane / nx;

    const size_t xp = (x + 1 < nx) ? index + 1 : index;
    const size_t xn = (x > 0) ? index - 1 : index;
    const size_t yp = (y + 1 < ny) ? index + nx : index;
    const size_t yn = (y > 0) ? index - nx : index;

    const double center = field[index];
    const double cxx = (field[xp] + field[xn] - 2.0 * center) / (dx * dx);
    const double cyy = (field[yp] + field[yn] - 2.0 * center) / (dy * dy);
    const double czz = (field[index + plane] + field[index - plane] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration, double* __restrict__ potential,
                                        const size_t plane, const size_t nx, const size_t ny,
                                        const size_t firstPlane, const size_t planeCount, const double dx,
                                        const double dy, const double dz, const double gamma, const double eAA,
                                        const double eBB, const double eAB) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = planeCount * plane;
    if (element >= elements) {
        return;
    }

    const size_t index = firstPlane * plane + element;
    const double cv = concentration[index];
    potential[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) + 3.0 * cv +
                       cv * cv * cv - gamma * laplacian(concentration, index, plane, nx, ny, dx, dy, dz);
}

__global__ void updateKernel(const double* __restrict__ oldConcentration, const double* __restrict__ potential,
                             double* __restrict__ newConcentration, const size_t plane, const size_t nx,
                             const size_t ny, const size_t firstPlane, const size_t planeCount, const double D,
                             const double dt, const double dx, const double dy, const double dz) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = planeCount * plane;
    if (element >= elements) {
        return;
    }

    const size_t index = firstPlane * plane + element;
    newConcentration[index] = oldConcentration[index] +
                              dt * D * laplacian(potential, index, plane, nx, ny, dx, dy, dz);
}

void launchChemicalPotential(const double* concentration, double* potential, const size_t plane, const size_t nx,
                             const size_t ny, const size_t firstPlane, const size_t planeCount, const double dx,
                             const double dy, const double dz, const double gamma, const double eAA,
                             const double eBB, const double eAB, const cudaStream_t stream, const int rank) {
    if (planeCount == 0) {
        return;
    }
    const size_t elements = planeCount * plane;
    const size_t blocks = (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        abortWithMessage(rank, "CUDA grid exceeds the supported launch size");
    }
    chemicalPotentialKernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(
        concentration, potential, plane, nx, ny, firstPlane, planeCount, dx, dy, dz, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdate(const double* oldConcentration, const double* potential, double* newConcentration,
                  const size_t plane, const size_t nx, const size_t ny, const size_t firstPlane,
                  const size_t planeCount, const double D, const double dt, const double dx, const double dy,
                  const double dz, const cudaStream_t stream, const int rank) {
    if (planeCount == 0) {
        return;
    }
    const size_t elements = planeCount * plane;
    const size_t blocks = (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        abortWithMessage(rank, "CUDA grid exceeds the supported launch size");
    }
    updateKernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(
        oldConcentration, potential, newConcentration, plane, nx, ny, firstPlane, planeCount, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaPeekAtLastError());
}

struct StagedHaloBuffers {
    double* sendLower = nullptr;
    double* sendUpper = nullptr;
    double* receiveLower = nullptr;
    double* receiveUpper = nullptr;
};

struct HaloExchange {
    MPI_Request requests[4];
    int requestCount = 0;
};

bool hasCudaAwareMpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() != 0;
#else
    // CUDA-awareness is not part of the MPI standard.  Without an explicit
    // implementation capability query, use portable pinned staging rather
    // than passing a device pointer to an MPI library that may dereference it.
    return false;
#endif
}

void allocateStagedHaloBuffers(StagedHaloBuffers& buffers, const size_t plane, const int rank) {
    const size_t bytes = plane * sizeof(double);
    CUDA_CHECK(cudaHostAlloc(&buffers.sendLower, bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.sendUpper, bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.receiveLower, bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.receiveUpper, bytes, cudaHostAllocPortable));
}

void freeStagedHaloBuffers(StagedHaloBuffers& buffers, const int rank) {
    if (buffers.sendLower != nullptr) CUDA_CHECK(cudaFreeHost(buffers.sendLower));
    if (buffers.sendUpper != nullptr) CUDA_CHECK(cudaFreeHost(buffers.sendUpper));
    if (buffers.receiveLower != nullptr) CUDA_CHECK(cudaFreeHost(buffers.receiveLower));
    if (buffers.receiveUpper != nullptr) CUDA_CHECK(cudaFreeHost(buffers.receiveUpper));
    buffers = {};
}

HaloExchange beginHaloExchange(double* field, const size_t localNz, const size_t plane, const int lowerRank,
                               const int upperRank, const bool cudaAwareMpi, StagedHaloBuffers& staging,
                               const cudaStream_t stream, const int rank) {
    // The preceding CUDA phase produces the send planes.  Synchronizing here
    // gives MPI ownership of fully produced device data while its transfers
    // overlap the interior-plane CUDA kernel launched by the caller.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const size_t bytes = plane * sizeof(double);
    if (lowerRank == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field, field + plane, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    if (upperRank == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, field + localNz * plane, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    if (!cudaAwareMpi) {
        if (lowerRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(staging.sendLower, field + plane, bytes, cudaMemcpyDeviceToHost, stream));
        }
        if (upperRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(staging.sendUpper, field + localNz * plane, bytes, cudaMemcpyDeviceToHost,
                                       stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    HaloExchange exchange{};
    const int count = static_cast<int>(plane);
    if (lowerRank != MPI_PROC_NULL) {
        double* receive = cudaAwareMpi ? field : staging.receiveLower;
        const double* send = cudaAwareMpi ? field + plane : staging.sendLower;
        MPI_CHECK(MPI_Irecv(receive, count, MPI_DOUBLE, lowerRank, kLowerToUpperTag, MPI_COMM_WORLD,
                            &exchange.requests[exchange.requestCount++]));
        MPI_CHECK(MPI_Isend(send, count, MPI_DOUBLE, lowerRank, kUpperToLowerTag, MPI_COMM_WORLD,
                            &exchange.requests[exchange.requestCount++]));
    }
    if (upperRank != MPI_PROC_NULL) {
        double* receive = cudaAwareMpi ? field + (localNz + 1) * plane : staging.receiveUpper;
        const double* send = cudaAwareMpi ? field + localNz * plane : staging.sendUpper;
        MPI_CHECK(MPI_Irecv(receive, count, MPI_DOUBLE, upperRank, kUpperToLowerTag, MPI_COMM_WORLD,
                            &exchange.requests[exchange.requestCount++]));
        MPI_CHECK(MPI_Isend(send, count, MPI_DOUBLE, upperRank, kLowerToUpperTag, MPI_COMM_WORLD,
                            &exchange.requests[exchange.requestCount++]));
    }
    return exchange;
}

void finishHaloExchange(HaloExchange& exchange, double* field, const size_t localNz, const size_t plane,
                        const int lowerRank, const int upperRank, const bool cudaAwareMpi, StagedHaloBuffers& staging,
                        const cudaStream_t stream, const int rank) {
    if (exchange.requestCount != 0) {
        MPI_CHECK(MPI_Waitall(exchange.requestCount, exchange.requests, MPI_STATUSES_IGNORE));
    }
    if (!cudaAwareMpi) {
        const size_t bytes = plane * sizeof(double);
        if (lowerRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(field, staging.receiveLower, bytes, cudaMemcpyHostToDevice, stream));
        }
        if (upperRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, staging.receiveUpper, bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
    }
}

void initializeConcentration(std::vector<double>& concentration, const size_t plane, const size_t localNz,
                             const size_t globalZStart, const size_t globalVolume) {
#pragma omp parallel for schedule(static)
    for (long long localZ = 0; localZ < static_cast<long long>(localNz); ++localZ) {
        const size_t globalZ = globalZStart + static_cast<size_t>(localZ);
        const size_t base = globalZ * plane;
        const size_t localBase = (static_cast<size_t>(localZ) + 1) * plane;
        for (size_t inPlane = 0; inPlane < plane; ++inPlane) {
            const size_t linearId = base + inPlane;
            const double pseudo = (((linearId + 1) * 1299709) % globalVolume) /
                                  static_cast<double>(globalVolume);
            concentration[localBase + inPlane] = -1.0 + 2.0 * pseudo;
        }
    }
}

bool validateResult(const std::vector<double>& concentration, const int rank) {
    int localInvalid = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();

#pragma omp parallel for reduction(+ : localInvalid) reduction(min : localMin) reduction(max : localMax) schedule(static)
    for (long long i = 0; i < static_cast<long long>(concentration.size()); ++i) {
        const double value = concentration[static_cast<size_t>(i)];
        localInvalid += !std::isfinite(value);
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }

    int globalInvalid = 0;
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_CHECK(MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    int valid = globalInvalid == 0 && globalMax <= 10.0 && globalMin >= -10.0;
    if (rank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalInvalid == 0 && (globalMax > 10.0 || globalMin < -10.0)) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return valid != 0;
}

void printGlobalResults(const std::vector<double>& localConcentration, const size_t plane, const size_t nz,
                        const int worldSize, const int rank) {
    size_t globalElements = 0;
    if (!checkedMultiply(plane, nz, globalElements) || globalElements > static_cast<size_t>(INT_MAX)) {
        abortWithMessage(rank, "Result collection exceeds MPI_Gatherv's supported count range");
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> globalConcentration;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
        const size_t basePlanes = nz / static_cast<size_t>(worldSize);
        const size_t remainder = nz % static_cast<size_t>(worldSize);
        size_t displacement = 0;
        for (int process = 0; process < worldSize; ++process) {
            const size_t processPlanes = basePlanes + (static_cast<size_t>(process) < remainder ? 1 : 0);
            const size_t processElements = processPlanes * plane;
            counts[static_cast<size_t>(process)] = static_cast<int>(processElements);
            displacements[static_cast<size_t>(process)] = static_cast<int>(displacement);
            displacement += processElements;
        }
        globalConcentration.resize(globalElements);
    }

    MPI_CHECK(MPI_Gatherv(localConcentration.data(), static_cast<int>(localConcentration.size()), MPI_DOUBLE,
                          rank == 0 ? globalConcentration.data() : nullptr,
                          rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE,
                          0, MPI_COMM_WORLD));
    if (rank == 0) {
        print_results(globalConcentration, "Concentration");
    }
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI does not provide the thread support required by the hybrid solver");
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], nx) || nx == 0) abortWithMessage(rank, "-x must be a positive integer");
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], ny)) abortWithMessage(rank, "-y must be a non-negative integer");
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], nz)) abortWithMessage(rank, "-z must be a non-negative integer");
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIterations(argv[++i], iterations)) abortWithMessage(rank, "-i must be a non-negative integer");
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (static_cast<size_t>(worldSize) > nz) {
        abortWithMessage(rank, "The number of MPI ranks cannot exceed the global Z dimension");
    }

    size_t plane = 0;
    size_t globalElements = 0;
    if (!checkedMultiply(nx, ny, plane) || !checkedMultiply(plane, nz, globalElements) ||
        plane > static_cast<size_t>(INT_MAX)) {
        abortWithMessage(rank, "Grid dimensions exceed supported allocation or MPI halo count limits");
    }

    const size_t basePlanes = nz / static_cast<size_t>(worldSize);
    const size_t remainder = nz % static_cast<size_t>(worldSize);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZStart = basePlanes * static_cast<size_t>(rank) +
                                std::min(static_cast<size_t>(rank), remainder);
    size_t localElements = 0;
    size_t storageElements = 0;
    if (!checkedMultiply(localNz, plane, localElements) || !checkedMultiply(localNz + 2, plane, storageElements) ||
        localElements > static_cast<size_t>(INT_MAX)) {
        abortWithMessage(rank, "Local grid dimensions exceed supported allocation limits");
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithMessage(rank, "No CUDA accelerator is available");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    const bool cudaAwareMpi = hasCudaAwareMpi();
    const int lowerRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upperRank = rank + 1 == worldSize ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, CUDA-aware halo transport: %s\n", worldSize,
                    cudaAwareMpi ? "enabled" : "pinned-host staging");
        std::printf("Initializing concentration field...\n");
    }

    // Host initialization and validation/reduction use all host cores, while
    // the time-stepping stencil itself remains entirely on each rank's GPU.
    std::vector<double> hostConcentration(storageElements, 0.0);
    initializeConcentration(hostConcentration, plane, localNz, globalZStart, globalElements);

    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, storageElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, storageElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, storageElements * sizeof(double)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(cold + plane, hostConcentration.data() + plane, localElements * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    // Initialization is deliberately outside the measured evolution loop.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    StagedHaloBuffers staging;
    if (!cudaAwareMpi) {
        allocateStagedHaloBuffers(staging, plane, rank);
    }

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = 2.0 / 9.0;
    const double gamma = 0.5;
    const double D = 1.0;
    const size_t interiorPlanes = localNz > 2 ? localNz - 2 : 0;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");

    for (int step = 0; step < iterations; ++step) {
        HaloExchange concentrationExchange = beginHaloExchange(cold, localNz, plane, lowerRank, upperRank,
                                                                 cudaAwareMpi, staging, stream, rank);
        // Planes 2..localNz-1 do not touch an MPI halo and execute while the
        // neighboring ranks transfer their boundary planes.
        launchChemicalPotential(cold, mu, plane, nx, ny, 2, interiorPlanes, dx, dy, dz, gamma, eAA, eBB, eAB,
                                stream, rank);
        finishHaloExchange(concentrationExchange, cold, localNz, plane, lowerRank, upperRank, cudaAwareMpi, staging,
                           stream, rank);
        launchChemicalPotential(cold, mu, plane, nx, ny, 1, 1, dx, dy, dz, gamma, eAA, eBB, eAB, stream, rank);
        if (localNz > 1) {
            launchChemicalPotential(cold, mu, plane, nx, ny, localNz, 1, dx, dy, dz, gamma, eAA, eBB, eAB, stream,
                                    rank);
        }

        HaloExchange potentialExchange = beginHaloExchange(mu, localNz, plane, lowerRank, upperRank, cudaAwareMpi,
                                                             staging, stream, rank);
        launchUpdate(cold, mu, cnew, plane, nx, ny, 2, interiorPlanes, D, dt, dx, dy, dz, stream, rank);
        finishHaloExchange(potentialExchange, mu, localNz, plane, lowerRank, upperRank, cudaAwareMpi, staging, stream,
                           rank);
        launchUpdate(cold, mu, cnew, plane, nx, ny, 1, 1, D, dt, dx, dy, dz, stream, rank);
        if (localNz > 1) {
            launchUpdate(cold, mu, cnew, plane, nx, ny, localNz, 1, D, dt, dx, dy, dz, stream, rank);
        }
        std::swap(cold, cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_CHECK(MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double mcups = static_cast<double>(globalElements) * static_cast<double>(iterations) / duration / 1.0e6;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    CUDA_CHECK(cudaMemcpyAsync(hostConcentration.data() + plane, cold + plane, localElements * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> localResult(hostConcentration.begin() + static_cast<std::ptrdiff_t>(plane),
                                    hostConcentration.begin() + static_cast<std::ptrdiff_t>(plane + localElements));

    if (printResults) {
        printGlobalResults(localResult, plane, nz, worldSize, rank);
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(localResult, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (!cudaAwareMpi) {
        freeStagedHaloBuffers(staging, rank);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(cold));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
