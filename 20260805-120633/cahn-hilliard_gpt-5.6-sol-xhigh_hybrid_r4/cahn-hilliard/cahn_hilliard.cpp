#include <algorithm>
#include <cerrno>
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

namespace {

constexpr int kLowerTag = 100;
constexpr int kUpperTag = 101;

[[noreturn]] void fail(const char* message, int rank, int exitCode = 1) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, exitCode);
    std::abort();
}

void checkCuda(cudaError_t error, const char* expression, const char* file, int line, int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error at %s:%d for %s: %s\n", rank, file, line,
                     expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }
}

void checkMpi(int error, const char* expression, const char* file, int line, int rank) {
    if (error != MPI_SUCCESS) {
        char errorString[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(error, errorString, &length);
        std::fprintf(stderr, "Rank %d: MPI error at %s:%d for %s: %.*s\n", rank, file, line,
                     expression, length, errorString);
        MPI_Abort(MPI_COMM_WORLD, 3);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__, rank)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__, rank)

bool multiplyWouldOverflow(size_t left, size_t right) {
    return right != 0 && left > std::numeric_limits<size_t>::max() / right;
}

bool parseSize(const char* text, size_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Options {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool help = false;
};

bool parseOptions(int argc, char** argv, Options& options, int rank) {
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-x") == 0 && argument + 1 < argc) {
            if (!parseSize(argv[++argument], options.nx)) {
                if (rank == 0) std::fprintf(stderr, "Invalid X dimension\n");
                return false;
            }
        } else if (std::strcmp(argv[argument], "-y") == 0 && argument + 1 < argc) {
            if (!parseSize(argv[++argument], options.ny)) {
                if (rank == 0) std::fprintf(stderr, "Invalid Y dimension\n");
                return false;
            }
        } else if (std::strcmp(argv[argument], "-z") == 0 && argument + 1 < argc) {
            if (!parseSize(argv[++argument], options.nz)) {
                if (rank == 0) std::fprintf(stderr, "Invalid Z dimension\n");
                return false;
            }
        } else if (std::strcmp(argv[argument], "-i") == 0 && argument + 1 < argc) {
            if (!parseIterations(argv[++argument], options.iterations)) {
                if (rank == 0) std::fprintf(stderr, "Invalid iteration count\n");
                return false;
            }
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            options.help = true;
            return true;
        } else {
            if (rank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[argument]);
            return false;
        }
    }

    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    return true;
}

struct Slab {
    size_t localNz;
    size_t globalZOffset;
};

Slab decomposeZ(size_t nz, int rank, int ranks) {
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZOffset = static_cast<size_t>(rank) * base +
                                 std::min(static_cast<size_t>(rank), remainder);
    return {localNz, globalZOffset};
}

void initializeConcentration(std::vector<double>& concentration, size_t globalZOffset,
                             size_t planeCells, size_t globalCells) {
    const size_t localCells = concentration.size();
    const size_t firstGlobalCell = globalZOffset * planeCells;

#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(localCells); ++local) {
        const size_t globalLinear = firstGlobalCell + static_cast<size_t>(local);
        const double pseudo = (((globalLinear + 1) * size_t{1299709}) % globalCells) /
                              static_cast<double>(globalCells);
        concentration[static_cast<size_t>(local)] = -1.0 + 2.0 * pseudo;
    }
}

__device__ __forceinline__ double laplacianAt(const double* __restrict__ field, size_t index,
                                               size_t x, size_t y, size_t nx, size_t ny,
                                               size_t planeCells, double inverseDx2,
                                               double inverseDy2, double inverseDz2) {
    const size_t negativeX = x == 0 ? index : index - 1;
    const size_t positiveX = x + 1 == nx ? index : index + 1;
    const size_t negativeY = y == 0 ? index : index - nx;
    const size_t positiveY = y + 1 == ny ? index : index + nx;
    const double center = field[index];

    return (field[positiveX] + field[negativeX] - 2.0 * center) * inverseDx2 +
           (field[positiveY] + field[negativeY] - 2.0 * center) * inverseDy2 +
           (field[index + planeCells] + field[index - planeCells] - 2.0 * center) * inverseDz2;
}

__device__ __forceinline__ void computeChemicalCell(
    const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
    size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t planeCells,
    double inverseDx2, double inverseDy2, double inverseDz2, double gamma,
    double eAA, double eBB, double eAB) {
    const size_t index = z * planeCells + y * nx + x;
    const double value = concentration[index];
    const double laplacian = laplacianAt(concentration, index, x, y, nx, ny, planeCells,
                                         inverseDx2, inverseDy2, inverseDz2);
    chemicalPotential[index] =
        4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB) +
        3.0 * value + value * value * value - gamma * laplacian;
}

__device__ __forceinline__ void updateCell(
    const double* __restrict__ oldConcentration, double* __restrict__ newConcentration,
    const double* __restrict__ chemicalPotential, size_t x, size_t y, size_t z,
    size_t nx, size_t ny, size_t planeCells, double inverseDx2, double inverseDy2,
    double inverseDz2, double dtTimesD) {
    const size_t index = z * planeCells + y * nx + x;
    const double laplacian = laplacianAt(chemicalPotential, index, x, y, nx, ny, planeCells,
                                         inverseDx2, inverseDy2, inverseDz2);
    newConcentration[index] = oldConcentration[index] + dtTimesD * laplacian;
}

__global__ __launch_bounds__(256) void chemicalBulkKernel(
    const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
    size_t nx, size_t ny, size_t planeCells, int zBegin, int zEnd,
    double inverseDx2, double inverseDy2, double inverseDz2, double gamma,
    double eAA, double eBB, double eAB) {
    for (size_t z = static_cast<size_t>(zBegin) + blockIdx.z; z <= static_cast<size_t>(zEnd);
         z += gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny;
             y += blockDim.y * gridDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx;
                 x += static_cast<size_t>(blockDim.x) * gridDim.x) {
                computeChemicalCell(concentration, chemicalPotential, x, y, z, nx, ny,
                                    planeCells, inverseDx2, inverseDy2, inverseDz2, gamma,
                                    eAA, eBB, eAB);
            }
        }
    }
}

__global__ __launch_bounds__(256) void chemicalBoundaryKernel(
    const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
    size_t nx, size_t ny, size_t planeCells, int localNz,
    double inverseDx2, double inverseDy2, double inverseDz2, double gamma,
    double eAA, double eBB, double eAB) {
    const size_t z = blockIdx.z == 0 ? 1 : static_cast<size_t>(localNz);
    for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny;
         y += blockDim.y * gridDim.y) {
        for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx;
             x += static_cast<size_t>(blockDim.x) * gridDim.x) {
            computeChemicalCell(concentration, chemicalPotential, x, y, z, nx, ny,
                                planeCells, inverseDx2, inverseDy2, inverseDz2, gamma,
                                eAA, eBB, eAB);
        }
    }
}

__global__ __launch_bounds__(256) void updateBulkKernel(
    const double* __restrict__ oldConcentration, double* __restrict__ newConcentration,
    const double* __restrict__ chemicalPotential, size_t nx, size_t ny, size_t planeCells,
    int zBegin, int zEnd, double inverseDx2, double inverseDy2, double inverseDz2,
    double dtTimesD) {
    for (size_t z = static_cast<size_t>(zBegin) + blockIdx.z; z <= static_cast<size_t>(zEnd);
         z += gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny;
             y += blockDim.y * gridDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx;
                 x += static_cast<size_t>(blockDim.x) * gridDim.x) {
                updateCell(oldConcentration, newConcentration, chemicalPotential, x, y, z,
                           nx, ny, planeCells, inverseDx2, inverseDy2, inverseDz2, dtTimesD);
            }
        }
    }
}

__global__ __launch_bounds__(256) void updateBoundaryKernel(
    const double* __restrict__ oldConcentration, double* __restrict__ newConcentration,
    const double* __restrict__ chemicalPotential, size_t nx, size_t ny, size_t planeCells,
    int localNz, double inverseDx2, double inverseDy2, double inverseDz2,
    double dtTimesD) {
    const size_t z = blockIdx.z == 0 ? 1 : static_cast<size_t>(localNz);
    for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny;
         y += blockDim.y * gridDim.y) {
        for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx;
             x += static_cast<size_t>(blockDim.x) * gridDim.x) {
            updateCell(oldConcentration, newConcentration, chemicalPotential, x, y, z,
                       nx, ny, planeCells, inverseDx2, inverseDy2, inverseDz2, dtTimesD);
        }
    }
}

dim3 stencilBlock() {
    return dim3(32, 8, 1);
}

dim3 stencilGrid(size_t nx, size_t ny, size_t zPlanes) {
    const size_t xBlocks = (nx + 31) / 32;
    const size_t yBlocks = (ny + 7) / 8;
    return dim3(static_cast<unsigned int>(std::min<size_t>(xBlocks, 2147483647u)),
                static_cast<unsigned int>(std::min<size_t>(yBlocks, 65535u)),
                static_cast<unsigned int>(std::min<size_t>(zPlanes, 65535u)));
}

void launchChemicalBulk(const double* concentration, double* chemicalPotential,
                        size_t nx, size_t ny, size_t planeCells, int localNz,
                        double inverseDx2, double inverseDy2, double inverseDz2,
                        double gamma, double eAA, double eBB, double eAB,
                        cudaStream_t stream, int rank) {
    if (localNz <= 2) return;
    chemicalBulkKernel<<<stencilGrid(nx, ny, static_cast<size_t>(localNz - 2)), stencilBlock(), 0, stream>>>(
        concentration, chemicalPotential, nx, ny, planeCells, 2, localNz - 1,
        inverseDx2, inverseDy2, inverseDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchChemicalBoundary(const double* concentration, double* chemicalPotential,
                            size_t nx, size_t ny, size_t planeCells, int localNz,
                            double inverseDx2, double inverseDy2, double inverseDz2,
                            double gamma, double eAA, double eBB, double eAB,
                            cudaStream_t stream, int rank) {
    const size_t boundaryPlanes = localNz == 1 ? 1 : 2;
    chemicalBoundaryKernel<<<stencilGrid(nx, ny, boundaryPlanes), stencilBlock(), 0, stream>>>(
        concentration, chemicalPotential, nx, ny, planeCells, localNz,
        inverseDx2, inverseDy2, inverseDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdateBulk(const double* oldConcentration, double* newConcentration,
                      const double* chemicalPotential, size_t nx, size_t ny,
                      size_t planeCells, int localNz, double inverseDx2,
                      double inverseDy2, double inverseDz2, double dtTimesD,
                      cudaStream_t stream, int rank) {
    if (localNz <= 2) return;
    updateBulkKernel<<<stencilGrid(nx, ny, static_cast<size_t>(localNz - 2)), stencilBlock(), 0, stream>>>(
        oldConcentration, newConcentration, chemicalPotential, nx, ny, planeCells,
        2, localNz - 1, inverseDx2, inverseDy2, inverseDz2, dtTimesD);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdateBoundary(const double* oldConcentration, double* newConcentration,
                          const double* chemicalPotential, size_t nx, size_t ny,
                          size_t planeCells, int localNz, double inverseDx2,
                          double inverseDy2, double inverseDz2, double dtTimesD,
                          cudaStream_t stream, int rank) {
    const size_t boundaryPlanes = localNz == 1 ? 1 : 2;
    updateBoundaryKernel<<<stencilGrid(nx, ny, boundaryPlanes), stencilBlock(), 0, stream>>>(
        oldConcentration, newConcentration, chemicalPotential, nx, ny, planeCells,
        localNz, inverseDx2, inverseDy2, inverseDz2, dtTimesD);
    CUDA_CHECK(cudaPeekAtLastError());
}

struct HaloBuffers {
    double* sendLower = nullptr;
    double* sendUpper = nullptr;
    double* receiveLower = nullptr;
    double* receiveUpper = nullptr;
};

void allocateHaloBuffers(HaloBuffers& buffers, size_t planeBytes, int rank) {
    CUDA_CHECK(cudaHostAlloc(&buffers.sendLower, planeBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.sendUpper, planeBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.receiveLower, planeBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.receiveUpper, planeBytes, cudaHostAllocPortable));
}

void freeHaloBuffers(HaloBuffers& buffers, int rank) {
    if (buffers.sendLower != nullptr) CUDA_CHECK(cudaFreeHost(buffers.sendLower));
    if (buffers.sendUpper != nullptr) CUDA_CHECK(cudaFreeHost(buffers.sendUpper));
    if (buffers.receiveLower != nullptr) CUDA_CHECK(cudaFreeHost(buffers.receiveLower));
    if (buffers.receiveUpper != nullptr) CUDA_CHECK(cudaFreeHost(buffers.receiveUpper));
}

void postAndWaitHaloExchange(double* receiveLower, double* receiveUpper,
                             const double* sendLower, const double* sendUpper,
                             int planeCount, int lowerRank, int upperRank,
                             MPI_Comm communicator, int rank) {
    MPI_Request requests[4];
    int requestCount = 0;
    if (lowerRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(receiveLower, planeCount, MPI_DOUBLE, lowerRank, kUpperTag,
                            communicator, &requests[requestCount++]));
    }
    if (upperRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(receiveUpper, planeCount, MPI_DOUBLE, upperRank, kLowerTag,
                            communicator, &requests[requestCount++]));
    }
    if (lowerRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Isend(sendLower, planeCount, MPI_DOUBLE, lowerRank, kLowerTag,
                            communicator, &requests[requestCount++]));
    }
    if (upperRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Isend(sendUpper, planeCount, MPI_DOUBLE, upperRank, kUpperTag,
                            communicator, &requests[requestCount++]));
    }
    if (requestCount != 0) {
        MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE));
    }
}

template <typename LaunchBulk>
void exchangeHalosAndLaunchBulk(double* field, size_t planeCells, size_t planeBytes,
                                int localNz, int lowerRank, int upperRank,
                                bool cudaAwareMpi, const HaloBuffers& buffers,
                                MPI_Comm communicator, cudaStream_t computeStream,
                                cudaStream_t communicationStream, cudaEvent_t fieldReady,
                                LaunchBulk&& launchBulk, int rank) {
    double* const lowerHalo = field;
    double* const firstPlane = field + planeCells;
    double* const lastPlane = field + static_cast<size_t>(localNz) * planeCells;
    double* const upperHalo = field + static_cast<size_t>(localNz + 1) * planeCells;
    const int planeCount = static_cast<int>(planeCells);

    CUDA_CHECK(cudaEventRecord(fieldReady, computeStream));
    CUDA_CHECK(cudaStreamWaitEvent(communicationStream, fieldReady, 0));

    if (cudaAwareMpi) {
        // A CUDA-aware MPI implementation may access device pointers only after the
        // producing stream has made the boundary planes visible.
        CUDA_CHECK(cudaEventSynchronize(fieldReady));
        if (lowerRank == MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(lowerHalo, firstPlane, planeBytes,
                                       cudaMemcpyDeviceToDevice, communicationStream));
        }
        if (upperRank == MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(upperHalo, lastPlane, planeBytes,
                                       cudaMemcpyDeviceToDevice, communicationStream));
        }

        MPI_Request requests[4];
        int requestCount = 0;
        if (lowerRank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(lowerHalo, planeCount, MPI_DOUBLE, lowerRank, kUpperTag,
                                communicator, &requests[requestCount++]));
        }
        if (upperRank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(upperHalo, planeCount, MPI_DOUBLE, upperRank, kLowerTag,
                                communicator, &requests[requestCount++]));
        }
        if (lowerRank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Isend(firstPlane, planeCount, MPI_DOUBLE, lowerRank, kLowerTag,
                                communicator, &requests[requestCount++]));
        }
        if (upperRank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Isend(lastPlane, planeCount, MPI_DOUBLE, upperRank, kUpperTag,
                                communicator, &requests[requestCount++]));
        }

        launchBulk();
        if (requestCount != 0) {
            MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE));
        }
        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
        return;
    }

    if (lowerRank == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(lowerHalo, firstPlane, planeBytes,
                                   cudaMemcpyDeviceToDevice, communicationStream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendLower, firstPlane, planeBytes,
                                   cudaMemcpyDeviceToHost, communicationStream));
    }
    if (upperRank == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(upperHalo, lastPlane, planeBytes,
                                   cudaMemcpyDeviceToDevice, communicationStream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendUpper, lastPlane, planeBytes,
                                   cudaMemcpyDeviceToHost, communicationStream));
    }

    launchBulk();
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));
    postAndWaitHaloExchange(buffers.receiveLower, buffers.receiveUpper,
                            buffers.sendLower, buffers.sendUpper, planeCount,
                            lowerRank, upperRank, communicator, rank);

    if (lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(lowerHalo, buffers.receiveLower, planeBytes,
                                   cudaMemcpyHostToDevice, communicationStream));
    }
    if (upperRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(upperHalo, buffers.receiveUpper, planeBytes,
                                   cudaMemcpyHostToDevice, communicationStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));
}

bool validateDistributed(const std::vector<double>& localConcentration,
                         MPI_Comm communicator, int rank) {
    double localMinimum = std::numeric_limits<double>::infinity();
    double localMaximum = -std::numeric_limits<double>::infinity();
    int localInvalid = 0;

#pragma omp parallel for reduction(min : localMinimum) reduction(max : localMaximum) reduction(| : localInvalid) schedule(static)
    for (long long index = 0; index < static_cast<long long>(localConcentration.size()); ++index) {
        const double value = localConcentration[static_cast<size_t>(index)];
        if (!std::isfinite(value)) localInvalid = 1;
        localMinimum = std::min(localMinimum, value);
        localMaximum = std::max(localMaximum, value);
    }

    double globalMinimum = 0.0;
    double globalMaximum = 0.0;
    int globalInvalid = 0;
    MPI_CHECK(MPI_Allreduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, communicator));
    MPI_CHECK(MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, communicator));
    MPI_CHECK(MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX, communicator));

    if (rank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
            if (globalMaximum > 10.0 || globalMinimum < -10.0) {
                std::printf("Validation failed: values out of expected range\n");
            }
        }
    }
    return globalInvalid == 0 && globalMaximum <= 10.0 && globalMinimum >= -10.0;
}

std::vector<double> gatherConcentration(const std::vector<double>& localConcentration,
                                        size_t globalCells, size_t localCells,
                                        size_t firstGlobalCell, MPI_Comm communicator,
                                        int rank, int ranks) {
    if (globalCells > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        localCells > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        firstGlobalCell > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fail("-r output currently requires at most INT_MAX global cells", rank);
    }

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<double> globalConcentration;
    if (rank == 0) {
        receiveCounts.resize(static_cast<size_t>(ranks));
        displacements.resize(static_cast<size_t>(ranks));
        globalConcentration.resize(globalCells);
    }

    const int count = static_cast<int>(localCells);
    const int displacement = static_cast<int>(firstGlobalCell);
    MPI_CHECK(MPI_Gather(&count, 1, MPI_INT, rank == 0 ? receiveCounts.data() : nullptr,
                         1, MPI_INT, 0, communicator));
    MPI_CHECK(MPI_Gather(&displacement, 1, MPI_INT, rank == 0 ? displacements.data() : nullptr,
                         1, MPI_INT, 0, communicator));
    MPI_CHECK(MPI_Gatherv(localConcentration.data(), count, MPI_DOUBLE,
                          rank == 0 ? globalConcentration.data() : nullptr,
                          rank == 0 ? receiveCounts.data() : nullptr,
                          rank == 0 ? displacements.data() : nullptr,
                          MPI_DOUBLE, 0, communicator));
    return globalConcentration;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    int mpiError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (mpiError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        fail("MPI does not provide the required MPI_THREAD_FUNNELED support", rank);
    }

    Options options;
    const bool validOptions = parseOptions(argc, argv, options, rank);
    if (!validOptions || options.help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return validOptions ? 0 : 1;
    }

    if (static_cast<size_t>(ranks) > options.nz) {
        fail("the number of MPI ranks must not exceed the number of Z planes", rank);
    }
    if (multiplyWouldOverflow(options.nx, options.ny)) {
        fail("grid dimensions overflow size_t", rank);
    }
    const size_t planeCells = options.nx * options.ny;
    if (multiplyWouldOverflow(planeCells, options.nz)) {
        fail("grid dimensions overflow size_t", rank);
    }
    const size_t globalCells = planeCells * options.nz;
    if (globalCells > static_cast<size_t>(std::numeric_limits<long long>::max())) {
        fail("grid is too large for OpenMP loop indexing", rank);
    }
    if (planeCells > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fail("an XY plane is too large for an MPI message", rank);
    }

    const Slab slab = decomposeZ(options.nz, rank, ranks);
    if (slab.localNz > static_cast<size_t>(std::numeric_limits<int>::max() - 2)) {
        fail("local Z extent exceeds supported CUDA kernel indexing", rank);
    }
    if (multiplyWouldOverflow(slab.localNz + 2, planeCells) ||
        multiplyWouldOverflow((slab.localNz + 2) * planeCells, sizeof(double))) {
        fail("local grid allocation size overflows size_t", rank);
    }
    const size_t localCells = slab.localNz * planeCells;
    const size_t allocatedCells = (slab.localNz + 2) * planeCells;
    const size_t allocationBytes = allocatedCells * sizeof(double);
    const size_t planeBytes = planeCells * sizeof(double);

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &nodeCommunicator));
    int localRank = 0;
    int localRanks = 1;
    MPI_CHECK(MPI_Comm_rank(nodeCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeCommunicator, &localRanks));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) fail("no CUDA device is available", rank);
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    const char* cudaAwareEnvironment = std::getenv("CAHN_HILLIARD_CUDA_AWARE_MPI");
    const bool cudaAwareMpi = cudaAwareEnvironment != nullptr &&
                              std::strcmp(cudaAwareEnvironment, "0") != 0;

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Time steps: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank, CUDA\n",
                    ranks, omp_get_max_threads());
        std::printf("MPI halo path: %s\n", cudaAwareMpi ? "CUDA-aware direct" : "pinned host staging");
        if (localRanks > deviceCount) {
            std::printf("Warning: %d local MPI ranks share %d CUDA device(s)\n", localRanks, deviceCount);
        }
    }

    // Physical parameters are intentionally identical to the serial benchmark.
    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;
    constexpr double inverseDx2 = 1.0 / (dx * dx);
    constexpr double inverseDy2 = 1.0 / (dy * dy);
    constexpr double inverseDz2 = 1.0 / (dz * dz);
    constexpr double dtTimesD = dt * diffusion;

    double* oldConcentration = nullptr;
    double* newConcentration = nullptr;
    double* chemicalPotential = nullptr;
    CUDA_CHECK(cudaMalloc(&oldConcentration, allocationBytes));
    CUDA_CHECK(cudaMalloc(&newConcentration, allocationBytes));
    CUDA_CHECK(cudaMalloc(&chemicalPotential, allocationBytes));

    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    cudaEvent_t fieldReady = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&fieldReady, cudaEventDisableTiming));

    HaloBuffers haloBuffers;
    if (!cudaAwareMpi) allocateHaloBuffers(haloBuffers, planeBytes, rank);

    if (rank == 0) std::printf("Initializing concentration field...\n");
    std::vector<double> initialConcentration(localCells);
    initializeConcentration(initialConcentration, slab.globalZOffset, planeCells, globalCells);
    CUDA_CHECK(cudaMemcpy(oldConcentration + planeCells, initialConcentration.data(),
                          localCells * sizeof(double), cudaMemcpyHostToDevice));
    initialConcentration.clear();
    initialConcentration.shrink_to_fit();

    const int lowerRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upperRank = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int localNz = static_cast<int>(slab.localNz);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    const double startTime = MPI_Wtime();

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        exchangeHalosAndLaunchBulk(
            oldConcentration, planeCells, planeBytes, localNz, lowerRank, upperRank,
            cudaAwareMpi, haloBuffers, MPI_COMM_WORLD, computeStream, communicationStream,
            fieldReady,
            [&] {
                launchChemicalBulk(oldConcentration, chemicalPotential, options.nx, options.ny,
                                   planeCells, localNz, inverseDx2, inverseDy2, inverseDz2,
                                   gamma, eAA, eBB, eAB, computeStream, rank);
            },
            rank);
        launchChemicalBoundary(oldConcentration, chemicalPotential, options.nx, options.ny,
                               planeCells, localNz, inverseDx2, inverseDy2, inverseDz2,
                               gamma, eAA, eBB, eAB, computeStream, rank);

        exchangeHalosAndLaunchBulk(
            chemicalPotential, planeCells, planeBytes, localNz, lowerRank, upperRank,
            cudaAwareMpi, haloBuffers, MPI_COMM_WORLD, computeStream, communicationStream,
            fieldReady,
            [&] {
                launchUpdateBulk(oldConcentration, newConcentration, chemicalPotential,
                                 options.nx, options.ny, planeCells, localNz, inverseDx2,
                                 inverseDy2, inverseDz2, dtTimesD, computeStream, rank);
            },
            rank);
        launchUpdateBoundary(oldConcentration, newConcentration, chemicalPotential,
                             options.nx, options.ny, planeCells, localNz, inverseDx2,
                             inverseDy2, inverseDz2, dtTimesD, computeStream, rank);
        std::swap(oldConcentration, newConcentration);
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double cellUpdates = static_cast<double>(globalCells) * options.iterations;
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    bool valid = true;
    std::vector<double> localConcentration;
    if (options.validate || options.printResults) {
        localConcentration.resize(localCells);
        CUDA_CHECK(cudaMemcpy(localConcentration.data(), oldConcentration + planeCells,
                              localCells * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (options.printResults) {
        std::vector<double> globalConcentration = gatherConcentration(
            localConcentration, globalCells, localCells, slab.globalZOffset * planeCells,
            MPI_COMM_WORLD, rank, ranks);
        if (rank == 0) print_results(globalConcentration, "Concentration");
    }

    if (options.validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(localConcentration, MPI_COMM_WORLD, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (!cudaAwareMpi) freeHaloBuffers(haloBuffers, rank);
    CUDA_CHECK(cudaEventDestroy(fieldReady));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(chemicalPotential));
    CUDA_CHECK(cudaFree(newConcentration));
    CUDA_CHECK(cudaFree(oldConcentration));
    MPI_CHECK(MPI_Comm_free(&nodeCommunicator));
    MPI_Finalize();
    return valid ? 0 : 1;
}
