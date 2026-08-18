#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

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

namespace {

constexpr int HALO_TO_LOWER = 100;
constexpr int HALO_TO_UPPER = 101;

[[noreturn]] void abortRun(const char* message, const char* file, int line) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s (%s:%d)\n", rank, message, file, line);
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
        char mpiMessage[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, mpiMessage, &length);
        char message[1024];
        std::snprintf(message, sizeof(message), "MPI call '%s' failed: %.*s", expression,
                      length, mpiMessage);
        abortRun(message, file, line);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

bool checkedMultiply(size_t a, size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool parseSize(const char* text, size_t& value) {
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
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
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

__device__ __forceinline__ double chemicalPotentialAt(
    const double* __restrict__ concentration, size_t nx, size_t ny, size_t plane,
    size_t x, size_t y, size_t z, double invDx2, double invDy2, double invDz2,
    double gamma, double eAA, double eBB, double eAB) {
    const size_t index = z * plane + y * nx + x;
    const size_t xm = (x == 0) ? index : index - 1;
    const size_t xp = (x + 1 == nx) ? index : index + 1;
    const size_t ym = (y == 0) ? index : index - nx;
    const size_t yp = (y + 1 == ny) ? index : index + nx;

    const double center = concentration[index];
    const double laplacian =
        (concentration[xp] + concentration[xm] - 2.0 * center) * invDx2 +
        (concentration[yp] + concentration[ym] - 2.0 * center) * invDy2 +
        (concentration[index + plane] + concentration[index - plane] - 2.0 * center) * invDz2;

    return 4.5 * ((center + 1.0) * eAA + (center - 1.0) * eBB -
                  2.0 * center * eAB) +
           3.0 * center + center * center * center - gamma * laplacian;
}

__device__ __forceinline__ double updatedConcentrationAt(
    const double* __restrict__ concentration, const double* __restrict__ chemicalPotential,
    size_t nx, size_t ny, size_t plane, size_t x, size_t y, size_t z,
    double invDx2, double invDy2, double invDz2, double dtDiffusion) {
    const size_t index = z * plane + y * nx + x;
    const size_t xm = (x == 0) ? index : index - 1;
    const size_t xp = (x + 1 == nx) ? index : index + 1;
    const size_t ym = (y == 0) ? index : index - nx;
    const size_t yp = (y + 1 == ny) ? index : index + nx;

    const double center = chemicalPotential[index];
    const double cxx = (chemicalPotential[xp] + chemicalPotential[xm] - 2.0 * center) * invDx2;
    const double cyy = (chemicalPotential[yp] + chemicalPotential[ym] - 2.0 * center) * invDy2;
    const double czz = (chemicalPotential[index + plane] + chemicalPotential[index - plane] -
                        2.0 * center) * invDz2;
    return concentration[index] + dtDiffusion * (cxx + cyy + czz);
}

__global__ void chemicalPotentialRangeKernel(
    const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
    size_t nx, size_t ny, size_t plane, size_t zBegin, size_t zCount,
    double invDx2, double invDy2, double invDz2, double gamma,
    double eAA, double eBB, double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t zOffset = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zOffset >= zCount) {
        return;
    }
    const size_t z = zBegin + zOffset;
    chemicalPotential[z * plane + y * nx + x] = chemicalPotentialAt(
        concentration, nx, ny, plane, x, y, z, invDx2, invDy2, invDz2,
        gamma, eAA, eBB, eAB);
}

__global__ void chemicalPotentialBoundaryKernel(
    const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
    size_t nx, size_t ny, size_t plane, size_t localNz,
    double invDx2, double invDy2, double invDz2, double gamma,
    double eAA, double eBB, double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }
    const size_t z = (blockIdx.z == 0) ? 1 : localNz;
    chemicalPotential[z * plane + y * nx + x] = chemicalPotentialAt(
        concentration, nx, ny, plane, x, y, z, invDx2, invDy2, invDz2,
        gamma, eAA, eBB, eAB);
}

__global__ void updateRangeKernel(
    const double* __restrict__ concentration, const double* __restrict__ chemicalPotential,
    double* __restrict__ updated, size_t nx, size_t ny, size_t plane,
    size_t zBegin, size_t zCount, double invDx2, double invDy2, double invDz2,
    double dtDiffusion) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t zOffset = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zOffset >= zCount) {
        return;
    }
    const size_t z = zBegin + zOffset;
    updated[z * plane + y * nx + x] = updatedConcentrationAt(
        concentration, chemicalPotential, nx, ny, plane, x, y, z,
        invDx2, invDy2, invDz2, dtDiffusion);
}

__global__ void updateBoundaryKernel(
    const double* __restrict__ concentration, const double* __restrict__ chemicalPotential,
    double* __restrict__ updated, size_t nx, size_t ny, size_t plane,
    size_t localNz, double invDx2, double invDy2, double invDz2,
    double dtDiffusion) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }
    const size_t z = (blockIdx.z == 0) ? 1 : localNz;
    updated[z * plane + y * nx + x] = updatedConcentrationAt(
        concentration, chemicalPotential, nx, ny, plane, x, y, z,
        invDx2, invDy2, invDz2, dtDiffusion);
}

struct HaloWorkspace {
    bool cudaAware = false;
    size_t plane = 0;
    size_t localNz = 0;
    size_t bytes = 0;
    int lower = MPI_PROC_NULL;
    int upper = MPI_PROC_NULL;
    double* staging = nullptr;
    MPI_Request requests[4]{};
    int requestCount = 0;

    double* sendLower() const { return staging; }
    double* sendUpper() const { return staging + plane; }
    double* receiveLower() const { return staging + 2 * plane; }
    double* receiveUpper() const { return staging + 3 * plane; }
};

void beginHaloExchange(double* field, HaloWorkspace& workspace, cudaStream_t communicationStream) {
    workspace.requestCount = 0;

    if (workspace.lower == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field, field + workspace.plane, workspace.bytes,
                                   cudaMemcpyDeviceToDevice, communicationStream));
    }
    if (workspace.upper == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field + (workspace.localNz + 1) * workspace.plane,
                                   field + workspace.localNz * workspace.plane,
                                   workspace.bytes, cudaMemcpyDeviceToDevice,
                                   communicationStream));
    }

    double* lowerReceive = field;
    double* upperReceive = field + (workspace.localNz + 1) * workspace.plane;
    double* lowerSend = field + workspace.plane;
    double* upperSend = field + workspace.localNz * workspace.plane;

    if (!workspace.cudaAware && (workspace.lower != MPI_PROC_NULL || workspace.upper != MPI_PROC_NULL)) {
        if (workspace.lower != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(workspace.sendLower(), lowerSend, workspace.bytes,
                                       cudaMemcpyDeviceToHost, communicationStream));
        }
        if (workspace.upper != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(workspace.sendUpper(), upperSend, workspace.bytes,
                                       cudaMemcpyDeviceToHost, communicationStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
        lowerReceive = workspace.receiveLower();
        upperReceive = workspace.receiveUpper();
        lowerSend = workspace.sendLower();
        upperSend = workspace.sendUpper();
    }

    const int count = static_cast<int>(workspace.plane);
    if (workspace.lower != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(lowerReceive, count, MPI_DOUBLE, workspace.lower, HALO_TO_UPPER,
                            MPI_COMM_WORLD, &workspace.requests[workspace.requestCount++]));
    }
    if (workspace.upper != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(upperReceive, count, MPI_DOUBLE, workspace.upper, HALO_TO_LOWER,
                            MPI_COMM_WORLD, &workspace.requests[workspace.requestCount++]));
    }
    if (workspace.lower != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Isend(lowerSend, count, MPI_DOUBLE, workspace.lower, HALO_TO_LOWER,
                            MPI_COMM_WORLD, &workspace.requests[workspace.requestCount++]));
    }
    if (workspace.upper != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Isend(upperSend, count, MPI_DOUBLE, workspace.upper, HALO_TO_UPPER,
                            MPI_COMM_WORLD, &workspace.requests[workspace.requestCount++]));
    }
}

void finishHaloExchange(double* field, HaloWorkspace& workspace, cudaStream_t communicationStream) {
    if (workspace.requestCount != 0) {
        MPI_CHECK(MPI_Waitall(workspace.requestCount, workspace.requests, MPI_STATUSES_IGNORE));
    }
    if (!workspace.cudaAware) {
        if (workspace.lower != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(field, workspace.receiveLower(), workspace.bytes,
                                       cudaMemcpyHostToDevice, communicationStream));
        }
        if (workspace.upper != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(field + (workspace.localNz + 1) * workspace.plane,
                                       workspace.receiveUpper(), workspace.bytes,
                                       cudaMemcpyHostToDevice, communicationStream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));
}

void launchChemicalInterior(const double* concentration, double* chemicalPotential,
                            size_t nx, size_t ny, size_t plane, size_t localNz,
                            double invDx2, double invDy2, double invDz2, double gamma,
                            double eAA, double eBB, double eAB, cudaStream_t stream) {
    if (localNz <= 2) {
        return;
    }
    constexpr dim3 block(32, 4, 2);
    const size_t zCount = localNz - 2;
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>((zCount + block.z - 1) / block.z));
    chemicalPotentialRangeKernel<<<grid, block, 0, stream>>>(
        concentration, chemicalPotential, nx, ny, plane, 2, zCount,
        invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaGetLastError());
}

void launchChemicalBoundary(const double* concentration, double* chemicalPotential,
                            size_t nx, size_t ny, size_t plane, size_t localNz,
                            double invDx2, double invDy2, double invDz2, double gamma,
                            double eAA, double eBB, double eAB, cudaStream_t stream) {
    constexpr dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(localNz == 1 ? 1 : 2));
    chemicalPotentialBoundaryKernel<<<grid, block, 0, stream>>>(
        concentration, chemicalPotential, nx, ny, plane, localNz,
        invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdateInterior(const double* concentration, const double* chemicalPotential,
                          double* updated, size_t nx, size_t ny, size_t plane, size_t localNz,
                          double invDx2, double invDy2, double invDz2,
                          double dtDiffusion, cudaStream_t stream) {
    if (localNz <= 2) {
        return;
    }
    constexpr dim3 block(32, 4, 2);
    const size_t zCount = localNz - 2;
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>((zCount + block.z - 1) / block.z));
    updateRangeKernel<<<grid, block, 0, stream>>>(
        concentration, chemicalPotential, updated, nx, ny, plane, 2, zCount,
        invDx2, invDy2, invDz2, dtDiffusion);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdateBoundary(const double* concentration, const double* chemicalPotential,
                          double* updated, size_t nx, size_t ny, size_t plane, size_t localNz,
                          double invDx2, double invDy2, double invDz2,
                          double dtDiffusion, cudaStream_t stream) {
    constexpr dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(localNz == 1 ? 1 : 2));
    updateBoundaryKernel<<<grid, block, 0, stream>>>(
        concentration, chemicalPotential, updated, nx, ny, plane, localNz,
        invDx2, invDy2, invDz2, dtDiffusion);
    CUDA_CHECK(cudaGetLastError());
}

void initializeConcentration(std::vector<double>& concentration, size_t plane,
                             size_t zOffset, size_t globalCells) {
    if (concentration.size() > static_cast<size_t>(std::numeric_limits<long long>::max())) {
        abortRun("Local grid is too large for the OpenMP initialization loop", __FILE__, __LINE__);
    }
    const long long localCells = static_cast<long long>(concentration.size());
#pragma omp parallel for schedule(static)
    for (long long localIndex = 0; localIndex < localCells; ++localIndex) {
        const size_t linearId = zOffset * plane + static_cast<size_t>(localIndex);
        const double pseudo = (((linearId + 1) * static_cast<size_t>(1299709)) % globalCells) /
                              static_cast<double>(globalCells);
        concentration[static_cast<size_t>(localIndex)] = -1.0 + 2.0 * pseudo;
    }
}

bool validateDistributed(const std::vector<double>& localConcentration, int rank) {
    if (localConcentration.size() > static_cast<size_t>(std::numeric_limits<long long>::max())) {
        abortRun("Local grid is too large for the OpenMP validation loop", __FILE__, __LINE__);
    }

    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    int localInvalid = 0;
    const long long localCells = static_cast<long long>(localConcentration.size());
#pragma omp parallel for reduction(min : localMin) reduction(max : localMax) reduction(| : localInvalid) schedule(static)
    for (long long i = 0; i < localCells; ++i) {
        const double value = localConcentration[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
            localInvalid = 1;
        } else {
            if (value < localMin) localMin = value;
            if (value > localMax) localMax = value;
        }
    }

    int globalInvalid = 0;
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_CHECK(MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    int valid = 1;
    if (rank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (globalMax > 10.0 || globalMin < -10.0) {
                std::printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return valid != 0;
}

std::vector<double> gatherConcentration(const std::vector<double>& localConcentration,
                                        size_t plane, size_t nz, int rank, int worldSize) {
    size_t globalCells = 0;
    if (!checkedMultiply(plane, nz, globalCells) || globalCells > static_cast<size_t>(INT_MAX)) {
        abortRun("-r currently requires no more than INT_MAX global cells", __FILE__, __LINE__);
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> globalConcentration;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
        size_t displacement = 0;
        const size_t base = nz / static_cast<size_t>(worldSize);
        const size_t remainder = nz % static_cast<size_t>(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t rankNz = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[static_cast<size_t>(r)] = static_cast<int>(rankNz * plane);
            displacements[static_cast<size_t>(r)] = static_cast<int>(displacement);
            displacement += rankNz * plane;
        }
        globalConcentration.resize(globalCells);
    }

    MPI_CHECK(MPI_Gatherv(localConcentration.data(), static_cast<int>(localConcentration.size()),
                          MPI_DOUBLE, rank == 0 ? globalConcentration.data() : nullptr,
                          rank == 0 ? counts.data() : nullptr,
                          rank == 0 ? displacements.data() : nullptr,
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));
    return globalConcentration;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortRun("MPI does not provide the required MPI_THREAD_FUNNELED support", __FILE__, __LINE__);
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            argumentsValid = parseSize(argv[++i], nx) && argumentsValid;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            argumentsValid = parseSize(argv[++i], ny) && argumentsValid;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            argumentsValid = parseSize(argv[++i], nz) && argumentsValid;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentsValid = parseIterations(argv[++i], iterations) && argumentsValid;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) std::printf("Unknown or incomplete option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!argumentsValid || nx == 0 || ny == 0 || nz == 0 ||
        static_cast<size_t>(worldSize) > nz) {
        if (rank == 0) {
            if (argumentsValid && static_cast<size_t>(worldSize) > nz) {
                std::printf("The number of MPI ranks must not exceed the Z dimension.\n");
            } else if (argumentsValid) {
                std::printf("Grid dimensions must be positive.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    size_t plane = 0;
    size_t globalCells = 0;
    if (!checkedMultiply(nx, ny, plane) || !checkedMultiply(plane, nz, globalCells) ||
        plane > static_cast<size_t>(INT_MAX)) {
        abortRun("Grid size overflows addressable storage or MPI halo count", __FILE__, __LINE__);
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &nodeCommunicator));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeCommunicator, &localSize));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortRun("No CUDA-capable device is visible", __FILE__, __LINE__);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    CUDA_CHECK(cudaFree(nullptr));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    if (localRank == 0 && localSize > deviceCount) {
        char processor[MPI_MAX_PROCESSOR_NAME];
        int processorLength = 0;
        MPI_CHECK(MPI_Get_processor_name(processor, &processorLength));
        std::fprintf(stderr,
                     "Warning: node %.*s has %d MPI ranks sharing %d CUDA devices; "
                     "one rank per GPU is recommended.\n",
                     processorLength, processor, localSize, deviceCount);
    }
    MPI_CHECK(MPI_Comm_free(&nodeCommunicator));

    int cudaAwareLocal = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    cudaAwareLocal = MPIX_Query_cuda_support() != 0 ? 1 : 0;
#endif
    int cudaAwareAll = 0;
    MPI_CHECK(MPI_Allreduce(&cudaAwareLocal, &cudaAwareAll, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
    const bool cudaAware = cudaAwareAll != 0;

    const size_t baseNz = nz / static_cast<size_t>(worldSize);
    const size_t remainder = nz % static_cast<size_t>(worldSize);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t zOffset = static_cast<size_t>(rank) * baseNz +
                           std::min(static_cast<size_t>(rank), remainder);
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == worldSize ? MPI_PROC_NULL : rank + 1;

    size_t localCells = 0;
    size_t allocatedCells = 0;
    if (!checkedMultiply(plane, localNz, localCells) ||
        !checkedMultiply(plane, localNz + 2, allocatedCells) ||
        localCells > static_cast<size_t>(std::numeric_limits<long long>::max())) {
        abortRun("Local grid size overflows addressable storage", __FILE__, __LINE__);
    }
    const size_t planeBytes = plane * sizeof(double);
    if (allocatedCells > std::numeric_limits<size_t>::max() / sizeof(double)) {
        abortRun("Local CUDA allocation size overflow", __FILE__, __LINE__);
    }
    const size_t allocationBytes = allocatedCells * sizeof(double);

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank, CUDA\n",
                    worldSize, omp_get_max_threads());
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        std::printf("MPI halo transport: %s\n",
                    cudaAware ? "CUDA-aware device buffers" : "pinned-host staging");
    }

    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;
    constexpr double invDx2 = 1.0 / (dx * dx);
    constexpr double invDy2 = 1.0 / (dy * dy);
    constexpr double invDz2 = 1.0 / (dz * dz);
    constexpr double dtDiffusion = dt * diffusion;

    double* concentration = nullptr;
    double* updated = nullptr;
    double* chemicalPotential = nullptr;
    CUDA_CHECK(cudaMalloc(&concentration, allocationBytes));
    CUDA_CHECK(cudaMalloc(&updated, allocationBytes));
    CUDA_CHECK(cudaMalloc(&chemicalPotential, allocationBytes));

    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));

    HaloWorkspace halo;
    halo.cudaAware = cudaAware;
    halo.plane = plane;
    halo.localNz = localNz;
    halo.bytes = planeBytes;
    halo.lower = lower;
    halo.upper = upper;
    if (!cudaAware && worldSize > 1) {
        size_t stagingCells = 0;
        if (!checkedMultiply(plane, static_cast<size_t>(4), stagingCells)) {
            abortRun("Halo staging allocation size overflow", __FILE__, __LINE__);
        }
        CUDA_CHECK(cudaMallocHost(&halo.staging, stagingCells * sizeof(double)));
    }

    if (rank == 0) std::printf("Initializing concentration field...\n");
    std::vector<double> initialConcentration(localCells);
    initializeConcentration(initialConcentration, plane, zOffset, globalCells);
    CUDA_CHECK(cudaMemcpyAsync(concentration + plane, initialConcentration.data(),
                               localCells * sizeof(double), cudaMemcpyHostToDevice,
                               computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    initialConcentration.clear();
    initialConcentration.shrink_to_fit();

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    for (int step = 0; step < iterations; ++step) {
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        if (cudaAware) beginHaloExchange(concentration, halo, communicationStream);
        launchChemicalInterior(concentration, chemicalPotential, nx, ny, plane, localNz,
                               invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB, computeStream);
        if (!cudaAware) beginHaloExchange(concentration, halo, communicationStream);
        finishHaloExchange(concentration, halo, communicationStream);
        launchChemicalBoundary(concentration, chemicalPotential, nx, ny, plane, localNz,
                               invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB, computeStream);

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        if (cudaAware) beginHaloExchange(chemicalPotential, halo, communicationStream);
        launchUpdateInterior(concentration, chemicalPotential, updated, nx, ny, plane,
                             localNz, invDx2, invDy2, invDz2, dtDiffusion, computeStream);
        if (!cudaAware) beginHaloExchange(chemicalPotential, halo, communicationStream);
        finishHaloExchange(chemicalPotential, halo, communicationStream);
        launchUpdateBoundary(concentration, chemicalPotential, updated, nx, ny, plane,
                             localNz, invDx2, invDy2, invDz2, dtDiffusion, computeStream);
        std::swap(concentration, updated);
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        const long elapsedMilliseconds = static_cast<long>(elapsed * 1000.0);
        const double cellUpdates = static_cast<double>(globalCells) * iterations;
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %ld ms\n", elapsedMilliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> localConcentration;
    if (printResults || validate) {
        localConcentration.resize(localCells);
        CUDA_CHECK(cudaMemcpy(localConcentration.data(), concentration + plane,
                              localCells * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        std::vector<double> globalConcentration = gatherConcentration(
            localConcentration, plane, nz, rank, worldSize);
        if (rank == 0) print_results(globalConcentration, "Concentration");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(localConcentration, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (halo.staging != nullptr) CUDA_CHECK(cudaFreeHost(halo.staging));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(chemicalPotential));
    CUDA_CHECK(cudaFree(updated));
    CUDA_CHECK(cudaFree(concentration));

    MPI_Finalize();
    return valid ? 0 : 1;
}
