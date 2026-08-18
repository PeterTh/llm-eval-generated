#include <algorithm>
#include <cmath>
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

namespace {

constexpr int kLowerHaloTag = 101;
constexpr int kUpperHaloTag = 102;
constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 8;
constexpr size_t kMaxGridZ = 65535;

void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error == cudaSuccess) {
        return;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA error at %s:%d for %s: %s\n", rank, file, line, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

[[noreturn]] void abortRun(const int rank, const char* message) {
    if (rank == 0) {
        std::fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkMpi(const int error, const char* expression, const int rank) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char errorText[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, errorText, &length);
    if (rank == 0) {
        std::fprintf(stderr, "MPI error for %s: %.*s\n", expression, length, errorText);
    }
    MPI_Abort(MPI_COMM_WORLD, error);
}

#define MPI_CHECK(call, rank) checkMpi((call), #call, (rank))

bool multiplicationOverflows(const size_t left, const size_t right) {
    return right != 0 && left > std::numeric_limits<size_t>::max() / right;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
                                        const size_t nx, const size_t ny, const size_t planeSize,
                                        const size_t firstLocalZ, const size_t zCount, const double invDx2,
                                        const double invDy2, const double invDz2, const double gamma,
                                        const double eAA, const double eBB, const double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = firstLocalZ + blockIdx.z;
    if (x >= nx || y >= ny || localZ >= firstLocalZ + zCount) {
        return;
    }

    const size_t index = localZ * planeSize + y * nx + x;
    const size_t left = x == 0 ? index : index - 1;
    const size_t right = x + 1 == nx ? index : index + 1;
    const size_t down = y == 0 ? index : index - nx;
    const size_t up = y + 1 == ny ? index : index + nx;
    const double value = concentration[index];
    const double laplacian = (concentration[right] + concentration[left] - 2.0 * value) * invDx2 +
                             (concentration[up] + concentration[down] - 2.0 * value) * invDy2 +
                             (concentration[index + planeSize] + concentration[index - planeSize] - 2.0 * value) * invDz2;

    chemicalPotential[index] = 4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB) +
                               3.0 * value + value * value * value - gamma * laplacian;
}

__global__ void updateKernel(const double* __restrict__ concentration, const double* __restrict__ chemicalPotential,
                             double* __restrict__ updatedConcentration, const size_t nx, const size_t ny,
                             const size_t planeSize, const size_t firstLocalZ, const size_t zCount,
                             const double invDx2, const double invDy2, const double invDz2, const double diffusion,
                             const double timeStep) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = firstLocalZ + blockIdx.z;
    if (x >= nx || y >= ny || localZ >= firstLocalZ + zCount) {
        return;
    }

    const size_t index = localZ * planeSize + y * nx + x;
    const size_t left = x == 0 ? index : index - 1;
    const size_t right = x + 1 == nx ? index : index + 1;
    const size_t down = y == 0 ? index : index - nx;
    const size_t up = y + 1 == ny ? index : index + nx;
    const double value = chemicalPotential[index];
    const double laplacian = (chemicalPotential[right] + chemicalPotential[left] - 2.0 * value) * invDx2 +
                             (chemicalPotential[up] + chemicalPotential[down] - 2.0 * value) * invDy2 +
                             (chemicalPotential[index + planeSize] + chemicalPotential[index - planeSize] - 2.0 * value) * invDz2;

    updatedConcentration[index] = concentration[index] + timeStep * diffusion * laplacian;
}

dim3 blockShape() {
    return dim3(kBlockX, kBlockY, 1);
}

dim3 gridShape(const size_t nx, const size_t ny, const size_t zCount) {
    return dim3(static_cast<unsigned int>((nx + kBlockX - 1) / kBlockX),
                static_cast<unsigned int>((ny + kBlockY - 1) / kBlockY), static_cast<unsigned int>(zCount));
}

void launchChemicalPotential(const double* concentration, double* chemicalPotential, const size_t nx, const size_t ny,
                             const size_t planeSize, const size_t firstLocalZ, const size_t zCount,
                             const double invDx2, const double invDy2, const double invDz2, const double gamma,
                             const double eAA, const double eBB, const double eAB, const cudaStream_t stream) {
    for (size_t processedPlanes = 0; processedPlanes < zCount; processedPlanes += kMaxGridZ) {
        const size_t launchPlanes = std::min(kMaxGridZ, zCount - processedPlanes);
        chemicalPotentialKernel<<<gridShape(nx, ny, launchPlanes), blockShape(), 0, stream>>>(
            concentration, chemicalPotential, nx, ny, planeSize, firstLocalZ + processedPlanes, launchPlanes, invDx2,
            invDy2, invDz2, gamma, eAA, eBB, eAB);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launchUpdate(const double* concentration, const double* chemicalPotential, double* updatedConcentration,
                  const size_t nx, const size_t ny, const size_t planeSize, const size_t firstLocalZ,
                  const size_t zCount, const double invDx2, const double invDy2, const double invDz2,
                  const double diffusion, const double timeStep, const cudaStream_t stream) {
    for (size_t processedPlanes = 0; processedPlanes < zCount; processedPlanes += kMaxGridZ) {
        const size_t launchPlanes = std::min(kMaxGridZ, zCount - processedPlanes);
        updateKernel<<<gridShape(nx, ny, launchPlanes), blockShape(), 0, stream>>>(
            concentration, chemicalPotential, updatedConcentration, nx, ny, planeSize, firstLocalZ + processedPlanes,
            launchPlanes, invDx2, invDy2, invDz2, diffusion, timeStep);
        CUDA_CHECK(cudaGetLastError());
    }
}

struct HaloBuffers {
    double* sendLower = nullptr;
    double* sendUpper = nullptr;
    double* receiveLower = nullptr;
    double* receiveUpper = nullptr;

    void allocate(const size_t planeSize) {
        const size_t bytes = planeSize * sizeof(double);
        CUDA_CHECK(cudaHostAlloc(&sendLower, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&sendUpper, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&receiveLower, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&receiveUpper, bytes, cudaHostAllocDefault));
    }

    void release() {
        if (sendLower != nullptr) {
            CUDA_CHECK(cudaFreeHost(sendLower));
            sendLower = nullptr;
        }
        if (sendUpper != nullptr) {
            CUDA_CHECK(cudaFreeHost(sendUpper));
            sendUpper = nullptr;
        }
        if (receiveLower != nullptr) {
            CUDA_CHECK(cudaFreeHost(receiveLower));
            receiveLower = nullptr;
        }
        if (receiveUpper != nullptr) {
            CUDA_CHECK(cudaFreeHost(receiveUpper));
            receiveUpper = nullptr;
        }
    }
};

// The host-staged path is intentionally used instead of assuming a vendor-specific CUDA-aware MPI extension.
// Pinned buffers and independent streams retain overlap between communication and the interior CUDA work.
void beginHaloExchange(const double* deviceField, const size_t localNz, const size_t planeSize, const int lowerRank,
                       const int upperRank, HaloBuffers& buffers, const cudaStream_t communicationStream) {
    const size_t bytes = planeSize * sizeof(double);
    if (lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendLower, deviceField + planeSize, bytes, cudaMemcpyDeviceToHost,
                                   communicationStream));
    }
    if (upperRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendUpper, deviceField + localNz * planeSize, bytes,
                                   cudaMemcpyDeviceToHost, communicationStream));
    }
}

void finishHaloExchange(double* deviceField, const size_t localNz, const size_t planeSize, const int lowerRank,
                        const int upperRank, HaloBuffers& buffers, const cudaStream_t communicationStream,
                        const int rank) {
    const size_t bytes = planeSize * sizeof(double);
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));

    MPI_Request requests[4];
    int requestCount = 0;
    if (lowerRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(buffers.receiveLower, static_cast<int>(planeSize), MPI_DOUBLE, lowerRank, kUpperHaloTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
        MPI_CHECK(MPI_Isend(buffers.sendLower, static_cast<int>(planeSize), MPI_DOUBLE, lowerRank, kLowerHaloTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
    }
    if (upperRank != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(buffers.receiveUpper, static_cast<int>(planeSize), MPI_DOUBLE, upperRank, kLowerHaloTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
        MPI_CHECK(MPI_Isend(buffers.sendUpper, static_cast<int>(planeSize), MPI_DOUBLE, upperRank, kUpperHaloTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
    }
    if (requestCount != 0) {
        MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE), rank);
    }

    if (lowerRank == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(deviceField, deviceField + planeSize, bytes, cudaMemcpyDeviceToDevice,
                                   communicationStream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(deviceField, buffers.receiveLower, bytes, cudaMemcpyHostToDevice,
                                   communicationStream));
    }
    if (upperRank == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(deviceField + (localNz + 1) * planeSize, deviceField + localNz * planeSize,
                                   bytes, cudaMemcpyDeviceToDevice, communicationStream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(deviceField + (localNz + 1) * planeSize, buffers.receiveUpper, bytes,
                                   cudaMemcpyHostToDevice, communicationStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));
}

void initializeConcentration(std::vector<double>& concentration, const size_t nx, const size_t ny,
                             const size_t globalZOffset, const size_t globalVolume) {
    const size_t planeSize = nx * ny;
    const size_t localVolume = concentration.size();
#pragma omp parallel for schedule(static)
    for (long long localIndex = 0; localIndex < static_cast<long long>(localVolume); ++localIndex) {
        const size_t offset = static_cast<size_t>(localIndex);
        const size_t z = globalZOffset + offset / planeSize;
        const size_t linearId = z * planeSize + offset % planeSize;
        const double pseudo = (((linearId + 1) * 1299709ULL) % globalVolume) / static_cast<double>(globalVolume);
        concentration[offset] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& concentration, const int rank) {
    double localMinimum = std::numeric_limits<double>::infinity();
    double localMaximum = -std::numeric_limits<double>::infinity();
    int localNonFiniteCount = 0;
#pragma omp parallel for schedule(static) reduction(min : localMinimum) reduction(max : localMaximum) reduction(+ : localNonFiniteCount)
    for (long long index = 0; index < static_cast<long long>(concentration.size()); ++index) {
        const double value = concentration[static_cast<size_t>(index)];
        if (!std::isfinite(value)) {
            ++localNonFiniteCount;
        } else {
            localMinimum = std::min(localMinimum, value);
            localMaximum = std::max(localMaximum, value);
        }
    }

    double globalMinimum = 0.0;
    double globalMaximum = 0.0;
    int globalNonFiniteCount = 0;
    MPI_CHECK(MPI_Allreduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD), rank);
    MPI_CHECK(MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), rank);
    MPI_CHECK(MPI_Allreduce(&localNonFiniteCount, &globalNonFiniteCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD), rank);

    if (rank == 0) {
        if (globalNonFiniteCount != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
    }
    return globalNonFiniteCount == 0 && globalMaximum <= 10.0 && globalMinimum >= -10.0;
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

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortRun(rank, "MPI does not provide the MPI_THREAD_FUNNELED level required by this benchmark.");
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-x") == 0 && argument + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++argument], nullptr, 10));
        } else if (std::strcmp(argv[argument], "-y") == 0 && argument + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++argument], nullptr, 10));
        } else if (std::strcmp(argv[argument], "-z") == 0 && argument + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++argument], nullptr, 10));
        } else if (std::strcmp(argv[argument], "-i") == 0 && argument + 1 < argc) {
            iterations = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[argument]);
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
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        abortRun(rank, "All grid dimensions must be positive and the iteration count must be non-negative.");
    }
    if (worldSize > static_cast<int>(nz)) {
        abortRun(rank, "The MPI world size cannot exceed the number of z planes.");
    }
    if (multiplicationOverflows(nx, ny)) {
        abortRun(rank, "The x and y dimensions overflow the supported allocation size.");
    }
    const size_t planeSize = nx * ny;
    if (planeSize > static_cast<size_t>(std::numeric_limits<int>::max()) || multiplicationOverflows(planeSize, nz)) {
        abortRun(rank, "The requested grid is too large for this MPI halo exchange.");
    }
    const size_t globalVolume = planeSize * nz;

    const size_t baseLocalNz = nz / static_cast<size_t>(worldSize);
    const size_t extraPlanes = nz % static_cast<size_t>(worldSize);
    const size_t localNz = baseLocalNz + (static_cast<size_t>(rank) < extraPlanes ? 1 : 0);
    const size_t globalZOffset = static_cast<size_t>(rank) * baseLocalNz +
                                 std::min(static_cast<size_t>(rank), extraPlanes);
    if (multiplicationOverflows(planeSize, localNz + 2)) {
        abortRun(rank, "The requested local grid is too large for CUDA allocation.");
    }
    const size_t localVolume = planeSize * localNz;
    const size_t allocatedVolume = planeSize * (localNz + 2);
    if (localVolume > static_cast<size_t>(std::numeric_limits<int>::max())) {
        abortRun(rank, "The requested local grid is too large for MPI result collection.");
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm), rank);
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank), rank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0 || localRank >= deviceCount) {
        abortRun(rank, "Each MPI rank must have an exclusive visible CUDA device on its node.");
    }
    CUDA_CHECK(cudaSetDevice(localRank));

    cudaDeviceProp deviceProperties {};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, localRank));
    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize, omp_get_max_threads());
        std::printf("CUDA device mapping: one MPI rank per GPU (rank 0: %s)\n", deviceProperties.name);
        std::printf("Local z decomposition: rank 0 owns %zu planes\n", localNz);
    }

    // Physical parameters, retained from the serial benchmark.
    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double timeStep = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;
    constexpr double invDx2 = 1.0 / (dx * dx);
    constexpr double invDy2 = 1.0 / (dy * dy);
    constexpr double invDz2 = 1.0 / (dz * dz);

    std::vector<double> hostConcentration(localVolume);
    if (rank == 0) {
        std::printf("Initializing concentration field...\n");
    }
    initializeConcentration(hostConcentration, nx, ny, globalZOffset, globalVolume);

    double* deviceCold = nullptr;
    double* deviceNew = nullptr;
    double* deviceMu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCold), allocatedVolume * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceNew), allocatedVolume * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMu), allocatedVolume * sizeof(double)));

    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    cudaEvent_t muReady = nullptr;
    cudaEvent_t updateReady = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&muReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&updateReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemcpyAsync(deviceCold + planeSize, hostConcentration.data(), localVolume * sizeof(double),
                               cudaMemcpyHostToDevice, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    HaloBuffers haloBuffers;
    haloBuffers.allocate(planeSize);
    const int lowerRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upperRank = rank + 1 == worldSize ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD), rank);
    const double startTime = MPI_Wtime();

    for (int step = 0; step < iterations; ++step) {
        // Exchange concentration halos while the CUDA compute stream evaluates planes that do not use them.
        if (step != 0) {
            CUDA_CHECK(cudaStreamWaitEvent(communicationStream, updateReady, 0));
        }
        beginHaloExchange(deviceCold, localNz, planeSize, lowerRank, upperRank, haloBuffers, communicationStream);
        launchChemicalPotential(deviceCold, deviceMu, nx, ny, planeSize, 2, localNz > 2 ? localNz - 2 : 0,
                                invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB, computeStream);
        finishHaloExchange(deviceCold, localNz, planeSize, lowerRank, upperRank, haloBuffers, communicationStream, rank);
        launchChemicalPotential(deviceCold, deviceMu, nx, ny, planeSize, 1, 1, invDx2, invDy2, invDz2, gamma, eAA,
                                eBB, eAB, computeStream);
        if (localNz > 1) {
            launchChemicalPotential(deviceCold, deviceMu, nx, ny, planeSize, localNz, 1, invDx2, invDy2, invDz2,
                                    gamma, eAA, eBB, eAB, computeStream);
        }

        // The completed chemical potential is exchanged in the same overlapped fashion before the update stencil.
        CUDA_CHECK(cudaEventRecord(muReady, computeStream));
        CUDA_CHECK(cudaStreamWaitEvent(communicationStream, muReady, 0));
        beginHaloExchange(deviceMu, localNz, planeSize, lowerRank, upperRank, haloBuffers, communicationStream);
        launchUpdate(deviceCold, deviceMu, deviceNew, nx, ny, planeSize, 2, localNz > 2 ? localNz - 2 : 0, invDx2,
                     invDy2, invDz2, diffusion, timeStep, computeStream);
        finishHaloExchange(deviceMu, localNz, planeSize, lowerRank, upperRank, haloBuffers, communicationStream, rank);
        launchUpdate(deviceCold, deviceMu, deviceNew, nx, ny, planeSize, 1, 1, invDx2, invDy2, invDz2, diffusion,
                     timeStep, computeStream);
        if (localNz > 1) {
            launchUpdate(deviceCold, deviceMu, deviceNew, nx, ny, planeSize, localNz, 1, invDx2, invDy2, invDz2,
                         diffusion, timeStep, computeStream);
        }
        CUDA_CHECK(cudaEventRecord(updateReady, computeStream));
        std::swap(deviceCold, deviceNew);
    }

    if (iterations != 0) {
        CUDA_CHECK(cudaEventSynchronize(updateReady));
    }
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), rank);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double cellUpdates = static_cast<double>(globalVolume) * static_cast<double>(iterations);
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(hostConcentration.data(), deviceCold + planeSize, localVolume * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        const int localElementCount = static_cast<int>(localVolume);
        std::vector<int> receiveCounts;
        std::vector<int> receiveDisplacements;
        std::vector<double> globalConcentration;
        if (rank == 0) {
            receiveCounts.resize(static_cast<size_t>(worldSize));
            receiveDisplacements.resize(static_cast<size_t>(worldSize));
        }
        MPI_CHECK(MPI_Gather(&localElementCount, 1, MPI_INT, rank == 0 ? receiveCounts.data() : nullptr, 1, MPI_INT, 0,
                             MPI_COMM_WORLD),
                  rank);
        if (rank == 0) {
            int displacement = 0;
            for (int process = 0; process < worldSize; ++process) {
                receiveDisplacements[static_cast<size_t>(process)] = displacement;
                displacement += receiveCounts[static_cast<size_t>(process)];
            }
            globalConcentration.resize(globalVolume);
        }
        MPI_CHECK(MPI_Gatherv(hostConcentration.data(), localElementCount, MPI_DOUBLE,
                              rank == 0 ? globalConcentration.data() : nullptr,
                              rank == 0 ? receiveCounts.data() : nullptr,
                              rank == 0 ? receiveDisplacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                  rank);
        if (rank == 0) {
            print_results(globalConcentration, "Concentration");
        }
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateResult(hostConcentration, rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    haloBuffers.release();
    CUDA_CHECK(cudaEventDestroy(updateReady));
    CUDA_CHECK(cudaEventDestroy(muReady));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(deviceMu));
    CUDA_CHECK(cudaFree(deviceNew));
    CUDA_CHECK(cudaFree(deviceCold));
    MPI_CHECK(MPI_Comm_free(&localComm), rank);
    MPI_Finalize();
    return valid ? 0 : 1;
}
