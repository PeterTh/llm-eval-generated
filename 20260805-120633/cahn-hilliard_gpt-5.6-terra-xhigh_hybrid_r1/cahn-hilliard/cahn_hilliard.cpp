#include <algorithm>
#include <chrono>
#include <climits>
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

constexpr int kThreadsPerBlock = 256;
constexpr int kTagLowPlane = 3101;
constexpr int kTagHighPlane = 3102;

[[noreturn]] void cudaCheck(const cudaError_t status, const char* const expression,
                            const char* const file, const int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d while evaluating %s: %s\n", file, line,
                 expression, cudaGetErrorString(status));
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    }
    std::abort();
}

#define CUDA_CHECK(expression)                                                               \
    do {                                                                                     \
        const cudaError_t cuda_status_ = (expression);                                      \
        if (cuda_status_ != cudaSuccess) {                                                   \
            cudaCheck(cuda_status_, #expression, __FILE__, __LINE__);                       \
        }                                                                                    \
    } while (false)

void mpiCheck(const int status, const char* const expression, const int rank) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING] = {};
    int errorLength = 0;
    MPI_Error_string(status, error, &errorLength);
    std::fprintf(stderr, "MPI failure on rank %d while evaluating %s: %.*s\n", rank,
                 expression, errorLength, error);
    MPI_Abort(MPI_COMM_WORLD, status);
}

#define MPI_CHECK(expression) mpiCheck((expression), #expression, rank)

// The device arrays include one z-plane on either side of the rank-local slab.
// A physical boundary ghost is a copy of its adjacent interior plane, exactly
// matching the original clamped-boundary stencil.
struct HaloBuffers {
    double* sendLow = nullptr;
    double* sendHigh = nullptr;
    double* receiveLow = nullptr;
    double* receiveHigh = nullptr;
};

__device__ __forceinline__ double laplacian(const double* const field, const size_t point,
                                             const size_t inPlane, const size_t nx, const size_t ny,
                                             const size_t planeElements,
                                             const double inverseDx2,
                                             const double inverseDy2,
                                             const double inverseDz2) {
    const double center = field[point];
    const size_t x = inPlane % nx;
    const size_t positiveX = (x + 1U < nx) ? 1U : 0U;
    const size_t negativeX = (x > 0U) ? 1U : 0U;
    const size_t y = inPlane / nx;
    const size_t positiveY = (y + 1U < ny) ? nx : 0U;
    const size_t negativeY = (y > 0U) ? nx : 0U;

    const double xx = (field[point + positiveX] + field[point - negativeX] - 2.0 * center) *
                      inverseDx2;
    const double yy = (field[point + positiveY] + field[point - negativeY] - 2.0 * center) *
                      inverseDy2;
    const double zz = (field[point + planeElements] + field[point - planeElements] -
                       2.0 * center) * inverseDz2;
    return xx + yy + zz;
}

__global__ void chemicalRangeKernel(const double* __restrict__ concentration,
                                    double* __restrict__ chemicalPotential, const size_t nx,
                                    const size_t ny,
                                    const size_t planeElements, const size_t firstLocalZ,
                                    const size_t planeCount, const double inverseDx2,
                                    const double inverseDy2, const double inverseDz2,
                                    const double gamma, const double eAA, const double eBB,
                                    const double eAB) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = planeElements * planeCount;
    if (linear >= elements) {
        return;
    }

    const size_t inPlane = linear % planeElements;
    const size_t localZ = firstLocalZ + linear / planeElements;
    const size_t point = localZ * planeElements + inPlane;
    const double c = concentration[point];

    chemicalPotential[point] =
        4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB) + 3.0 * c + c * c * c -
        gamma * laplacian(concentration, point, inPlane, nx, ny, planeElements, inverseDx2, inverseDy2,
                          inverseDz2);
}

__global__ void chemicalBoundaryKernel(const double* __restrict__ concentration,
                                       double* __restrict__ chemicalPotential, const size_t nx,
                                       const size_t ny,
                                       const size_t planeElements, const size_t localNz,
                                       const double inverseDx2, const double inverseDy2,
                                       const double inverseDz2, const double gamma,
                                       const double eAA, const double eBB, const double eAB) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t boundaryPlanes = (localNz == 1U) ? 1U : 2U;
    if (linear >= planeElements * boundaryPlanes) {
        return;
    }

    const size_t inPlane = linear % planeElements;
    const size_t localZ = (linear < planeElements) ? 1U : localNz;
    const size_t point = localZ * planeElements + inPlane;
    const double c = concentration[point];

    chemicalPotential[point] =
        4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB) + 3.0 * c + c * c * c -
        gamma * laplacian(concentration, point, inPlane, nx, ny, planeElements, inverseDx2, inverseDy2,
                          inverseDz2);
}

__global__ void updateRangeKernel(double* __restrict__ nextConcentration,
                                  const double* __restrict__ concentration,
                                  const double* __restrict__ chemicalPotential, const size_t nx,
                                  const size_t ny,
                                  const size_t planeElements, const size_t firstLocalZ,
                                  const size_t planeCount, const double inverseDx2,
                                  const double inverseDy2, const double inverseDz2,
                                  const double diffusion, const double dt) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = planeElements * planeCount;
    if (linear >= elements) {
        return;
    }

    const size_t inPlane = linear % planeElements;
    const size_t localZ = firstLocalZ + linear / planeElements;
    const size_t point = localZ * planeElements + inPlane;
    nextConcentration[point] =
        concentration[point] + dt * diffusion *
                                   laplacian(chemicalPotential, point, inPlane, nx, ny, planeElements,
                                             inverseDx2, inverseDy2, inverseDz2);
}

__global__ void updateBoundaryKernel(double* __restrict__ nextConcentration,
                                     const double* __restrict__ concentration,
                                     const double* __restrict__ chemicalPotential, const size_t nx,
                                     const size_t ny,
                                     const size_t planeElements, const size_t localNz,
                                     const double inverseDx2, const double inverseDy2,
                                     const double inverseDz2, const double diffusion,
                                     const double dt) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t boundaryPlanes = (localNz == 1U) ? 1U : 2U;
    if (linear >= planeElements * boundaryPlanes) {
        return;
    }

    const size_t inPlane = linear % planeElements;
    const size_t localZ = (linear < planeElements) ? 1U : localNz;
    const size_t point = localZ * planeElements + inPlane;
    nextConcentration[point] =
        concentration[point] + dt * diffusion *
                                   laplacian(chemicalPotential, point, inPlane, nx, ny, planeElements,
                                             inverseDx2, inverseDy2, inverseDz2);
}

inline dim3 gridFor(const size_t elementCount) {
    const size_t blocks = (elementCount + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    return dim3(static_cast<unsigned int>(blocks));
}

void launchChemicalInterior(const double* const concentration, double* const chemicalPotential,
                            const size_t nx, const size_t ny, const size_t planeElements, const size_t localNz,
                            const double inverseDx2, const double inverseDy2,
                            const double inverseDz2, const double gamma, const double eAA,
                            const double eBB, const double eAB, const cudaStream_t stream) {
    if (localNz <= 2U) {
        return;
    }
    const size_t interiorPlanes = localNz - 2U;
    chemicalRangeKernel<<<gridFor(interiorPlanes * planeElements), kThreadsPerBlock, 0, stream>>>(
        concentration, chemicalPotential, nx, ny, planeElements, 2U, interiorPlanes, inverseDx2,
        inverseDy2, inverseDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchChemicalBoundary(const double* const concentration, double* const chemicalPotential,
                            const size_t nx, const size_t ny, const size_t planeElements, const size_t localNz,
                            const double inverseDx2, const double inverseDy2,
                            const double inverseDz2, const double gamma, const double eAA,
                            const double eBB, const double eAB, const cudaStream_t stream) {
    const size_t boundaryPlanes = (localNz == 1U) ? 1U : 2U;
    chemicalBoundaryKernel<<<gridFor(boundaryPlanes * planeElements), kThreadsPerBlock, 0, stream>>>(
        concentration, chemicalPotential, nx, ny, planeElements, localNz, inverseDx2, inverseDy2,
        inverseDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdateInterior(double* const nextConcentration, const double* const concentration,
                          const double* const chemicalPotential, const size_t nx,
                          const size_t ny, const size_t planeElements, const size_t localNz,
                          const double inverseDx2, const double inverseDy2,
                          const double inverseDz2, const double diffusion, const double dt,
                          const cudaStream_t stream) {
    if (localNz <= 2U) {
        return;
    }
    const size_t interiorPlanes = localNz - 2U;
    updateRangeKernel<<<gridFor(interiorPlanes * planeElements), kThreadsPerBlock, 0, stream>>>(
        nextConcentration, concentration, chemicalPotential, nx, ny, planeElements, 2U,
        interiorPlanes, inverseDx2, inverseDy2, inverseDz2, diffusion, dt);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdateBoundary(double* const nextConcentration, const double* const concentration,
                          const double* const chemicalPotential, const size_t nx,
                          const size_t ny, const size_t planeElements, const size_t localNz,
                          const double inverseDx2, const double inverseDy2,
                          const double inverseDz2, const double diffusion, const double dt,
                          const cudaStream_t stream) {
    const size_t boundaryPlanes = (localNz == 1U) ? 1U : 2U;
    updateBoundaryKernel<<<gridFor(boundaryPlanes * planeElements), kThreadsPerBlock, 0, stream>>>(
        nextConcentration, concentration, chemicalPotential, nx, ny, planeElements, localNz,
        inverseDx2, inverseDy2, inverseDz2, diffusion, dt);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Queue the device-to-host leg.  The caller can use a producer event when the
// boundary planes were just generated on a different stream.
void queueHaloCopiesToHost(const double* const field, const size_t localNz,
                           const size_t planeElements, HaloBuffers& halos,
                           const cudaStream_t communicationStream,
                           const cudaEvent_t producerReady = nullptr) {
    if (producerReady != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(communicationStream, producerReady, 0));
    }
    const size_t planeBytes = planeElements * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(halos.sendLow, field + planeElements, planeBytes,
                               cudaMemcpyDeviceToHost, communicationStream));
    CUDA_CHECK(cudaMemcpyAsync(halos.sendHigh, field + localNz * planeElements, planeBytes,
                               cudaMemcpyDeviceToHost, communicationStream));
}

// MPI deliberately sees pinned host buffers.  This works with every standard
// MPI implementation, avoids depending on CUDA-aware MPI, and still leaves the
// GPU interior kernels free to run while the host waits for halo traffic.
void completeHaloExchange(double* const field, const size_t localNz, const size_t planeElements,
                          HaloBuffers& halos, const cudaStream_t communicationStream,
                          const cudaEvent_t haloReady, const int rank, const int worldSize) {
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));

    MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL};
    int requestCount = 0;
    const int planeCount = static_cast<int>(planeElements);

    if (rank > 0) {
        MPI_CHECK(MPI_Irecv(halos.receiveLow, planeCount, MPI_DOUBLE, rank - 1, kTagHighPlane,
                            MPI_COMM_WORLD, &requests[requestCount++]));
        MPI_CHECK(MPI_Isend(halos.sendLow, planeCount, MPI_DOUBLE, rank - 1, kTagLowPlane,
                            MPI_COMM_WORLD, &requests[requestCount++]));
    }
    if (rank + 1 < worldSize) {
        MPI_CHECK(MPI_Irecv(halos.receiveHigh, planeCount, MPI_DOUBLE, rank + 1, kTagLowPlane,
                            MPI_COMM_WORLD, &requests[requestCount++]));
        MPI_CHECK(MPI_Isend(halos.sendHigh, planeCount, MPI_DOUBLE, rank + 1, kTagHighPlane,
                            MPI_COMM_WORLD, &requests[requestCount++]));
    }
    if (requestCount > 0) {
        MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE));
    }

    const size_t planeBytes = planeElements * sizeof(double);
    if (rank == 0) {
        CUDA_CHECK(cudaMemcpyAsync(field, field + planeElements, planeBytes, cudaMemcpyDeviceToDevice,
                                   communicationStream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(field, halos.receiveLow, planeBytes, cudaMemcpyHostToDevice,
                                   communicationStream));
    }
    if (rank + 1 == worldSize) {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1U) * planeElements,
                                   field + localNz * planeElements, planeBytes,
                                   cudaMemcpyDeviceToDevice, communicationStream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1U) * planeElements, halos.receiveHigh,
                                   planeBytes, cudaMemcpyHostToDevice, communicationStream));
    }
    CUDA_CHECK(cudaEventRecord(haloReady, communicationStream));
}

void initializeConcentration(double* const concentration, const size_t nx, const size_t ny,
                             const size_t localNz, const size_t firstGlobalZ,
                             const size_t globalNz) {
    const size_t planeElements = nx * ny;
    const size_t localElements = localNz * planeElements;
    const size_t volume = globalNz * planeElements;

#pragma omp parallel for schedule(static)
    for (long long localLinear = 0; localLinear < static_cast<long long>(localElements);
         ++localLinear) {
        const size_t local = static_cast<size_t>(localLinear);
        const size_t localZ = local / planeElements;
        const size_t inPlane = local % planeElements;
        const size_t globalLinear = (firstGlobalZ + localZ) * planeElements + inPlane;
        const double pseudo =
            (((globalLinear + 1U) * static_cast<size_t>(1299709U)) % volume) /
            static_cast<double>(volume);
        concentration[local] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& concentration) {
    int invalid = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();

#pragma omp parallel for reduction(| : invalid) reduction(min : minimum) reduction(max : maximum) schedule(static)
    for (long long i = 0; i < static_cast<long long>(concentration.size()); ++i) {
        const double value = concentration[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
            invalid = 1;
        } else {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }

    if (invalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
    if (maximum > 10.0 || minimum < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* const programName) {
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

bool multiplicationOverflows(const size_t left, const size_t right) {
    return left != 0U && right > std::numeric_limits<size_t>::max() / left;
}

} // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0U) {
        ny = nx;
    }
    if (nz == 0U) {
        nz = nx;
    }
    if (nx == 0U || ny == 0U || nz == 0U || iterations < 0 || multiplicationOverflows(nx, ny) ||
        multiplicationOverflows(nx * ny, nz)) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid dimensions must be positive and fit in addressable memory; iterations must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (static_cast<size_t>(worldSize) > nz) {
        if (rank == 0) {
            std::fprintf(stderr, "The number of MPI ranks (%d) cannot exceed the z dimension (%zu)\n",
                         worldSize, nz);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeElements = nx * ny;
    const size_t globalElements = planeElements * nz;
    if (planeElements > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "An x-y plane is too large for the MPI count interface\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t basePlanes = nz / static_cast<size_t>(worldSize);
    const size_t extraPlanes = nz % static_cast<size_t>(worldSize);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < extraPlanes ? 1U : 0U);
    const size_t firstGlobalZ = static_cast<size_t>(rank) * basePlanes +
                                std::min(static_cast<size_t>(rank), extraPlanes);
    const size_t localElements = localNz * planeElements;
    if (localElements > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "A rank-local slab is too large for MPI_Gatherv\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localNodeComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &localNodeComm));
    int localNodeRank = 0;
    MPI_CHECK(MPI_Comm_rank(localNodeComm, &localNodeRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localNodeRank % deviceCount));

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize, omp_get_max_threads());
    }

    // Physical parameters, retained from the original finite-difference model.
    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;
    constexpr double inverseDx2 = 1.0 / (dx * dx);
    constexpr double inverseDy2 = 1.0 / (dy * dy);
    constexpr double inverseDz2 = 1.0 / (dz * dz);

    double* deviceConcentration = nullptr;
    double* deviceNextConcentration = nullptr;
    double* deviceChemicalPotential = nullptr;
    const size_t allocationElements = (localNz + 2U) * planeElements;
    const size_t allocationBytes = allocationElements * sizeof(double);
    CUDA_CHECK(cudaMalloc(&deviceConcentration, allocationBytes));
    CUDA_CHECK(cudaMalloc(&deviceNextConcentration, allocationBytes));
    CUDA_CHECK(cudaMalloc(&deviceChemicalPotential, allocationBytes));

    HaloBuffers halos;
    const size_t haloBytes = planeElements * sizeof(double);
    CUDA_CHECK(cudaMallocHost(&halos.sendLow, haloBytes));
    CUDA_CHECK(cudaMallocHost(&halos.sendHigh, haloBytes));
    CUDA_CHECK(cudaMallocHost(&halos.receiveLow, haloBytes));
    CUDA_CHECK(cudaMallocHost(&halos.receiveHigh, haloBytes));

    cudaStream_t interiorStream = nullptr;
    cudaStream_t boundaryStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    cudaEvent_t concentrationHaloReady = nullptr;
    cudaEvent_t chemicalBoundaryReady = nullptr;
    cudaEvent_t chemicalHaloReady = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&interiorStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&boundaryStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&concentrationHaloReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&chemicalBoundaryReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&chemicalHaloReady, cudaEventDisableTiming));

    std::vector<double> hostLocalConcentration(localElements);
    if (rank == 0) {
        std::printf("Initializing concentration field...\n");
    }
    initializeConcentration(hostLocalConcentration.data(), nx, ny, localNz, firstGlobalZ, nz);
    CUDA_CHECK(cudaMemcpy(deviceConcentration + planeElements, hostLocalConcentration.data(),
                          localElements * sizeof(double), cudaMemcpyHostToDevice));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    const auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < iterations; ++step) {
        // Concentration halos and chemical-potential interior execute concurrently.
        queueHaloCopiesToHost(deviceConcentration, localNz, planeElements, halos,
                              communicationStream);
        launchChemicalInterior(deviceConcentration, deviceChemicalPotential, nx, ny, planeElements,
                               localNz, inverseDx2, inverseDy2, inverseDz2, gamma, eAA, eBB,
                               eAB, interiorStream);
        completeHaloExchange(deviceConcentration, localNz, planeElements, halos, communicationStream,
                             concentrationHaloReady, rank, worldSize);

        CUDA_CHECK(cudaStreamWaitEvent(boundaryStream, concentrationHaloReady, 0));
        launchChemicalBoundary(deviceConcentration, deviceChemicalPotential, nx, ny, planeElements,
                               localNz, inverseDx2, inverseDy2, inverseDz2, gamma, eAA, eBB, eAB,
                               boundaryStream);
        CUDA_CHECK(cudaEventRecord(chemicalBoundaryReady, boundaryStream));

        // The second exchange overlaps MPI with the purely local concentration update.
        queueHaloCopiesToHost(deviceChemicalPotential, localNz, planeElements, halos,
                              communicationStream, chemicalBoundaryReady);
        launchUpdateInterior(deviceNextConcentration, deviceConcentration, deviceChemicalPotential,
                             nx, ny, planeElements, localNz, inverseDx2, inverseDy2, inverseDz2,
                             diffusion, dt, interiorStream);
        completeHaloExchange(deviceChemicalPotential, localNz, planeElements, halos,
                             communicationStream, chemicalHaloReady, rank, worldSize);

        CUDA_CHECK(cudaStreamWaitEvent(boundaryStream, chemicalHaloReady, 0));
        launchUpdateBoundary(deviceNextConcentration, deviceConcentration, deviceChemicalPotential,
                             nx, ny, planeElements, localNz, inverseDx2, inverseDy2, inverseDz2,
                             diffusion, dt, boundaryStream);

        CUDA_CHECK(cudaStreamSynchronize(interiorStream));
        CUDA_CHECK(cudaStreamSynchronize(boundaryStream));
        std::swap(deviceConcentration, deviceNextConcentration);
    }

    const auto end = std::chrono::steady_clock::now();
    const double localMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMilliseconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    std::vector<double> globalConcentration;
    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(hostLocalConcentration.data(), deviceConcentration + planeElements,
                              localElements * sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            globalConcentration.resize(globalElements);
            counts.resize(static_cast<size_t>(worldSize));
            displacements.resize(static_cast<size_t>(worldSize));
            for (int process = 0; process < worldSize; ++process) {
                const size_t processPlanes =
                    basePlanes + (static_cast<size_t>(process) < extraPlanes ? 1U : 0U);
                const size_t processFirstZ = static_cast<size_t>(process) * basePlanes +
                                             std::min(static_cast<size_t>(process), extraPlanes);
                counts[static_cast<size_t>(process)] =
                    static_cast<int>(processPlanes * planeElements);
                displacements[static_cast<size_t>(process)] =
                    static_cast<int>(processFirstZ * planeElements);
            }
        }
        MPI_CHECK(MPI_Gatherv(hostLocalConcentration.data(), static_cast<int>(localElements), MPI_DOUBLE,
                              rank == 0 ? globalConcentration.data() : nullptr,
                              rank == 0 ? counts.data() : nullptr,
                              rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                              MPI_COMM_WORLD));
    }

    int exitCode = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        const double cellUpdates = static_cast<double>(globalElements) * iterations;
        const double mcups = elapsedMilliseconds > 0.0 ? cellUpdates / elapsedMilliseconds / 1.0e3 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(globalConcentration, "Concentration");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(globalConcentration)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaEventDestroy(chemicalHaloReady));
    CUDA_CHECK(cudaEventDestroy(chemicalBoundaryReady));
    CUDA_CHECK(cudaEventDestroy(concentrationHaloReady));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(boundaryStream));
    CUDA_CHECK(cudaStreamDestroy(interiorStream));
    CUDA_CHECK(cudaFreeHost(halos.receiveHigh));
    CUDA_CHECK(cudaFreeHost(halos.receiveLow));
    CUDA_CHECK(cudaFreeHost(halos.sendHigh));
    CUDA_CHECK(cudaFreeHost(halos.sendLow));
    CUDA_CHECK(cudaFree(deviceChemicalPotential));
    CUDA_CHECK(cudaFree(deviceNextConcentration));
    CUDA_CHECK(cudaFree(deviceConcentration));
    MPI_CHECK(MPI_Comm_free(&localNodeComm));
    MPI_Finalize();
    return exitCode;
}
