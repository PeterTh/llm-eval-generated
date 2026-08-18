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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kHaloTagLower = 100;
constexpr int kHaloTagUpper = 101;

[[noreturn]] void abortWithMessage(const MPI_Comm comm, const int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(comm, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* expression, const MPI_Comm comm, const int rank) {
    if (error != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA call '%s' failed: %s", expression,
                      cudaGetErrorString(error));
        abortWithMessage(comm, rank, message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, comm, rank)

size_t checkedMultiply(const size_t a, const size_t b, const MPI_Comm comm, const int rank) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        abortWithMessage(comm, rank, "Grid dimensions overflow the addressable size");
    }
    return a * b;
}

int checkedMpiCount(const size_t count, const MPI_Comm comm, const int rank) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(comm, rank, "A grid plane is too large for the MPI count interface");
    }
    return static_cast<int>(count);
}

int gridBlocks(const size_t elements) {
    const size_t required = (elements + static_cast<size_t>(kThreadsPerBlock) - 1) /
                            static_cast<size_t>(kThreadsPerBlock);
    // The kernels use a grid-stride loop, so limiting the grid keeps launch overhead
    // bounded for very large grids without limiting the amount of work processed.
    return static_cast<int>(std::min<size_t>(required, 65535));
}

struct HaloBuffers {
    double* sendLower = nullptr;
    double* sendUpper = nullptr;
    double* recvLower = nullptr;
    double* recvUpper = nullptr;

    void allocate(const size_t bytes, const MPI_Comm comm, const int rank) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendLower), bytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendUpper), bytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recvLower), bytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recvUpper), bytes, cudaHostAllocPortable));
    }

    void release(const MPI_Comm comm, const int rank) {
        if (sendLower != nullptr) CUDA_CHECK(cudaFreeHost(sendLower));
        if (sendUpper != nullptr) CUDA_CHECK(cudaFreeHost(sendUpper));
        if (recvLower != nullptr) CUDA_CHECK(cudaFreeHost(recvLower));
        if (recvUpper != nullptr) CUDA_CHECK(cudaFreeHost(recvUpper));
        sendLower = sendUpper = recvLower = recvUpper = nullptr;
    }
};

// Every rank owns a contiguous slab in z.  The two extra planes are ghost
// planes; they are refreshed before each stencil phase by exchangeHalos().
void exchangeHalos(double* const field, const size_t planeSize, const size_t localNz,
                   const int rank, const int worldSize, const MPI_Comm comm,
                   HaloBuffers& halos, const cudaStream_t stream) {
    const size_t planeBytes = planeSize * sizeof(double);
    const int planeCount = checkedMpiCount(planeSize, comm, rank);
    const int lowerRank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upperRank = (rank + 1 < worldSize) ? rank + 1 : MPI_PROC_NULL;

    // Queue all device-to-host transfers together.  Pinned buffers permit the
    // copies to overlap with MPI progress on systems that support it.
    if (lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(halos.sendLower, field + planeSize, planeBytes,
                                   cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(field, field + planeSize, planeBytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (upperRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(halos.sendUpper, field + localNz * planeSize, planeBytes,
                                   cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * planeSize,
                                   field + localNz * planeSize, planeBytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The tags are directional: a lower-plane send is received as the upper
    // halo by the rank below, and vice versa.  Nonblocking operations allow
    // both neighbors to communicate concurrently.
    MPI_Request requests[4];
    MPI_Irecv(halos.recvLower, planeCount, MPI_DOUBLE, lowerRank, kHaloTagUpper,
              comm, &requests[0]);
    MPI_Isend(halos.sendLower, planeCount, MPI_DOUBLE, lowerRank, kHaloTagLower,
              comm, &requests[1]);
    MPI_Irecv(halos.recvUpper, planeCount, MPI_DOUBLE, upperRank, kHaloTagLower,
              comm, &requests[2]);
    MPI_Isend(halos.sendUpper, planeCount, MPI_DOUBLE, upperRank, kHaloTagUpper,
              comm, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    if (lowerRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field, halos.recvLower, planeBytes,
                                   cudaMemcpyHostToDevice, stream));
    }
    if (upperRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * planeSize, halos.recvUpper,
                                   planeBytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t localNz,
    const double dxSquared, const double dySquared, const double dzSquared,
    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t planeSize = nx * ny;
    const size_t ownedCells = planeSize * localNz;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < ownedCells; linear += stride) {
        const size_t z = linear / planeSize + 1;
        const size_t inPlane = linear % planeSize;
        const size_t y = inPlane / nx;
        const size_t x = inPlane % nx;
        const size_t center = z * planeSize + inPlane;

        const size_t xp = (x + 1 < nx) ? x + 1 : x;
        const size_t xn = (x > 0) ? x - 1 : x;
        const size_t yp = (y + 1 < ny) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : y;
        const size_t zp = z + 1;
        const size_t zn = z - 1;

        const double cxx = (c[z * planeSize + y * nx + xp] +
                            c[z * planeSize + y * nx + xn] - 2.0 * c[center]) / dxSquared;
        const double cyy = (c[z * planeSize + yp * nx + x] +
                            c[z * planeSize + yn * nx + x] - 2.0 * c[center]) / dySquared;
        const double czz = (c[zp * planeSize + inPlane] +
                            c[zn * planeSize + inPlane] - 2.0 * c[center]) / dzSquared;
        const double laplacian = cxx + cyy + czz;
        const double cv = c[center];

        mu[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
                     3.0 * cv + cv * cv * cv - gamma * laplacian;
    }
}

__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu, const size_t nx, const size_t ny, const size_t localNz,
    const double coefficient, const double dxSquared, const double dySquared,
    const double dzSquared) {
    const size_t planeSize = nx * ny;
    const size_t ownedCells = planeSize * localNz;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < ownedCells; linear += stride) {
        const size_t z = linear / planeSize + 1;
        const size_t inPlane = linear % planeSize;
        const size_t y = inPlane / nx;
        const size_t x = inPlane % nx;
        const size_t center = z * planeSize + inPlane;

        const size_t xp = (x + 1 < nx) ? x + 1 : x;
        const size_t xn = (x > 0) ? x - 1 : x;
        const size_t yp = (y + 1 < ny) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : y;
        const size_t zp = z + 1;
        const size_t zn = z - 1;

        const double mux = mu[z * planeSize + y * nx + xp] +
                           mu[z * planeSize + y * nx + xn] - 2.0 * mu[center];
        const double muy = mu[z * planeSize + yp * nx + x] +
                           mu[z * planeSize + yn * nx + x] - 2.0 * mu[center];
        const double muz = mu[zp * planeSize + inPlane] +
                           mu[zn * planeSize + inPlane] - 2.0 * mu[center];

        cnew[center] = cold[center] + coefficient *
                       (mux / dxSquared + muy / dySquared + muz / dzSquared);
    }
}

void launchChemicalPotential(const double* const d_c, double* const d_mu,
                             const size_t nx, const size_t ny, const size_t localNz,
                             const double dx, const double dy, const double dz,
                             const double gamma, const double e_AA, const double e_BB,
                             const double e_AB, const cudaStream_t stream,
                             const MPI_Comm comm, const int rank) {
    computeChemicalPotentialKernel<<<gridBlocks(nx * ny * localNz), kThreadsPerBlock, 0, stream>>>(
        d_c, d_mu, nx, ny, localNz, dx * dx, dy * dy, dz * dz,
        gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdate(double* const d_cnew, const double* const d_cold, const double* const d_mu,
                  const size_t nx, const size_t ny, const size_t localNz,
                  const double coefficient, const double dx, const double dy, const double dz,
                  const cudaStream_t stream, const MPI_Comm comm, const int rank) {
    cahnHilliardUpdateKernel<<<gridBlocks(nx * ny * localNz), kThreadsPerBlock, 0, stream>>>(
        d_cnew, d_cold, d_mu, nx, ny, localNz, coefficient,
        dx * dx, dy * dy, dz * dz);
    CUDA_CHECK(cudaGetLastError());
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t globalNz, const size_t globalZStart) {
    const size_t planeSize = nx * ny;
    const size_t volume = planeSize * globalNz;

    // This is deliberately OpenMP-parallel: initialization is outside the
    // timed solver and scales independently on ranks with many CPU cores.
    #pragma omp parallel for schedule(static)
    for (std::int64_t localLinear = 0;
         localLinear < static_cast<std::int64_t>(c.size()); ++localLinear) {
        const size_t localId = static_cast<size_t>(localLinear);
        const size_t globalId = globalZStart * planeSize + localId;
        const size_t pseudoNumerator = ((globalId + 1) * static_cast<size_t>(1299709)) % volume;
        const double pseudo = static_cast<double>(pseudoNumerator) / static_cast<double>(volume);
        c[localId] = -1.0 + 2.0 * pseudo;
    }
}

void copyInteriorToHost(const double* const d_field, std::vector<double>& hostField,
                        const size_t planeSize, const cudaStream_t stream,
                        const MPI_Comm comm, const int rank) {
    CUDA_CHECK(cudaMemcpyAsync(hostField.data(), d_field + planeSize,
                               hostField.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool validateDistributed(const std::vector<double>& localField, const MPI_Comm comm,
                         const int rank) {
    int localInvalid = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();

    #pragma omp parallel for schedule(static) reduction(|:localInvalid) reduction(min:localMin) reduction(max:localMax)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(localField.size()); ++i) {
        const double value = localField[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
            localInvalid = 1;
        } else {
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    }

    int globalInvalid = 0;
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_LOR, comm);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    int valid = (globalInvalid == 0 && globalMax <= 10.0 && globalMin >= -10.0) ? 1 : 0;
    if (rank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (valid == 0) {
                std::printf("Validation failed: values out of expected range\n");
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
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

int localRankOnNode(const MPI_Comm comm, const int globalRank) {
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = globalRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    return localRank;
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        abortWithMessage(comm, rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }
    if (static_cast<size_t>(worldSize) > nz) {
        if (rank == 0) {
            std::fprintf(stderr, "The z dimension (%zu) must be at least the MPI rank count (%d)\n",
                         nz, worldSize);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeSize = checkedMultiply(nx, ny, comm, rank);
    const size_t gridSize = checkedMultiply(planeSize, nz, comm, rank);
    const size_t baseNz = nz / static_cast<size_t>(worldSize);
    const size_t remainder = nz % static_cast<size_t>(worldSize);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZStart = static_cast<size_t>(rank) * baseNz +
                               std::min(static_cast<size_t>(rank), remainder);
    const size_t localCells = checkedMultiply(planeSize, localNz, comm, rank);
    const size_t devicePlanes = localNz + 2;
    const size_t deviceCells = checkedMultiply(planeSize, devicePlanes, comm, rank);
    const size_t deviceBytes = checkedMultiply(deviceCells, sizeof(double), comm, rank);

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks, %d OpenMP threads/rank, CUDA\n",
                    worldSize, omp_get_max_threads());
    }

    // Physical parameters, identical to the original implementation.
    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double D = 1.0;

    const int localRank = localRankOnNode(comm, rank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) abortWithMessage(comm, rank, "No CUDA device is visible to this MPI rank");
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr)); // Establish the CUDA context before timing.

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), deviceBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), deviceBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), deviceBytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    HaloBuffers halos;
    halos.allocate(planeSize * sizeof(double), comm, rank);

    std::vector<double> initial(localCells);
    if (rank == 0) std::printf("Initializing concentration field...\n");
    initializeConcentration(initial, nx, ny, nz, globalZStart);
    CUDA_CHECK(cudaMemcpyAsync(d_cold + planeSize, initial.data(), localCells * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(d_cold, planeSize, localNz, rank, worldSize, comm, halos, stream);
        launchChemicalPotential(d_cold, d_mu, nx, ny, localNz, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB, stream, comm, rank);
        exchangeHalos(d_mu, planeSize, localNz, rank, worldSize, comm, halos, stream);
        launchUpdate(d_cnew, d_cold, d_mu, nx, ny, localNz, dt * D,
                     dx, dy, dz, stream, comm, rank);
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(comm);
    const double elapsedSeconds = MPI_Wtime() - start;
    if (rank == 0) {
        const long long durationMs = static_cast<long long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %lld ms\n", durationMs);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = (elapsedSeconds > 0.0) ? cellUpdates / elapsedSeconds / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> localResult;
    if (printResults || validate) {
        localResult.resize(localCells);
        copyInteriorToHost(d_cold, localResult, planeSize, stream, comm, rank);
    }

    if (printResults) {
        if (localCells > static_cast<size_t>(std::numeric_limits<int>::max())) {
            abortWithMessage(comm, rank, "Result gathering exceeds the MPI count interface");
        }
        std::vector<double> result;
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            result.resize(gridSize);
            counts.resize(static_cast<size_t>(worldSize));
            displacements.resize(static_cast<size_t>(worldSize));
            for (int r = 0; r < worldSize; ++r) {
                const size_t rankNz = baseNz + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t rankStart = static_cast<size_t>(r) * baseNz +
                                         std::min(static_cast<size_t>(r), remainder);
                const size_t rankCells = checkedMultiply(planeSize, rankNz, comm, rank);
                counts[static_cast<size_t>(r)] = checkedMpiCount(rankCells, comm, rank);
                displacements[static_cast<size_t>(r)] = checkedMpiCount(
                    checkedMultiply(planeSize, rankStart, comm, rank), comm, rank);
            }
        }
        MPI_Gatherv(localResult.data(), static_cast<int>(localCells), MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(result, "Concentration");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(localResult, comm, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    halos.release(comm, rank);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Finalize();
    return (validate && !valid) ? 1 : 0;
}
