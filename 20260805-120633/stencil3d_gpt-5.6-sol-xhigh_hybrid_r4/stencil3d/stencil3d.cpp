#include <mpi.h>
#include <omp.h>

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kZTile = 8;
constexpr int kLowerTag = 1701;
constexpr int kUpperTag = 1702;

int worldRank = 0;

[[noreturn]] void abortRun(const char* message, const int errorCode = 1) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, errorCode);
    std::abort();
}

void checkMpi(const int status, const char* expression) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    std::fprintf(stderr, "Rank %d: MPI failure in %s: %.*s\n",
                 worldRank, expression, length, error);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* expression,
               const char* file, const int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "Rank %d: CUDA failure in %s at %s:%d: %s\n",
                 worldRank, expression, file, line, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

#define MPI_CHECK(call) checkMpi((call), #call)
#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

bool checkedMultiply(const std::size_t a, const std::size_t b,
                     std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool parseSize(const char* text, std::size_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

struct Slab {
    std::size_t globalBegin;
    std::size_t planes;
};

Slab slabForRank(const std::size_t nz, const int rank, const int ranks) {
    const std::size_t rankCount = static_cast<std::size_t>(ranks);
    const std::size_t base = nz / rankCount;
    const std::size_t remainder = nz % rankCount;
    const std::size_t rankIndex = static_cast<std::size_t>(rank);
    return {
        rankIndex * base + std::min(rankIndex, remainder),
        base + (rankIndex < remainder ? 1U : 0U)
    };
}

// Each CUDA block owns an x/y tile and up to kZTile consecutive planes.  The
// z loop retains bottom/center values in registers, while the current x/y
// plane is shared by the block.  This reduces the steady-state global-memory
// traffic to approximately one load and one store per cell update.
__global__ void stencilRange(const Real* __restrict__ input,
                             Real* __restrict__ output,
                             const std::size_t nx,
                             const std::size_t ny,
                             const std::size_t firstZ,
                             const std::size_t lastZ) {
    __shared__ Real tile[kBlockY + 2][kBlockX + 2];

    const std::size_t rawX = 1 + static_cast<std::size_t>(blockIdx.x) * kBlockX
                           + threadIdx.x;
    const std::size_t rawY = 1 + static_cast<std::size_t>(blockIdx.y) * kBlockY
                           + threadIdx.y;
    const std::size_t x = min(rawX, nx - 1);
    const std::size_t y = min(rawY, ny - 1);
    const bool active = rawX < nx - 1 && rawY < ny - 1;
    const std::size_t plane = nx * ny;
    const std::size_t rowOffset = y * nx + x;

    const std::size_t zStride = static_cast<std::size_t>(gridDim.z) * kZTile;
    for (std::size_t tileBegin = firstZ
                                   + static_cast<std::size_t>(blockIdx.z) * kZTile;
         tileBegin <= lastZ; tileBegin += zStride) {
        const std::size_t tileEnd = min(tileBegin + kZTile - 1, lastZ);
        std::size_t index = tileBegin * plane + rowOffset;
        Real bottom = input[index - plane];
        Real center = input[index];

        for (std::size_t z = tileBegin; z <= tileEnd; ++z) {
            const Real top = input[index + plane];
            tile[threadIdx.y + 1][threadIdx.x + 1] = center;

            if (threadIdx.x == 0) {
                tile[threadIdx.y + 1][0] = input[index - 1];
            }
            if (threadIdx.x == kBlockX - 1) {
                const std::size_t rightX = min(rawX + 1, nx - 1);
                tile[threadIdx.y + 1][kBlockX + 1] =
                    input[z * plane + y * nx + rightX];
            }
            if (threadIdx.y == 0) {
                tile[0][threadIdx.x + 1] = input[index - nx];
            }
            if (threadIdx.y == kBlockY - 1) {
                const std::size_t backY = min(rawY + 1, ny - 1);
                tile[kBlockY + 1][threadIdx.x + 1] =
                    input[z * plane + backY * nx + x];
            }
            __syncthreads();

            if (active) {
                const Real left = tile[threadIdx.y + 1][threadIdx.x];
                const Real right = tile[threadIdx.y + 1][threadIdx.x + 2];
                const Real front = tile[threadIdx.y][threadIdx.x + 1];
                const Real back = tile[threadIdx.y + 2][threadIdx.x + 1];
                output[index] =
                    (center + left + right + front + back + bottom + top) / 7.0;
            }
            __syncthreads();

            bottom = center;
            center = top;
            index += plane;
        }
    }
}

void launchStencil(const Real* input, Real* output,
                   const std::size_t nx, const std::size_t ny,
                   const std::size_t firstZ, const std::size_t lastZ,
                   cudaStream_t stream) {
    if (firstZ > lastZ) {
        return;
    }

    const std::size_t zTiles = (lastZ - firstZ + kZTile) / kZTile;
    const unsigned int gridZ = static_cast<unsigned int>(
        std::min<std::size_t>(zTiles, 65535));
    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid(static_cast<unsigned int>((nx - 2 + kBlockX - 1) / kBlockX),
                    static_cast<unsigned int>((ny - 2 + kBlockY - 1) / kBlockY),
                    gridZ);
    stencilRange<<<grid, block, 0, stream>>>(input, output, nx, ny,
                                             firstZ, lastZ);
    CUDA_CHECK(cudaGetLastError());
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
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
    int threadSupport = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadSupport));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    if (threadSupport < MPI_THREAD_FUNNELED) {
        abortRun("MPI does not provide the required MPI_THREAD_FUNNELED support");
    }

    std::size_t nx = 128;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool parseOk = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parseOk = parseSize(argv[++i], nx) && parseOk;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parseOk = parseSize(argv[++i], ny) && parseOk;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parseOk = parseSize(argv[++i], nz) && parseOk;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parseOk = parseIterations(argv[++i], iterations) && parseOk;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
            parseOk = false;
        }
    }

    if (showHelp) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    std::size_t planeElements = 0;
    std::size_t globalElements = 0;
    if (!parseOk || nx < 3 || ny < 3 || nz < 3 ||
        !checkedMultiply(nx, ny, planeElements) ||
        !checkedMultiply(planeElements, nz, globalElements) ||
        globalElements > static_cast<std::size_t>(LLONG_MAX) ||
        planeElements > static_cast<std::size_t>(INT_MAX)) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "Grid dimensions must be integers >= 3, fit in memory, "
                         "and each x/y plane must fit in one MPI message.\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }
    if (static_cast<std::size_t>(worldSize) > nz) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "The number of MPI ranks (%d) cannot exceed nz (%zu).\n",
                         worldSize, nz);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                  worldRank, MPI_INFO_NULL, &localComm));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_size(localComm, &localSize));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortRun("no CUDA device is visible");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    if (localRank == 0 && localSize > deviceCount) {
        std::fprintf(stderr,
                     "Warning: %d local MPI ranks share %d CUDA device(s); "
                     "one rank per GPU is recommended.\n",
                     localSize, deviceCount);
    }

    // Open MPI exposes a runtime query for CUDA-aware buffer support.  Use
    // device buffers directly when every rank supports it; pinned staging is
    // the portable fallback for MPI implementations without that extension.
    int localCudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA)
    localCudaAware = MPIX_Query_cuda_support() != 0;
#endif
    int cudaAware = 0;
    MPI_CHECK(MPI_Allreduce(&localCudaAware, &cudaAware, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));

    omp_set_dynamic(0);
    const Slab slab = slabForRank(nz, worldRank, worldSize);
    std::size_t localElements = 0;
    std::size_t allocationElements = 0;
    if (!checkedMultiply(slab.planes, planeElements, localElements) ||
        !checkedMultiply(slab.planes + 2, planeElements, allocationElements) ||
        allocationElements > std::numeric_limits<std::size_t>::max() / sizeof(Real)) {
        abortRun("local grid allocation size overflow");
    }
    const std::size_t allocationBytes = allocationElements * sizeof(Real);
    const std::size_t localBytes = localElements * sizeof(Real);
    const std::size_t planeBytes = planeElements * sizeof(Real);

    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, CUDA GPU(s) (%s on rank 0)\n",
                    worldSize, omp_get_max_threads(), deviceProperties.name);
        std::printf("MPI halo transport: %s\n",
                    cudaAware ? "CUDA-aware device buffers" :
                                "portable pinned-host buffers");
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> hostLocal(localElements);
    const std::size_t globalOffset = slab.globalBegin * planeElements;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(localElements); ++i) {
        const std::size_t localIndex = static_cast<std::size_t>(i);
        hostLocal[localIndex] = static_cast<Real>((globalOffset + localIndex) % 19);
    }

    Real* deviceA = nullptr;
    Real* deviceB = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB), allocationBytes));
    CUDA_CHECK(cudaMemset(deviceA, 0, allocationBytes));
    CUDA_CHECK(cudaMemset(deviceB, 0, allocationBytes));
    CUDA_CHECK(cudaMemcpy(deviceA + planeElements, hostLocal.data(), localBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB + planeElements, hostLocal.data(), localBytes,
                          cudaMemcpyHostToDevice));

    Real* sendLower = nullptr;
    Real* sendUpper = nullptr;
    Real* recvLower = nullptr;
    Real* recvUpper = nullptr;
    if (!cudaAware) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendLower), planeBytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendUpper), planeBytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvLower), planeBytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvUpper), planeBytes));
    }

    cudaStream_t computeStream{};
    cudaStream_t communicationStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream,
                                         cudaStreamNonBlocking));
    cudaEvent_t haloReady{};
    cudaEvent_t iterationDone[2]{};
    CUDA_CHECK(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&iterationDone[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&iterationDone[1], cudaEventDisableTiming));

    const int lowerRank = worldRank == 0 ? MPI_PROC_NULL : worldRank - 1;
    const int upperRank = worldRank + 1 == worldSize ? MPI_PROC_NULL : worldRank + 1;
    const int planeCount = static_cast<int>(planeElements);

    // Convert global interior z bounds to local indices.  Local owned planes
    // are numbered [1, slab.planes], with receive halos at 0 and planes + 1.
    const std::size_t validBegin = std::max<std::size_t>(
        1, slab.globalBegin == 0 ? 2 : 1);
    const std::size_t validEnd = std::min<std::size_t>(
        slab.planes, nz - 1 - slab.globalBegin);
    const std::size_t independentBegin = std::max<std::size_t>(validBegin, 2);
    const std::size_t independentEnd = slab.planes > 1
        ? std::min<std::size_t>(validEnd, slab.planes - 1) : 0;

    if (worldRank == 0) {
        std::printf("Running stencil computation...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    Real* input = deviceA;
    Real* output = deviceB;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        MPI_Request requests[4]{};
        Real* const lowerReceive = cudaAware ? input : recvLower;
        Real* const upperReceive = cudaAware
            ? input + (slab.planes + 1) * planeElements : recvUpper;
        MPI_CHECK(MPI_Irecv(lowerReceive, planeCount, MPI_DOUBLE, lowerRank,
                            kUpperTag, MPI_COMM_WORLD, &requests[0]));
        MPI_CHECK(MPI_Irecv(upperReceive, planeCount, MPI_DOUBLE, upperRank,
                            kLowerTag, MPI_COMM_WORLD, &requests[1]));

        // These planes depend only on locally owned data and run while halo
        // copies and MPI transfers progress.
        if (independentBegin <= independentEnd) {
            launchStencil(input, output, nx, ny, independentBegin,
                          independentEnd, computeStream);
        }

        if (cudaAware) {
            // The previous edge kernels produced the outgoing device faces.
            // Interior work for this iteration has already been queued and can
            // overlap the MPI transfer once that event becomes ready.
            if (iteration != 0) {
                CUDA_CHECK(cudaEventSynchronize(
                    iterationDone[(iteration - 1) & 1]));
            }
            MPI_CHECK(MPI_Isend(input + planeElements, planeCount, MPI_DOUBLE,
                                lowerRank, kLowerTag, MPI_COMM_WORLD,
                                &requests[2]));
            MPI_CHECK(MPI_Isend(input + slab.planes * planeElements,
                                planeCount, MPI_DOUBLE, upperRank, kUpperTag,
                                MPI_COMM_WORLD, &requests[3]));
            MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
        } else {
            if (iteration != 0) {
                CUDA_CHECK(cudaStreamWaitEvent(
                    communicationStream, iterationDone[(iteration - 1) & 1], 0));
            }
            if (lowerRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sendLower, input + planeElements,
                                           planeBytes, cudaMemcpyDeviceToHost,
                                           communicationStream));
            }
            if (upperRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sendUpper,
                                           input + slab.planes * planeElements,
                                           planeBytes, cudaMemcpyDeviceToHost,
                                           communicationStream));
            }
            CUDA_CHECK(cudaStreamSynchronize(communicationStream));
            MPI_CHECK(MPI_Isend(sendLower, planeCount, MPI_DOUBLE, lowerRank,
                                kLowerTag, MPI_COMM_WORLD, &requests[2]));
            MPI_CHECK(MPI_Isend(sendUpper, planeCount, MPI_DOUBLE, upperRank,
                                kUpperTag, MPI_COMM_WORLD, &requests[3]));
            MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));

            if (lowerRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(input, recvLower, planeBytes,
                                           cudaMemcpyHostToDevice,
                                           communicationStream));
            }
            if (upperRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(
                    input + (slab.planes + 1) * planeElements, recvUpper,
                    planeBytes, cudaMemcpyHostToDevice, communicationStream));
            }
            CUDA_CHECK(cudaEventRecord(haloReady, communicationStream));
            CUDA_CHECK(cudaStreamWaitEvent(computeStream, haloReady, 0));
        }

        // The first and last owned planes consume the newly received halos.
        if (validBegin == 1 && validBegin <= validEnd) {
            launchStencil(input, output, nx, ny, 1, 1, computeStream);
        }
        if (validEnd == slab.planes && slab.planes != 1 &&
            validBegin <= validEnd) {
            launchStencil(input, output, nx, ny, slab.planes, slab.planes,
                          computeStream);
        }
        CUDA_CHECK(cudaEventRecord(iterationDone[iteration & 1], computeStream));
        std::swap(input, output);
    }

    if (iterations > 0) {
        CUDA_CHECK(cudaEventSynchronize(iterationDone[(iterations - 1) & 1]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (worldRank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double updates = static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
        const double mcups = elapsed > 0.0 ? updates / elapsed / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(hostLocal.data(), input + planeElements,
                              localBytes, cudaMemcpyDeviceToHost));
    }

    std::vector<Real> finalGrid;
    if (printResults) {
        if (globalElements > static_cast<std::size_t>(INT_MAX)) {
            abortRun("global result is too large for MPI_Gatherv");
        }
        std::vector<int> counts;
        std::vector<int> displacements;
        if (worldRank == 0) {
            finalGrid.resize(globalElements);
            counts.resize(worldSize);
            displacements.resize(worldSize);
            for (int rank = 0; rank < worldSize; ++rank) {
                const Slab rankSlab = slabForRank(nz, rank, worldSize);
                counts[rank] = static_cast<int>(rankSlab.planes * planeElements);
                displacements[rank] =
                    static_cast<int>(rankSlab.globalBegin * planeElements);
            }
        }
        MPI_CHECK(MPI_Gatherv(hostLocal.data(), static_cast<int>(localElements),
                              MPI_DOUBLE, finalGrid.data(), counts.data(),
                              displacements.data(), MPI_DOUBLE, 0,
                              MPI_COMM_WORLD));
        if (worldRank == 0) {
            print_results(finalGrid, "Grid");
        }
    }

    int returnCode = 0;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int localInvalid = 0;
#pragma omp parallel for reduction(min:localMin) reduction(max:localMax) \
                         reduction(|:localInvalid) schedule(static)
        for (long long i = 0; i < static_cast<long long>(localElements); ++i) {
            const Real value = hostLocal[static_cast<std::size_t>(i)];
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
            localInvalid |= !std::isfinite(value);
        }

        Real globalMin = 0.0;
        Real globalMax = 0.0;
        int globalInvalid = 0;
        MPI_CHECK(MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0,
                             MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0,
                             MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX,
                             0, MPI_COMM_WORLD));
        if (worldRank == 0) {
            std::printf("Validating result...\n");
            if (globalInvalid != 0) {
                std::printf("Validation failed: found NaN or Inf value\n");
            }
            std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            const bool valid = globalInvalid == 0 && globalMax <= 1.0e6 &&
                               globalMin >= -1.0e6;
            if (!valid && globalInvalid == 0) {
                std::printf("Validation failed: values out of expected range\n");
            }
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? 0 : 1;
        }
        MPI_CHECK(MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    CUDA_CHECK(cudaEventDestroy(iterationDone[1]));
    CUDA_CHECK(cudaEventDestroy(iterationDone[0]));
    CUDA_CHECK(cudaEventDestroy(haloReady));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    if (!cudaAware) {
        CUDA_CHECK(cudaFreeHost(recvUpper));
        CUDA_CHECK(cudaFreeHost(recvLower));
        CUDA_CHECK(cudaFreeHost(sendUpper));
        CUDA_CHECK(cudaFreeHost(sendLower));
    }
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceA));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());
    return returnCode;
}
