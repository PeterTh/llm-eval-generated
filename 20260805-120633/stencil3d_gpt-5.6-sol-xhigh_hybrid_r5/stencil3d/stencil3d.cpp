#include <cuda_runtime.h>
#include <mpi.h>

#if defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi-ext.h>
#    define STENCIL3D_HAS_MPI_EXT 1
#  endif
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
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kHaloFromLowerTag = 1101;
constexpr int kHaloFromUpperTag = 1102;
constexpr int kResultTag = 1201;

[[noreturn]] void abortRun(const char* message, const char* file, int line) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    int rank = 0;
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "Rank %d: %s (%s:%d)\n", rank, message, file, line);
    std::fflush(stderr);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        char message[1024];
        std::snprintf(message, sizeof(message), "CUDA call '%s' failed: %s", expression,
                      cudaGetErrorString(status));
        abortRun(message, file, line);
    }
}

void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(status, error, &length);
        char message[1024];
        std::snprintf(message, sizeof(message), "MPI call '%s' failed: %.*s", expression,
                      length, error);
        abortRun(message, file, line);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

bool checkedMultiply(size_t lhs, size_t rhs, size_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool parseSize(const char* text, size_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    if (text == nullptr || text[0] == '\0') {
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
    std::printf("Environment:\n");
    std::printf("  STENCIL3D_CUDA_AWARE_MPI=0|1  Override automatic CUDA-aware MPI detection\n");
}

struct Slab {
    size_t globalZBegin;
    size_t planes;
};

Slab slabForRank(size_t nz, int ranks, int rank) {
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t quotient = nz / rankCount;
    const size_t remainder = nz % rankCount;
    return {rankIndex * quotient + std::min(rankIndex, remainder),
            quotient + (rankIndex < remainder ? 1U : 0U)};
}

void initializeLocalGrid(std::vector<Real>& grid, size_t planeElements,
                         size_t globalZBegin, size_t localPlanes) {
    const long long count = static_cast<long long>(localPlanes);
#pragma omp parallel for schedule(static)
    for (long long plane = 0; plane < count; ++plane) {
        const size_t localZ = static_cast<size_t>(plane) + 1;
        const size_t globalZ = globalZBegin + static_cast<size_t>(plane);
        const size_t localOffset = localZ * planeElements;
        const size_t globalOffset = globalZ * planeElements;
        for (size_t cell = 0; cell < planeElements; ++cell) {
            grid[localOffset + cell] = static_cast<Real>((globalOffset + cell) % 19U);
        }
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t globalNz,
                              size_t globalZBegin, size_t localZBegin,
                              size_t localZEnd) {
    __shared__ Real tile[kBlockY + 2][kBlockX + 2];

    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y + 1;
    const size_t localZ = localZBegin + blockIdx.z;
    const bool valid = x + 1 < nx && y + 1 < ny && localZ <= localZEnd;
    const size_t planeElements = nx * ny;

    size_t index = 0;
    if (valid) {
        index = localZ * planeElements + y * nx + x;
        tile[threadIdx.y + 1][threadIdx.x + 1] = input[index];

        if (threadIdx.x == 0) {
            tile[threadIdx.y + 1][0] = input[index - 1];
        }
        if (threadIdx.x == kBlockX - 1 || x == nx - 2) {
            tile[threadIdx.y + 1][threadIdx.x + 2] = input[index + 1];
        }
        if (threadIdx.y == 0) {
            tile[0][threadIdx.x + 1] = input[index - nx];
        }
        if (threadIdx.y == kBlockY - 1 || y == ny - 2) {
            tile[threadIdx.y + 2][threadIdx.x + 1] = input[index + nx];
        }
    }
    __syncthreads();

    if (!valid) {
        return;
    }

    const size_t globalZ = globalZBegin + localZ - 1;
    if (globalZ == 0 || globalZ + 1 == globalNz) {
        return;
    }

    Real sum = tile[threadIdx.y + 1][threadIdx.x + 1];
    sum += tile[threadIdx.y + 1][threadIdx.x];
    sum += tile[threadIdx.y + 1][threadIdx.x + 2];
    sum += tile[threadIdx.y][threadIdx.x + 1];
    sum += tile[threadIdx.y + 2][threadIdx.x + 1];
    sum += input[index - planeElements];
    sum += input[index + planeElements];
    output[index] = sum / Real{7.0};
}

void launchStencilRange(const Real* input, Real* output, size_t nx, size_t ny,
                        size_t globalNz, size_t globalZBegin, size_t localZBegin,
                        size_t localZEnd, cudaStream_t stream) {
    if (localZBegin > localZEnd || nx <= 2 || ny <= 2) {
        return;
    }

    const dim3 block(kBlockX, kBlockY, 1);
    const unsigned int gridX = static_cast<unsigned int>((nx - 2 + kBlockX - 1) / kBlockX);
    const unsigned int gridY = static_cast<unsigned int>((ny - 2 + kBlockY - 1) / kBlockY);
    constexpr size_t maxGridZ = 65535;

    for (size_t first = localZBegin; first <= localZEnd;) {
        const size_t planes = std::min(maxGridZ, localZEnd - first + 1);
        const dim3 grid(gridX, gridY, static_cast<unsigned int>(planes));
        stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, globalNz,
                                                  globalZBegin, first,
                                                  first + planes - 1);
        CUDA_CHECK(cudaGetLastError());
        first += planes;
    }
}

bool environmentFlag(const char* value, bool& parsedValue) {
    if (value == nullptr) {
        return false;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0) {
        parsedValue = true;
        return true;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "FALSE") == 0 || std::strcmp(value, "off") == 0 ||
        std::strcmp(value, "OFF") == 0) {
        parsedValue = false;
        return true;
    }
    return false;
}

bool detectCudaAwareMpi(int worldRank) {
    bool enabled = false;
#if defined(STENCIL3D_HAS_MPI_EXT) && defined(MPIX_CUDA_AWARE_SUPPORT)
    enabled = MPIX_Query_cuda_support() != 0;
#endif

    const char* overrideValue = std::getenv("STENCIL3D_CUDA_AWARE_MPI");
    if (overrideValue != nullptr) {
        bool requested = false;
        if (environmentFlag(overrideValue, requested)) {
            enabled = requested;
        } else if (worldRank == 0) {
            std::fprintf(stderr,
                         "Warning: ignoring invalid STENCIL3D_CUDA_AWARE_MPI='%s'\n",
                         overrideValue);
        }
    }

    int local = enabled ? 1 : 0;
    int allEnabled = 0;
    MPI_CHECK(MPI_Allreduce(&local, &allEnabled, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    return allEnabled != 0;
}

void appendReceives(std::vector<MPI_Request>& requests, Real* buffer, size_t elements,
                    int peer, int tag) {
    if (peer == MPI_PROC_NULL) {
        return;
    }
    for (size_t offset = 0; offset < elements;) {
        const int count = static_cast<int>(std::min(elements - offset,
                                                    static_cast<size_t>(INT_MAX)));
        requests.emplace_back(MPI_REQUEST_NULL);
        MPI_CHECK(MPI_Irecv(buffer + offset, count, MPI_DOUBLE, peer, tag,
                            MPI_COMM_WORLD, &requests.back()));
        offset += static_cast<size_t>(count);
    }
}

void appendSends(std::vector<MPI_Request>& requests, const Real* buffer, size_t elements,
                 int peer, int tag) {
    if (peer == MPI_PROC_NULL) {
        return;
    }
    for (size_t offset = 0; offset < elements;) {
        const int count = static_cast<int>(std::min(elements - offset,
                                                    static_cast<size_t>(INT_MAX)));
        requests.emplace_back(MPI_REQUEST_NULL);
        MPI_CHECK(MPI_Isend(buffer + offset, count, MPI_DOUBLE, peer, tag,
                            MPI_COMM_WORLD, &requests.back()));
        offset += static_cast<size_t>(count);
    }
}

void postHaloExchange(std::vector<MPI_Request>& requests,
                      Real* receiveLower, Real* receiveUpper,
                      const Real* sendLower, const Real* sendUpper,
                      size_t planeElements, int lowerRank, int upperRank) {
    requests.clear();
    appendReceives(requests, receiveLower, planeElements, lowerRank,
                   kHaloFromLowerTag);
    appendReceives(requests, receiveUpper, planeElements, upperRank,
                   kHaloFromUpperTag);
    appendSends(requests, sendUpper, planeElements, upperRank,
                kHaloFromLowerTag);
    appendSends(requests, sendLower, planeElements, lowerRank,
                kHaloFromUpperTag);
}

void waitForRequests(std::vector<MPI_Request>& requests) {
    if (requests.empty()) {
        return;
    }
    if (requests.size() > static_cast<size_t>(INT_MAX)) {
        abortRun("too many MPI requests", __FILE__, __LINE__);
    }
    MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                          MPI_STATUSES_IGNORE));
}

bool validateDistributed(const std::vector<Real>& localResult, int worldRank) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localInvalid = 0;
    const long long elements = static_cast<long long>(localResult.size());

#pragma omp parallel for schedule(static) reduction(min : localMin) reduction(max : localMax) reduction(| : localInvalid)
    for (long long i = 0; i < elements; ++i) {
        const Real value = localResult[static_cast<size_t>(i)];
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
        localInvalid |= !std::isfinite(value);
    }

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    int globalInvalid = 0;
    MPI_CHECK(MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN,
                            MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX,
                            MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX,
                            MPI_COMM_WORLD));

    if (worldRank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    return globalInvalid == 0 && globalMax <= 1e6 && globalMin >= -1e6;
}

void sendInChunks(const Real* data, size_t elements, int destination, int tag) {
    for (size_t offset = 0; offset < elements;) {
        const int count = static_cast<int>(std::min(elements - offset,
                                                    static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Send(data + offset, count, MPI_DOUBLE, destination, tag,
                           MPI_COMM_WORLD));
        offset += static_cast<size_t>(count);
    }
}

void receiveInChunks(Real* data, size_t elements, int source, int tag) {
    for (size_t offset = 0; offset < elements;) {
        const int count = static_cast<int>(std::min(elements - offset,
                                                    static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Recv(data + offset, count, MPI_DOUBLE, source, tag,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += static_cast<size_t>(count);
    }
}

std::vector<Real> gatherResult(const std::vector<Real>& localResult, size_t nx,
                               size_t ny, size_t nz, int worldRank, int worldSize) {
    size_t planeElements = 0;
    size_t globalElements = 0;
    if (!checkedMultiply(nx, ny, planeElements) ||
        !checkedMultiply(planeElements, nz, globalElements)) {
        abortRun("global result size overflow", __FILE__, __LINE__);
    }

    if (worldRank != 0) {
        sendInChunks(localResult.data(), localResult.size(), 0, kResultTag);
        return {};
    }

    std::vector<Real> result(globalElements);
    const Slab rootSlab = slabForRank(nz, worldSize, 0);
    std::memcpy(result.data() + rootSlab.globalZBegin * planeElements,
                localResult.data(), localResult.size() * sizeof(Real));

    for (int rank = 1; rank < worldSize; ++rank) {
        const Slab slab = slabForRank(nz, worldSize, rank);
        receiveInChunks(result.data() + slab.globalZBegin * planeElements,
                        slab.planes * planeElements, rank, kResultTag);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initStatus = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initStatus != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int worldRank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) {
        abortRun("MPI implementation does not provide MPI_THREAD_FUNNELED", __FILE__,
                 __LINE__);
    }

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

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
            showHelp = true;
        } else {
            argumentsValid = false;
            if (worldRank == 0) {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    size_t planeElements = 0;
    size_t globalElements = 0;
    argumentsValid &= nx >= 2 && ny >= 2 && nz >= 2;
    argumentsValid &= static_cast<size_t>(worldSize) <= nz;
    argumentsValid &= checkedMultiply(nx, ny, planeElements);
    argumentsValid &= argumentsValid && checkedMultiply(planeElements, nz, globalElements);
    argumentsValid &= nz <= static_cast<size_t>(LLONG_MAX);

    if (!argumentsValid) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "Invalid configuration: dimensions must be at least 2, sizes must "
                         "not overflow, and MPI ranks must not exceed Z planes.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &nodeComm));
    int nodeRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &nodeRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortRun("no CUDA device is visible", __FILE__, __LINE__);
    }
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    const size_t gridX = (nx - 2 + kBlockX - 1) / kBlockX;
    const size_t gridY = (ny - 2 + kBlockY - 1) / kBlockY;
    if (gridX > static_cast<size_t>(deviceProperties.maxGridSize[0]) ||
        gridY > static_cast<size_t>(deviceProperties.maxGridSize[1])) {
        abortRun("X/Y dimensions exceed CUDA launch limits", __FILE__, __LINE__);
    }

    const bool cudaAwareMpi = detectCudaAwareMpi(worldRank);
    const Slab localSlab = slabForRank(nz, worldSize, worldRank);
    size_t allocationPlanes = localSlab.planes + 2;
    size_t allocationElements = 0;
    size_t allocationBytes = 0;
    size_t planeBytes = 0;
    if (!checkedMultiply(allocationPlanes, planeElements, allocationElements) ||
        !checkedMultiply(allocationElements, sizeof(Real), allocationBytes) ||
        !checkedMultiply(planeElements, sizeof(Real), planeBytes) ||
        localSlab.planes > static_cast<size_t>(LLONG_MAX)) {
        abortRun("local allocation size overflow", __FILE__, __LINE__);
    }

    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelization: %d MPI rank%s, up to %d OpenMP thread%s/rank, CUDA\n",
                    worldSize, worldSize == 1 ? "" : "s", omp_get_max_threads(),
                    omp_get_max_threads() == 1 ? "" : "s");
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        std::printf("Halo transport: %s\n",
                    worldSize == 1
                        ? "none (single MPI rank)"
                        : (cudaAwareMpi ? "CUDA-aware MPI device buffers"
                                        : "MPI with pinned-host staging"));
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> initial(allocationElements, Real{0.0});
    initializeLocalGrid(initial, planeElements, localSlab.globalZBegin,
                        localSlab.planes);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), allocationBytes));
    CUDA_CHECK(cudaMemcpy(deviceGrid1, initial.data(), allocationBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGrid2, initial.data(), allocationBytes,
                          cudaMemcpyHostToDevice));
    std::vector<Real>().swap(initial);

    Real* pinnedBase = nullptr;
    Real* sendLowerHost = nullptr;
    Real* sendUpperHost = nullptr;
    Real* receiveLowerHost = nullptr;
    Real* receiveUpperHost = nullptr;
    if (!cudaAwareMpi && worldSize > 1) {
        size_t stagingElements = 0;
        size_t stagingBytes = 0;
        if (!checkedMultiply(planeElements, size_t{4}, stagingElements) ||
            !checkedMultiply(stagingElements, sizeof(Real), stagingBytes)) {
            abortRun("halo staging size overflow", __FILE__, __LINE__);
        }
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinnedBase), stagingBytes));
        sendLowerHost = pinnedBase;
        sendUpperHost = pinnedBase + planeElements;
        receiveLowerHost = pinnedBase + 2 * planeElements;
        receiveUpperHost = pinnedBase + 3 * planeElements;
    }

    cudaStream_t interiorStream = nullptr;
    cudaStream_t boundaryStream = nullptr;
    cudaStream_t transferStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&interiorStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&boundaryStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking));
    cudaEvent_t interiorDone[2] = {nullptr, nullptr};
    if (worldSize > 1) {
        CUDA_CHECK(cudaEventCreateWithFlags(&interiorDone[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&interiorDone[1], cudaEventDisableTiming));
    }

    const int lowerRank = worldRank == 0 ? MPI_PROC_NULL : worldRank - 1;
    const int upperRank = worldRank + 1 == worldSize ? MPI_PROC_NULL : worldRank + 1;
    const size_t chunkCount = (planeElements + static_cast<size_t>(INT_MAX) - 1) /
                              static_cast<size_t>(INT_MAX);
    std::vector<MPI_Request> requests;
    requests.reserve(4 * chunkCount);

    Real* input = deviceGrid1;
    Real* output = deviceGrid2;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (worldRank == 0) {
        std::printf("Running stencil computation...\n");
    }
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (worldSize == 1) {
            // A single rank has no halos: one stream supplies all temporal
            // dependencies without a host synchronization per iteration.
            launchStencilRange(input, output, nx, ny, nz, localSlab.globalZBegin,
                               2, localSlab.planes - 1, interiorStream);
            std::swap(input, output);
            continue;
        }

        // These edge planes are produced on boundaryStream.  The middle from the
        // preceding iteration may still finish while halo transfer is initiated.
        CUDA_CHECK(cudaStreamSynchronize(boundaryStream));

        Real* receiveLower = input;
        Real* receiveUpper = input + (localSlab.planes + 1) * planeElements;
        const Real* sendLower = input + planeElements;
        const Real* sendUpper = input + localSlab.planes * planeElements;

        if (!cudaAwareMpi) {
            if (lowerRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sendLowerHost, sendLower, planeBytes,
                                           cudaMemcpyDeviceToHost, transferStream));
            }
            if (upperRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sendUpperHost, sendUpper, planeBytes,
                                           cudaMemcpyDeviceToHost, transferStream));
            }
            CUDA_CHECK(cudaStreamSynchronize(transferStream));
            receiveLower = receiveLowerHost;
            receiveUpper = receiveUpperHost;
            sendLower = sendLowerHost;
            sendUpper = sendUpperHost;
        }

        postHaloExchange(requests, receiveLower, receiveUpper, sendLower, sendUpper,
                         planeElements, lowerRank, upperRank);

        if (localSlab.planes > 2) {
            launchStencilRange(input, output, nx, ny, nz, localSlab.globalZBegin,
                               2, localSlab.planes - 1, interiorStream);
        }
        CUDA_CHECK(cudaEventRecord(interiorDone[iteration & 1], interiorStream));

        waitForRequests(requests);

        if (!cudaAwareMpi) {
            if (lowerRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(input, receiveLowerHost, planeBytes,
                                           cudaMemcpyHostToDevice, boundaryStream));
            }
            if (upperRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(
                    input + (localSlab.planes + 1) * planeElements,
                    receiveUpperHost, planeBytes, cudaMemcpyHostToDevice,
                                           boundaryStream));
            }
        }

        // An edge plane reads the neighboring local plane.  Preserve that
        // dependency on the preceding iteration on-device, allowing the CPU to
        // drive MPI while the preceding and current middle kernels stay queued.
        if (iteration != 0 && localSlab.planes > 2) {
            CUDA_CHECK(cudaStreamWaitEvent(boundaryStream,
                                           interiorDone[(iteration - 1) & 1], 0));
        }

        const bool firstPlaneIsInterior = localSlab.globalZBegin != 0;
        const bool lastPlaneIsInterior =
            localSlab.globalZBegin + localSlab.planes != nz;
        if (localSlab.planes <= 2) {
            const size_t first = firstPlaneIsInterior ? 1 : 2;
            const size_t last = lastPlaneIsInterior
                                    ? localSlab.planes
                                    : localSlab.planes - 1;
            launchStencilRange(input, output, nx, ny, nz, localSlab.globalZBegin,
                               first, last, boundaryStream);
        } else {
            if (firstPlaneIsInterior) {
                launchStencilRange(input, output, nx, ny, nz,
                                   localSlab.globalZBegin, 1, 1,
                                   boundaryStream);
            }
            if (lastPlaneIsInterior) {
                launchStencilRange(input, output, nx, ny, nz,
                                   localSlab.globalZBegin, localSlab.planes,
                                   localSlab.planes, boundaryStream);
            }
        }

        std::swap(input, output);
    }

    CUDA_CHECK(cudaStreamSynchronize(interiorStream));
    CUDA_CHECK(cudaStreamSynchronize(boundaryStream));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (worldRank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double cellUpdates = static_cast<double>(nx - 2) *
                                   static_cast<double>(ny - 2) *
                                   static_cast<double>(nz - 2) *
                                   static_cast<double>(iterations);
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> localResult;
    if (printResults || validate) {
        size_t localResultElements = 0;
        if (!checkedMultiply(localSlab.planes, planeElements, localResultElements) ||
            localResultElements > static_cast<size_t>(LLONG_MAX)) {
            abortRun("local result size overflow", __FILE__, __LINE__);
        }
        localResult.resize(localResultElements);
        CUDA_CHECK(cudaMemcpy(localResult.data(), input + planeElements,
                              localResultElements * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        std::vector<Real> globalResult = gatherResult(localResult, nx, ny, nz,
                                                       worldRank, worldSize);
        if (worldRank == 0) {
            print_results(globalResult, "Grid");
        }
    }

    bool valid = true;
    if (validate) {
        if (worldRank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateDistributed(localResult, worldRank);
        if (worldRank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaStreamDestroy(transferStream));
    CUDA_CHECK(cudaStreamDestroy(boundaryStream));
    CUDA_CHECK(cudaStreamDestroy(interiorStream));
    if (interiorDone[0] != nullptr) {
        CUDA_CHECK(cudaEventDestroy(interiorDone[1]));
        CUDA_CHECK(cudaEventDestroy(interiorDone[0]));
    }
    if (pinnedBase != nullptr) {
        CUDA_CHECK(cudaFreeHost(pinnedBase));
    }
    CUDA_CHECK(cudaFree(deviceGrid2));
    CUDA_CHECK(cudaFree(deviceGrid1));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    MPI_CHECK(MPI_Finalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
