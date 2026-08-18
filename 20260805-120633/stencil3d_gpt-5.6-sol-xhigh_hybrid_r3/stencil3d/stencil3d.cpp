#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kTileX = 32;
constexpr int kTileY = 8;
constexpr int kTagToLower = 101;
constexpr int kTagToUpper = 102;

int worldRank = 0;

[[noreturn]] void abortBenchmark(const char* message) {
    if (worldRank == 0) {
        std::fprintf(stderr, "Error: %s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d while executing %s: %s\n",
                     worldRank, file, line, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

bool checkedMultiply(size_t lhs, size_t rhs, size_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool parseSize(const char* text, size_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
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

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
};

bool parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nx)) return false;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.ny)) return false;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nz)) return false;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIterations(argv[++i], options.iterations)) return false;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            if (worldRank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return false;
        }
    }

    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    return true;
}

struct Slab {
    int localNz;
    int globalZBegin;
};

Slab decomposeZ(int globalNz, int rank, int ranks) {
    const int base = globalNz / ranks;
    const int remainder = globalNz % ranks;
    return {base + (rank < remainder ? 1 : 0),
            rank * base + std::min(rank, remainder)};
}

// ENDPOINTS=false maps block rows to a contiguous range of local Z planes.
// ENDPOINTS=true maps them to the first and last owned planes.  The latter lets
// the two halo-dependent planes share one launch.
template <bool ENDPOINTS>
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              int nx, int ny, int globalNz,
                              int localNz, int globalZBegin,
                              int localZBegin, int zPlaneCount,
                              int yTileCount) {
    __shared__ Real tile[kTileY + 2][kTileX + 2];

    const unsigned int yAndZTile = blockIdx.y;
    const int zOrdinal = static_cast<int>(yAndZTile / static_cast<unsigned int>(yTileCount));
    if (zOrdinal >= zPlaneCount) return;

    const int yTile = static_cast<int>(yAndZTile % static_cast<unsigned int>(yTileCount));
    const int localZ = ENDPOINTS
        ? ((zOrdinal == 0) ? 1 : localNz)
        : localZBegin + zOrdinal;
    const int x = 1 + static_cast<int>(blockIdx.x) * kTileX + static_cast<int>(threadIdx.x);
    const int y = 1 + yTile * kTileY + static_cast<int>(threadIdx.y);
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const size_t planeElements = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t planeOffset = static_cast<size_t>(localZ) * planeElements;

    const bool inPlane = x < nx && y < ny;
    tile[ty + 1][tx + 1] = inPlane ? input[planeOffset + static_cast<size_t>(y) * nx + x] : Real{0};

    if (tx == 0) {
        tile[ty + 1][0] = inPlane ? input[planeOffset + static_cast<size_t>(y) * nx + x - 1] : Real{0};
    }
    if (tx == kTileX - 1) {
        tile[ty + 1][kTileX + 1] = (x < nx - 1 && y < ny)
            ? input[planeOffset + static_cast<size_t>(y) * nx + x + 1] : Real{0};
    }
    if (ty == 0) {
        tile[0][tx + 1] = inPlane ? input[planeOffset + static_cast<size_t>(y - 1) * nx + x] : Real{0};
    }
    if (ty == kTileY - 1) {
        tile[kTileY + 1][tx + 1] = (x < nx && y < ny - 1)
            ? input[planeOffset + static_cast<size_t>(y + 1) * nx + x] : Real{0};
    }
    __syncthreads();

    const int globalZ = globalZBegin + localZ - 1;
    if (x >= nx - 1 || y >= ny - 1 || globalZ == 0 || globalZ == globalNz - 1) {
        return;
    }

    const size_t offset = planeOffset + static_cast<size_t>(y) * nx + x;
    const Real center = tile[ty + 1][tx + 1];
    const Real left = tile[ty + 1][tx];
    const Real right = tile[ty + 1][tx + 2];
    const Real front = tile[ty][tx + 1];
    const Real back = tile[ty + 2][tx + 1];
    const Real bottom = input[offset - planeElements];
    const Real top = input[offset + planeElements];
    output[offset] = (center + left + right + front + back + bottom + top) / Real{7};
}

template <bool ENDPOINTS>
void launchStencil(const Real* input, Real* output,
                   int nx, int ny, int globalNz, int localNz, int globalZBegin,
                   int localZBegin, int zPlaneCount, cudaStream_t stream) {
    if (zPlaneCount <= 0) return;
    const int xTileCount = (nx - 2 + kTileX - 1) / kTileX;
    const int yTileCount = (ny - 2 + kTileY - 1) / kTileY;
    const size_t combinedTiles = static_cast<size_t>(yTileCount) * static_cast<size_t>(zPlaneCount);
    if (combinedTiles > static_cast<size_t>(INT_MAX)) {
        abortBenchmark("CUDA launch grid is too large");
    }
    const dim3 block(kTileX, kTileY, 1);
    const dim3 grid(static_cast<unsigned int>(xTileCount),
                    static_cast<unsigned int>(combinedTiles), 1);
    stencilKernel<ENDPOINTS><<<grid, block, 0, stream>>>(
        input, output, nx, ny, globalNz, localNz, globalZBegin,
        localZBegin, zPlaneCount, yTileCount);
    CUDA_CHECK(cudaGetLastError());
}

bool queryCudaAwareMpi() {
    const char* overrideValue = std::getenv("STENCIL3D_GPU_AWARE_MPI");
    if (overrideValue != nullptr) {
        if (std::strcmp(overrideValue, "1") == 0 || std::strcmp(overrideValue, "on") == 0 ||
            std::strcmp(overrideValue, "ON") == 0 || std::strcmp(overrideValue, "true") == 0) {
            return true;
        }
        if (std::strcmp(overrideValue, "0") == 0 || std::strcmp(overrideValue, "off") == 0 ||
            std::strcmp(overrideValue, "OFF") == 0 || std::strcmp(overrideValue, "false") == 0) {
            return false;
        }
    }
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

class HaloExchange {
  public:
    HaloExchange(size_t planeElements, int lowerRank, int upperRank, bool cudaAware)
        : planeElements_(planeElements), planeBytes_(planeElements * sizeof(Real)),
          lowerRank_(lowerRank), upperRank_(upperRank), cudaAware_(cudaAware) {
        if (!cudaAware_) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostBuffers_), 4 * planeBytes_));
            sendLower_ = hostBuffers_;
            sendUpper_ = hostBuffers_ + planeElements_;
            receiveLower_ = hostBuffers_ + 2 * planeElements_;
            receiveUpper_ = hostBuffers_ + 3 * planeElements_;
        }
    }

    ~HaloExchange() {
        if (hostBuffers_ != nullptr) cudaFreeHost(hostBuffers_);
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    void begin(const Real* input, int localNz, cudaStream_t communicationStream) {
        const Real* firstOwned = input + planeElements_;
        const Real* lastOwned = input + static_cast<size_t>(localNz) * planeElements_;
        Real* lowerGhost = const_cast<Real*>(input);
        Real* upperGhost = const_cast<Real*>(input) + static_cast<size_t>(localNz + 1) * planeElements_;

        if (cudaAware_) {
            MPI_Irecv(lowerGhost, static_cast<int>(planeElements_), MPI_DOUBLE, lowerRank_,
                      kTagToUpper, MPI_COMM_WORLD, &requests_[0]);
            MPI_Irecv(upperGhost, static_cast<int>(planeElements_), MPI_DOUBLE, upperRank_,
                      kTagToLower, MPI_COMM_WORLD, &requests_[1]);
            MPI_Isend(firstOwned, static_cast<int>(planeElements_), MPI_DOUBLE, lowerRank_,
                      kTagToLower, MPI_COMM_WORLD, &requests_[2]);
            MPI_Isend(lastOwned, static_cast<int>(planeElements_), MPI_DOUBLE, upperRank_,
                      kTagToUpper, MPI_COMM_WORLD, &requests_[3]);
            return;
        }

        MPI_Irecv(receiveLower_, static_cast<int>(planeElements_), MPI_DOUBLE, lowerRank_,
                  kTagToUpper, MPI_COMM_WORLD, &requests_[0]);
        MPI_Irecv(receiveUpper_, static_cast<int>(planeElements_), MPI_DOUBLE, upperRank_,
                  kTagToLower, MPI_COMM_WORLD, &requests_[1]);
        if (lowerRank_ != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendLower_, firstOwned, planeBytes_, cudaMemcpyDeviceToHost,
                                       communicationStream));
        }
        if (upperRank_ != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendUpper_, lastOwned, planeBytes_, cudaMemcpyDeviceToHost,
                                       communicationStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
        MPI_Isend(sendLower_, static_cast<int>(planeElements_), MPI_DOUBLE, lowerRank_,
                  kTagToLower, MPI_COMM_WORLD, &requests_[2]);
        MPI_Isend(sendUpper_, static_cast<int>(planeElements_), MPI_DOUBLE, upperRank_,
                  kTagToUpper, MPI_COMM_WORLD, &requests_[3]);
    }

    void finish(Real* input, int localNz, cudaStream_t communicationStream) {
        MPI_Waitall(4, requests_, MPI_STATUSES_IGNORE);
        if (cudaAware_) return;

        Real* lowerGhost = input;
        Real* upperGhost = input + static_cast<size_t>(localNz + 1) * planeElements_;
        if (lowerRank_ != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(lowerGhost, receiveLower_, planeBytes_, cudaMemcpyHostToDevice,
                                       communicationStream));
        }
        if (upperRank_ != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(upperGhost, receiveUpper_, planeBytes_, cudaMemcpyHostToDevice,
                                       communicationStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
    }

  private:
    size_t planeElements_;
    size_t planeBytes_;
    int lowerRank_;
    int upperRank_;
    bool cudaAware_;
    MPI_Request requests_[4]{};
    Real* hostBuffers_ = nullptr;
    Real* sendLower_ = nullptr;
    Real* sendUpper_ = nullptr;
    Real* receiveLower_ = nullptr;
    Real* receiveUpper_ = nullptr;
};

void initializeDeviceGrids(Real* gridA, Real* gridB, size_t planeElements,
                           int localNz, int globalZBegin) {
    size_t ownedElements = 0;
    if (!checkedMultiply(planeElements, static_cast<size_t>(localNz), ownedElements) ||
        ownedElements > static_cast<size_t>(std::numeric_limits<std::int64_t>::max())) {
        abortBenchmark("local grid is too large for host initialization");
    }

    Real* initial = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&initial), ownedElements * sizeof(Real)));
    const size_t globalOffset = static_cast<size_t>(globalZBegin) * planeElements;
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(ownedElements); ++i) {
        initial[i] = static_cast<Real>((globalOffset + static_cast<size_t>(i)) % 19);
    }

    const size_t allocationElements = planeElements * static_cast<size_t>(localNz + 2);
    const size_t allocationBytes = allocationElements * sizeof(Real);
    CUDA_CHECK(cudaMemset(gridA, 0, allocationBytes));
    CUDA_CHECK(cudaMemcpy(gridA + planeElements, initial, ownedElements * sizeof(Real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gridB, gridA, allocationBytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFreeHost(initial));
}

bool validateDistributed(const std::vector<Real>& localGrid) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    unsigned long long localInvalid = 0;

#pragma omp parallel for schedule(static) reduction(min : localMin) reduction(max : localMax) reduction(+ : localInvalid)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(localGrid.size()); ++i) {
        const Real value = localGrid[static_cast<size_t>(i)];
        if (!std::isfinite(value)) ++localInvalid;
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }

    Real globalMin = 0;
    Real globalMax = 0;
    unsigned long long globalInvalid = 0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localInvalid, &globalInvalid, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (worldRank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        } else if (globalMax > 1e6 || globalMin < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

void gatherAndPrint(const std::vector<Real>& localGrid, size_t planeElements,
                    int globalNz, int worldSize) {
    const size_t globalElements = planeElements * static_cast<size_t>(globalNz);
    if (globalElements > static_cast<size_t>(INT_MAX)) {
        abortBenchmark("-r output currently requires at most INT_MAX grid elements");
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<Real> globalGrid;
    if (worldRank == 0) {
        counts.resize(worldSize);
        displacements.resize(worldSize);
        for (int rank = 0; rank < worldSize; ++rank) {
            const Slab slab = decomposeZ(globalNz, rank, worldSize);
            counts[rank] = static_cast<int>(static_cast<size_t>(slab.localNz) * planeElements);
            displacements[rank] = static_cast<int>(static_cast<size_t>(slab.globalZBegin) * planeElements);
        }
        globalGrid.resize(globalElements);
    }

    MPI_Gatherv(localGrid.data(), static_cast<int>(localGrid.size()), MPI_DOUBLE,
                worldRank == 0 ? globalGrid.data() : nullptr,
                worldRank == 0 ? counts.data() : nullptr,
                worldRank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (worldRank == 0) print_results(globalGrid, "Grid");
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortBenchmark("MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    Options options;
    if (!parseOptions(argc, argv, options)) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (options.help) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    if (options.nx < 3 || options.ny < 3 || options.nz < 3) {
        abortBenchmark("all grid dimensions must be at least 3");
    }
    if (options.nx > static_cast<size_t>(INT_MAX) ||
        options.ny > static_cast<size_t>(INT_MAX) ||
        options.nz > static_cast<size_t>(INT_MAX)) {
        abortBenchmark("grid dimensions exceed the supported integer range");
    }
    if (worldSize > static_cast<int>(options.nz)) {
        abortBenchmark("the number of MPI ranks must not exceed the Z dimension");
    }

    size_t planeElements = 0;
    size_t globalElements = 0;
    if (!checkedMultiply(options.nx, options.ny, planeElements) ||
        !checkedMultiply(planeElements, options.nz, globalElements) ||
        planeElements > static_cast<size_t>(INT_MAX)) {
        abortBenchmark("grid is too large (a halo plane must fit in an MPI count)");
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL,
                        &nodeCommunicator);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeCommunicator, &nodeRank);
    MPI_Comm_size(nodeCommunicator, &nodeSize);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) abortBenchmark("no CUDA accelerator is available");
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    const bool cudaAwareMpi = queryCudaAwareMpi();

    const int nx = static_cast<int>(options.nx);
    const int ny = static_cast<int>(options.ny);
    const int nz = static_cast<int>(options.nz);
    const Slab slab = decomposeZ(nz, worldRank, worldSize);
    size_t allocationElements = 0;
    if (!checkedMultiply(planeElements, static_cast<size_t>(slab.localNz + 2),
                         allocationElements) ||
        allocationElements > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        abortBenchmark("local device allocation size overflow");
    }
    const size_t allocationBytes = allocationElements * sizeof(Real);

    Real* gridA = nullptr;
    Real* gridB = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gridA), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gridB), allocationBytes));

    omp_set_dynamic(0);
    initializeDeviceGrids(gridA, gridB, planeElements, slab.localNz, slab.globalZBegin);

    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));

    const int lowerRank = (worldRank == 0) ? MPI_PROC_NULL : worldRank - 1;
    const int upperRank = (worldRank == worldSize - 1) ? MPI_PROC_NULL : worldRank + 1;
    HaloExchange halo(planeElements, lowerRank, upperRank, cudaAwareMpi);

    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), %d OpenMP thread(s)/rank, "
                    "%d rank(s) on this node\n", worldSize, omp_get_max_threads(), nodeSize);
        std::printf("Rank 0 CUDA device: %s; halo transport: %s\n",
                    deviceProperties.name,
                    cudaAwareMpi ? "CUDA-aware MPI" : "pinned-host MPI staging");
        std::printf("Initializing grid...\n");
        std::printf("Running stencil computation...\n");
    }

    Real* input = gridA;
    Real* output = gridB;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        // Planes not adjacent to a rank boundary do not depend on the incoming
        // ghosts.  Enqueue them first so both CUDA-aware communication and the
        // portable device-to-host staging copies overlap useful GPU work.
        launchStencil<false>(input, output, nx, ny, nz, slab.localNz,
                             slab.globalZBegin, 2, slab.localNz - 2, computeStream);

        halo.begin(input, slab.localNz, communicationStream);
        halo.finish(input, slab.localNz, communicationStream);
        const int endpointCount = (slab.localNz == 1) ? 1 : 2;
        launchStencil<true>(input, output, nx, ny, nz, slab.localNz,
                            slab.globalZBegin, 1, endpointCount, computeStream);
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        std::swap(input, output);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double updates = static_cast<double>(options.nx - 2) *
                               static_cast<double>(options.ny - 2) *
                               static_cast<double>(options.nz - 2) *
                               static_cast<double>(options.iterations);
        const double mcups = elapsed > 0.0 ? updates / elapsed / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> localFinal;
    if (options.validate || options.printResults) {
        const size_t ownedElements = planeElements * static_cast<size_t>(slab.localNz);
        localFinal.resize(ownedElements);
        CUDA_CHECK(cudaMemcpy(localFinal.data(), input + planeElements,
                              ownedElements * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    if (options.printResults) {
        gatherAndPrint(localFinal, planeElements, nz, worldSize);
    }

    bool valid = true;
    if (options.validate) {
        if (worldRank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(localFinal);
        if (worldRank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(gridB));
    CUDA_CHECK(cudaFree(gridA));
    MPI_Comm_free(&nodeCommunicator);
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
