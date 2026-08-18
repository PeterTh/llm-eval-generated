#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
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

constexpr int Dimensions = 3;
constexpr int XMinus = 0;
constexpr int XPlus = 1;
constexpr int YMinus = 2;
constexpr int YPlus = 3;
constexpr int ZMinus = 4;
constexpr int ZPlus = 5;
constexpr int FaceCount = 6;

int worldRank = 0;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "Rank %d: CUDA error at %s:%d while executing %s: %s\n",
                 worldRank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call) do { \
    const cudaError_t cudaStatus = (call); \
    if (cudaStatus != cudaSuccess) cudaFailure(cudaStatus, #call, __FILE__, __LINE__); \
} while (false)

__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Partition {
    std::array<size_t, Dimensions> offset{};
    std::array<size_t, Dimensions> extent{};
};

struct FaceBuffers {
    size_t count = 0;
    Real* deviceSend = nullptr;
    Real* deviceReceive = nullptr;
    Real* hostSend = nullptr;
    Real* hostReceive = nullptr;
};

bool safeMultiply(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool gridElementCount(const std::array<size_t, Dimensions>& dimensions, size_t& count) {
    size_t xy = 0;
    return safeMultiply(dimensions[0], dimensions[1], xy) &&
           safeMultiply(xy, dimensions[2], count);
}

bool parseSize(const char* text, size_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    size_t parsed = 0;
    if (!parseSize(text, parsed) || parsed > static_cast<size_t>(INT_MAX)) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Finds a factorization that is both geometry-aware and valid for very thin grids.
bool chooseCartesianDimensions(const int ranks,
                               const std::array<size_t, Dimensions>& global,
                               std::array<int, Dimensions>& result) {
    double bestSurface = std::numeric_limits<double>::infinity();
    bool found = false;

    for (int dx = 1; dx <= ranks; ++dx) {
        if (ranks % dx != 0 || static_cast<size_t>(dx) > global[0]) continue;
        const int remainder = ranks / dx;
        for (int dy = 1; dy <= remainder; ++dy) {
            if (remainder % dy != 0 || static_cast<size_t>(dy) > global[1]) continue;
            const int dz = remainder / dy;
            if (static_cast<size_t>(dz) > global[2]) continue;

            const double localX = static_cast<double>(global[0]) / dx;
            const double localY = static_cast<double>(global[1]) / dy;
            const double localZ = static_cast<double>(global[2]) / dz;
            const double surface = localY * localZ + localX * localZ + localX * localY;
            if (surface < bestSurface) {
                bestSurface = surface;
                result = {dx, dy, dz};
                found = true;
            }
        }
    }
    return found;
}

Partition partitionForCoordinates(const std::array<size_t, Dimensions>& global,
                                  const std::array<int, Dimensions>& processGrid,
                                  const int coordinates[Dimensions]) {
    Partition partition;
    for (int axis = 0; axis < Dimensions; ++axis) {
        const size_t base = global[axis] / static_cast<size_t>(processGrid[axis]);
        const size_t remainder = global[axis] % static_cast<size_t>(processGrid[axis]);
        const size_t coordinate = static_cast<size_t>(coordinates[axis]);
        partition.extent[axis] = base + (coordinate < remainder ? 1 : 0);
        partition.offset[axis] = coordinate * base + std::min(coordinate, remainder);
    }
    return partition;
}

size_t faceElementCount(const int face, const Partition& partition) {
    if (face == XMinus || face == XPlus) {
        return partition.extent[1] * partition.extent[2];
    }
    if (face == YMinus || face == YPlus) {
        return partition.extent[0] * partition.extent[2];
    }
    return partition.extent[0] * partition.extent[1];
}

constexpr int oppositeFace(const int face) noexcept {
    return face ^ 1;
}

__global__ void initializeOwnedKernel(Real* grid, const size_t localX, const size_t localY,
                                      const size_t localZ, const size_t globalX,
                                      const size_t globalY, const size_t offsetX,
                                      const size_t offsetY, const size_t offsetZ) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= localX || y >= localY || z >= localZ) return;

    const size_t pitchX = localX + 2;
    const size_t plane = pitchX * (localY + 2);
    const size_t localIndex = (z + 1) * plane + (y + 1) * pitchX + (x + 1);
    const size_t globalIndex = idx3(offsetX + x, offsetY + y, offsetZ + z, globalX, globalY);
    grid[localIndex] = static_cast<Real>(globalIndex % 19);
}

__global__ void packFaceKernel(const Real* grid, Real* packed, const int face,
                               const size_t localX, const size_t localY, const size_t localZ) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = (face == XMinus || face == XPlus) ? localY * localZ :
                         (face == YMinus || face == YPlus) ? localX * localZ : localX * localY;
    if (element >= count) return;

    const size_t pitchX = localX + 2;
    const size_t plane = pitchX * (localY + 2);
    size_t index = 0;
    if (face == XMinus || face == XPlus) {
        const size_t y = element % localY;
        const size_t z = element / localY;
        const size_t x = face == XMinus ? 1 : localX;
        index = (z + 1) * plane + (y + 1) * pitchX + x;
    } else if (face == YMinus || face == YPlus) {
        const size_t x = element % localX;
        const size_t z = element / localX;
        const size_t y = face == YMinus ? 1 : localY;
        index = (z + 1) * plane + y * pitchX + (x + 1);
    } else {
        const size_t x = element % localX;
        const size_t y = element / localX;
        const size_t z = face == ZMinus ? 1 : localZ;
        index = z * plane + (y + 1) * pitchX + (x + 1);
    }
    packed[element] = grid[index];
}

__global__ void unpackFaceKernel(Real* grid, const Real* packed, const int face,
                                 const size_t localX, const size_t localY, const size_t localZ) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = (face == XMinus || face == XPlus) ? localY * localZ :
                         (face == YMinus || face == YPlus) ? localX * localZ : localX * localY;
    if (element >= count) return;

    const size_t pitchX = localX + 2;
    const size_t plane = pitchX * (localY + 2);
    size_t index = 0;
    if (face == XMinus || face == XPlus) {
        const size_t y = element % localY;
        const size_t z = element / localY;
        const size_t x = face == XMinus ? 0 : localX + 1;
        index = (z + 1) * plane + (y + 1) * pitchX + x;
    } else if (face == YMinus || face == YPlus) {
        const size_t x = element % localX;
        const size_t z = element / localX;
        const size_t y = face == YMinus ? 0 : localY + 1;
        index = (z + 1) * plane + y * pitchX + (x + 1);
    } else {
        const size_t x = element % localX;
        const size_t y = element / localX;
        const size_t z = face == ZMinus ? 0 : localZ + 1;
        index = z * plane + (y + 1) * pitchX + (x + 1);
    }
    grid[index] = packed[element];
}

__global__ void stencilKernel(const Real* input, Real* output,
                              const size_t localX, const size_t localY, const size_t localZ,
                              const size_t globalX, const size_t globalY, const size_t globalZ,
                              const size_t offsetX, const size_t offsetY, const size_t offsetZ,
                              const bool boundaryLayer) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    if (x > localX || y > localY || z > localZ) return;

    const bool strictInterior = x > 1 && x < localX && y > 1 && y < localY && z > 1 && z < localZ;
    if (boundaryLayer ? strictInterior : !strictInterior) return;

    const size_t pitchX = localX + 2;
    const size_t plane = pitchX * (localY + 2);
    const size_t index = z * plane + y * pitchX + x;
    const size_t gx = offsetX + x - 1;
    const size_t gy = offsetY + y - 1;
    const size_t gz = offsetZ + z - 1;

    if (gx == 0 || gx + 1 == globalX || gy == 0 || gy + 1 == globalY || gz == 0 || gz + 1 == globalZ) {
        output[index] = input[index];
        return;
    }

    output[index] = (input[index] + input[index - 1] + input[index + 1] +
                     input[index - pitchX] + input[index + pitchX] +
                     input[index - plane] + input[index + plane]) / 7.0;
}

__global__ void packOwnedKernel(const Real* grid, Real* packed,
                                const size_t localX, const size_t localY, const size_t localZ) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= localX || y >= localY || z >= localZ) return;

    const size_t pitchX = localX + 2;
    const size_t plane = pitchX * (localY + 2);
    packed[idx3(x, y, z, localX, localY)] = grid[(z + 1) * plane + (y + 1) * pitchX + (x + 1)];
}

dim3 makeGrid(const size_t localX, const size_t localY, const size_t localZ, const dim3 block) {
    const size_t blocksX = (localX + block.x - 1) / block.x;
    const size_t blocksY = (localY + block.y - 1) / block.y;
    const size_t blocksZ = (localZ + block.z - 1) / block.z;
    if (blocksX > std::numeric_limits<unsigned int>::max() ||
        blocksY > std::numeric_limits<unsigned int>::max() ||
        blocksZ > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Rank %d: local CUDA launch grid is too large\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
    return dim3(static_cast<unsigned int>(blocksX), static_cast<unsigned int>(blocksY),
                static_cast<unsigned int>(blocksZ));
}

void launchStencil(const Real* input, Real* output, const Partition& partition,
                   const std::array<size_t, Dimensions>& global, const bool boundaryLayer,
                   cudaStream_t stream) {
    constexpr dim3 block(8, 8, 4);
    const dim3 grid = makeGrid(partition.extent[0], partition.extent[1], partition.extent[2], block);
    stencilKernel<<<grid, block, 0, stream>>>(input, output,
        partition.extent[0], partition.extent[1], partition.extent[2],
        global[0], global[1], global[2],
        partition.offset[0], partition.offset[1], partition.offset[2], boundaryLayer);
    CUDA_CHECK(cudaGetLastError());
}

void copyOwnedToHost(const Real* deviceGrid, std::vector<Real>& hostGrid,
                     const Partition& partition, cudaStream_t stream) {
    size_t localElements = 0;
    if (!gridElementCount(partition.extent, localElements)) {
        std::fprintf(stderr, "Rank %d: local grid is too large\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
    Real* packed = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&packed), localElements * sizeof(Real)));
    constexpr dim3 block(8, 8, 4);
    const dim3 grid = makeGrid(partition.extent[0], partition.extent[1], partition.extent[2], block);
    packOwnedKernel<<<grid, block, 0, stream>>>(deviceGrid, packed,
        partition.extent[0], partition.extent[1], partition.extent[2]);
    CUDA_CHECK(cudaGetLastError());
    hostGrid.resize(localElements);
    CUDA_CHECK(cudaMemcpyAsync(hostGrid.data(), packed, localElements * sizeof(Real),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(packed));
}

bool validateLocal(const std::vector<Real>& grid, Real& minimum, Real& maximum) {
    minimum = std::numeric_limits<Real>::infinity();
    maximum = -std::numeric_limits<Real>::infinity();
    int finite = 1;

#pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(&:finite) schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
            finite = 0;
        } else {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    return finite != 0;
}

void releaseFaceBuffers(std::array<FaceBuffers, FaceCount>& faces) {
    for (FaceBuffers& face : faces) {
        if (face.deviceSend != nullptr) CUDA_CHECK(cudaFree(face.deviceSend));
        if (face.deviceReceive != nullptr) CUDA_CHECK(cudaFree(face.deviceReceive));
        if (face.hostSend != nullptr) CUDA_CHECK(cudaFreeHost(face.hostSend));
        if (face.hostReceive != nullptr) CUDA_CHECK(cudaFreeHost(face.hostReceive));
    }
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED support\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    std::array<size_t, Dimensions> global{128, 0, 0};
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], global[0]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], global[1]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], global[2]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parseError = !parseIterations(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            parseError = true;
        }
        if (parseError) break;
    }
    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];

    size_t globalElements = 0;
    if (parseError || global[0] == 0 || global[1] == 0 || global[2] == 0 ||
        !gridElementCount(global, globalElements)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int activeRanks = static_cast<int>(std::min(globalElements, static_cast<size_t>(worldSize)));
    std::array<int, Dimensions> processGrid{};
    while (activeRanks > 1 && !chooseCartesianDimensions(activeRanks, global, processGrid)) {
        --activeRanks;
    }
    if (!chooseCartesianDimensions(activeRanks, global, processGrid)) {
        if (worldRank == 0) std::fprintf(stderr, "Unable to construct an MPI process grid\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm activeComm = MPI_COMM_NULL;
    const int color = worldRank < activeRanks ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &activeComm);
    if (activeComm == MPI_COMM_NULL) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    int activeRank = 0;
    MPI_Comm_rank(activeComm, &activeRank);
    MPI_Comm cartComm = MPI_COMM_NULL;
    const int dimensions[Dimensions] = {processGrid[0], processGrid[1], processGrid[2]};
    const int periods[Dimensions] = {0, 0, 0};
    MPI_Cart_create(activeComm, Dimensions, dimensions, periods, 0, &cartComm);
    if (cartComm == MPI_COMM_NULL) {
        if (worldRank == 0) std::fprintf(stderr, "Unable to create Cartesian MPI communicator\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int cartRank = 0;
    int coordinates[Dimensions] = {};
    MPI_Comm_rank(cartComm, &cartRank);
    MPI_Cart_coords(cartComm, cartRank, Dimensions, coordinates);
    const Partition partition = partitionForCoordinates(global, processGrid, coordinates);

    int neighbours[FaceCount] = {};
    MPI_Cart_shift(cartComm, 0, 1, &neighbours[XMinus], &neighbours[XPlus]);
    MPI_Cart_shift(cartComm, 1, 1, &neighbours[YMinus], &neighbours[YPlus]);
    MPI_Cart_shift(cartComm, 2, 1, &neighbours[ZMinus], &neighbours[ZPlus]);

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(activeComm, MPI_COMM_TYPE_SHARED, activeRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (worldRank == 0) std::fprintf(stderr, "No CUDA device is visible to this MPI rank\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));
    MPI_Comm_free(&nodeComm);

    size_t paddedXY = 0;
    size_t paddedElements = 0;
    if (!safeMultiply(partition.extent[0] + 2, partition.extent[1] + 2, paddedXY) ||
        !safeMultiply(paddedXY, partition.extent[2] + 2, paddedElements)) {
        if (worldRank == 0) std::fprintf(stderr, "Local grid is too large\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    Real* current = nullptr;
    Real* next = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&current), paddedElements * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next), paddedElements * sizeof(Real)));

    cudaStream_t computeStream = nullptr;
    cudaStream_t transferStream = nullptr;
    cudaEvent_t computationDone = nullptr;
    cudaEvent_t sendBuffersReady = nullptr;
    cudaEvent_t halosReady = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&computationDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&sendBuffersReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));

    constexpr dim3 stencilBlock(8, 8, 4);
    const dim3 stencilGrid = makeGrid(partition.extent[0], partition.extent[1], partition.extent[2], stencilBlock);
    initializeOwnedKernel<<<stencilGrid, stencilBlock, 0, computeStream>>>(current,
        partition.extent[0], partition.extent[1], partition.extent[2], global[0], global[1],
        partition.offset[0], partition.offset[1], partition.offset[2]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(computationDone, computeStream));

    std::array<FaceBuffers, FaceCount> faces{};
    for (int face = 0; face < FaceCount; ++face) {
        if (neighbours[face] == MPI_PROC_NULL) continue;
        FaceBuffers& buffers = faces[face];
        buffers.count = faceElementCount(face, partition);
        if (buffers.count > static_cast<size_t>(INT_MAX) ||
            buffers.count > std::numeric_limits<size_t>::max() / sizeof(Real)) {
            std::fprintf(stderr, "Rank %d: MPI halo is too large\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }
        const size_t bytes = buffers.count * sizeof(Real);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.deviceSend), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.deviceReceive), bytes));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.hostSend), bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.hostReceive), bytes, cudaHostAllocDefault));
    }

    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", global[0], global[1], global[2]);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI process grid: %d x %d x %d (%d active ranks)\n",
                    processGrid[0], processGrid[1], processGrid[2], activeRanks);
        std::printf("Running stencil computation...\n");
    }

    CUDA_CHECK(cudaEventSynchronize(computationDone));
    MPI_Barrier(activeComm);
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::array<MPI_Request, FaceCount * 2> requests{};
        int requestCount = 0;
        for (int face = 0; face < FaceCount; ++face) {
            if (neighbours[face] == MPI_PROC_NULL) continue;
            FaceBuffers& buffers = faces[face];
            MPI_Irecv(buffers.hostReceive, static_cast<int>(buffers.count), MPI_DOUBLE,
                      neighbours[face], oppositeFace(face), cartComm, &requests[requestCount++]);
        }

        launchStencil(current, next, partition, global, false, computeStream);

        CUDA_CHECK(cudaStreamWaitEvent(transferStream, computationDone, 0));
        for (int face = 0; face < FaceCount; ++face) {
            if (neighbours[face] == MPI_PROC_NULL) continue;
            FaceBuffers& buffers = faces[face];
            const int threads = 256;
            const unsigned int blocks = static_cast<unsigned int>((buffers.count + threads - 1) / threads);
            packFaceKernel<<<blocks, threads, 0, transferStream>>>(current, buffers.deviceSend, face,
                partition.extent[0], partition.extent[1], partition.extent[2]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(buffers.hostSend, buffers.deviceSend, buffers.count * sizeof(Real),
                                       cudaMemcpyDeviceToHost, transferStream));
        }
        CUDA_CHECK(cudaEventRecord(sendBuffersReady, transferStream));
        CUDA_CHECK(cudaEventSynchronize(sendBuffersReady));

        for (int face = 0; face < FaceCount; ++face) {
            if (neighbours[face] == MPI_PROC_NULL) continue;
            FaceBuffers& buffers = faces[face];
            MPI_Isend(buffers.hostSend, static_cast<int>(buffers.count), MPI_DOUBLE,
                      neighbours[face], face, cartComm, &requests[requestCount++]);
        }
        MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE);

        for (int face = 0; face < FaceCount; ++face) {
            if (neighbours[face] == MPI_PROC_NULL) continue;
            FaceBuffers& buffers = faces[face];
            CUDA_CHECK(cudaMemcpyAsync(buffers.deviceReceive, buffers.hostReceive, buffers.count * sizeof(Real),
                                       cudaMemcpyHostToDevice, transferStream));
            const int threads = 256;
            const unsigned int blocks = static_cast<unsigned int>((buffers.count + threads - 1) / threads);
            unpackFaceKernel<<<blocks, threads, 0, transferStream>>>(current, buffers.deviceReceive, face,
                partition.extent[0], partition.extent[1], partition.extent[2]);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(halosReady, transferStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, halosReady, 0));
        launchStencil(current, next, partition, global, true, computeStream);
        CUDA_CHECK(cudaEventRecord(computationDone, computeStream));
        std::swap(current, next);
    }

    CUDA_CHECK(cudaEventSynchronize(computationDone));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);

    if (worldRank == 0) {
        const long milliseconds = static_cast<long>(std::llround(elapsed * 1000.0));
        size_t interiorCells = 0;
        if (global[0] > 2 && global[1] > 2 && global[2] > 2) {
            const std::array<size_t, Dimensions> interior{global[0] - 2, global[1] - 2, global[2] - 2};
            gridElementCount(interior, interiorCells);
        }
        const double updates = static_cast<double>(interiorCells) * static_cast<double>(iterations);
        const double mcups = elapsed > 0.0 ? updates / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %ld ms\n", milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> localResult;
    if (validate || printResults) {
        copyOwnedToHost(current, localResult, partition, computeStream);
    }

    if (printResults) {
        const int localCount = static_cast<int>(localResult.size());
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<int> allCoordinates;
        if (activeRank == 0) {
            receiveCounts.resize(activeRanks);
            allCoordinates.resize(activeRanks * Dimensions);
        }
        MPI_Gather(&localCount, 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, 0, activeComm);
        MPI_Gather(coordinates, Dimensions, MPI_INT, allCoordinates.data(), Dimensions, MPI_INT, 0, activeComm);

        std::vector<Real> receivedBlocks;
        std::vector<Real> finalGrid;
        if (activeRank == 0) {
            displacements.resize(activeRanks);
            int totalCount = 0;
            for (int rank = 0; rank < activeRanks; ++rank) {
                displacements[rank] = totalCount;
                totalCount += receiveCounts[rank];
            }
            receivedBlocks.resize(static_cast<size_t>(totalCount));
            finalGrid.resize(globalElements);
        }
        MPI_Gatherv(localResult.data(), localCount, MPI_DOUBLE, receivedBlocks.data(), receiveCounts.data(),
                    displacements.data(), MPI_DOUBLE, 0, activeComm);

        if (activeRank == 0) {
#pragma omp parallel for schedule(static)
            for (int rank = 0; rank < activeRanks; ++rank) {
                const int* rankCoordinates = allCoordinates.data() + rank * Dimensions;
                const Partition rankPartition = partitionForCoordinates(global, processGrid, rankCoordinates);
                const Real* source = receivedBlocks.data() + displacements[rank];
                for (size_t z = 0; z < rankPartition.extent[2]; ++z) {
                    for (size_t y = 0; y < rankPartition.extent[1]; ++y) {
                        Real* destination = finalGrid.data() + idx3(rankPartition.offset[0],
                            rankPartition.offset[1] + y, rankPartition.offset[2] + z,
                            global[0], global[1]);
                        const Real* sourceRow = source + idx3(0, y, z,
                            rankPartition.extent[0], rankPartition.extent[1]);
                        std::memcpy(destination, sourceRow, rankPartition.extent[0] * sizeof(Real));
                    }
                }
            }
            print_results(finalGrid, "Grid");
        }
    }

    int exitCode = EXIT_SUCCESS;
    if (validate) {
        Real localMinimum = 0.0;
        Real localMaximum = 0.0;
        const int localValid = validateLocal(localResult, localMinimum, localMaximum) ? 1 : 0;
        int allFinite = 0;
        Real minimum = 0.0;
        Real maximum = 0.0;
        MPI_Reduce(&localValid, &allFinite, 1, MPI_INT, MPI_MIN, 0, activeComm);
        MPI_Reduce(&localMinimum, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, activeComm);
        MPI_Reduce(&localMaximum, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);

        if (activeRank == 0) {
            std::printf("Validating result...\n");
            if (!allFinite) {
                std::printf("Validation failed: found NaN or Inf value\n");
                exitCode = EXIT_FAILURE;
            } else {
                std::printf("Value range: [%.6f, %.6f]\n", minimum, maximum);
                if (maximum > 1.0e6 || minimum < -1.0e6) {
                    std::printf("Validation failed: values out of expected range\n");
                    exitCode = EXIT_FAILURE;
                }
            }
            std::printf("Validation: %s\n", exitCode == EXIT_SUCCESS ? "PASSED" : "FAILED");
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, activeComm);
    releaseFaceBuffers(faces);
    CUDA_CHECK(cudaEventDestroy(halosReady));
    CUDA_CHECK(cudaEventDestroy(sendBuffersReady));
    CUDA_CHECK(cudaEventDestroy(computationDone));
    CUDA_CHECK(cudaStreamDestroy(transferStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFree(current));
    MPI_Comm_free(&cartComm);
    MPI_Comm_free(&activeComm);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
