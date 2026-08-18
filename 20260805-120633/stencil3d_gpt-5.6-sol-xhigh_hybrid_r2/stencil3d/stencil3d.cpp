#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
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

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kGatherTag = 700;

enum Face : int { XMinus = 0, XPlus, YMinus, YPlus, ZMinus, ZPlus, FaceCount };

[[noreturn]] void abortMpi(const char* expression, const char* file, int line,
                          int errorCode) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(errorCode, message, &length);
    std::fprintf(stderr, "Rank %d: MPI call %s failed at %s:%d: %.*s\n", rank,
                 expression, file, line, length, message);
    MPI_Abort(MPI_COMM_WORLD, errorCode);
    std::abort();
}

void checkMpi(int errorCode, const char* expression, const char* file, int line) {
    if (errorCode != MPI_SUCCESS) {
        abortMpi(expression, file, line, errorCode);
    }
}

#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

[[noreturn]] void abortCuda(const char* expression, const char* file, int line,
                           cudaError_t errorCode) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA call %s failed at %s:%d: %s\n", rank,
                 expression, file, line, cudaGetErrorString(errorCode));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(errorCode));
    std::abort();
}

void checkCuda(cudaError_t errorCode, const char* expression, const char* file,
               int line) {
    if (errorCode != cudaSuccess) {
        abortCuda(expression, file, line, errorCode);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

struct AxisPartition {
    std::size_t begin = 0;
    std::size_t size = 0;
};

AxisPartition partitionAxis(std::size_t globalSize, int partitions, int coordinate) {
    const std::size_t p = static_cast<std::size_t>(partitions);
    const std::size_t c = static_cast<std::size_t>(coordinate);
    const std::size_t base = globalSize / p;
    const std::size_t remainder = globalSize % p;
    return {c * base + std::min(c, remainder), base + (c < remainder ? 1U : 0U)};
}

bool checkedMultiply(std::size_t a, std::size_t b, std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

// Select the rectangular process grid with the least total internal interface
// area. Unlike MPI_Dims_create, this also respects small or anisotropic grids.
bool chooseProcessGrid(int ranks, std::size_t nx, std::size_t ny, std::size_t nz,
                       std::array<int, 3>& best) {
    long double bestSurface = std::numeric_limits<long double>::infinity();
    long double bestAspect = std::numeric_limits<long double>::infinity();
    bool found = false;

    for (int px = 1; px <= ranks; ++px) {
        if (ranks % px != 0 || static_cast<std::size_t>(px) > nx) {
            continue;
        }
        const int remaining = ranks / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0 || static_cast<std::size_t>(py) > ny) {
                continue;
            }
            const int pz = remaining / py;
            if (static_cast<std::size_t>(pz) > nz) {
                continue;
            }

            const long double surface =
                static_cast<long double>(px - 1) * ny * nz +
                static_cast<long double>(py - 1) * nx * nz +
                static_cast<long double>(pz - 1) * nx * ny;
            const long double sx = static_cast<long double>(nx) / px;
            const long double sy = static_cast<long double>(ny) / py;
            const long double sz = static_cast<long double>(nz) / pz;
            const long double largest = std::max({sx, sy, sz});
            const long double smallest = std::min({sx, sy, sz});
            const long double aspect = largest / smallest;

            if (!found || surface < bestSurface ||
                (surface == bestSurface && aspect < bestAspect)) {
                best = {px, py, pz};
                bestSurface = surface;
                bestAspect = aspect;
                found = true;
            }
        }
    }
    return found;
}

__host__ __device__ constexpr std::size_t localIndex(std::size_t x, std::size_t y,
                                                      std::size_t z, std::size_t mx,
                                                      std::size_t my) noexcept {
    return (z * my + y) * mx + x;
}

// A 2-D shared-memory tile reuses the four in-plane neighbors. Z neighbors are
// naturally coalesced and generally served by L2 while adjacent planes run.
__global__ void stencilRegion(const Real* __restrict__ input,
                              Real* __restrict__ output, std::size_t mx,
                              std::size_t my, std::size_t nx, std::size_t ny,
                              std::size_t nz, std::size_t globalXBegin,
                              std::size_t globalYBegin, std::size_t globalZBegin,
                              std::size_t xBegin, std::size_t xEnd,
                              std::size_t yBegin, std::size_t yEnd,
                              std::size_t zBegin) {
    __shared__ Real tile[kBlockY + 2][kBlockX + 2];

    const std::size_t x = xBegin + static_cast<std::size_t>(blockIdx.x) * kBlockX +
                          threadIdx.x;
    const std::size_t y = yBegin + static_cast<std::size_t>(blockIdx.y) * kBlockY +
                          threadIdx.y;
    const std::size_t z = zBegin + blockIdx.z;
    const int sx = static_cast<int>(threadIdx.x) + 1;
    const int sy = static_cast<int>(threadIdx.y) + 1;
    const bool active = x <= xEnd && y <= yEnd;

    std::size_t centerIndex = 0;
    if (active) {
        centerIndex = localIndex(x, y, z, mx, my);
        tile[sy][sx] = input[centerIndex];
        if (threadIdx.x == 0) {
            tile[sy][0] = input[centerIndex - 1];
        }
        if (threadIdx.x == kBlockX - 1 || x == xEnd) {
            tile[sy][sx + 1] = input[centerIndex + 1];
        }
        if (threadIdx.y == 0) {
            tile[0][sx] = input[centerIndex - mx];
        }
        if (threadIdx.y == kBlockY - 1 || y == yEnd) {
            tile[sy + 1][sx] = input[centerIndex + mx];
        }
    }
    __syncthreads();

    if (!active) {
        return;
    }

    const std::size_t globalX = globalXBegin + x - 1;
    const std::size_t globalY = globalYBegin + y - 1;
    const std::size_t globalZ = globalZBegin + z - 1;
    const Real center = tile[sy][sx];
    if (globalX == 0 || globalX + 1 == nx || globalY == 0 || globalY + 1 == ny ||
        globalZ == 0 || globalZ + 1 == nz) {
        output[centerIndex] = center;
        return;
    }

    const Real left = tile[sy][sx - 1];
    const Real right = tile[sy][sx + 1];
    const Real front = tile[sy - 1][sx];
    const Real back = tile[sy + 1][sx];
    const Real bottom = input[centerIndex - mx * my];
    const Real top = input[centerIndex + mx * my];
    output[centerIndex] =
        (center + left + right + front + back + bottom + top) / Real{7.0};
}

void launchRegion(const Real* input, Real* output, std::size_t mx, std::size_t my,
                  std::size_t nx, std::size_t ny, std::size_t nz,
                  const AxisPartition& xPart, const AxisPartition& yPart,
                  const AxisPartition& zPart, std::size_t xBegin,
                  std::size_t xEnd, std::size_t yBegin, std::size_t yEnd,
                  std::size_t zBegin, std::size_t zEnd, cudaStream_t stream) {
    if (xBegin > xEnd || yBegin > yEnd || zBegin > zEnd) {
        return;
    }

    const dim3 block(kBlockX, kBlockY, 1);
    const unsigned int gridX = static_cast<unsigned int>((xEnd - xBegin + kBlockX) /
                                                          kBlockX);
    const unsigned int gridY = static_cast<unsigned int>((yEnd - yBegin + kBlockY) /
                                                          kBlockY);
    if (gridY > 65535U) {
        std::fprintf(stderr, "Local Y extent exceeds the CUDA launch limit\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    constexpr std::size_t maxGridZ = 65535;
    for (std::size_t firstZ = zBegin; firstZ <= zEnd;) {
        const std::size_t depth = std::min(maxGridZ, zEnd - firstZ + 1);
        const dim3 grid(gridX, gridY, static_cast<unsigned int>(depth));
        stencilRegion<<<grid, block, 0, stream>>>(
            input, output, mx, my, nx, ny, nz, xPart.begin, yPart.begin,
            zPart.begin, xBegin, xEnd, yBegin, yEnd, firstZ);
        CUDA_CHECK(cudaPeekAtLastError());
        firstZ += depth;
    }
}

struct HaloBuffers {
    std::array<Real*, FaceCount> send{};
    std::array<Real*, FaceCount> receive{};
    std::array<std::size_t, FaceCount> count{};
};

int oppositeFace(int face) { return face ^ 1; }

void allocateHaloBuffers(HaloBuffers& buffers,
                         const std::array<int, FaceCount>& neighbors,
                         std::size_t mx, std::size_t my, std::size_t mz) {
    buffers.count[XMinus] = buffers.count[XPlus] = my * mz;
    buffers.count[YMinus] = buffers.count[YPlus] = mx * mz;
    buffers.count[ZMinus] = buffers.count[ZPlus] = mx * my;

    for (int face = 0; face < FaceCount; ++face) {
        if (neighbors[face] == MPI_PROC_NULL) {
            continue;
        }
        if (buffers.count[face] > static_cast<std::size_t>(INT_MAX)) {
            std::fprintf(stderr, "A halo face exceeds the MPI count limit\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.send[face]),
                                 buffers.count[face] * sizeof(Real),
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.receive[face]),
                                 buffers.count[face] * sizeof(Real),
                                 cudaHostAllocPortable));
    }
}

void freeHaloBuffers(HaloBuffers& buffers) {
    for (int face = 0; face < FaceCount; ++face) {
        if (buffers.send[face] != nullptr) {
            CUDA_CHECK(cudaFreeHost(buffers.send[face]));
        }
        if (buffers.receive[face] != nullptr) {
            CUDA_CHECK(cudaFreeHost(buffers.receive[face]));
        }
    }
}

void packHalos(const Real* deviceGrid, HaloBuffers& buffers,
               const std::array<int, FaceCount>& neighbors, std::size_t lx,
               std::size_t ly, std::size_t lz, std::size_t mx, std::size_t my,
               std::size_t mz, cudaStream_t stream) {
    const std::size_t rowBytes = mx * sizeof(Real);
    const std::size_t planeBytes = mx * my * sizeof(Real);

    if (neighbors[XMinus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(buffers.send[XMinus], sizeof(Real),
                                     deviceGrid + localIndex(1, 0, 0, mx, my),
                                     rowBytes, sizeof(Real), my * mz,
                                     cudaMemcpyDeviceToHost, stream));
    }
    if (neighbors[XPlus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(buffers.send[XPlus], sizeof(Real),
                                     deviceGrid + localIndex(lx, 0, 0, mx, my),
                                     rowBytes, sizeof(Real), my * mz,
                                     cudaMemcpyDeviceToHost, stream));
    }
    if (neighbors[YMinus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(buffers.send[YMinus], rowBytes,
                                     deviceGrid + localIndex(0, 1, 0, mx, my),
                                     planeBytes, rowBytes, mz,
                                     cudaMemcpyDeviceToHost, stream));
    }
    if (neighbors[YPlus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(buffers.send[YPlus], rowBytes,
                                     deviceGrid + localIndex(0, ly, 0, mx, my),
                                     planeBytes, rowBytes, mz,
                                     cudaMemcpyDeviceToHost, stream));
    }
    if (neighbors[ZMinus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.send[ZMinus],
                                   deviceGrid + localIndex(0, 0, 1, mx, my),
                                   planeBytes, cudaMemcpyDeviceToHost, stream));
    }
    if (neighbors[ZPlus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.send[ZPlus],
                                   deviceGrid + localIndex(0, 0, lz, mx, my),
                                   planeBytes, cudaMemcpyDeviceToHost, stream));
    }
}

void unpackHalos(Real* deviceGrid, const HaloBuffers& buffers,
                 const std::array<int, FaceCount>& neighbors, std::size_t lx,
                 std::size_t ly, std::size_t lz, std::size_t mx, std::size_t my,
                 std::size_t mz, cudaStream_t stream) {
    const std::size_t rowBytes = mx * sizeof(Real);
    const std::size_t planeBytes = mx * my * sizeof(Real);

    if (neighbors[XMinus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(deviceGrid + localIndex(0, 0, 0, mx, my),
                                     rowBytes, buffers.receive[XMinus], sizeof(Real),
                                     sizeof(Real), my * mz, cudaMemcpyHostToDevice,
                                     stream));
    }
    if (neighbors[XPlus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            deviceGrid + localIndex(lx + 1, 0, 0, mx, my), rowBytes,
            buffers.receive[XPlus], sizeof(Real), sizeof(Real), my * mz,
            cudaMemcpyHostToDevice, stream));
    }
    if (neighbors[YMinus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(deviceGrid + localIndex(0, 0, 0, mx, my),
                                     planeBytes, buffers.receive[YMinus], rowBytes,
                                     rowBytes, mz, cudaMemcpyHostToDevice, stream));
    }
    if (neighbors[YPlus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            deviceGrid + localIndex(0, ly + 1, 0, mx, my), planeBytes,
            buffers.receive[YPlus], rowBytes, rowBytes, mz,
            cudaMemcpyHostToDevice, stream));
    }
    if (neighbors[ZMinus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(deviceGrid + localIndex(0, 0, 0, mx, my),
                                   buffers.receive[ZMinus], planeBytes,
                                   cudaMemcpyHostToDevice, stream));
    }
    if (neighbors[ZPlus] != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(
            deviceGrid + localIndex(0, 0, lz + 1, mx, my),
            buffers.receive[ZPlus], planeBytes, cudaMemcpyHostToDevice, stream));
    }
}

void launchShell(const Real* input, Real* output, std::size_t mx, std::size_t my,
                 std::size_t nx, std::size_t ny, std::size_t nz,
                 const AxisPartition& xPart, const AxisPartition& yPart,
                 const AxisPartition& zPart, cudaStream_t stream) {
    const std::size_t lx = xPart.size;
    const std::size_t ly = yPart.size;
    const std::size_t lz = zPart.size;

    launchRegion(input, output, mx, my, nx, ny, nz, xPart, yPart, zPart, 1, 1, 1,
                 ly, 1, lz, stream);
    if (lx > 1) {
        launchRegion(input, output, mx, my, nx, ny, nz, xPart, yPart, zPart, lx,
                     lx, 1, ly, 1, lz, stream);
    }
    if (lx > 2) {
        launchRegion(input, output, mx, my, nx, ny, nz, xPart, yPart, zPart, 2,
                     lx - 1, 1, 1, 1, lz, stream);
        if (ly > 1) {
            launchRegion(input, output, mx, my, nx, ny, nz, xPart, yPart, zPart,
                         2, lx - 1, ly, ly, 1, lz, stream);
        }
    }
    if (lx > 2 && ly > 2) {
        launchRegion(input, output, mx, my, nx, ny, nz, xPart, yPart, zPart, 2,
                     lx - 1, 2, ly - 1, 1, 1, stream);
        if (lz > 1) {
            launchRegion(input, output, mx, my, nx, ny, nz, xPart, yPart, zPart,
                         2, lx - 1, 2, ly - 1, lz, lz, stream);
        }
    }
}

std::vector<Real> downloadOwned(const Real* deviceGrid, std::size_t lx,
                                std::size_t ly, std::size_t lz, std::size_t mx,
                                std::size_t my) {
    std::vector<Real> compact(lx * ly * lz);
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(const_cast<Real*>(deviceGrid),
                                      mx * sizeof(Real), mx, my);
    copy.srcPos = make_cudaPos(sizeof(Real), 1, 1);
    copy.dstPtr = make_cudaPitchedPtr(compact.data(), lx * sizeof(Real), lx, ly);
    copy.extent = make_cudaExtent(lx * sizeof(Real), ly, lz);
    copy.kind = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy3D(&copy));
    return compact;
}

void sendLarge(const Real* data, std::size_t count, int destination, MPI_Comm comm) {
    std::size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(
            std::min<std::size_t>(count - offset, static_cast<std::size_t>(INT_MAX)));
        MPI_CHECK(MPI_Send(data + offset, chunk, MPI_DOUBLE, destination, kGatherTag,
                           comm));
        offset += static_cast<std::size_t>(chunk);
    }
}

void receiveLarge(Real* data, std::size_t count, int source, MPI_Comm comm) {
    std::size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(
            std::min<std::size_t>(count - offset, static_cast<std::size_t>(INT_MAX)));
        MPI_CHECK(MPI_Recv(data + offset, chunk, MPI_DOUBLE, source, kGatherTag,
                           comm, MPI_STATUS_IGNORE));
        offset += static_cast<std::size_t>(chunk);
    }
}

void insertBlock(std::vector<Real>& global, const std::vector<Real>& block,
                 std::size_t nx, std::size_t ny, const AxisPartition& xPart,
                 const AxisPartition& yPart, const AxisPartition& zPart) {
    const std::int64_t ly = static_cast<std::int64_t>(yPart.size);
    const std::int64_t lz = static_cast<std::int64_t>(zPart.size);
#pragma omp parallel for collapse(2) schedule(static)
    for (std::int64_t z = 0; z < lz; ++z) {
        for (std::int64_t y = 0; y < ly; ++y) {
            const std::size_t source =
                (static_cast<std::size_t>(z) * yPart.size +
                 static_cast<std::size_t>(y)) *
                xPart.size;
            const std::size_t destination =
                ((zPart.begin + static_cast<std::size_t>(z)) * ny + yPart.begin +
                 static_cast<std::size_t>(y)) *
                    nx +
                xPart.begin;
            std::memcpy(global.data() + destination, block.data() + source,
                        xPart.size * sizeof(Real));
        }
    }
}

std::vector<Real> gatherGlobal(const std::vector<Real>& local, std::size_t nx,
                               std::size_t ny, std::size_t nz,
                               const std::array<int, 3>& processGrid, int rank,
                               int ranks, MPI_Comm cartesian) {
    if (rank != 0) {
        sendLarge(local.data(), local.size(), 0, cartesian);
        return {};
    }

    std::vector<Real> global(nx * ny * nz);
    for (int source = 0; source < ranks; ++source) {
        int coordinates[3] = {};
        MPI_CHECK(MPI_Cart_coords(cartesian, source, 3, coordinates));
        const AxisPartition xPart = partitionAxis(nx, processGrid[0], coordinates[0]);
        const AxisPartition yPart = partitionAxis(ny, processGrid[1], coordinates[1]);
        const AxisPartition zPart = partitionAxis(nz, processGrid[2], coordinates[2]);

        if (source == 0) {
            insertBlock(global, local, nx, ny, xPart, yPart, zPart);
        } else {
            std::vector<Real> block(xPart.size * yPart.size * zPart.size);
            receiveLarge(block.data(), block.size(), source, cartesian);
            insertBlock(global, block, nx, ny, xPart, yPart, zPart);
        }
    }
    return global;
}

bool validateDistributed(const std::vector<Real>& local, int rank,
                         MPI_Comm communicator) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localInvalid = 0;
    const std::int64_t count = static_cast<std::int64_t>(local.size());
#pragma omp parallel for reduction(min : localMin) reduction(max : localMax) \
    reduction(| : localInvalid) schedule(static)
    for (std::int64_t i = 0; i < count; ++i) {
        const Real value = local[static_cast<std::size_t>(i)];
        if (!std::isfinite(value)) {
            localInvalid = 1;
        } else {
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    }

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    int globalInvalid = 0;
    MPI_CHECK(MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN,
                            communicator));
    MPI_CHECK(MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX,
                            communicator));
    MPI_CHECK(MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX,
                            communicator));

    if (globalInvalid != 0) {
        if (rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }
    if (rank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    }
    if (globalMax > 1e6 || globalMin < -1e6) {
        if (rank == 0) {
            std::printf("Validation failed: values out of expected range\n");
        }
        return false;
    }
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
}

bool parseSize(const char* text, std::size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || text[0] == '-') {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return static_cast<unsigned long long>(value) == parsed;
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

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI does not provide the required FUNNELED thread level\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    std::size_t nx = 128;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 10;
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
            argumentsValid = false;
            if (rank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    std::size_t xy = 0;
    std::size_t globalCells = 0;
    argumentsValid = argumentsValid && nx >= 2 && ny >= 2 && nz >= 2 &&
                     checkedMultiply(nx, ny, xy) &&
                     checkedMultiply(xy, nz, globalCells);
    if (!argumentsValid) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Invalid arguments: dimensions must be at least 2 and all "
                         "values must fit their supported ranges\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    std::array<int, 3> processGrid{};
    if (!chooseProcessGrid(ranks, nx, ny, nz, processGrid)) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Cannot map %d MPI ranks to nonempty rectangular subdomains\n",
                         ranks);
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[3] = {processGrid[0], processGrid[1], processGrid[2]};
    int periods[3] = {0, 0, 0};
    MPI_Comm cartesian = MPI_COMM_NULL;
    MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions, periods, 0,
                              &cartesian));
    MPI_CHECK(MPI_Comm_set_errhandler(cartesian, MPI_ERRORS_RETURN));
    int coordinates[3] = {};
    MPI_CHECK(MPI_Cart_coords(cartesian, rank, 3, coordinates));

    const AxisPartition xPart = partitionAxis(nx, processGrid[0], coordinates[0]);
    const AxisPartition yPart = partitionAxis(ny, processGrid[1], coordinates[1]);
    const AxisPartition zPart = partitionAxis(nz, processGrid[2], coordinates[2]);
    const std::size_t lx = xPart.size;
    const std::size_t ly = yPart.size;
    const std::size_t lz = zPart.size;
    const std::size_t mx = lx + 2;
    const std::size_t my = ly + 2;
    const std::size_t mz = lz + 2;

    std::array<int, FaceCount> neighbors{};
    MPI_CHECK(MPI_Cart_shift(cartesian, 0, 1, &neighbors[XMinus],
                             &neighbors[XPlus]));
    MPI_CHECK(MPI_Cart_shift(cartesian, 1, 1, &neighbors[YMinus],
                             &neighbors[YPlus]));
    MPI_CHECK(MPI_Cart_shift(cartesian, 2, 1, &neighbors[ZMinus],
                             &neighbors[ZPlus]));

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localRanks = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localRanks));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0 || localRanks > deviceCount) {
        if (localRank == 0) {
            std::fprintf(stderr,
                         "This program requires one CUDA device per local MPI rank "
                         "(%d ranks, %d visible devices)\n",
                         localRanks, deviceCount);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const int device = localRank;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    cudaFuncAttributes stencilAttributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&stencilAttributes, stencilRegion));
    CUDA_CHECK(cudaFuncSetCacheConfig(stencilRegion, cudaFuncCachePreferL1));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    omp_set_dynamic(0);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks (%d x %d x %d), up to %d "
                    "OpenMP threads/rank, CUDA %s\n",
                    ranks, processGrid[0], processGrid[1], processGrid[2],
                    omp_get_max_threads(), deviceProperties.name);
        std::printf("Initializing grid...\n");
    }

    std::size_t localPlane = 0;
    std::size_t localAllocatedCells = 0;
    if (!checkedMultiply(mx, my, localPlane) ||
        !checkedMultiply(localPlane, mz, localAllocatedCells)) {
        std::fprintf(stderr, "Rank %d: local allocation size overflow\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    std::vector<Real> initial(localAllocatedCells, Real{0.0});
    const std::int64_t initializeX = static_cast<std::int64_t>(lx);
    const std::int64_t initializeY = static_cast<std::int64_t>(ly);
    const std::int64_t initializeZ = static_cast<std::int64_t>(lz);
#pragma omp parallel for collapse(3) schedule(static)
    for (std::int64_t z = 1; z <= initializeZ; ++z) {
        for (std::int64_t y = 1; y <= initializeY; ++y) {
            for (std::int64_t x = 1; x <= initializeX; ++x) {
                const std::size_t globalX = xPart.begin + static_cast<std::size_t>(x - 1);
                const std::size_t globalY = yPart.begin + static_cast<std::size_t>(y - 1);
                const std::size_t globalZ = zPart.begin + static_cast<std::size_t>(z - 1);
                const std::size_t globalIndex = (globalZ * ny + globalY) * nx + globalX;
                initial[localIndex(static_cast<std::size_t>(x),
                                   static_cast<std::size_t>(y),
                                   static_cast<std::size_t>(z), mx, my)] =
                    static_cast<Real>(globalIndex % 19);
            }
        }
    }

    Real* deviceGridA = nullptr;
    Real* deviceGridB = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGridA),
                          localAllocatedCells * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGridB),
                          localAllocatedCells * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(deviceGridA, initial.data(),
                          localAllocatedCells * sizeof(Real),
                          cudaMemcpyHostToDevice));
    initial.clear();
    initial.shrink_to_fit();

    HaloBuffers haloBuffers;
    allocateHaloBuffers(haloBuffers, neighbors, mx, my, mz);
    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));

    Real* current = deviceGridA;
    Real* next = deviceGridB;
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(cartesian));
    if (rank == 0) {
        std::printf("Running stencil computation...\n");
    }
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::array<MPI_Request, FaceCount * 2> requests{};
        int requestCount = 0;
        for (int face = 0; face < FaceCount; ++face) {
            if (neighbors[face] != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(haloBuffers.receive[face],
                                    static_cast<int>(haloBuffers.count[face]), MPI_DOUBLE,
                                    neighbors[face], oppositeFace(face), cartesian,
                                    &requests[requestCount++]));
            }
        }

        packHalos(current, haloBuffers, neighbors, lx, ly, lz, mx, my, mz,
                   communicationStream);

        // The strict interior does not consume halo cells and can overlap all
        // device-to-host copies and the subsequent MPI transfers.
        if (lx > 2 && ly > 2 && lz > 2) {
            launchRegion(current, next, mx, my, nx, ny, nz, xPart, yPart, zPart,
                         2, lx - 1, 2, ly - 1, 2, lz - 1, computeStream);
        }

        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
        for (int face = 0; face < FaceCount; ++face) {
            if (neighbors[face] != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend(haloBuffers.send[face],
                                    static_cast<int>(haloBuffers.count[face]), MPI_DOUBLE,
                                    neighbors[face], face, cartesian,
                                    &requests[requestCount++]));
            }
        }
        if (requestCount != 0) {
            MPI_CHECK(MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE));
        }

        unpackHalos(current, haloBuffers, neighbors, lx, ly, lz, mx, my, mz,
                     communicationStream);
        launchShell(current, next, mx, my, nx, ny, nz, xPart, yPart, zPart,
                    communicationStream);

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
        std::swap(current, next);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         cartesian));

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        const long double xUpdates = nx > 2 ? static_cast<long double>(nx - 2) : 0.0L;
        const long double yUpdates = ny > 2 ? static_cast<long double>(ny - 2) : 0.0L;
        const long double zUpdates = nz > 2 ? static_cast<long double>(nz - 2) : 0.0L;
        const long double updates =
            xUpdates * yUpdates * zUpdates * static_cast<long double>(iterations);
        const double mcups = elapsed > 0.0 ? static_cast<double>(updates / elapsed / 1e6L)
                                           : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> localResult;
    if (printResults || validate) {
        localResult = downloadOwned(current, lx, ly, lz, mx, my);
    }
    if (printResults) {
        std::vector<Real> globalResult =
            gatherGlobal(localResult, nx, ny, nz, processGrid, rank, ranks, cartesian);
        if (rank == 0) {
            print_results(globalResult, "Grid");
        }
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateDistributed(localResult, rank, cartesian);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    freeHaloBuffers(haloBuffers);
    CUDA_CHECK(cudaFree(deviceGridA));
    CUDA_CHECK(cudaFree(deviceGridB));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_CHECK(MPI_Comm_free(&cartesian));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
