#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kBlockSize = 256;
constexpr int kFaceCount = 6;

enum Face : int {
    XMinus = 0,
    XPlus = 1,
    YMinus = 2,
    YPlus = 3,
    ZMinus = 4,
    ZPlus = 5,
};

struct Extent {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t x0;
    size_t y0;
    size_t z0;
};

struct DeviceGrid {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t pitchX;
    size_t pitchY;
    size_t storageCells;
};

struct HaloBuffers {
    std::array<double*, kFaceCount> send{};
    std::array<double*, kFaceCount> receive{};
    std::array<double*, kFaceCount> hostSend{};
    std::array<double*, kFaceCount> hostReceive{};
    std::array<size_t, kFaceCount> count{};
    bool deviceDirect = false;
};

[[noreturn]] void failCuda(const cudaError_t error, const char* expression, const char* file, const int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d while executing %s: %s\n", file, line, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

void checkMpi(const int error, const char* expression, const char* file, const int line) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char errorText[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(error, errorText, &length);
    std::fprintf(stderr, "MPI failure at %s:%d while executing %s: %.*s\n", file, line, expression, length,
                 errorText);
    MPI_Abort(MPI_COMM_WORLD, error);
}

#define CUDA_CHECK(expression)                                                                       \
    do {                                                                                             \
        const cudaError_t cudaError = (expression);                                                  \
        if (cudaError != cudaSuccess) {                                                              \
            failCuda(cudaError, #expression, __FILE__, __LINE__);                                   \
        }                                                                                            \
    } while (false)

#define MPI_CHECK(expression) checkMpi((expression), #expression, __FILE__, __LINE__)

constexpr int oppositeFace(const int face) {
    return (face & 1) == 0 ? face + 1 : face - 1;
}

constexpr size_t paddedIndex(const size_t x, const size_t y, const size_t z, const size_t pitchX,
                             const size_t pitchY) {
    return (z * pitchY + y) * pitchX + x;
}

size_t checkedProduct(const size_t first, const size_t second, const char* description) {
    if (first != 0 && second > std::numeric_limits<size_t>::max() / first) {
        std::fprintf(stderr, "Size overflow while calculating %s\n", description);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return first * second;
}

int checkedMpiCount(const size_t count, const char* description) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "%s exceeds the maximum MPI count supported by this benchmark\n", description);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(count);
}

__global__ void initializeConcentrationKernel(double* concentration, const size_t nx, const size_t ny,
                                              const size_t nz, const size_t x0, const size_t y0,
                                              const size_t z0, const size_t globalNx, const size_t globalNy,
                                              const size_t globalVolume, const size_t pitchX, const size_t pitchY) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localVolume = nx * ny * nz;
    if (linear >= localVolume) {
        return;
    }

    const size_t localZ = linear / (nx * ny);
    const size_t remainder = linear - localZ * nx * ny;
    const size_t localY = remainder / nx;
    const size_t localX = remainder - localY * nx;
    const size_t globalLinear = (z0 + localZ) * (globalNx * globalNy) + (y0 + localY) * globalNx + x0 + localX;
    const double pseudo = (((globalLinear + 1) * static_cast<size_t>(1299709)) % globalVolume) /
                          static_cast<double>(globalVolume);
    concentration[paddedIndex(localX + 1, localY + 1, localZ + 1, pitchX, pitchY)] = -1.0 + 2.0 * pseudo;
}

__global__ void chemicalPotentialKernel(const double* concentration, double* chemicalPotential, const size_t xFirst,
                                        const size_t xCount, const size_t yFirst, const size_t yCount,
                                        const size_t zFirst, const size_t zCount, const size_t pitchX,
                                        const size_t pitchY, const double invDx2,
                                        const double invDy2, const double invDz2, const double gamma,
                                        const double eAA, const double eBB, const double eAB) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = xCount * yCount * zCount;
    if (linear >= cells) {
        return;
    }

    const size_t localZ = zFirst + linear / (xCount * yCount);
    const size_t remainder = linear - (localZ - zFirst) * xCount * yCount;
    const size_t localY = yFirst + remainder / xCount;
    const size_t localX = xFirst + remainder - (localY - yFirst) * xCount;
    const size_t index = paddedIndex(localX, localY, localZ, pitchX, pitchY);
    const double value = concentration[index];
    const double laplacian = (concentration[index + 1] + concentration[index - 1] - 2.0 * value) * invDx2 +
                             (concentration[index + pitchX] + concentration[index - pitchX] - 2.0 * value) * invDy2 +
                             (concentration[index + pitchX * pitchY] +
                              concentration[index - pitchX * pitchY] - 2.0 * value) * invDz2;

    chemicalPotential[index] = 4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB) +
                               3.0 * value + value * value * value - gamma * laplacian;
}

__global__ void updateConcentrationKernel(const double* concentration, const double* chemicalPotential,
                                          double* updatedConcentration, const size_t xFirst, const size_t xCount,
                                          const size_t yFirst, const size_t yCount, const size_t zFirst,
                                          const size_t zCount, const size_t pitchX, const size_t pitchY,
                                          const double invDx2, const double invDy2,
                                          const double invDz2, const double diffusion, const double dt) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = xCount * yCount * zCount;
    if (linear >= cells) {
        return;
    }

    const size_t localZ = zFirst + linear / (xCount * yCount);
    const size_t remainder = linear - (localZ - zFirst) * xCount * yCount;
    const size_t localY = yFirst + remainder / xCount;
    const size_t localX = xFirst + remainder - (localY - yFirst) * xCount;
    const size_t index = paddedIndex(localX, localY, localZ, pitchX, pitchY);
    const double value = chemicalPotential[index];
    const double laplacian = (chemicalPotential[index + 1] + chemicalPotential[index - 1] - 2.0 * value) * invDx2 +
                             (chemicalPotential[index + pitchX] + chemicalPotential[index - pitchX] - 2.0 * value) * invDy2 +
                             (chemicalPotential[index + pitchX * pitchY] +
                              chemicalPotential[index - pitchX * pitchY] - 2.0 * value) * invDz2;
    updatedConcentration[index] = concentration[index] + dt * diffusion * laplacian;
}

__global__ void packFaceKernel(const double* field, double* packed, const int face, const size_t nx, const size_t ny,
                               const size_t nz, const size_t pitchX, const size_t pitchY) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t faceCells = face < YMinus ? ny * nz : (face < ZMinus ? nx * nz : nx * ny);
    if (linear >= faceCells) {
        return;
    }

    size_t x = 1;
    size_t y = 1;
    size_t z = 1;
    if (face < YMinus) {
        z += linear / ny;
        y += linear - (z - 1) * ny;
        x = face == XMinus ? 1 : nx;
    } else if (face < ZMinus) {
        z += linear / nx;
        x += linear - (z - 1) * nx;
        y = face == YMinus ? 1 : ny;
    } else {
        y += linear / nx;
        x += linear - (y - 1) * nx;
        z = face == ZMinus ? 1 : nz;
    }
    packed[linear] = field[paddedIndex(x, y, z, pitchX, pitchY)];
}

__global__ void unpackFaceKernel(double* field, const double* packed, const int face, const size_t nx,
                                 const size_t ny, const size_t nz, const size_t pitchX, const size_t pitchY) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t faceCells = face < YMinus ? ny * nz : (face < ZMinus ? nx * nz : nx * ny);
    if (linear >= faceCells) {
        return;
    }

    size_t x = 1;
    size_t y = 1;
    size_t z = 1;
    if (face < YMinus) {
        z += linear / ny;
        y += linear - (z - 1) * ny;
        x = face == XMinus ? 0 : nx + 1;
    } else if (face < ZMinus) {
        z += linear / nx;
        x += linear - (z - 1) * nx;
        y = face == YMinus ? 0 : ny + 1;
    } else {
        y += linear / nx;
        x += linear - (y - 1) * nx;
        z = face == ZMinus ? 0 : nz + 1;
    }
    field[paddedIndex(x, y, z, pitchX, pitchY)] = packed[linear];
}

__global__ void clampBoundaryFaceKernel(double* field, const int face, const size_t nx, const size_t ny,
                                        const size_t nz, const size_t pitchX, const size_t pitchY) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t faceCells = face < YMinus ? ny * nz : (face < ZMinus ? nx * nz : nx * ny);
    if (linear >= faceCells) {
        return;
    }

    size_t sourceX = 1;
    size_t sourceY = 1;
    size_t sourceZ = 1;
    size_t targetX = 1;
    size_t targetY = 1;
    size_t targetZ = 1;
    if (face < YMinus) {
        sourceZ = targetZ = 1 + linear / ny;
        sourceY = targetY = 1 + linear - (sourceZ - 1) * ny;
        sourceX = face == XMinus ? 1 : nx;
        targetX = face == XMinus ? 0 : nx + 1;
    } else if (face < ZMinus) {
        sourceZ = targetZ = 1 + linear / nx;
        sourceX = targetX = 1 + linear - (sourceZ - 1) * nx;
        sourceY = face == YMinus ? 1 : ny;
        targetY = face == YMinus ? 0 : ny + 1;
    } else {
        sourceY = targetY = 1 + linear / nx;
        sourceX = targetX = 1 + linear - (sourceY - 1) * nx;
        sourceZ = face == ZMinus ? 1 : nz;
        targetZ = face == ZMinus ? 0 : nz + 1;
    }
    field[paddedIndex(targetX, targetY, targetZ, pitchX, pitchY)] =
        field[paddedIndex(sourceX, sourceY, sourceZ, pitchX, pitchY)];
}

__global__ void packInteriorKernel(const double* field, double* packed, const size_t nx, const size_t ny,
                                   const size_t nz, const size_t pitchX, const size_t pitchY) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = nx * ny * nz;
    if (linear >= cells) {
        return;
    }
    const size_t z = linear / (nx * ny);
    const size_t remainder = linear - z * nx * ny;
    const size_t y = remainder / nx;
    const size_t x = remainder - y * nx;
    packed[linear] = field[paddedIndex(x + 1, y + 1, z + 1, pitchX, pitchY)];
}

unsigned int gridSizeFor(const size_t cells) {
    const size_t blocks = (cells + kBlockSize - 1) / kBlockSize;
    if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        std::fprintf(stderr, "CUDA launch grid is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<unsigned int>(blocks);
}

void launchFacePack(const double* field, double* packed, const int face, const DeviceGrid& grid, cudaStream_t stream) {
    const size_t cells = face < YMinus ? grid.ny * grid.nz : (face < ZMinus ? grid.nx * grid.nz : grid.nx * grid.ny);
    packFaceKernel<<<gridSizeFor(cells), kBlockSize, 0, stream>>>(field, packed, face, grid.nx, grid.ny, grid.nz,
                                                                   grid.pitchX, grid.pitchY);
    CUDA_CHECK(cudaGetLastError());
}

void launchFaceUnpack(double* field, const double* packed, const int face, const DeviceGrid& grid, cudaStream_t stream) {
    const size_t cells = face < YMinus ? grid.ny * grid.nz : (face < ZMinus ? grid.nx * grid.nz : grid.nx * grid.ny);
    unpackFaceKernel<<<gridSizeFor(cells), kBlockSize, 0, stream>>>(field, packed, face, grid.nx, grid.ny, grid.nz,
                                                                     grid.pitchX, grid.pitchY);
    CUDA_CHECK(cudaGetLastError());
}

void launchBoundaryClamp(double* field, const int face, const DeviceGrid& grid, cudaStream_t stream) {
    const size_t cells = face < YMinus ? grid.ny * grid.nz : (face < ZMinus ? grid.nx * grid.nz : grid.nx * grid.ny);
    clampBoundaryFaceKernel<<<gridSizeFor(cells), kBlockSize, 0, stream>>>(field, face, grid.nx, grid.ny, grid.nz,
                                                                            grid.pitchX, grid.pitchY);
    CUDA_CHECK(cudaGetLastError());
}

void beginHaloExchange(const double* field, const DeviceGrid& grid, const std::array<int, kFaceCount>& neighbours,
                       HaloBuffers& buffers, MPI_Comm cartesianCommunicator, cudaStream_t communicationStream,
                       std::array<MPI_Request, 2 * kFaceCount>& requests, int& requestCount) {
    for (int face = 0; face < kFaceCount; ++face) {
        if (neighbours[face] == MPI_PROC_NULL) {
            launchBoundaryClamp(const_cast<double*>(field), face, grid, communicationStream);
        } else {
            launchFacePack(field, buffers.send[face], face, grid, communicationStream);
        }
    }

    // MPI must not observe a pack kernel that is still executing.  When the
    // MPI library has no CUDA support, this also serializes the D2H staging.
    CUDA_CHECK(cudaStreamSynchronize(communicationStream));

    if (!buffers.deviceDirect) {
        for (int face = 0; face < kFaceCount; ++face) {
            if (neighbours[face] != MPI_PROC_NULL) {
                const size_t bytes = checkedProduct(buffers.count[face], sizeof(double), "staged halo bytes");
                CUDA_CHECK(cudaMemcpyAsync(buffers.hostSend[face], buffers.send[face], bytes, cudaMemcpyDeviceToHost,
                                           communicationStream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(communicationStream));
    }

    requestCount = 0;
    for (int face = 0; face < kFaceCount; ++face) {
        if (neighbours[face] == MPI_PROC_NULL) {
            continue;
        }
        const int count = checkedMpiCount(buffers.count[face], "Halo face");
        double* receiveBuffer = buffers.deviceDirect ? buffers.receive[face] : buffers.hostReceive[face];
        double* sendBuffer = buffers.deviceDirect ? buffers.send[face] : buffers.hostSend[face];
        MPI_CHECK(MPI_Irecv(receiveBuffer, count, MPI_DOUBLE, neighbours[face], 100 + oppositeFace(face),
                            cartesianCommunicator, &requests[requestCount++]));
        MPI_CHECK(MPI_Isend(sendBuffer, count, MPI_DOUBLE, neighbours[face], 100 + face, cartesianCommunicator,
                            &requests[requestCount++]));
    }
}

void finishHaloExchange(double* field, const DeviceGrid& grid, const std::array<int, kFaceCount>& neighbours,
                        HaloBuffers& buffers, cudaStream_t communicationStream, cudaEvent_t halosReady,
                        std::array<MPI_Request, 2 * kFaceCount>& requests, const int requestCount) {
    MPI_CHECK(MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE));
    if (!buffers.deviceDirect) {
        for (int face = 0; face < kFaceCount; ++face) {
            if (neighbours[face] != MPI_PROC_NULL) {
                const size_t bytes = checkedProduct(buffers.count[face], sizeof(double), "staged halo bytes");
                CUDA_CHECK(cudaMemcpyAsync(buffers.receive[face], buffers.hostReceive[face], bytes,
                                           cudaMemcpyHostToDevice, communicationStream));
            }
        }
    }
    for (int face = 0; face < kFaceCount; ++face) {
        if (neighbours[face] != MPI_PROC_NULL) {
            launchFaceUnpack(field, buffers.receive[face], face, grid, communicationStream);
        }
    }
    CUDA_CHECK(cudaEventRecord(halosReady, communicationStream));
}

void launchChemicalBox(const double* concentration, double* chemicalPotential, const DeviceGrid& grid,
                       const size_t xFirst, const size_t xCount, const size_t yFirst, const size_t yCount,
                       const size_t zFirst, const size_t zCount, const double invDx2, const double invDy2,
                       const double invDz2, const double gamma, const double eAA, const double eBB,
                       const double eAB, cudaStream_t stream) {
    if (xCount == 0 || yCount == 0 || zCount == 0) {
        return;
    }
    const size_t cells = xCount * yCount * zCount;
    chemicalPotentialKernel<<<gridSizeFor(cells), kBlockSize, 0, stream>>>(
        concentration, chemicalPotential, xFirst, xCount, yFirst, yCount, zFirst, zCount, grid.pitchX, grid.pitchY,
        invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdateBox(const double* concentration, const double* chemicalPotential, double* updatedConcentration,
                     const DeviceGrid& grid, const size_t xFirst, const size_t xCount, const size_t yFirst,
                     const size_t yCount, const size_t zFirst, const size_t zCount, const double invDx2,
                     const double invDy2, const double invDz2, const double diffusion, const double dt,
                     cudaStream_t stream) {
    if (xCount == 0 || yCount == 0 || zCount == 0) {
        return;
    }
    const size_t cells = xCount * yCount * zCount;
    updateConcentrationKernel<<<gridSizeFor(cells), kBlockSize, 0, stream>>>(
        concentration, chemicalPotential, updatedConcentration, xFirst, xCount, yFirst, yCount, zFirst, zCount,
        grid.pitchX, grid.pitchY, invDx2, invDy2, invDz2, diffusion, dt);
    CUDA_CHECK(cudaGetLastError());
}

void launchBoundaryBoxesForChemical(const double* concentration, double* chemicalPotential, const DeviceGrid& grid,
                                    const double invDx2, const double invDy2, const double invDz2,
                                    const double gamma, const double eAA, const double eBB, const double eAB,
                                    cudaStream_t stream) {
    launchChemicalBox(concentration, chemicalPotential, grid, 1, 1, 1, grid.ny, 1, grid.nz, invDx2, invDy2, invDz2,
                      gamma, eAA, eBB, eAB, stream);
    if (grid.nx > 1) {
        launchChemicalBox(concentration, chemicalPotential, grid, grid.nx, 1, 1, grid.ny, 1, grid.nz, invDx2, invDy2,
                          invDz2, gamma, eAA, eBB, eAB, stream);
    }
    const size_t innerX = grid.nx > 2 ? grid.nx - 2 : 0;
    launchChemicalBox(concentration, chemicalPotential, grid, 2, innerX, 1, 1, 1, grid.nz, invDx2, invDy2, invDz2,
                      gamma, eAA, eBB, eAB, stream);
    if (grid.ny > 1) {
        launchChemicalBox(concentration, chemicalPotential, grid, 2, innerX, grid.ny, 1, 1, grid.nz, invDx2, invDy2,
                          invDz2, gamma, eAA, eBB, eAB, stream);
    }
    const size_t innerY = grid.ny > 2 ? grid.ny - 2 : 0;
    launchChemicalBox(concentration, chemicalPotential, grid, 2, innerX, 2, innerY, 1, 1, invDx2, invDy2, invDz2,
                      gamma, eAA, eBB, eAB, stream);
    if (grid.nz > 1) {
        launchChemicalBox(concentration, chemicalPotential, grid, 2, innerX, 2, innerY, grid.nz, 1, invDx2, invDy2,
                          invDz2, gamma, eAA, eBB, eAB, stream);
    }
}

void launchBoundaryBoxesForUpdate(const double* concentration, const double* chemicalPotential,
                                  double* updatedConcentration, const DeviceGrid& grid, const double invDx2,
                                  const double invDy2, const double invDz2, const double diffusion, const double dt,
                                  cudaStream_t stream) {
    launchUpdateBox(concentration, chemicalPotential, updatedConcentration, grid, 1, 1, 1, grid.ny, 1, grid.nz,
                    invDx2, invDy2, invDz2, diffusion, dt, stream);
    if (grid.nx > 1) {
        launchUpdateBox(concentration, chemicalPotential, updatedConcentration, grid, grid.nx, 1, 1, grid.ny, 1,
                        grid.nz, invDx2, invDy2, invDz2, diffusion, dt, stream);
    }
    const size_t innerX = grid.nx > 2 ? grid.nx - 2 : 0;
    launchUpdateBox(concentration, chemicalPotential, updatedConcentration, grid, 2, innerX, 1, 1, 1, grid.nz,
                    invDx2, invDy2, invDz2, diffusion, dt, stream);
    if (grid.ny > 1) {
        launchUpdateBox(concentration, chemicalPotential, updatedConcentration, grid, 2, innerX, grid.ny, 1, 1,
                        grid.nz, invDx2, invDy2, invDz2, diffusion, dt, stream);
    }
    const size_t innerY = grid.ny > 2 ? grid.ny - 2 : 0;
    launchUpdateBox(concentration, chemicalPotential, updatedConcentration, grid, 2, innerX, 2, innerY, 1, 1,
                    invDx2, invDy2, invDz2, diffusion, dt, stream);
    if (grid.nz > 1) {
        launchUpdateBox(concentration, chemicalPotential, updatedConcentration, grid, 2, innerX, 2, innerY, grid.nz,
                        1, invDx2, invDy2, invDz2, diffusion, dt, stream);
    }
}

Extent localExtent(const std::array<size_t, 3>& global, const std::array<int, 3>& dimensions,
                   const std::array<int, 3>& coordinates) {
    Extent result{};
    size_t* localSizes[3] = {&result.nx, &result.ny, &result.nz};
    size_t* starts[3] = {&result.x0, &result.y0, &result.z0};
    for (int axis = 0; axis < 3; ++axis) {
        const size_t partitions = static_cast<size_t>(dimensions[axis]);
        const size_t base = global[axis] / partitions;
        const size_t remainder = global[axis] % partitions;
        const size_t coordinate = static_cast<size_t>(coordinates[axis]);
        *localSizes[axis] = base + (coordinate < remainder ? 1 : 0);
        *starts[axis] = coordinate * base + std::min(coordinate, remainder);
    }
    return result;
}

DeviceGrid makeDeviceGrid(const Extent& local) {
    DeviceGrid grid{};
    grid.nx = local.nx;
    grid.ny = local.ny;
    grid.nz = local.nz;
    grid.pitchX = grid.nx + 2;
    grid.pitchY = grid.ny + 2;
    grid.storageCells = checkedProduct(checkedProduct(grid.pitchX, grid.pitchY, "padded XY plane"), grid.nz + 2,
                                       "padded local volume");
    return grid;
}

size_t faceCells(const DeviceGrid& grid, const int face) {
    if (face < YMinus) {
        return grid.ny * grid.nz;
    }
    if (face < ZMinus) {
        return grid.nx * grid.nz;
    }
    return grid.nx * grid.ny;
}

void allocateHaloBuffers(HaloBuffers& buffers, const DeviceGrid& grid,
                         const std::array<int, kFaceCount>& neighbours) {
    for (int face = 0; face < kFaceCount; ++face) {
        buffers.count[face] = faceCells(grid, face);
        if (neighbours[face] != MPI_PROC_NULL) {
            const size_t bytes = checkedProduct(buffers.count[face], sizeof(double), "halo buffer bytes");
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.send[face]), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.receive[face]), bytes));
            if (!buffers.deviceDirect) {
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.hostSend[face]), bytes, cudaHostAllocDefault));
                CUDA_CHECK(
                    cudaHostAlloc(reinterpret_cast<void**>(&buffers.hostReceive[face]), bytes, cudaHostAllocDefault));
            }
        }
    }
}

void freeHaloBuffers(HaloBuffers& buffers) {
    for (int face = 0; face < kFaceCount; ++face) {
        if (buffers.send[face] != nullptr) {
            CUDA_CHECK(cudaFree(buffers.send[face]));
        }
        if (buffers.receive[face] != nullptr) {
            CUDA_CHECK(cudaFree(buffers.receive[face]));
        }
        if (buffers.hostSend[face] != nullptr) {
            CUDA_CHECK(cudaFreeHost(buffers.hostSend[face]));
        }
        if (buffers.hostReceive[face] != nullptr) {
            CUDA_CHECK(cudaFreeHost(buffers.hostReceive[face]));
        }
    }
}

bool cudaAwareMpiAvailable() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

std::vector<double> downloadInterior(const double* field, const DeviceGrid& grid, cudaStream_t stream) {
    const size_t cells = checkedProduct(checkedProduct(grid.nx, grid.ny, "local result XY plane"), grid.nz,
                                        "local result volume");
    const size_t bytes = checkedProduct(cells, sizeof(double), "local result bytes");
    double* packedDevice = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&packedDevice), bytes));
    packInteriorKernel<<<gridSizeFor(cells), kBlockSize, 0, stream>>>(field, packedDevice, grid.nx, grid.ny, grid.nz,
                                                                        grid.pitchX, grid.pitchY);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> host(cells);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), packedDevice, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(packedDevice));
    return host;
}

bool validateResult(const std::vector<double>& localResult, MPI_Comm communicator, const int rank) {
    int localFinite = 1;
    double localMinimum = std::numeric_limits<double>::infinity();
    double localMaximum = -std::numeric_limits<double>::infinity();

#pragma omp parallel
    {
        int threadFinite = 1;
        double threadMinimum = std::numeric_limits<double>::infinity();
        double threadMaximum = -std::numeric_limits<double>::infinity();
#pragma omp for nowait schedule(static)
        for (size_t index = 0; index < localResult.size(); ++index) {
            const double value = localResult[index];
            if (!std::isfinite(value)) {
                threadFinite = 0;
            } else {
                threadMinimum = std::min(threadMinimum, value);
                threadMaximum = std::max(threadMaximum, value);
            }
        }
#pragma omp critical
        {
            localFinite = std::min(localFinite, threadFinite);
            localMinimum = std::min(localMinimum, threadMinimum);
            localMaximum = std::max(localMaximum, threadMaximum);
        }
    }

    int globallyFinite = 0;
    double globalMinimum = 0.0;
    double globalMaximum = 0.0;
    MPI_CHECK(MPI_Allreduce(&localFinite, &globallyFinite, 1, MPI_INT, MPI_MIN, communicator));
    MPI_CHECK(MPI_Reduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, 0, communicator));
    MPI_CHECK(MPI_Reduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, 0, communicator));

    int result = 1;
    if (rank == 0) {
        if (globallyFinite == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
            result = 0;
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
        }
        if (result != 0 && (globalMaximum > 10.0 || globalMinimum < -10.0)) {
            std::printf("Validation failed: values out of expected range\n");
            result = 0;
        }
    }
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, communicator));
    return result != 0;
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
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return 1;
    }
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int worldRank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-x") == 0 && argument + 1 < argc) {
            nx = static_cast<size_t>(std::atoi(argv[++argument]));
        } else if (std::strcmp(argv[argument], "-y") == 0 && argument + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++argument]));
        } else if (std::strcmp(argv[argument], "-z") == 0 && argument + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++argument]));
        } else if (std::strcmp(argv[argument], "-i") == 0 && argument + 1 < argc) {
            iterations = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (worldRank == 0) {
            if (parseError) {
                std::printf("Unknown or incomplete option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Grid dimensions must be positive and time steps must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (worldRank == 0) {
            std::fprintf(stderr, "No CUDA device is visible to this MPI job\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // Establish the selected device before any device-direct MPI operation.

    std::array<size_t, 3> global = {nx, ny, nz};
    std::array<int, 3> dimensions = {0, 0, 0};
    MPI_CHECK(MPI_Dims_create(worldSize, 3, dimensions.data()));
    for (int axis = 0; axis < 3; ++axis) {
        if (static_cast<size_t>(dimensions[axis]) > global[axis]) {
            if (worldRank == 0) {
                std::fprintf(stderr,
                             "MPI Cartesian decomposition would create an empty local domain; use no more ranks or a larger grid\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const std::array<int, 3> periodic = {0, 0, 0};
    MPI_Comm cartesianCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions.data(), periodic.data(), 0, &cartesianCommunicator));
    int cartesianRank = 0;
    std::array<int, 3> coordinates{};
    MPI_CHECK(MPI_Comm_rank(cartesianCommunicator, &cartesianRank));
    MPI_CHECK(MPI_Cart_coords(cartesianCommunicator, cartesianRank, 3, coordinates.data()));

    const Extent local = localExtent(global, dimensions, coordinates);
    const DeviceGrid grid = makeDeviceGrid(local);
    const size_t localCells = checkedProduct(checkedProduct(grid.nx, grid.ny, "local XY plane"), grid.nz,
                                             "local volume");
    const size_t globalCells = checkedProduct(checkedProduct(nx, ny, "global XY plane"), nz, "global volume");
    const size_t allocationBytes = checkedProduct(grid.storageCells, sizeof(double), "device field bytes");

    std::array<int, kFaceCount> neighbours{};
    for (int axis = 0; axis < 3; ++axis) {
        MPI_CHECK(MPI_Cart_shift(cartesianCommunicator, axis, 1, &neighbours[2 * axis], &neighbours[2 * axis + 1]));
    }

    int openmpThreads = 1;
#pragma omp parallel
    {
#pragma omp single
        openmpThreads = omp_get_num_threads();
    }

    if (worldRank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks, %d OpenMP threads/rank, CUDA %s halos\n", worldSize,
                    openmpThreads, cudaAwareMpiAvailable() ? "device-direct" : "pinned-staged");
        std::printf("Initializing concentration field...\n");
    }

    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    cudaEvent_t halosReady = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));

    double* concentration = nullptr;
    double* updatedConcentration = nullptr;
    double* chemicalPotential = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&concentration), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&updatedConcentration), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&chemicalPotential), allocationBytes));
    CUDA_CHECK(cudaMemsetAsync(concentration, 0, allocationBytes, computeStream));
    CUDA_CHECK(cudaMemsetAsync(updatedConcentration, 0, allocationBytes, computeStream));
    CUDA_CHECK(cudaMemsetAsync(chemicalPotential, 0, allocationBytes, computeStream));
    initializeConcentrationKernel<<<gridSizeFor(localCells), kBlockSize, 0, computeStream>>>(
        concentration, grid.nx, grid.ny, grid.nz, local.x0, local.y0, local.z0, nx, ny, globalCells, grid.pitchX,
        grid.pitchY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    HaloBuffers haloBuffers;
    int localCudaAwareMpi = cudaAwareMpiAvailable() ? 1 : 0;
    int cudaAwareMpi = 0;
    MPI_CHECK(MPI_Allreduce(&localCudaAwareMpi, &cudaAwareMpi, 1, MPI_INT, MPI_MIN, cartesianCommunicator));
    haloBuffers.deviceDirect = cudaAwareMpi != 0;
    allocateHaloBuffers(haloBuffers, grid, neighbours);

    // Physical parameters are retained exactly from the serial benchmark.
    const double invDx2 = 1.0;
    const double invDy2 = 1.0;
    const double invDz2 = 1.0;
    const double dt = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = 2.0 / 9.0;
    const double gamma = 0.5;
    const double diffusion = 1.0;

    if (worldRank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_CHECK(MPI_Barrier(cartesianCommunicator));
    const double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        std::array<MPI_Request, 2 * kFaceCount> requests{};
        int requestCount = 0;
        beginHaloExchange(concentration, grid, neighbours, haloBuffers, cartesianCommunicator, communicationStream,
                          requests, requestCount);
        launchChemicalBox(concentration, chemicalPotential, grid, 2, grid.nx > 2 ? grid.nx - 2 : 0, 2,
                          grid.ny > 2 ? grid.ny - 2 : 0, 2, grid.nz > 2 ? grid.nz - 2 : 0, invDx2, invDy2, invDz2,
                          gamma, eAA, eBB, eAB, computeStream);
        finishHaloExchange(concentration, grid, neighbours, haloBuffers, communicationStream, halosReady, requests,
                           requestCount);
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, halosReady, 0));
        launchBoundaryBoxesForChemical(concentration, chemicalPotential, grid, invDx2, invDy2, invDz2, gamma, eAA,
                                       eBB, eAB, computeStream);
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        beginHaloExchange(chemicalPotential, grid, neighbours, haloBuffers, cartesianCommunicator, communicationStream,
                          requests, requestCount);
        launchUpdateBox(concentration, chemicalPotential, updatedConcentration, grid, 2, grid.nx > 2 ? grid.nx - 2 : 0,
                        2, grid.ny > 2 ? grid.ny - 2 : 0, 2, grid.nz > 2 ? grid.nz - 2 : 0, invDx2, invDy2, invDz2,
                        diffusion, dt, computeStream);
        finishHaloExchange(chemicalPotential, grid, neighbours, haloBuffers, communicationStream, halosReady, requests,
                           requestCount);
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, halosReady, 0));
        launchBoundaryBoxesForUpdate(concentration, chemicalPotential, updatedConcentration, grid, invDx2, invDy2,
                                     invDz2, diffusion, dt, computeStream);
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        std::swap(concentration, updatedConcentration);
    }
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, cartesianCommunicator));

    if (worldRank == 0) {
        const long elapsedMilliseconds = static_cast<long>(std::llround(elapsedSeconds * 1000.0));
        std::printf("Computation time: %ld ms\n", elapsedMilliseconds);
        const double mcups = elapsedSeconds > 0.0
                                 ? static_cast<double>(globalCells) * static_cast<double>(iterations) / elapsedSeconds / 1.0e6
                                 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> localResult;
    if (printResults || validate) {
        localResult = downloadInterior(concentration, grid, computeStream);
    }

    if (printResults) {
        const int sendCount = checkedMpiCount(localCells, "Local result");
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<double> gathered;
        if (worldRank == 0) {
            checkedMpiCount(globalCells, "Global result");
            receiveCounts.resize(worldSize);
        }
        MPI_CHECK(MPI_Gather(&sendCount, 1, MPI_INT, worldRank == 0 ? receiveCounts.data() : nullptr, 1, MPI_INT, 0,
                             cartesianCommunicator));
        if (worldRank == 0) {
            displacements.resize(worldSize);
            int displacement = 0;
            for (int rank = 0; rank < worldSize; ++rank) {
                displacements[rank] = displacement;
                if (receiveCounts[rank] > std::numeric_limits<int>::max() - displacement) {
                    std::fprintf(stderr, "Global result exceeds MPI_Gatherv displacement range\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                displacement += receiveCounts[rank];
            }
            gathered.resize(globalCells);
        }
        MPI_CHECK(MPI_Gatherv(localResult.data(), sendCount, MPI_DOUBLE, worldRank == 0 ? gathered.data() : nullptr,
                              worldRank == 0 ? receiveCounts.data() : nullptr,
                              worldRank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, cartesianCommunicator));

        if (worldRank == 0) {
            std::vector<double> globalResult(globalCells);
            for (int rank = 0; rank < worldSize; ++rank) {
                std::array<int, 3> rankCoordinates{};
                MPI_CHECK(MPI_Cart_coords(cartesianCommunicator, rank, 3, rankCoordinates.data()));
                const Extent rankExtent = localExtent(global, dimensions, rankCoordinates);
                const size_t sourceOffset = static_cast<size_t>(displacements[rank]);
                for (size_t localZ = 0; localZ < rankExtent.nz; ++localZ) {
                    for (size_t localY = 0; localY < rankExtent.ny; ++localY) {
                        for (size_t localX = 0; localX < rankExtent.nx; ++localX) {
                            const size_t source = sourceOffset + (localZ * rankExtent.ny + localY) * rankExtent.nx + localX;
                            const size_t destination = ((rankExtent.z0 + localZ) * ny + rankExtent.y0 + localY) * nx +
                                                       rankExtent.x0 + localX;
                            globalResult[destination] = gathered[source];
                        }
                    }
                }
            }
            print_results(globalResult, "Concentration");
        }
    }

    bool valid = true;
    if (validate) {
        if (worldRank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateResult(localResult, cartesianCommunicator, worldRank);
        if (worldRank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    freeHaloBuffers(haloBuffers);
    CUDA_CHECK(cudaFree(concentration));
    CUDA_CHECK(cudaFree(updatedConcentration));
    CUDA_CHECK(cudaFree(chemicalPotential));
    CUDA_CHECK(cudaEventDestroy(halosReady));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    MPI_CHECK(MPI_Comm_free(&cartesianCommunicator));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_Finalize();
    return valid ? 0 : 1;
}
