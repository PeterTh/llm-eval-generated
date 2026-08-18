#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int TAG_X_NEG = 10;
constexpr int TAG_X_POS = 11;
constexpr int TAG_Y_NEG = 12;
constexpr int TAG_Y_POS = 13;
constexpr int TAG_Z_NEG = 14;
constexpr int TAG_Z_POS = 15;
constexpr int TAG_GATHER = 20;

struct Block {
    size_t begin;
    size_t size;
};

struct FaceTypes {
    MPI_Datatype x = MPI_DATATYPE_NULL;
    MPI_Datatype y = MPI_DATATYPE_NULL;
    MPI_Datatype z = MPI_DATATYPE_NULL;
};

[[noreturn]] void abortMpi(const char* operation, const int errorCode) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(errorCode, message, &length);
    std::fprintf(stderr, "MPI error in %s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, errorCode);
    std::abort();
}

inline void mpiCheck(const int errorCode, const char* operation) {
    if (errorCode != MPI_SUCCESS) {
        abortMpi(operation, errorCode);
    }
}

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline constexpr size_t localIndex(const size_t x, const size_t y, const size_t z,
                                   const size_t pitch, const size_t slice) noexcept {
    return z * slice + y * pitch + x;
}

bool checkedMultiply(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool parseSize(const char* text, size_t& value) {
    if (text[0] == '-') {
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
    if (text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > INT_MAX) {
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

Block splitInterior(const size_t extent, const int parts, const int coordinate) {
    const size_t quotient = extent / static_cast<size_t>(parts);
    const size_t remainder = extent % static_cast<size_t>(parts);
    const size_t coord = static_cast<size_t>(coordinate);
    return {1 + coord * quotient + std::min(coord, remainder),
            quotient + (coord < remainder ? 1u : 0u)};
}

bool chooseProcessGrid(const int processCount, const size_t nx, const size_t ny,
                       const size_t nz, std::array<int, 3>& bestDimensions) {
    long double bestHalo = std::numeric_limits<long double>::infinity();
    long double bestVolume = std::numeric_limits<long double>::infinity();
    int bestSplitAxes = INT_MAX;
    bool found = false;

    for (int px = 1; px <= processCount; ++px) {
        if (processCount % px != 0 || static_cast<size_t>(px) > nx) {
            continue;
        }
        const int remaining = processCount / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0 || static_cast<size_t>(py) > ny) {
                continue;
            }
            const int pz = remaining / py;
            if (static_cast<size_t>(pz) > nz) {
                continue;
            }

            const long double lx = static_cast<long double>((nx + px - 1) / px);
            const long double ly = static_cast<long double>((ny + py - 1) / py);
            const long double lz = static_cast<long double>((nz + pz - 1) / pz);
            // A two-way split gives every rank one neighbor on that axis;
            // larger splits put two faces on the critical-path interior ranks.
            const int xFaces = px == 1 ? 0 : (px == 2 ? 1 : 2);
            const int yFaces = py == 1 ? 0 : (py == 2 ? 1 : 2);
            const int zFaces = pz == 1 ? 0 : (pz == 2 ? 1 : 2);
            const long double halo = xFaces * ly * lz +
                                     yFaces * lx * lz +
                                     zFaces * lx * ly;
            const long double volume = lx * ly * lz;
            const int splitAxes = (px > 1) + (py > 1) + (pz > 1);

            if (!found || halo < bestHalo ||
                (halo == bestHalo && volume < bestVolume) ||
                (halo == bestHalo && volume == bestVolume && splitAxes < bestSplitAxes)) {
                found = true;
                bestHalo = halo;
                bestVolume = volume;
                bestSplitAxes = splitAxes;
                bestDimensions = {px, py, pz};
            }
        }
    }
    return found;
}

int selectActiveProcesses(const int worldSize, const size_t nx, const size_t ny,
                          const size_t nz, std::array<int, 3>& dimensions) {
    size_t interiorCells = 0;
    size_t xy = 0;
    if (!checkedMultiply(nx, ny, xy) || !checkedMultiply(xy, nz, interiorCells)) {
        interiorCells = std::numeric_limits<size_t>::max();
    }
    const int upper = static_cast<int>(
        std::min(static_cast<size_t>(worldSize), interiorCells));
    for (int active = upper; active >= 1; --active) {
        if (chooseProcessGrid(active, nx, ny, nz, dimensions)) {
            return active;
        }
    }
    dimensions = {1, 1, 1};
    return 1;
}

void initializeLocal(Real* grid1, Real* grid2, const size_t localNx,
                     const size_t localNy, const size_t localNz,
                     const size_t startX, const size_t startY, const size_t startZ,
                     const size_t globalNx, const size_t globalNy) {
    const size_t pitch = localNx + 2;
    const size_t slice = pitch * (localNy + 2);
    const size_t globalSlice = globalNx * globalNy;
    for (size_t z = 0; z < localNz + 2; ++z) {
        const size_t globalZ = startZ + z - 1;
        for (size_t y = 0; y < localNy + 2; ++y) {
            const size_t globalY = startY + y - 1;
            const size_t globalBase = globalZ * globalSlice + globalY * globalNx;
            const size_t localBase = z * slice + y * pitch;
            for (size_t x = 0; x < localNx + 2; ++x) {
                const size_t globalX = startX + x - 1;
                const Real value = static_cast<Real>((globalBase + globalX) % 19);
                grid1[localBase + x] = value;
                grid2[localBase + x] = value;
            }
        }
    }
}

void initializeGlobal(std::vector<Real>& grid, const size_t nx, const size_t ny,
                      const size_t nz) {
    const size_t cells = nx * ny * nz;
    grid.resize(cells);
    for (size_t index = 0; index < cells; ++index) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

inline void updateBox(const Real* __restrict input, Real* __restrict output,
                      const size_t xBegin, const size_t xEnd,
                      const size_t yBegin, const size_t yEnd,
                      const size_t zBegin, const size_t zEnd,
                      const size_t pitch, const size_t slice) noexcept {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = yBegin; y < yEnd; ++y) {
            const size_t begin = localIndex(xBegin, y, z, pitch, slice);
            const size_t end = begin + (xEnd - xBegin);
#if defined(__clang__)
#pragma clang loop vectorize(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (size_t index = begin; index < end; ++index) {
                output[index] =
                    (input[index] + input[index - 1] + input[index + 1] +
                     input[index - pitch] + input[index + pitch] +
                     input[index - slice] + input[index + slice]) /
                    7.0;
            }
        }
    }
}

FaceTypes createFaceTypes(const size_t localNx, const size_t localNy,
                          const size_t localNz, const size_t pitch,
                          const size_t slice) {
    FaceTypes faces;
    MPI_Datatype column = MPI_DATATYPE_NULL;
    MPI_Datatype row = MPI_DATATYPE_NULL;
    const MPI_Aint pitchBytes = static_cast<MPI_Aint>(pitch * sizeof(Real));
    const MPI_Aint sliceBytes = static_cast<MPI_Aint>(slice * sizeof(Real));

    mpiCheck(MPI_Type_create_hvector(static_cast<int>(localNy), 1, pitchBytes,
                                     MPI_DOUBLE, &column),
             "MPI_Type_create_hvector(x column)");
    mpiCheck(MPI_Type_create_hvector(static_cast<int>(localNz), 1, sliceBytes,
                                     column, &faces.x),
             "MPI_Type_create_hvector(x face)");
    mpiCheck(MPI_Type_commit(&faces.x), "MPI_Type_commit(x face)");
    mpiCheck(MPI_Type_free(&column), "MPI_Type_free(x column)");

    mpiCheck(MPI_Type_contiguous(static_cast<int>(localNx), MPI_DOUBLE, &row),
             "MPI_Type_contiguous(face row)");
    mpiCheck(MPI_Type_create_hvector(static_cast<int>(localNz), 1, sliceBytes,
                                     row, &faces.y),
             "MPI_Type_create_hvector(y face)");
    mpiCheck(MPI_Type_commit(&faces.y), "MPI_Type_commit(y face)");
    mpiCheck(MPI_Type_create_hvector(static_cast<int>(localNy), 1, pitchBytes,
                                     row, &faces.z),
             "MPI_Type_create_hvector(z face)");
    mpiCheck(MPI_Type_commit(&faces.z), "MPI_Type_commit(z face)");
    mpiCheck(MPI_Type_free(&row), "MPI_Type_free(face row)");
    return faces;
}

void freeFaceTypes(FaceTypes& faces) {
    mpiCheck(MPI_Type_free(&faces.x), "MPI_Type_free(x face)");
    mpiCheck(MPI_Type_free(&faces.y), "MPI_Type_free(y face)");
    mpiCheck(MPI_Type_free(&faces.z), "MPI_Type_free(z face)");
}

std::vector<MPI_Request> createPersistentHalo(
    Real* grid, const size_t localNx, const size_t localNy, const size_t localNz,
    const size_t pitch, const size_t slice, const std::array<int, 6>& neighbors,
    const FaceTypes& faces, MPI_Comm communicator) {
    std::vector<MPI_Request> requests;
    requests.reserve(12);
    auto receive = [&](Real* address, MPI_Datatype type, const int source,
                       const int tag) {
        if (source != MPI_PROC_NULL) {
            requests.push_back(MPI_REQUEST_NULL);
            mpiCheck(MPI_Recv_init(address, 1, type, source, tag, communicator,
                                   &requests.back()),
                     "MPI_Recv_init");
        }
    };
    auto send = [&](Real* address, MPI_Datatype type, const int destination,
                    const int tag) {
        if (destination != MPI_PROC_NULL) {
            requests.push_back(MPI_REQUEST_NULL);
            mpiCheck(MPI_Send_init(address, 1, type, destination, tag, communicator,
                                   &requests.back()),
                     "MPI_Send_init");
        }
    };

    receive(grid + localIndex(0, 1, 1, pitch, slice), faces.x, neighbors[0], TAG_X_POS);
    receive(grid + localIndex(localNx + 1, 1, 1, pitch, slice), faces.x,
            neighbors[1], TAG_X_NEG);
    receive(grid + localIndex(1, 0, 1, pitch, slice), faces.y, neighbors[2], TAG_Y_POS);
    receive(grid + localIndex(1, localNy + 1, 1, pitch, slice), faces.y,
            neighbors[3], TAG_Y_NEG);
    receive(grid + localIndex(1, 1, 0, pitch, slice), faces.z, neighbors[4], TAG_Z_POS);
    receive(grid + localIndex(1, 1, localNz + 1, pitch, slice), faces.z,
            neighbors[5], TAG_Z_NEG);

    send(grid + localIndex(1, 1, 1, pitch, slice), faces.x, neighbors[0], TAG_X_NEG);
    send(grid + localIndex(localNx, 1, 1, pitch, slice), faces.x,
         neighbors[1], TAG_X_POS);
    send(grid + localIndex(1, 1, 1, pitch, slice), faces.y, neighbors[2], TAG_Y_NEG);
    send(grid + localIndex(1, localNy, 1, pitch, slice), faces.y,
         neighbors[3], TAG_Y_POS);
    send(grid + localIndex(1, 1, 1, pitch, slice), faces.z, neighbors[4], TAG_Z_NEG);
    send(grid + localIndex(1, 1, localNz, pitch, slice), faces.z,
         neighbors[5], TAG_Z_POS);
    return requests;
}

void freePersistentRequests(std::vector<MPI_Request>& requests) {
    for (MPI_Request& request : requests) {
        mpiCheck(MPI_Request_free(&request), "MPI_Request_free");
    }
}

void stencilIteration(const Real* input, Real* output,
                      std::vector<MPI_Request>& haloRequests,
                      const size_t localNx, const size_t localNy,
                      const size_t localNz, const size_t pitch, const size_t slice,
                      const std::array<int, 6>& neighbors) {
    if (!haloRequests.empty()) {
        mpiCheck(MPI_Startall(static_cast<int>(haloRequests.size()),
                              haloRequests.data()),
                 "MPI_Startall");
    }

    const size_t xBegin = 1;
    const size_t xEnd = localNx + 1;
    const size_t yBegin = 1;
    const size_t yEnd = localNy + 1;
    const size_t zBegin = 1;
    const size_t zEnd = localNz + 1;

    size_t coreXBegin = xBegin + (neighbors[0] != MPI_PROC_NULL ? 1u : 0u);
    size_t coreXEnd = xEnd - (neighbors[1] != MPI_PROC_NULL ? 1u : 0u);
    size_t coreYBegin = yBegin + (neighbors[2] != MPI_PROC_NULL ? 1u : 0u);
    size_t coreYEnd = yEnd - (neighbors[3] != MPI_PROC_NULL ? 1u : 0u);
    size_t coreZBegin = zBegin + (neighbors[4] != MPI_PROC_NULL ? 1u : 0u);
    size_t coreZEnd = zEnd - (neighbors[5] != MPI_PROC_NULL ? 1u : 0u);
    if (coreXBegin > coreXEnd) coreXEnd = coreXBegin;
    if (coreYBegin > coreYEnd) coreYEnd = coreYBegin;
    if (coreZBegin > coreZEnd) coreZEnd = coreZBegin;

    updateBox(input, output, coreXBegin, coreXEnd, coreYBegin, coreYEnd,
              coreZBegin, coreZEnd, pitch, slice);

    if (!haloRequests.empty()) {
        mpiCheck(MPI_Waitall(static_cast<int>(haloRequests.size()),
                             haloRequests.data(), MPI_STATUSES_IGNORE),
                 "MPI_Waitall");
    }

    // Six disjoint boxes cover the halo-dependent shell around the core.
    updateBox(input, output, xBegin, xEnd, yBegin, yEnd, zBegin, coreZBegin,
              pitch, slice);
    updateBox(input, output, xBegin, xEnd, yBegin, yEnd, coreZEnd, zEnd,
              pitch, slice);
    updateBox(input, output, xBegin, xEnd, yBegin, coreYBegin, coreZBegin,
              coreZEnd, pitch, slice);
    updateBox(input, output, xBegin, xEnd, coreYEnd, yEnd, coreZBegin,
              coreZEnd, pitch, slice);
    updateBox(input, output, xBegin, coreXBegin, coreYBegin, coreYEnd,
              coreZBegin, coreZEnd, pitch, slice);
    updateBox(input, output, coreXEnd, xEnd, coreYBegin, coreYEnd,
              coreZBegin, coreZEnd, pitch, slice);
}

void addBoundaryValue(const size_t x, const size_t y, const size_t z,
                      const size_t nx, const size_t ny, Real& minimum,
                      Real& maximum) {
    const Real value = static_cast<Real>(idx3(x, y, z, nx, ny) % 19);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
}

bool validateDistributed(const Real* grid, const size_t localNx,
                         const size_t localNy, const size_t localNz,
                         const size_t pitch, const size_t slice,
                         const size_t globalNx, const size_t globalNy,
                         const size_t globalNz, MPI_Comm communicator,
                         const int rank) {
    int allFinite = 1;
    Real localMinimum = std::numeric_limits<Real>::infinity();
    Real localMaximum = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= localNz; ++z) {
        for (size_t y = 1; y <= localNy; ++y) {
            const size_t begin = localIndex(1, y, z, pitch, slice);
            for (size_t x = 0; x < localNx; ++x) {
                const Real value = grid[begin + x];
                allFinite &= std::isfinite(value) ? 1 : 0;
                localMinimum = std::min(localMinimum, value);
                localMaximum = std::max(localMaximum, value);
            }
        }
    }

    // Rank zero supplies fixed physical boundaries to the global reduction.
    if (rank == 0) {
        for (size_t y = 0; y < globalNy; ++y) {
            for (size_t x = 0; x < globalNx; ++x) {
                addBoundaryValue(x, y, 0, globalNx, globalNy,
                                 localMinimum, localMaximum);
                addBoundaryValue(x, y, globalNz - 1, globalNx, globalNy,
                                 localMinimum, localMaximum);
            }
        }
        for (size_t z = 1; z + 1 < globalNz; ++z) {
            for (size_t x = 0; x < globalNx; ++x) {
                addBoundaryValue(x, 0, z, globalNx, globalNy,
                                 localMinimum, localMaximum);
                addBoundaryValue(x, globalNy - 1, z, globalNx, globalNy,
                                 localMinimum, localMaximum);
            }
            for (size_t y = 1; y + 1 < globalNy; ++y) {
                addBoundaryValue(0, y, z, globalNx, globalNy,
                                 localMinimum, localMaximum);
                addBoundaryValue(globalNx - 1, y, z, globalNx, globalNy,
                                 localMinimum, localMaximum);
            }
        }
    }

    int globallyFinite = 0;
    Real globalMinimum = 0.0;
    Real globalMaximum = 0.0;
    mpiCheck(MPI_Allreduce(&allFinite, &globallyFinite, 1, MPI_INT, MPI_MIN,
                           communicator),
             "MPI_Allreduce(finite)");
    if (!globallyFinite) {
        if (rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }
    mpiCheck(MPI_Allreduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN,
                           communicator),
             "MPI_Allreduce(minimum)");
    mpiCheck(MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX,
                           communicator),
             "MPI_Allreduce(maximum)");
    if (rank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
    }
    const bool valid = globalMaximum <= 1e6 && globalMinimum >= -1e6;
    if (!valid && rank == 0) {
        std::printf("Validation failed: values out of expected range\n");
    }
    return valid;
}

MPI_Datatype createVolumeType(const size_t localNx, const size_t localNy,
                              const size_t localNz, const size_t pitch,
                              const size_t slice) {
    MPI_Datatype row = MPI_DATATYPE_NULL;
    MPI_Datatype plane = MPI_DATATYPE_NULL;
    MPI_Datatype volume = MPI_DATATYPE_NULL;
    mpiCheck(MPI_Type_contiguous(static_cast<int>(localNx), MPI_DOUBLE, &row),
             "MPI_Type_contiguous(gather row)");
    mpiCheck(MPI_Type_create_hvector(static_cast<int>(localNy), 1,
                                     static_cast<MPI_Aint>(pitch * sizeof(Real)),
                                     row, &plane),
             "MPI_Type_create_hvector(gather plane)");
    mpiCheck(MPI_Type_create_hvector(static_cast<int>(localNz), 1,
                                     static_cast<MPI_Aint>(slice * sizeof(Real)),
                                     plane, &volume),
             "MPI_Type_create_hvector(gather volume)");
    mpiCheck(MPI_Type_commit(&volume), "MPI_Type_commit(gather volume)");
    mpiCheck(MPI_Type_free(&plane), "MPI_Type_free(gather plane)");
    mpiCheck(MPI_Type_free(&row), "MPI_Type_free(gather row)");
    return volume;
}

std::vector<Real> gatherGlobal(const Real* localGrid, const size_t localNx,
                               const size_t localNy, const size_t localNz,
                               const size_t pitch, const size_t slice,
                               const size_t globalNx, const size_t globalNy,
                               const size_t globalNz,
                               const std::array<int, 3>& processDimensions,
                               MPI_Comm cartesian, const int rank,
                               const int processCount) {
    std::vector<Real> globalGrid;
    if (rank == 0) {
        initializeGlobal(globalGrid, globalNx, globalNy, globalNz);
        int coordinates[3] = {};
        mpiCheck(MPI_Cart_coords(cartesian, 0, 3, coordinates),
                 "MPI_Cart_coords(root gather)");
        const Block xBlock = splitInterior(globalNx - 2, processDimensions[0],
                                           coordinates[0]);
        const Block yBlock = splitInterior(globalNy - 2, processDimensions[1],
                                           coordinates[1]);
        const Block zBlock = splitInterior(globalNz - 2, processDimensions[2],
                                           coordinates[2]);
        for (size_t z = 0; z < zBlock.size; ++z) {
            for (size_t y = 0; y < yBlock.size; ++y) {
                const Real* source = localGrid + localIndex(1, y + 1, z + 1,
                                                            pitch, slice);
                Real* destination = globalGrid.data() +
                    idx3(xBlock.begin, yBlock.begin + y, zBlock.begin + z,
                         globalNx, globalNy);
                std::copy_n(source, xBlock.size, destination);
            }
        }

        std::vector<MPI_Request> requests;
        std::vector<MPI_Datatype> receiveTypes;
        requests.reserve(static_cast<size_t>(processCount - 1));
        receiveTypes.reserve(static_cast<size_t>(processCount - 1));
        const int globalSizes[3] = {static_cast<int>(globalNz),
                                    static_cast<int>(globalNy),
                                    static_cast<int>(globalNx)};
        for (int sourceRank = 1; sourceRank < processCount; ++sourceRank) {
            mpiCheck(MPI_Cart_coords(cartesian, sourceRank, 3, coordinates),
                     "MPI_Cart_coords(gather source)");
            const Block sourceX = splitInterior(globalNx - 2, processDimensions[0],
                                                coordinates[0]);
            const Block sourceY = splitInterior(globalNy - 2, processDimensions[1],
                                                coordinates[1]);
            const Block sourceZ = splitInterior(globalNz - 2, processDimensions[2],
                                                coordinates[2]);
            const int subSizes[3] = {static_cast<int>(sourceZ.size),
                                     static_cast<int>(sourceY.size),
                                     static_cast<int>(sourceX.size)};
            const int starts[3] = {static_cast<int>(sourceZ.begin),
                                   static_cast<int>(sourceY.begin),
                                   static_cast<int>(sourceX.begin)};
            receiveTypes.push_back(MPI_DATATYPE_NULL);
            mpiCheck(MPI_Type_create_subarray(3, globalSizes, subSizes, starts,
                                              MPI_ORDER_C, MPI_DOUBLE,
                                              &receiveTypes.back()),
                     "MPI_Type_create_subarray(gather receive)");
            mpiCheck(MPI_Type_commit(&receiveTypes.back()),
                     "MPI_Type_commit(gather receive)");
            requests.push_back(MPI_REQUEST_NULL);
            mpiCheck(MPI_Irecv(globalGrid.data(), 1, receiveTypes.back(), sourceRank,
                               TAG_GATHER, cartesian, &requests.back()),
                     "MPI_Irecv(gather)");
        }
        if (!requests.empty()) {
            mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                                 MPI_STATUSES_IGNORE),
                     "MPI_Waitall(gather)");
        }
        for (MPI_Datatype& type : receiveTypes) {
            mpiCheck(MPI_Type_free(&type), "MPI_Type_free(gather receive)");
        }
    } else {
        MPI_Datatype volume = createVolumeType(localNx, localNy, localNz,
                                               pitch, slice);
        mpiCheck(MPI_Send(localGrid + localIndex(1, 1, 1, pitch, slice), 1,
                          volume, 0, TAG_GATHER, cartesian),
                 "MPI_Send(gather)");
        mpiCheck(MPI_Type_free(&volume), "MPI_Type_free(gather volume)");
    }
    return globalGrid;
}

}  // namespace

int main(int argc, char** argv) {
    const int initError = MPI_Init(&argc, &argv);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
             "MPI_Comm_set_errhandler");

    int worldRank = 0;
    int worldSize = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank(world)");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size(world)");

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
            if (worldRank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    size_t globalSlice = 0;
    size_t globalCells = 0;
    if (!checkedMultiply(nx, ny, globalSlice) ||
        !checkedMultiply(globalSlice, nz, globalCells)) {
        argumentsValid = false;
    }
    if (nx < 2 || ny < 2 || nz < 2 || nx > static_cast<size_t>(INT_MAX) ||
        ny > static_cast<size_t>(INT_MAX) || nz > static_cast<size_t>(INT_MAX)) {
        argumentsValid = false;
    }
    if (!argumentsValid) {
        if (worldRank == 0) {
            std::printf("Grid sizes must be integers in [2, INT_MAX], the total size "
                        "must fit in memory, and iterations must be nonnegative.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t interiorNx = nx - 2;
    const size_t interiorNy = ny - 2;
    const size_t interiorNz = nz - 2;
    const bool hasInterior = interiorNx != 0 && interiorNy != 0 && interiorNz != 0;
    std::array<int, 3> processDimensions = {1, 1, 1};
    const int activeProcesses = hasInterior
        ? selectActiveProcesses(worldSize, interiorNx, interiorNy, interiorNz,
                                processDimensions)
        : 1;

    MPI_Comm activeCommunicator = MPI_COMM_NULL;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD,
                            worldRank < activeProcesses ? 0 : MPI_UNDEFINED,
                            worldRank, &activeCommunicator),
             "MPI_Comm_split");

    int exitCode = 0;
    if (activeCommunicator != MPI_COMM_NULL) {
        MPI_Comm computeCommunicator = activeCommunicator;
        MPI_Comm cartesian = MPI_COMM_NULL;
        if (hasInterior) {
            const int periods[3] = {0, 0, 0};
            mpiCheck(MPI_Cart_create(activeCommunicator, 3,
                                     processDimensions.data(), periods, 1,
                                     &cartesian),
                     "MPI_Cart_create");
            computeCommunicator = cartesian;
        }
        mpiCheck(MPI_Comm_set_errhandler(computeCommunicator, MPI_ERRORS_RETURN),
                 "MPI_Comm_set_errhandler(compute)");

        int rank = 0;
        mpiCheck(MPI_Comm_rank(computeCommunicator, &rank),
                 "MPI_Comm_rank(compute)");
        if (rank == 0) {
            std::printf("3D Stencil Benchmark\n");
            std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            std::printf("Iterations: %d\n", iterations);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
            std::printf("MPI processes: %d active of %d\n", activeProcesses,
                        worldSize);
            std::printf("Process grid: %d x %d x %d\n", processDimensions[0],
                        processDimensions[1], processDimensions[2]);
            std::printf("Initializing grid...\n");
        }

        std::unique_ptr<Real[]> grid1;
        std::unique_ptr<Real[]> grid2;
        Real* current = nullptr;
        Real* next = nullptr;
        size_t localNx = 0;
        size_t localNy = 0;
        size_t localNz = 0;
        size_t startX = 1;
        size_t startY = 1;
        size_t startZ = 1;
        size_t pitch = 0;
        size_t slice = 0;
        std::array<int, 6> neighbors = {MPI_PROC_NULL, MPI_PROC_NULL,
                                        MPI_PROC_NULL, MPI_PROC_NULL,
                                        MPI_PROC_NULL, MPI_PROC_NULL};
        FaceTypes faceTypes;
        std::array<std::vector<MPI_Request>, 2> haloRequests;

        try {
            if (hasInterior) {
                int coordinates[3] = {};
                mpiCheck(MPI_Cart_coords(cartesian, rank, 3, coordinates),
                         "MPI_Cart_coords(local)");
                const Block xBlock = splitInterior(interiorNx, processDimensions[0],
                                                   coordinates[0]);
                const Block yBlock = splitInterior(interiorNy, processDimensions[1],
                                                   coordinates[1]);
                const Block zBlock = splitInterior(interiorNz, processDimensions[2],
                                                   coordinates[2]);
                startX = xBlock.begin;
                startY = yBlock.begin;
                startZ = zBlock.begin;
                localNx = xBlock.size;
                localNy = yBlock.size;
                localNz = zBlock.size;
                pitch = localNx + 2;
                if (!checkedMultiply(pitch, localNy + 2, slice)) {
                    abortMpi("local slice size overflow", MPI_ERR_COUNT);
                }
                size_t localCells = 0;
                if (!checkedMultiply(slice, localNz + 2, localCells) ||
                    localCells > static_cast<size_t>(std::numeric_limits<MPI_Aint>::max()) /
                                     sizeof(Real)) {
                    abortMpi("local allocation size overflow", MPI_ERR_COUNT);
                }
                grid1.reset(new Real[localCells]);
                grid2.reset(new Real[localCells]);
                initializeLocal(grid1.get(), grid2.get(), localNx, localNy, localNz,
                                startX, startY, startZ, nx, ny);
                current = grid1.get();
                next = grid2.get();

                mpiCheck(MPI_Cart_shift(cartesian, 0, 1, &neighbors[0], &neighbors[1]),
                         "MPI_Cart_shift(x)");
                mpiCheck(MPI_Cart_shift(cartesian, 1, 1, &neighbors[2], &neighbors[3]),
                         "MPI_Cart_shift(y)");
                mpiCheck(MPI_Cart_shift(cartesian, 2, 1, &neighbors[4], &neighbors[5]),
                         "MPI_Cart_shift(z)");
                faceTypes = createFaceTypes(localNx, localNy, localNz, pitch, slice);
                haloRequests[0] = createPersistentHalo(
                    grid1.get(), localNx, localNy, localNz, pitch, slice,
                    neighbors, faceTypes, cartesian);
                haloRequests[1] = createPersistentHalo(
                    grid2.get(), localNx, localNy, localNz, pitch, slice,
                    neighbors, faceTypes, cartesian);
            }
        } catch (const std::bad_alloc&) {
            abortMpi("grid allocation", MPI_ERR_NO_MEM);
        }

        if (rank == 0) std::printf("Running stencil computation...\n");
        mpiCheck(MPI_Barrier(computeCommunicator), "MPI_Barrier(start)");
        const double startTime = MPI_Wtime();
        int currentBuffer = 0;
        if (hasInterior) {
            for (int iteration = 0; iteration < iterations; ++iteration) {
                stencilIteration(current, next, haloRequests[currentBuffer],
                                 localNx, localNy, localNz, pitch, slice, neighbors);
                std::swap(current, next);
                currentBuffer ^= 1;
            }
        }
        const double localElapsed = MPI_Wtime() - startTime;
        double elapsed = 0.0;
        mpiCheck(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                            computeCommunicator),
                 "MPI_Reduce(time)");

        if (hasInterior) {
            freePersistentRequests(haloRequests[0]);
            freePersistentRequests(haloRequests[1]);
            freeFaceTypes(faceTypes);
        }

        if (rank == 0) {
            const long long durationMs = static_cast<long long>(elapsed * 1000.0);
            const long double cellUpdates =
                static_cast<long double>(interiorNx) *
                static_cast<long double>(interiorNy) *
                static_cast<long double>(interiorNz) * iterations;
            const double mcups = elapsed > 0.0
                ? static_cast<double>(cellUpdates / elapsed / 1.0e6L)
                : 0.0;
            std::printf("Computation time: %lld ms\n", durationMs);
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            try {
                std::vector<Real> globalGrid;
                if (hasInterior) {
                    globalGrid = gatherGlobal(current, localNx, localNy, localNz,
                                              pitch, slice, nx, ny, nz,
                                              processDimensions, cartesian, rank,
                                              activeProcesses);
                } else if (rank == 0) {
                    initializeGlobal(globalGrid, nx, ny, nz);
                }
                if (rank == 0) print_results(globalGrid, "Grid");
            } catch (const std::bad_alloc&) {
                abortMpi("result assembly allocation", MPI_ERR_NO_MEM);
            }
        }

        if (validate) {
            if (rank == 0) std::printf("Validating result...\n");
            bool valid = true;
            if (hasInterior) {
                valid = validateDistributed(current, localNx, localNy, localNz,
                                            pitch, slice, nx, ny, nz,
                                            computeCommunicator, rank);
            } else if (rank == 0) {
                Real minimum = std::numeric_limits<Real>::infinity();
                Real maximum = -std::numeric_limits<Real>::infinity();
                for (size_t index = 0; index < globalCells; ++index) {
                    const Real value = static_cast<Real>(index % 19);
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
                std::printf("Value range: [%.6f, %.6f]\n", minimum, maximum);
                valid = maximum <= 1e6 && minimum >= -1e6;
            }
            int validInteger = valid ? 1 : 0;
            mpiCheck(MPI_Bcast(&validInteger, 1, MPI_INT, 0, computeCommunicator),
                     "MPI_Bcast(validation)");
            valid = validInteger != 0;
            if (rank == 0) {
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            exitCode = valid ? 0 : 1;
        }

        if (cartesian != MPI_COMM_NULL) {
            mpiCheck(MPI_Comm_free(&cartesian), "MPI_Comm_free(cartesian)");
        }
        mpiCheck(MPI_Comm_free(&activeCommunicator),
                 "MPI_Comm_free(active communicator)");
    }

    int globalExitCode = 0;
    mpiCheck(MPI_Allreduce(&exitCode, &globalExitCode, 1, MPI_INT, MPI_MAX,
                           MPI_COMM_WORLD),
             "MPI_Allreduce(exit code)");
    MPI_Finalize();
    return globalExitCode;
}
