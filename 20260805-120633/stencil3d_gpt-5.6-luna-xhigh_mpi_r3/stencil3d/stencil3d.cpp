#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation. The local arrays have one halo cell on every side.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

struct Range {
    size_t begin = 1;
    size_t end = 0;

    bool empty() const noexcept { return begin > end; }
};

struct Domain {
    size_t global_x = 0;
    size_t global_y = 0;
    size_t global_z = 0;

    size_t local_x = 0;
    size_t local_y = 0;
    size_t local_z = 0;

    size_t start_x = 0;
    size_t start_y = 0;
    size_t start_z = 0;

    // x-, x+, y-, y+, z-, z+
    std::array<int, 6> neighbors{};
    MPI_Comm communicator = MPI_COMM_NULL;
};

struct FaceTypes {
    MPI_Datatype x = MPI_DATATYPE_NULL;
    MPI_Datatype y = MPI_DATATYPE_NULL;
    MPI_Datatype z = MPI_DATATYPE_NULL;
};

struct GatherMetadata {
    uint64_t start_x;
    uint64_t start_y;
    uint64_t start_z;
    uint64_t local_x;
    uint64_t local_y;
    uint64_t local_z;
};

bool checkedMultiply(const size_t a, const size_t b, size_t& result) noexcept {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool checkedProduct3(const size_t a, const size_t b, const size_t c,
                     size_t& result) noexcept {
    size_t ab = 0;
    return checkedMultiply(a, b, ab) && checkedMultiply(ab, c, result);
}

bool parseSize(const char* text, size_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }

    value = static_cast<size_t>(parsed);
    return true;
}

bool parseInt(const char* text, int& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

bool parseArguments(const int argc, char** argv, Options& options,
                    const char*& badArgument) {
    for (int i = 1; i < argc; ++i) {
        const char* argument = argv[i];
        if (std::strcmp(argument, "-x") == 0 ||
            std::strcmp(argument, "-y") == 0 ||
            std::strcmp(argument, "-z") == 0) {
            badArgument = argument;
            if (i + 1 >= argc) {
                return false;
            }

            size_t value = 0;
            if (!parseSize(argv[++i], value)) {
                badArgument = argv[i];
                return false;
            }

            if (argument[1] == 'x') {
                options.nx = value;
            } else if (argument[1] == 'y') {
                options.ny = value;
            } else {
                options.nz = value;
            }
        } else if (std::strcmp(argument, "-i") == 0) {
            badArgument = argument;
            if (i + 1 >= argc || !parseInt(argv[i + 1], options.iterations)) {
                if (i + 1 < argc) {
                    badArgument = argv[i + 1];
                }
                return false;
            }
            ++i;
        } else if (std::strcmp(argument, "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argument, "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argument, "-h") == 0) {
            badArgument = argument;
            return false;
        } else {
            badArgument = argument;
            return false;
        }
    }
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

// Select a process grid that follows the shape of the problem. This is more
// useful for rectangular grids than an isotropic process grid and is fully
// deterministic, so every rank constructs the same Cartesian communicator.
std::array<int, 3> chooseProcessGrid(const int processCount,
                                     const size_t nx, const size_t ny,
                                     const size_t nz) {
    std::array<int, 3> best{0, 0, 0};
    long double bestSurface = std::numeric_limits<long double>::infinity();

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

            const long double localX = static_cast<long double>(nx) / px;
            const long double localY = static_cast<long double>(ny) / py;
            const long double localZ = static_cast<long double>(nz) / pz;

            long double surface = 0.0L;
            if (px > 1) {
                surface += localY * localZ;
            }
            if (py > 1) {
                surface += localX * localZ;
            }
            if (pz > 1) {
                surface += localX * localY;
            }

            if (surface < bestSurface) {
                bestSurface = surface;
                best = {px, py, pz};
            }
        }
    }
    return best;
}

size_t blockSize(const size_t globalSize, const int processCoordinate,
                 const int processDimension) noexcept {
    const size_t base = globalSize / static_cast<size_t>(processDimension);
    const size_t remainder = globalSize % static_cast<size_t>(processDimension);
    return base + (static_cast<size_t>(processCoordinate) < remainder ? 1 : 0);
}

size_t blockStart(const size_t globalSize, const int processCoordinate,
                  const int processDimension) noexcept {
    const size_t base = globalSize / static_cast<size_t>(processDimension);
    const size_t remainder = globalSize % static_cast<size_t>(processDimension);
    const size_t coordinate = static_cast<size_t>(processCoordinate);
    return coordinate * base + std::min(coordinate, remainder);
}

Range globalInteriorRange(const size_t start, const size_t local,
                          const size_t global) noexcept {
    if (global <= 2 || local == 0) {
        return {1, 0};
    }

    size_t begin = 1;
    size_t end = local;
    if (start == 0) {
        begin = 2;
    }
    if (start + local == global) {
        --end;
    }
    return {begin, end};
}

int mpiInt(const size_t value) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(value);
}

void initializeGrid(std::vector<Real>& grid, const Domain& domain) {
    const size_t localNx = domain.local_x + 2;
    const size_t localNy = domain.local_y + 2;

    for (size_t z = 1; z <= domain.local_z; ++z) {
        const size_t globalZ = domain.start_z + z - 1;
        for (size_t y = 1; y <= domain.local_y; ++y) {
            const size_t globalY = domain.start_y + y - 1;
            for (size_t x = 1; x <= domain.local_x; ++x) {
                const size_t globalX = domain.start_x + x - 1;
                const size_t globalIndex = idx3(globalX, globalY, globalZ,
                                                domain.global_x,
                                                domain.global_y);
                grid[idx3(x, y, z, localNx, localNy)] =
                    static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

void createFaceTypes(const Domain& domain, FaceTypes& faces) {
    const size_t localNx = domain.local_x + 2;
    const size_t localNy = domain.local_y + 2;
    const size_t planeSize = localNx * localNy;
    const int localX = mpiInt(domain.local_x);
    const int localY = mpiInt(domain.local_y);
    const int localZ = mpiInt(domain.local_z);
    const int pitch = mpiInt(localNx);
    const int plane = mpiInt(planeSize);

    if (localX < 1 || localY < 1 || localZ < 1 || pitch < 1 || plane < 1) {
        std::fprintf(stderr, "Local domain is too large for MPI datatype construction\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // An x face is a vector in y, repeated once per z plane. The nested type
    // avoids packing strided faces in user space.
    MPI_Datatype xLine = MPI_DATATYPE_NULL;
    MPI_Type_vector(localY, 1, pitch, MPI_DOUBLE, &xLine);
    MPI_Type_create_hvector(localZ, 1,
                            static_cast<MPI_Aint>(planeSize * sizeof(Real)),
                            xLine, &faces.x);
    MPI_Type_commit(&faces.x);
    MPI_Type_free(&xLine);

    // A y face is a contiguous x row repeated once per z plane.
    MPI_Type_vector(localZ, localX, plane, MPI_DOUBLE, &faces.y);
    MPI_Type_commit(&faces.y);

    // A z face is a set of contiguous x rows.
    MPI_Type_vector(localY, localX, localX, MPI_DOUBLE, &faces.z);
    MPI_Type_commit(&faces.z);
}

void destroyFaceTypes(FaceTypes& faces) {
    if (faces.x != MPI_DATATYPE_NULL) {
        MPI_Type_free(&faces.x);
    }
    if (faces.y != MPI_DATATYPE_NULL) {
        MPI_Type_free(&faces.y);
    }
    if (faces.z != MPI_DATATYPE_NULL) {
        MPI_Type_free(&faces.z);
    }
}

int beginHaloExchange(std::vector<Real>& input, const Domain& domain,
                      const FaceTypes& faces,
                      std::array<MPI_Request, 12>& requests) {
    const size_t localNx = domain.local_x + 2;
    const size_t localNy = domain.local_y + 2;
    int requestCount = 0;

    auto receive = [&](const size_t x, const size_t y, const size_t z,
                       const MPI_Datatype type, const int source,
                       const int tag) {
        if (source == MPI_PROC_NULL) {
            return;
        }
        MPI_Irecv(input.data() + idx3(x, y, z, localNx, localNy), 1, type,
                  source, tag, domain.communicator, &requests[requestCount++]);
    };
    auto send = [&](const size_t x, const size_t y, const size_t z,
                    const MPI_Datatype type, const int destination,
                    const int tag) {
        if (destination == MPI_PROC_NULL) {
            return;
        }
        MPI_Isend(input.data() + idx3(x, y, z, localNx, localNy), 1, type,
                  destination, tag, domain.communicator, &requests[requestCount++]);
    };

    // Receive first so all peers can immediately match their sends.
    receive(0, 1, 1, faces.x, domain.neighbors[0], 101);
    receive(domain.local_x + 1, 1, 1, faces.x, domain.neighbors[1], 100);
    receive(1, 0, 1, faces.y, domain.neighbors[2], 201);
    receive(1, domain.local_y + 1, 1, faces.y, domain.neighbors[3], 200);
    receive(1, 1, 0, faces.z, domain.neighbors[4], 301);
    receive(1, 1, domain.local_z + 1, faces.z, domain.neighbors[5], 300);

    send(1, 1, 1, faces.x, domain.neighbors[0], 100);
    send(domain.local_x, 1, 1, faces.x, domain.neighbors[1], 101);
    send(1, 1, 1, faces.y, domain.neighbors[2], 200);
    send(1, domain.local_y, 1, faces.y, domain.neighbors[3], 201);
    send(1, 1, 1, faces.z, domain.neighbors[4], 300);
    send(1, 1, domain.local_z, faces.z, domain.neighbors[5], 301);

    return requestCount;
}

void computeStencilRegion(const std::vector<Real>& input,
                          std::vector<Real>& output, const Domain& domain,
                          const Range xRange, const Range yRange,
                          const Range zRange) {
    if (xRange.empty() || yRange.empty() || zRange.empty()) {
        return;
    }

    const size_t localNx = domain.local_x + 2;
    const size_t localNy = domain.local_y + 2;
    const size_t planeSize = localNx * localNy;
    const Real* const in = input.data();
    Real* const out = output.data();

    for (size_t z = zRange.begin; z <= zRange.end; ++z) {
        for (size_t y = yRange.begin; y <= yRange.end; ++y) {
            size_t index = idx3(xRange.begin, y, z, localNx, localNy);
            for (size_t x = xRange.begin; x <= xRange.end; ++x, ++index) {
                const Real center = in[index];
                const Real left = in[index - 1];
                const Real right = in[index + 1];
                const Real front = in[index - localNx];
                const Real back = in[index + localNx];
                const Real bottom = in[index - planeSize];
                const Real top = in[index + planeSize];

                // Keep the arithmetic and operand order of the original
                // single-process stencil.
                out[index] =
                    (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

void copyBoundaryValues(const std::vector<Real>& input,
                        std::vector<Real>& output, const Domain& domain) {
    const size_t localNx = domain.local_x + 2;
    const size_t localNy = domain.local_y + 2;

    if (domain.start_x == 0) {
        for (size_t z = 1; z <= domain.local_z; ++z) {
            for (size_t y = 1; y <= domain.local_y; ++y) {
                const size_t index = idx3(1, y, z, localNx, localNy);
                output[index] = input[index];
            }
        }
    }
    if (domain.start_x + domain.local_x == domain.global_x) {
        for (size_t z = 1; z <= domain.local_z; ++z) {
            for (size_t y = 1; y <= domain.local_y; ++y) {
                const size_t index = idx3(domain.local_x, y, z,
                                          localNx, localNy);
                output[index] = input[index];
            }
        }
    }
    if (domain.start_y == 0) {
        for (size_t z = 1; z <= domain.local_z; ++z) {
            for (size_t x = 1; x <= domain.local_x; ++x) {
                const size_t index = idx3(x, 1, z, localNx, localNy);
                output[index] = input[index];
            }
        }
    }
    if (domain.start_y + domain.local_y == domain.global_y) {
        for (size_t z = 1; z <= domain.local_z; ++z) {
            for (size_t x = 1; x <= domain.local_x; ++x) {
                const size_t index = idx3(x, domain.local_y, z,
                                          localNx, localNy);
                output[index] = input[index];
            }
        }
    }
    if (domain.start_z == 0) {
        for (size_t y = 1; y <= domain.local_y; ++y) {
            for (size_t x = 1; x <= domain.local_x; ++x) {
                const size_t index = idx3(x, y, 1, localNx, localNy);
                output[index] = input[index];
            }
        }
    }
    if (domain.start_z + domain.local_z == domain.global_z) {
        for (size_t y = 1; y <= domain.local_y; ++y) {
            for (size_t x = 1; x <= domain.local_x; ++x) {
                const size_t index = idx3(x, y, domain.local_z,
                                          localNx, localNy);
                output[index] = input[index];
            }
        }
    }
}

void computeStencilShell(const std::vector<Real>& input,
                         std::vector<Real>& output, const Domain& domain,
                         const Range xRange, const Range yRange,
                         const Range zRange, const Range coreX,
                         const Range coreY, const Range coreZ) {
    if (xRange.empty() || yRange.empty() || zRange.empty()) {
        return;
    }
    if (coreX.empty() || coreY.empty() || coreZ.empty()) {
        computeStencilRegion(input, output, domain, xRange, yRange, zRange);
        return;
    }

    // z slabs outside the core
    if (zRange.begin < coreZ.begin) {
        computeStencilRegion(input, output, domain, xRange, yRange,
                             {zRange.begin, coreZ.begin - 1});
    }
    if (coreZ.end < zRange.end) {
        computeStencilRegion(input, output, domain, xRange, yRange,
                             {coreZ.end + 1, zRange.end});
    }

    // y slabs inside the core z range
    if (yRange.begin < coreY.begin) {
        computeStencilRegion(input, output, domain, xRange,
                             {yRange.begin, coreY.begin - 1}, coreZ);
    }
    if (coreY.end < yRange.end) {
        computeStencilRegion(input, output, domain, xRange,
                             {coreY.end + 1, yRange.end}, coreZ);
    }

    // x slabs inside the core y and z ranges
    if (xRange.begin < coreX.begin) {
        computeStencilRegion(input, output, domain,
                             {xRange.begin, coreX.begin - 1}, coreY, coreZ);
    }
    if (coreX.end < xRange.end) {
        computeStencilRegion(input, output, domain,
                             {coreX.end + 1, xRange.end}, coreY, coreZ);
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const Domain& domain, const FaceTypes& faces) {
    const Range xRange = globalInteriorRange(domain.start_x, domain.local_x,
                                             domain.global_x);
    const Range yRange = globalInteriorRange(domain.start_y, domain.local_y,
                                             domain.global_y);
    const Range zRange = globalInteriorRange(domain.start_z, domain.local_z,
                                             domain.global_z);

    std::array<MPI_Request, 12> requests{};
    const int requestCount = beginHaloExchange(input, domain, faces, requests);

    // Boundary values do not depend on halo data and can be copied while MPI
    // progresses the six face exchanges.
    copyBoundaryValues(input, output, domain);

    auto coreRange = [](Range range, const int minusNeighbor,
                        const int plusNeighbor) {
        if (range.empty()) {
            return range;
        }
        if (minusNeighbor != MPI_PROC_NULL) {
            ++range.begin;
        }
        if (plusNeighbor != MPI_PROC_NULL) {
            --range.end;
        }
        return range;
    };

    const Range coreX = coreRange(xRange, domain.neighbors[0],
                                  domain.neighbors[1]);
    const Range coreY = coreRange(yRange, domain.neighbors[2],
                                  domain.neighbors[3]);
    const Range coreZ = coreRange(zRange, domain.neighbors[4],
                                  domain.neighbors[5]);

    computeStencilRegion(input, output, domain, coreX, coreY, coreZ);

    if (requestCount > 0) {
        MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE);
    }

    computeStencilShell(input, output, domain, xRange, yRange, zRange,
                        coreX, coreY, coreZ);
}

bool validateDistributed(const std::vector<Real>& grid, const Domain& domain,
                         const MPI_Comm communicator, const int rank) {
    const size_t localNx = domain.local_x + 2;
    const size_t localNy = domain.local_y + 2;

    bool localInvalid = false;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= domain.local_z; ++z) {
        for (size_t y = 1; y <= domain.local_y; ++y) {
            for (size_t x = 1; x <= domain.local_x; ++x) {
                const Real value = grid[idx3(x, y, z, localNx, localNy)];
                if (std::isnan(value) || std::isinf(value)) {
                    localInvalid = true;
                }
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
    }

    int localInvalidInt = localInvalid ? 1 : 0;
    int globalInvalidInt = 0;
    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&localInvalidInt, &globalInvalidInt, 1, MPI_INT, MPI_LOR,
                  communicator);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN,
                  communicator);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX,
                  communicator);

    if (globalInvalidInt != 0) {
        if (rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    if (rank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    }
    return globalMax <= 1e6 && globalMin >= -1e6;
}

void gatherGlobalGrid(const std::vector<Real>& localGrid,
                      const Domain& domain, const int processCount,
                      const int rank, std::vector<Real>& globalGrid) {
    const size_t localNx = domain.local_x + 2;
    const size_t localNy = domain.local_y + 2;
    size_t localCount = 0;
    if (!checkedProduct3(domain.local_x, domain.local_y, domain.local_z,
                         localCount) || localCount >
                                             static_cast<size_t>(INT32_MAX)) {
        std::fprintf(stderr, "Local result is too large for MPI_Gatherv\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<Real> packed(localCount);
    size_t packedIndex = 0;
    for (size_t z = 1; z <= domain.local_z; ++z) {
        for (size_t y = 1; y <= domain.local_y; ++y) {
            const size_t source = idx3(1, y, z, localNx, localNy);
            std::memcpy(packed.data() + packedIndex,
                        localGrid.data() + source, domain.local_x * sizeof(Real));
            packedIndex += domain.local_x;
        }
    }

    const int sendCount = static_cast<int>(localCount);
    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<GatherMetadata> metadata;
    std::vector<Real> gathered;
    if (rank == 0) {
        receiveCounts.resize(processCount);
        displacements.resize(processCount);
        metadata.resize(processCount);
    }

    MPI_Gather(&sendCount, 1, MPI_INT,
               rank == 0 ? receiveCounts.data() : nullptr, 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    const GatherMetadata localMetadata{
        domain.start_x, domain.start_y, domain.start_z,
        domain.local_x, domain.local_y, domain.local_z};
    MPI_Gather(&localMetadata, static_cast<int>(sizeof(GatherMetadata)),
               MPI_BYTE,
               rank == 0 ? metadata.data() : nullptr,
               static_cast<int>(sizeof(GatherMetadata)), MPI_BYTE, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        size_t totalCount = 0;
        for (int process = 0; process < processCount; ++process) {
            if (receiveCounts[process] < 0 ||
                totalCount > static_cast<size_t>(INT32_MAX) -
                                 static_cast<size_t>(receiveCounts[process])) {
                std::fprintf(stderr, "Global result is too large for MPI_Gatherv\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            displacements[process] = static_cast<int>(totalCount);
            totalCount += static_cast<size_t>(receiveCounts[process]);
        }
        gathered.resize(totalCount);
        globalGrid.assign(domain.global_x * domain.global_y * domain.global_z,
                          0.0);
    }

    MPI_Gatherv(packed.data(), sendCount, MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? receiveCounts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        for (int process = 0; process < processCount; ++process) {
            const GatherMetadata& item = metadata[process];
            size_t sourceIndex = static_cast<size_t>(displacements[process]);
            for (size_t z = 0; z < item.local_z; ++z) {
                for (size_t y = 0; y < item.local_y; ++y) {
                    const size_t destination = idx3(
                        item.start_x, item.start_y + y, item.start_z + z,
                        domain.global_x, domain.global_y);
                    std::memcpy(globalGrid.data() + destination,
                                gathered.data() + sourceIndex,
                                item.local_x * sizeof(Real));
                    sourceIndex += item.local_x;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    Options options;
    const char* badArgument = nullptr;
    const bool parsed = parseArguments(argc, argv, options, badArgument);
    if (!parsed) {
        if (badArgument != nullptr && std::strcmp(badArgument, "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
        if (rank == 0) {
            std::printf("Unknown or invalid option: %s\n",
                        badArgument == nullptr ? "" : badArgument);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (options.ny == 0) {
        options.ny = options.nx;
    }
    if (options.nz == 0) {
        options.nz = options.nx;
    }

    size_t globalSize = 0;
    if (!checkedProduct3(options.nx, options.ny, options.nz, globalSize)) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid dimensions overflow the address space\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (static_cast<size_t>(processCount) > globalSize) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI process count cannot exceed the number of grid cells\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const std::array<int, 3> processGrid =
        chooseProcessGrid(processCount, options.nx, options.ny, options.nz);
    if (processGrid[0] == 0) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Unable to construct a non-empty 3D MPI decomposition\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int periods[3] = {0, 0, 0};
    MPI_Comm cartesian = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, processGrid.data(), periods, 0,
                    &cartesian);

    int coordinates[3] = {0, 0, 0};
    MPI_Cart_coords(cartesian, rank, 3, coordinates);

    Domain domain;
    domain.global_x = options.nx;
    domain.global_y = options.ny;
    domain.global_z = options.nz;
    domain.start_x = blockStart(options.nx, coordinates[0], processGrid[0]);
    domain.start_y = blockStart(options.ny, coordinates[1], processGrid[1]);
    domain.start_z = blockStart(options.nz, coordinates[2], processGrid[2]);
    domain.local_x = blockSize(options.nx, coordinates[0], processGrid[0]);
    domain.local_y = blockSize(options.ny, coordinates[1], processGrid[1]);
    domain.local_z = blockSize(options.nz, coordinates[2], processGrid[2]);
    domain.communicator = cartesian;

    MPI_Cart_shift(cartesian, 0, 1, &domain.neighbors[0],
                   &domain.neighbors[1]);
    MPI_Cart_shift(cartesian, 1, 1, &domain.neighbors[2],
                   &domain.neighbors[3]);
    MPI_Cart_shift(cartesian, 2, 1, &domain.neighbors[4],
                   &domain.neighbors[5]);

    size_t localWithHalos = 0;
    if (!checkedProduct3(domain.local_x + 2, domain.local_y + 2,
                         domain.local_z + 2, localWithHalos)) {
        std::fprintf(stderr, "Local allocation size overflow\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny,
                    options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n",
                    options.validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", processCount);
    }

    std::vector<Real> grid1(localWithHalos, 0.0);
    std::vector<Real> grid2(localWithHalos, 0.0);

    if (rank == 0) {
        std::printf("Initializing grid...\n");
    }
    initializeGrid(grid1, domain);

    FaceTypes faces;
    createFaceTypes(domain, faces);

    if (rank == 0) {
        std::printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        if ((iteration & 1) == 0) {
            stencilIteration(grid1, grid2, domain, faces);
        } else {
            stencilIteration(grid2, grid1, domain, faces);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const long computationMilliseconds =
        static_cast<long>(elapsed * 1000.0);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", computationMilliseconds);
    }

    const double interiorX = options.nx > 2 ?
                                  static_cast<double>(options.nx - 2) : 0.0;
    const double interiorY = options.ny > 2 ?
                                  static_cast<double>(options.ny - 2) : 0.0;
    const double interiorZ = options.nz > 2 ?
                                  static_cast<double>(options.nz - 2) : 0.0;
    const double cellUpdates =
        interiorX * interiorY * interiorZ * options.iterations;
    const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
    if (rank == 0) {
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocalGrid =
        (options.iterations & 1) == 0 ? grid1 : grid2;
    std::vector<Real> finalGlobalGrid;
    if (options.printResults) {
        gatherGlobalGrid(finalLocalGrid, domain, processCount, rank,
                         finalGlobalGrid);
        if (rank == 0) {
            print_results(finalGlobalGrid, "Grid");
        }
    }

    bool valid = true;
    if (options.validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateDistributed(finalLocalGrid, domain, MPI_COMM_WORLD,
                                     rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    destroyFaceTypes(faces);
    MPI_Comm_free(&cartesian);
    MPI_Finalize();
    return options.validate && !valid ? 1 : 0;
}
