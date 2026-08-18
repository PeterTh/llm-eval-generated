#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int X_TO_PLUS = 0;
constexpr int X_TO_MINUS = 1;
constexpr int Y_TO_PLUS = 2;
constexpr int Y_TO_MINUS = 3;
constexpr int Z_TO_PLUS = 4;
constexpr int Z_TO_MINUS = 5;

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

struct Decomposition {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    std::array<int, 3> dims{1, 1, 1};
    std::array<int, 3> coords{0, 0, 0};
    // x-, x+, y-, y+, z-, z+
    std::array<int, 6> neighbors{MPI_PROC_NULL, MPI_PROC_NULL,
                                 MPI_PROC_NULL, MPI_PROC_NULL,
                                 MPI_PROC_NULL, MPI_PROC_NULL};
    std::array<size_t, 3> globalInterior{0, 0, 0};
    std::array<size_t, 3> local{0, 0, 0};
    // Global coordinate of the first locally owned interior cell.
    std::array<size_t, 3> start{1, 1, 1};
    bool active = true;
};

struct FaceTypes {
    MPI_Datatype x = MPI_DATATYPE_NULL;
    MPI_Datatype y = MPI_DATATYPE_NULL;
    MPI_Datatype z = MPI_DATATYPE_NULL;
};

struct RequestSet {
    std::vector<MPI_Request> receives;
    std::vector<MPI_Request> sends;
};

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

bool parseSize(const char* text, size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text[0] == '\0' || *end != '\0' || text[0] == '-' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (text[0] == '\0' || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

// Returns 0 on success, 1 on error, and 2 for help.
int parseOptions(const int argc, char** argv, Options& options, const bool printMessages) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nx)) {
                if (printMessages) std::printf("Invalid X dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.ny)) {
                if (printMessages) std::printf("Invalid Y dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nz)) {
                if (printMessages) std::printf("Invalid Z dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIterations(argv[++i], options.iterations)) {
                if (printMessages) std::printf("Invalid iteration count: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (printMessages) printUsage(argv[0]);
            return 2;
        } else {
            if (printMessages) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    if (options.nx == 0 || options.ny == 0 || options.nz == 0) {
        if (printMessages) std::printf("All grid dimensions must be positive.\n");
        return 1;
    }
    return 0;
}

bool checkedGridSize(const Options& options, size_t& gridSize) {
    if (options.nx > std::numeric_limits<size_t>::max() / options.ny) return false;
    const size_t plane = options.nx * options.ny;
    if (plane > std::numeric_limits<size_t>::max() / options.nz) return false;
    gridSize = plane * options.nz;
    return true;
}

inline size_t ceilDiv(const size_t n, const size_t d) {
    return n / d + static_cast<size_t>(n % d != 0);
}

// Choose the factorization that first uses the most ranks, then minimizes the
// exact aggregate number of face elements exchanged by a 3-D decomposition.
std::array<int, 3> selectProcessGrid(const int processCount,
                                     const std::array<size_t, 3>& global) {
    // If any dimension has no interior, every point is a fixed boundary and
    // there is no stencil work to distribute.
    if (global[0] == 0 || global[1] == 0 || global[2] == 0) {
        return {1, 1, processCount};
    }
    std::array<int, 3> best{1, 1, processCount};
    long long bestActive = -1;
    long double bestTraffic = std::numeric_limits<long double>::infinity();
    long double bestWorstSurface = std::numeric_limits<long double>::infinity();

    for (int px = 1; px <= processCount; ++px) {
        if (processCount % px != 0) continue;
        const int remaining = processCount / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0) continue;
            const int pz = remaining / py;

            const size_t ax = std::min(global[0], static_cast<size_t>(px));
            const size_t ay = std::min(global[1], static_cast<size_t>(py));
            const size_t az = std::min(global[2], static_cast<size_t>(pz));
            const long long active = static_cast<long long>(ax * ay * az);

            const long double traffic =
                static_cast<long double>(ax - 1) * global[1] * global[2] +
                static_cast<long double>(ay - 1) * global[0] * global[2] +
                static_cast<long double>(az - 1) * global[0] * global[1];

            const size_t lx = ceilDiv(global[0], ax);
            const size_t ly = ceilDiv(global[1], ay);
            const size_t lz = ceilDiv(global[2], az);
            const long double worstSurface =
                static_cast<long double>(ax > 1 ? (ax == 2 ? 1 : 2) : 0) * ly * lz +
                static_cast<long double>(ay > 1 ? (ay == 2 ? 1 : 2) : 0) * lx * lz +
                static_cast<long double>(az > 1 ? (az == 2 ? 1 : 2) : 0) * lx * ly;

            if (active > bestActive ||
                (active == bestActive && traffic < bestTraffic) ||
                (active == bestActive && traffic == bestTraffic &&
                 worstSurface < bestWorstSurface)) {
                best = {px, py, pz};
                bestActive = active;
                bestTraffic = traffic;
                bestWorstSurface = worstSurface;
            }
        }
    }
    return best;
}

void block1D(const size_t globalCount, const int parts, const int coordinate,
             size_t& localCount, size_t& globalStart) {
    const size_t base = globalCount / static_cast<size_t>(parts);
    const size_t remainder = globalCount % static_cast<size_t>(parts);
    localCount = base +
                 (static_cast<size_t>(coordinate) < remainder ? size_t{1} : size_t{0});
    globalStart = 1 + static_cast<size_t>(coordinate) * base +
                  std::min(static_cast<size_t>(coordinate), remainder);
}

Decomposition createDecomposition(const Options& options, const int processCount) {
    Decomposition decomposition;
    decomposition.globalInterior = {
        options.nx > 2 ? options.nx - 2 : 0,
        options.ny > 2 ? options.ny - 2 : 0,
        options.nz > 2 ? options.nz - 2 : 0};
    decomposition.dims = selectProcessGrid(processCount, decomposition.globalInterior);
    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, decomposition.dims.data(), periods, 1,
                    &decomposition.comm);
    MPI_Comm_rank(decomposition.comm, &decomposition.rank);
    MPI_Comm_size(decomposition.comm, &decomposition.size);
    MPI_Cart_coords(decomposition.comm, decomposition.rank, 3,
                    decomposition.coords.data());

    for (int axis = 0; axis < 3; ++axis) {
        block1D(decomposition.globalInterior[axis], decomposition.dims[axis],
                decomposition.coords[axis], decomposition.local[axis],
                decomposition.start[axis]);
    }
    decomposition.active = decomposition.local[0] != 0 &&
                           decomposition.local[1] != 0 &&
                           decomposition.local[2] != 0;

    if (decomposition.active) {
        for (int axis = 0; axis < 3; ++axis) {
            int minus = MPI_PROC_NULL;
            int plus = MPI_PROC_NULL;
            MPI_Cart_shift(decomposition.comm, axis, 1, &minus, &plus);
            decomposition.neighbors[2 * axis] = minus;
            // When a process dimension is larger than the corresponding data
            // dimension, the trailing Cartesian ranks are deliberately idle.
            const size_t activeCoordinates = std::min(
                decomposition.globalInterior[axis],
                static_cast<size_t>(decomposition.dims[axis]));
            decomposition.neighbors[2 * axis + 1] =
                static_cast<size_t>(decomposition.coords[axis] + 1) < activeCoordinates
                    ? plus
                    : MPI_PROC_NULL;
        }
    }
    return decomposition;
}

inline size_t localIndex(const size_t x, const size_t y, const size_t z,
                         const size_t sx, const size_t plane) {
    return z * plane + y * sx + x;
}

void initializeLocalGrid(std::vector<Real>& first, std::vector<Real>& second,
                         const Decomposition& decomposition, const Options& options) {
    if (!decomposition.active) return;
    const size_t lx = decomposition.local[0];
    const size_t ly = decomposition.local[1];
    const size_t lz = decomposition.local[2];
    const size_t sx = lx + 2;
    const size_t plane = sx * (ly + 2);
    const size_t globalPlane = options.nx * options.ny;

    for (size_t z = 0; z < lz + 2; ++z) {
        const size_t gz = decomposition.start[2] + z - 1;
        for (size_t y = 0; y < ly + 2; ++y) {
            const size_t gy = decomposition.start[1] + y - 1;
            for (size_t x = 0; x < lx + 2; ++x) {
                const size_t gx = decomposition.start[0] + x - 1;
                const size_t globalIndex = gz * globalPlane + gy * options.nx + gx;
                const size_t index = localIndex(x, y, z, sx, plane);
                first[index] = static_cast<Real>(globalIndex % 19);
                // Physical boundary halos are invariant and either buffer can
                // become the input buffer after a swap.
                second[index] = first[index];
            }
        }
    }
}

FaceTypes createFaceTypes(const Decomposition& decomposition) {
    FaceTypes types;
    if (!decomposition.active) return types;

    const size_t lx = decomposition.local[0];
    const size_t ly = decomposition.local[1];
    const size_t lz = decomposition.local[2];
    if (lx + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ly + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        lz + 2 > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (decomposition.rank == 0) {
            std::fprintf(stderr, "Local dimensions exceed MPI datatype limits.\n");
        }
        MPI_Abort(decomposition.comm, 1);
    }

    const int sizes[3] = {static_cast<int>(lz + 2), static_cast<int>(ly + 2),
                          static_cast<int>(lx + 2)};
    const int xSubsizes[3] = {static_cast<int>(lz), static_cast<int>(ly), 1};
    const int xStarts[3] = {1, 1, 0};
    const int ySubsizes[3] = {static_cast<int>(lz), 1, static_cast<int>(lx)};
    const int yStarts[3] = {1, 0, 1};
    const int zSubsizes[3] = {1, static_cast<int>(ly), static_cast<int>(lx)};
    const int zStarts[3] = {0, 1, 1};

    MPI_Type_create_subarray(3, sizes, xSubsizes, xStarts, MPI_ORDER_C,
                             MPI_DOUBLE, &types.x);
    MPI_Type_create_subarray(3, sizes, ySubsizes, yStarts, MPI_ORDER_C,
                             MPI_DOUBLE, &types.y);
    MPI_Type_create_subarray(3, sizes, zSubsizes, zStarts, MPI_ORDER_C,
                             MPI_DOUBLE, &types.z);
    MPI_Type_commit(&types.x);
    MPI_Type_commit(&types.y);
    MPI_Type_commit(&types.z);
    return types;
}

void addPersistentReceive(RequestSet& requests, Real* address, MPI_Datatype type,
                          const int source, const int tag, MPI_Comm comm) {
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Recv_init(address, 1, type, source, tag, comm, &request);
    requests.receives.push_back(request);
}

void addPersistentSend(RequestSet& requests, Real* address, MPI_Datatype type,
                       const int destination, const int tag, MPI_Comm comm) {
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Send_init(address, 1, type, destination, tag, comm, &request);
    requests.sends.push_back(request);
}

RequestSet createRequests(std::vector<Real>& grid, const Decomposition& decomposition,
                          const FaceTypes& types) {
    RequestSet requests;
    requests.receives.reserve(6);
    requests.sends.reserve(6);
    if (!decomposition.active) return requests;

    const size_t lx = decomposition.local[0];
    const size_t ly = decomposition.local[1];
    const size_t lz = decomposition.local[2];
    const size_t sx = lx + 2;
    const size_t plane = sx * (ly + 2);
    Real* const data = grid.data();

    if (decomposition.neighbors[0] != MPI_PROC_NULL) {
        addPersistentReceive(requests, data, types.x, decomposition.neighbors[0],
                             X_TO_PLUS, decomposition.comm);
        addPersistentSend(requests, data + 1, types.x, decomposition.neighbors[0],
                          X_TO_MINUS, decomposition.comm);
    }
    if (decomposition.neighbors[1] != MPI_PROC_NULL) {
        addPersistentReceive(requests, data + lx + 1, types.x,
                             decomposition.neighbors[1], X_TO_MINUS,
                             decomposition.comm);
        addPersistentSend(requests, data + lx, types.x, decomposition.neighbors[1],
                          X_TO_PLUS, decomposition.comm);
    }
    if (decomposition.neighbors[2] != MPI_PROC_NULL) {
        addPersistentReceive(requests, data, types.y, decomposition.neighbors[2],
                             Y_TO_PLUS, decomposition.comm);
        addPersistentSend(requests, data + sx, types.y, decomposition.neighbors[2],
                          Y_TO_MINUS, decomposition.comm);
    }
    if (decomposition.neighbors[3] != MPI_PROC_NULL) {
        addPersistentReceive(requests, data + (ly + 1) * sx, types.y,
                             decomposition.neighbors[3], Y_TO_MINUS,
                             decomposition.comm);
        addPersistentSend(requests, data + ly * sx, types.y,
                          decomposition.neighbors[3], Y_TO_PLUS,
                          decomposition.comm);
    }
    if (decomposition.neighbors[4] != MPI_PROC_NULL) {
        addPersistentReceive(requests, data, types.z, decomposition.neighbors[4],
                             Z_TO_PLUS, decomposition.comm);
        addPersistentSend(requests, data + plane, types.z, decomposition.neighbors[4],
                          Z_TO_MINUS, decomposition.comm);
    }
    if (decomposition.neighbors[5] != MPI_PROC_NULL) {
        addPersistentReceive(requests, data + (lz + 1) * plane, types.z,
                             decomposition.neighbors[5], Z_TO_MINUS,
                             decomposition.comm);
        addPersistentSend(requests, data + lz * plane, types.z,
                          decomposition.neighbors[5], Z_TO_PLUS,
                          decomposition.comm);
    }
    return requests;
}

#if defined(__GNUC__) || defined(__clang__)
#define STENCIL_RESTRICT __restrict__
#else
#define STENCIL_RESTRICT
#endif

inline void computeBlock(const Real* STENCIL_RESTRICT input,
                         Real* STENCIL_RESTRICT output,
                         const size_t sx, const size_t plane,
                         const size_t xBegin, const size_t xEnd,
                         const size_t yBegin, const size_t yEnd,
                         const size_t zBegin, const size_t zEnd) {
    if (xBegin > xEnd || yBegin > yEnd || zBegin > zEnd) return;
    for (size_t z = zBegin; z <= zEnd; ++z) {
        for (size_t y = yBegin; y <= yEnd; ++y) {
            const size_t row = z * plane + y * sx;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t x = xBegin; x <= xEnd; ++x) {
                const size_t index = row + x;
                const Real center = input[index];
                const Real left = input[index - 1];
                const Real right = input[index + 1];
                const Real front = input[index - sx];
                const Real back = input[index + sx];
                const Real bottom = input[index - plane];
                const Real top = input[index + plane];
                output[index] =
                    (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                      RequestSet& requests, const Decomposition& decomposition) {
    if (!decomposition.active) return;
    if (!requests.receives.empty()) {
        MPI_Startall(static_cast<int>(requests.receives.size()), requests.receives.data());
    }
    if (!requests.sends.empty()) {
        MPI_Startall(static_cast<int>(requests.sends.size()), requests.sends.data());
    }

    const size_t lx = decomposition.local[0];
    const size_t ly = decomposition.local[1];
    const size_t lz = decomposition.local[2];
    const size_t sx = lx + 2;
    const size_t plane = sx * (ly + 2);

    // Cells adjacent to a physical boundary need no message; only trim a side
    // from the overlap region when that side actually has a remote neighbor.
    const size_t xBegin = decomposition.neighbors[0] == MPI_PROC_NULL ? 1 : 2;
    const size_t xEnd = decomposition.neighbors[1] == MPI_PROC_NULL ? lx : lx - 1;
    const size_t yBegin = decomposition.neighbors[2] == MPI_PROC_NULL ? 1 : 2;
    const size_t yEnd = decomposition.neighbors[3] == MPI_PROC_NULL ? ly : ly - 1;
    const size_t zBegin = decomposition.neighbors[4] == MPI_PROC_NULL ? 1 : 2;
    const size_t zEnd = decomposition.neighbors[5] == MPI_PROC_NULL ? lz : lz - 1;
    const bool hasIndependentCore =
        xBegin <= xEnd && yBegin <= yEnd && zBegin <= zEnd;

    if (hasIndependentCore) {
        computeBlock(input.data(), output.data(), sx, plane,
                     xBegin, xEnd, yBegin, yEnd, zBegin, zEnd);
    }

    if (!requests.receives.empty()) {
        MPI_Waitall(static_cast<int>(requests.receives.size()), requests.receives.data(),
                    MPI_STATUSES_IGNORE);
    }

    if (!hasIndependentCore) {
        computeBlock(input.data(), output.data(), sx, plane, 1, lx, 1, ly, 1, lz);
    } else {
        // Six non-overlapping slabs cover the communication-dependent shell.
        if (zBegin > 1) {
            computeBlock(input.data(), output.data(), sx, plane,
                         1, lx, 1, ly, 1, zBegin - 1);
        }
        if (zEnd < lz) {
            computeBlock(input.data(), output.data(), sx, plane,
                         1, lx, 1, ly, zEnd + 1, lz);
        }
        if (yBegin > 1) {
            computeBlock(input.data(), output.data(), sx, plane,
                         1, lx, 1, yBegin - 1, zBegin, zEnd);
        }
        if (yEnd < ly) {
            computeBlock(input.data(), output.data(), sx, plane,
                         1, lx, yEnd + 1, ly, zBegin, zEnd);
        }
        if (xBegin > 1) {
            computeBlock(input.data(), output.data(), sx, plane,
                         1, xBegin - 1, yBegin, yEnd, zBegin, zEnd);
        }
        if (xEnd < lx) {
            computeBlock(input.data(), output.data(), sx, plane,
                         xEnd + 1, lx, yBegin, yEnd, zBegin, zEnd);
        }
    }

    // The old input becomes an output buffer after the swap, so all sends that
    // refer to it must be complete before the next iteration begins.
    if (!requests.sends.empty()) {
        MPI_Waitall(static_cast<int>(requests.sends.size()), requests.sends.data(),
                    MPI_STATUSES_IGNORE);
    }
}

void freeRequests(RequestSet& requests) {
    for (MPI_Request& request : requests.receives) MPI_Request_free(&request);
    for (MPI_Request& request : requests.sends) MPI_Request_free(&request);
}

void freeFaceTypes(FaceTypes& types) {
    if (types.x != MPI_DATATYPE_NULL) MPI_Type_free(&types.x);
    if (types.y != MPI_DATATYPE_NULL) MPI_Type_free(&types.y);
    if (types.z != MPI_DATATYPE_NULL) MPI_Type_free(&types.z);
}

void initializeGlobalGrid(std::vector<Real>& grid, const Options& options) {
    const size_t plane = options.nx * options.ny;
    for (size_t z = 0; z < options.nz; ++z) {
        for (size_t y = 0; y < options.ny; ++y) {
            for (size_t x = 0; x < options.nx; ++x) {
                const size_t index = z * plane + y * options.nx + x;
                grid[index] = static_cast<Real>(index % 19);
            }
        }
    }
}

// Gather only for externally requested result reporting. The timed stencil and
// normal validation never replicate the global grid.
bool gatherGlobalGrid(const std::vector<Real>& localGrid,
                      std::vector<Real>& globalGrid,
                      const Decomposition& decomposition,
                      const Options& options, const size_t globalGridSize) {
    const size_t lx = decomposition.local[0];
    const size_t ly = decomposition.local[1];
    const size_t lz = decomposition.local[2];
    const size_t localCountSize = lx * ly * lz;
    int locallySupported = localCountSize <= static_cast<size_t>(std::numeric_limits<int>::max());
    int globallySupported = 0;
    MPI_Allreduce(&locallySupported, &globallySupported, 1, MPI_INT, MPI_MIN,
                  decomposition.comm);

    std::vector<int> counts;
    std::vector<int> displacements;
    size_t packedCount = 0;
    if (decomposition.rank == 0 && globallySupported) {
        counts.resize(decomposition.size);
        displacements.resize(decomposition.size);
        for (int rank = 0; rank < decomposition.size; ++rank) {
            std::array<int, 3> coordinates{};
            MPI_Cart_coords(decomposition.comm, rank, 3, coordinates.data());
            size_t count = 1;
            for (int axis = 0; axis < 3; ++axis) {
                size_t local = 0;
                size_t start = 0;
                block1D(decomposition.globalInterior[axis], decomposition.dims[axis],
                        coordinates[axis], local, start);
                count *= local;
            }
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                packedCount > static_cast<size_t>(std::numeric_limits<int>::max()) - count) {
                globallySupported = 0;
                break;
            }
            counts[rank] = static_cast<int>(count);
            displacements[rank] = static_cast<int>(packedCount);
            packedCount += count;
        }
    }
    MPI_Bcast(&globallySupported, 1, MPI_INT, 0, decomposition.comm);
    if (!globallySupported) return false;

    std::vector<Real> packedLocal(localCountSize);
    if (decomposition.active) {
        const size_t sx = lx + 2;
        const size_t plane = sx * (ly + 2);
        for (size_t z = 0; z < lz; ++z) {
            for (size_t y = 0; y < ly; ++y) {
                const Real* source = localGrid.data() +
                                     localIndex(1, y + 1, z + 1, sx, plane);
                Real* destination = packedLocal.data() + (z * ly + y) * lx;
                std::copy_n(source, lx, destination);
            }
        }
    }

    std::vector<Real> packedGlobal;
    if (decomposition.rank == 0) packedGlobal.resize(packedCount);
    MPI_Gatherv(packedLocal.data(), static_cast<int>(localCountSize), MPI_DOUBLE,
                decomposition.rank == 0 ? packedGlobal.data() : nullptr,
                decomposition.rank == 0 ? counts.data() : nullptr,
                decomposition.rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, decomposition.comm);

    if (decomposition.rank == 0) {
        globalGrid.resize(globalGridSize);
        initializeGlobalGrid(globalGrid, options);
        const size_t globalPlane = options.nx * options.ny;
        for (int rank = 0; rank < decomposition.size; ++rank) {
            std::array<int, 3> coordinates{};
            MPI_Cart_coords(decomposition.comm, rank, 3, coordinates.data());
            std::array<size_t, 3> local{};
            std::array<size_t, 3> start{};
            for (int axis = 0; axis < 3; ++axis) {
                block1D(decomposition.globalInterior[axis], decomposition.dims[axis],
                        coordinates[axis], local[axis], start[axis]);
            }
            if (counts[rank] == 0) continue;
            const Real* source = packedGlobal.data() + displacements[rank];
            for (size_t z = 0; z < local[2]; ++z) {
                for (size_t y = 0; y < local[1]; ++y) {
                    const size_t destination = (start[2] + z) * globalPlane +
                                               (start[1] + y) * options.nx + start[0];
                    const size_t sourceOffset = (z * local[1] + y) * local[0];
                    std::copy_n(source + sourceOffset, local[0],
                                globalGrid.data() + destination);
                }
            }
        }
    }
    return true;
}

bool validateResult(const std::vector<Real>& grid,
                    const Decomposition& decomposition, const Options& options) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localValid = 1;

    auto observe = [&](const Real value) {
        if (!std::isfinite(value)) localValid = 0;
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    };

    if (decomposition.active) {
        const size_t lx = decomposition.local[0];
        const size_t ly = decomposition.local[1];
        const size_t lz = decomposition.local[2];
        const size_t sx = lx + 2;
        const size_t plane = sx * (ly + 2);

        for (size_t z = 1; z <= lz; ++z) {
            for (size_t y = 1; y <= ly; ++y) {
                for (size_t x = 1; x <= lx; ++x) {
                    observe(grid[localIndex(x, y, z, sx, plane)]);
                }
            }
        }

        // Include only true physical halos. Internal halos contain the previous
        // iteration after the final exchange and are not part of the result.
        for (size_t z = 0; z < lz + 2; ++z) {
            const size_t gz = decomposition.start[2] + z - 1;
            for (size_t y = 0; y < ly + 2; ++y) {
                const size_t gy = decomposition.start[1] + y - 1;
                for (size_t x = 0; x < lx + 2; ++x) {
                    const size_t gx = decomposition.start[0] + x - 1;
                    if (gx == 0 || gx == options.nx - 1 ||
                        gy == 0 || gy == options.ny - 1 ||
                        gz == 0 || gz == options.nz - 1) {
                        observe(grid[localIndex(x, y, z, sx, plane)]);
                    }
                }
            }
        }
    } else if (decomposition.rank == 0 &&
               (decomposition.globalInterior[0] == 0 ||
                decomposition.globalInterior[1] == 0 ||
                decomposition.globalInterior[2] == 0)) {
        // With no interior cells, the whole grid is an immutable boundary.
        const size_t count = options.nx * options.ny * options.nz;
        for (size_t index = 0; index < count; ++index) {
            observe(static_cast<Real>(index % 19));
        }
    }

    int valid = 0;
    Real minimum = 0.0;
    Real maximum = 0.0;
    MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, decomposition.comm);
    MPI_Allreduce(&localMin, &minimum, 1, MPI_DOUBLE, MPI_MIN, decomposition.comm);
    MPI_Allreduce(&localMax, &maximum, 1, MPI_DOUBLE, MPI_MAX, decomposition.comm);

    if (decomposition.rank == 0) {
        if (!valid) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", minimum, maximum);
    }
    if (maximum > 1e6 || minimum < -1e6) {
        if (decomposition.rank == 0) {
            std::printf("Validation failed: values out of expected range\n");
        }
        valid = 0;
    }
    return valid != 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    Options options;
    const int parseResult = parseOptions(argc, argv, options, worldRank == 0);
    if (parseResult != 0) {
        MPI_Finalize();
        return parseResult == 2 ? 0 : 1;
    }

    size_t globalGridSize = 0;
    if (!checkedGridSize(options, globalGridSize)) {
        if (worldRank == 0) std::fprintf(stderr, "Grid size overflows size_t.\n");
        MPI_Finalize();
        return 1;
    }

    Decomposition decomposition = createDecomposition(options, processCount);
    const long long activeRanks =
        static_cast<long long>(std::min(decomposition.globalInterior[0],
                                        static_cast<size_t>(decomposition.dims[0]))) *
        static_cast<long long>(std::min(decomposition.globalInterior[1],
                                        static_cast<size_t>(decomposition.dims[1]))) *
        static_cast<long long>(std::min(decomposition.globalInterior[2],
                                        static_cast<size_t>(decomposition.dims[2])));

    if (decomposition.rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (Cartesian grid: %d x %d x %d)\n",
                    decomposition.size, decomposition.dims[0], decomposition.dims[1],
                    decomposition.dims[2]);
        if (activeRanks != decomposition.size) {
            std::printf("Active MPI processes: %lld (domain has fewer decomposition blocks)\n",
                        activeRanks);
        }
        std::printf("Initializing grid...\n");
    }

    size_t localStorageSize = 0;
    if (decomposition.active) {
        localStorageSize = (decomposition.local[0] + 2) *
                           (decomposition.local[1] + 2) *
                           (decomposition.local[2] + 2);
    }
    std::vector<Real> grid1(localStorageSize);
    std::vector<Real> grid2(localStorageSize);
    initializeLocalGrid(grid1, grid2, decomposition, options);

    FaceTypes faceTypes = createFaceTypes(decomposition);
    RequestSet grid1Requests = createRequests(grid1, decomposition, faceTypes);
    RequestSet grid2Requests = createRequests(grid2, decomposition, faceTypes);

    if (decomposition.rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(decomposition.comm);
    const double startTime = MPI_Wtime();

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        if (iteration % 2 == 0) {
            stencilIteration(grid1, grid2, grid1Requests, decomposition);
        } else {
            stencilIteration(grid2, grid1, grid2Requests, decomposition);
        }
    }

    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               decomposition.comm);

    if (decomposition.rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double cellUpdates =
            static_cast<double>(decomposition.globalInterior[0]) *
            static_cast<double>(decomposition.globalInterior[1]) *
            static_cast<double>(decomposition.globalInterior[2]) * options.iterations;
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid =
        options.iterations % 2 == 0 ? grid1 : grid2;
    int exitCode = 0;

    if (options.printResults) {
        std::vector<Real> globalGrid;
        if (!gatherGlobalGrid(finalGrid, globalGrid, decomposition, options,
                              globalGridSize)) {
            if (decomposition.rank == 0) {
                std::fprintf(stderr, "Result gathering exceeds MPI count limits.\n");
            }
            exitCode = 1;
        } else if (decomposition.rank == 0) {
            print_results(globalGrid, "Grid");
        }
    }

    if (options.validate && exitCode == 0) {
        if (decomposition.rank == 0) std::printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, decomposition, options);
        if (decomposition.rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (!valid) exitCode = 1;
    }

    freeRequests(grid1Requests);
    freeRequests(grid2Requests);
    freeFaceTypes(faceTypes);
    MPI_Comm_free(&decomposition.comm);
    MPI_Finalize();
    return exitCode;
}
