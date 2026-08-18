#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Cartesian coordinates are ordered z, y, x to match the storage order below.
enum Face : int {
    XMinus = 0,
    XPlus,
    YMinus,
    YPlus,
    ZMinus,
    ZPlus,
    FaceCount
};

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline constexpr Face opposite(const Face face) noexcept {
    return static_cast<Face>(static_cast<int>(face) ^ 1);
}

struct Block {
    size_t offset;
    size_t size;
};

struct Decomposition {
    std::array<int, 3> processGrid;  // x, y, z
    std::array<int, 3> coords;       // x, y, z
    Block x;
    Block y;
    Block z;
    std::array<int, FaceCount> neighbors;
};

struct HaloBuffers {
    std::array<std::vector<Real>, FaceCount> send;
    std::array<std::vector<Real>, FaceCount> receive;
};

Block blockForCoordinate(const size_t globalSize, const int coordinate,
                         const int processCount) {
    const size_t base = globalSize / static_cast<size_t>(processCount);
    const size_t remainder = globalSize % static_cast<size_t>(processCount);
    const size_t coordinateSize = static_cast<size_t>(coordinate);
    return {coordinateSize * base + std::min(coordinateSize, remainder),
            base + (coordinateSize < remainder ? 1 : 0)};
}

// Select the process grid that minimizes the per-rank halo surface area while
// respecting the dimensions of the global domain.
bool chooseProcessGrid(const int processCount, const size_t nx, const size_t ny,
                       const size_t nz, std::array<int, 3>& result) {
    double bestSurface = std::numeric_limits<double>::infinity();
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

            const double localX = static_cast<double>(nx) / px;
            const double localY = static_cast<double>(ny) / py;
            const double localZ = static_cast<double>(nz) / pz;
            const double surface = localY * localZ + localX * localZ + localX * localY;

            if (surface < bestSurface) {
                bestSurface = surface;
                result = {px, py, pz};
                found = true;
            }
        }
    }
    return found;
}

bool canRepresentGrid(const size_t nx, const size_t ny, const size_t nz) {
    return nx != 0 && ny != 0 && nz != 0 &&
           nx <= std::numeric_limits<size_t>::max() / ny &&
           nx * ny <= std::numeric_limits<size_t>::max() / nz;
}

inline size_t localIndex(const size_t x, const size_t y, const size_t z,
                         const Decomposition& decomposition) noexcept {
    // Each local allocation has a one-cell halo in every direction.
    return idx3(x, y, z, decomposition.x.size + 2, decomposition.y.size + 2);
}

void initializeGrid(std::vector<Real>& grid, const Decomposition& decomposition,
                    const size_t globalNx, const size_t globalNy) {
    for (size_t z = 1; z <= decomposition.z.size; ++z) {
        const size_t globalZ = decomposition.z.offset + z - 1;
        for (size_t y = 1; y <= decomposition.y.size; ++y) {
            const size_t globalY = decomposition.y.offset + y - 1;
            for (size_t x = 1; x <= decomposition.x.size; ++x) {
                const size_t globalX = decomposition.x.offset + x - 1;
                grid[localIndex(x, y, z, decomposition)] =
                    static_cast<Real>(idx3(globalX, globalY, globalZ, globalNx, globalNy) % 19);
            }
        }
    }
}

HaloBuffers makeHaloBuffers(const Decomposition& decomposition) {
    HaloBuffers buffers;
    const size_t xFaceSize = decomposition.y.size * decomposition.z.size;
    const size_t yFaceSize = decomposition.x.size * decomposition.z.size;
    const size_t zFaceSize = decomposition.x.size * decomposition.y.size;
    const std::array<size_t, FaceCount> faceSizes = {
        xFaceSize, xFaceSize, yFaceSize, yFaceSize, zFaceSize, zFaceSize};
    for (int face = 0; face < FaceCount; ++face) {
        if (decomposition.neighbors[face] != MPI_PROC_NULL) {
            buffers.send[face].resize(faceSizes[face]);
            buffers.receive[face].resize(faceSizes[face]);
        }
    }
    return buffers;
}

void packHalos(const std::vector<Real>& input, const Decomposition& decomposition,
               HaloBuffers& buffers) {
    if (decomposition.neighbors[XMinus] != MPI_PROC_NULL ||
        decomposition.neighbors[XPlus] != MPI_PROC_NULL) {
        size_t pos = 0;
        for (size_t z = 1; z <= decomposition.z.size; ++z) {
            for (size_t y = 1; y <= decomposition.y.size; ++y, ++pos) {
                if (decomposition.neighbors[XMinus] != MPI_PROC_NULL) {
                    buffers.send[XMinus][pos] = input[localIndex(1, y, z, decomposition)];
                }
                if (decomposition.neighbors[XPlus] != MPI_PROC_NULL) {
                    buffers.send[XPlus][pos] =
                        input[localIndex(decomposition.x.size, y, z, decomposition)];
                }
            }
        }
    }

    if (decomposition.neighbors[YMinus] != MPI_PROC_NULL ||
        decomposition.neighbors[YPlus] != MPI_PROC_NULL) {
        size_t pos = 0;
        for (size_t z = 1; z <= decomposition.z.size; ++z) {
            for (size_t x = 1; x <= decomposition.x.size; ++x, ++pos) {
                if (decomposition.neighbors[YMinus] != MPI_PROC_NULL) {
                    buffers.send[YMinus][pos] = input[localIndex(x, 1, z, decomposition)];
                }
                if (decomposition.neighbors[YPlus] != MPI_PROC_NULL) {
                    buffers.send[YPlus][pos] =
                        input[localIndex(x, decomposition.y.size, z, decomposition)];
                }
            }
        }
    }

    if (decomposition.neighbors[ZMinus] != MPI_PROC_NULL ||
        decomposition.neighbors[ZPlus] != MPI_PROC_NULL) {
        size_t pos = 0;
        for (size_t y = 1; y <= decomposition.y.size; ++y) {
            for (size_t x = 1; x <= decomposition.x.size; ++x, ++pos) {
                if (decomposition.neighbors[ZMinus] != MPI_PROC_NULL) {
                    buffers.send[ZMinus][pos] = input[localIndex(x, y, 1, decomposition)];
                }
                if (decomposition.neighbors[ZPlus] != MPI_PROC_NULL) {
                    buffers.send[ZPlus][pos] =
                        input[localIndex(x, y, decomposition.z.size, decomposition)];
                }
            }
        }
    }
}

void unpackHalos(std::vector<Real>& input, const Decomposition& decomposition,
                 const HaloBuffers& buffers) {
    size_t pos = 0;
    if (decomposition.neighbors[XMinus] != MPI_PROC_NULL) {
        for (size_t z = 1; z <= decomposition.z.size; ++z) {
            for (size_t y = 1; y <= decomposition.y.size; ++y) {
                input[localIndex(0, y, z, decomposition)] = buffers.receive[XMinus][pos++];
            }
        }
    }
    if (decomposition.neighbors[XPlus] != MPI_PROC_NULL) {
        pos = 0;
        for (size_t z = 1; z <= decomposition.z.size; ++z) {
            for (size_t y = 1; y <= decomposition.y.size; ++y) {
                input[localIndex(decomposition.x.size + 1, y, z, decomposition)] =
                    buffers.receive[XPlus][pos++];
            }
        }
    }

    if (decomposition.neighbors[YMinus] != MPI_PROC_NULL) {
        pos = 0;
        for (size_t z = 1; z <= decomposition.z.size; ++z) {
            for (size_t x = 1; x <= decomposition.x.size; ++x) {
                input[localIndex(x, 0, z, decomposition)] = buffers.receive[YMinus][pos++];
            }
        }
    }
    if (decomposition.neighbors[YPlus] != MPI_PROC_NULL) {
        pos = 0;
        for (size_t z = 1; z <= decomposition.z.size; ++z) {
            for (size_t x = 1; x <= decomposition.x.size; ++x) {
                input[localIndex(x, decomposition.y.size + 1, z, decomposition)] =
                    buffers.receive[YPlus][pos++];
            }
        }
    }

    if (decomposition.neighbors[ZMinus] != MPI_PROC_NULL) {
        pos = 0;
        for (size_t y = 1; y <= decomposition.y.size; ++y) {
            for (size_t x = 1; x <= decomposition.x.size; ++x) {
                input[localIndex(x, y, 0, decomposition)] = buffers.receive[ZMinus][pos++];
            }
        }
    }
    if (decomposition.neighbors[ZPlus] != MPI_PROC_NULL) {
        pos = 0;
        for (size_t y = 1; y <= decomposition.y.size; ++y) {
            for (size_t x = 1; x <= decomposition.x.size; ++x) {
                input[localIndex(x, y, decomposition.z.size + 1, decomposition)] =
                    buffers.receive[ZPlus][pos++];
            }
        }
    }
}

inline bool isPhysicalBoundary(const size_t x, const size_t y, const size_t z,
                               const Decomposition& decomposition, const size_t nx,
                               const size_t ny, const size_t nz) noexcept {
    const size_t globalX = decomposition.x.offset + x - 1;
    const size_t globalY = decomposition.y.offset + y - 1;
    const size_t globalZ = decomposition.z.offset + z - 1;
    return globalX == 0 || globalX + 1 == nx || globalY == 0 || globalY + 1 == ny ||
           globalZ == 0 || globalZ + 1 == nz;
}

inline void updateBoundaryCell(const std::vector<Real>& input, std::vector<Real>& output,
                               const size_t x, const size_t y, const size_t z,
                               const Decomposition& decomposition, const size_t nx,
                               const size_t ny, const size_t nz) {
    const size_t index = localIndex(x, y, z, decomposition);
    if (isPhysicalBoundary(x, y, z, decomposition, nx, ny, nz)) {
        output[index] = input[index];
    } else {
        output[index] = (input[index] + input[localIndex(x - 1, y, z, decomposition)] +
                         input[localIndex(x + 1, y, z, decomposition)] +
                         input[localIndex(x, y - 1, z, decomposition)] +
                         input[localIndex(x, y + 1, z, decomposition)] +
                         input[localIndex(x, y, z - 1, decomposition)] +
                         input[localIndex(x, y, z + 1, decomposition)]) /
                        7.0;
    }
}

void stencilCore(const std::vector<Real>& input, std::vector<Real>& output,
                 const Decomposition& decomposition) {
    const size_t strideX = decomposition.x.size + 2;
    const size_t strideZ = strideX * (decomposition.y.size + 2);
    const Real* const inputData = input.data();
    Real* const outputData = output.data();
    for (size_t z = 2; z < decomposition.z.size; ++z) {
        for (size_t y = 2; y < decomposition.y.size; ++y) {
            size_t index = z * strideZ + y * strideX + 2;
            for (size_t x = 2; x < decomposition.x.size; ++x, ++index) {
                outputData[index] = (inputData[index] + inputData[index - 1] + inputData[index + 1] +
                                     inputData[index - strideX] + inputData[index + strideX] +
                                     inputData[index - strideZ] + inputData[index + strideZ]) /
                                    7.0;
            }
        }
    }
}

void stencilBoundaryShell(const std::vector<Real>& input, std::vector<Real>& output,
                          const Decomposition& decomposition, const size_t nx, const size_t ny,
                          const size_t nz) {
    // X faces cover every cell when the local x extent is one or two.  The
    // remaining face passes deliberately exclude already written cells.
    for (size_t z = 1; z <= decomposition.z.size; ++z) {
        for (size_t y = 1; y <= decomposition.y.size; ++y) {
            updateBoundaryCell(input, output, 1, y, z, decomposition, nx, ny, nz);
            if (decomposition.x.size > 1) {
                updateBoundaryCell(input, output, decomposition.x.size, y, z, decomposition, nx, ny,
                                   nz);
            }
        }
    }

    if (decomposition.x.size > 2) {
        for (size_t z = 1; z <= decomposition.z.size; ++z) {
            for (size_t x = 2; x < decomposition.x.size; ++x) {
                updateBoundaryCell(input, output, x, 1, z, decomposition, nx, ny, nz);
                if (decomposition.y.size > 1) {
                    updateBoundaryCell(input, output, x, decomposition.y.size, z, decomposition, nx, ny,
                                       nz);
                }
            }
        }
    }

    if (decomposition.x.size > 2 && decomposition.y.size > 2) {
        for (size_t y = 2; y < decomposition.y.size; ++y) {
            for (size_t x = 2; x < decomposition.x.size; ++x) {
                updateBoundaryCell(input, output, x, y, 1, decomposition, nx, ny, nz);
                if (decomposition.z.size > 1) {
                    updateBoundaryCell(input, output, x, y, decomposition.z.size, decomposition, nx, ny,
                                       nz);
                }
            }
        }
    }
}

// Starts all transfers, evaluates the subdomain core while they progress, then
// completes the boundary shell after the received halos are available.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const Decomposition& decomposition, const size_t nx,
                      const size_t ny, const size_t nz, HaloBuffers& buffers,
                      const MPI_Comm communicator) {
    packHalos(input, decomposition, buffers);

    std::array<MPI_Request, 2 * FaceCount> requests;
    int requestCount = 0;
    for (int face = 0; face < FaceCount; ++face) {
        const Face direction = static_cast<Face>(face);
        const int neighbor = decomposition.neighbors[face];
        if (neighbor == MPI_PROC_NULL) {
            continue;
        }
        MPI_Irecv(buffers.receive[face].data(), static_cast<int>(buffers.receive[face].size()),
                  MPI_DOUBLE, neighbor, static_cast<int>(opposite(direction)), communicator,
                  &requests[requestCount++]);
    }
    for (int face = 0; face < FaceCount; ++face) {
        const int neighbor = decomposition.neighbors[face];
        if (neighbor == MPI_PROC_NULL) {
            continue;
        }
        MPI_Isend(buffers.send[face].data(), static_cast<int>(buffers.send[face].size()), MPI_DOUBLE,
                  neighbor, face, communicator, &requests[requestCount++]);
    }

    stencilCore(input, output, decomposition);
    MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE);
    unpackHalos(input, decomposition, buffers);
    stencilBoundaryShell(input, output, decomposition, nx, ny, nz);
}

void packLocalBlock(const std::vector<Real>& grid, const Decomposition& decomposition,
                    std::vector<Real>& packed) {
    packed.resize(decomposition.x.size * decomposition.y.size * decomposition.z.size);
    size_t pos = 0;
    for (size_t z = 1; z <= decomposition.z.size; ++z) {
        for (size_t y = 1; y <= decomposition.y.size; ++y) {
            for (size_t x = 1; x <= decomposition.x.size; ++x) {
                packed[pos++] = grid[localIndex(x, y, z, decomposition)];
            }
        }
    }
}

std::vector<Real> gatherGlobalGrid(const std::vector<Real>& localGrid,
                                   const Decomposition& decomposition, const size_t nx,
                                   const size_t ny, const size_t nz,
                                   const MPI_Comm communicator) {
    int rank = 0;
    int processCount = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processCount);

    std::vector<Real> packed;
    packLocalBlock(localGrid, decomposition, packed);
    const int localCount = static_cast<int>(packed.size());

    std::vector<int> counts;
    if (rank == 0) {
        counts.resize(processCount);
    }
    MPI_Gather(&localCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
               communicator);

    std::vector<int> displacements;
    std::vector<Real> gathered;
    if (rank == 0) {
        displacements.resize(processCount);
        int totalCount = 0;
        for (int process = 0; process < processCount; ++process) {
            displacements[process] = totalCount;
            totalCount += counts[process];
        }
        gathered.resize(static_cast<size_t>(totalCount));
    }
    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, communicator);

    if (rank != 0) {
        return {};
    }

    std::vector<Real> global(nx * ny * nz);
    for (int process = 0; process < processCount; ++process) {
        int cartCoords[3] = {};
        MPI_Cart_coords(communicator, process, 3, cartCoords);
        const Block blockX = blockForCoordinate(nx, cartCoords[2], decomposition.processGrid[0]);
        const Block blockY = blockForCoordinate(ny, cartCoords[1], decomposition.processGrid[1]);
        const Block blockZ = blockForCoordinate(nz, cartCoords[0], decomposition.processGrid[2]);
        size_t pos = static_cast<size_t>(displacements[process]);
        for (size_t z = 0; z < blockZ.size; ++z) {
            for (size_t y = 0; y < blockY.size; ++y) {
                for (size_t x = 0; x < blockX.size; ++x) {
                    global[idx3(blockX.offset + x, blockY.offset + y, blockZ.offset + z, nx, ny)] =
                        gathered[pos++];
                }
            }
        }
    }
    return global;
}

bool validateResult(const std::vector<Real>& grid, const Decomposition& decomposition,
                    const MPI_Comm communicator) {
    bool localFinite = true;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= decomposition.z.size; ++z) {
        for (size_t y = 1; y <= decomposition.y.size; ++y) {
            for (size_t x = 1; x <= decomposition.x.size; ++x) {
                const Real value = grid[localIndex(x, y, z, decomposition)];
                localFinite = localFinite && std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
    }

    int finite = localFinite ? 1 : 0;
    int globallyFinite = 0;
    Real globalMin = 0;
    Real globalMax = 0;
    MPI_Allreduce(&finite, &globallyFinite, 1, MPI_INT, MPI_LAND, communicator);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, communicator);

    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    if (rank == 0) {
        if (!globallyFinite) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }
    return globallyFinite != 0 && globalMax <= 1e6 && globalMin >= -1e6;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128;
    size_t ny = 0;  // Set to nx if not specified.
    size_t nz = 0;  // Set to nx if not specified.
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    int argumentStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            argumentStatus = 1;
        }
    }

    if (argumentStatus != 0) {
        MPI_Finalize();
        return argumentStatus;
    }
    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    if (!canRepresentGrid(nx, ny, nz) || iterations < 0) {
        if (worldRank == 0) {
            printf("Invalid grid dimensions or iteration count\n");
        }
        MPI_Finalize();
        return 1;
    }

    std::array<int, 3> processGrid{};
    int activeProcessCount = worldSize;
    while (activeProcessCount > 0 &&
           !chooseProcessGrid(activeProcessCount, nx, ny, nz, processGrid)) {
        --activeProcessCount;
    }
    if (activeProcessCount == 0) {
        if (worldRank == 0) {
            printf("Unable to decompose grid across MPI ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    const bool active = worldRank < activeProcessCount;
    MPI_Comm activeCommunicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeCommunicator);

    int exitStatus = 0;
    if (active) {
        int cartDimensions[3] = {processGrid[2], processGrid[1], processGrid[0]};
        int periods[3] = {0, 0, 0};
        MPI_Comm cartesianCommunicator = MPI_COMM_NULL;
        MPI_Cart_create(activeCommunicator, 3, cartDimensions, periods, 0, &cartesianCommunicator);

        int cartRank = 0;
        MPI_Comm_rank(cartesianCommunicator, &cartRank);
        int cartCoords[3] = {};
        MPI_Cart_coords(cartesianCommunicator, cartRank, 3, cartCoords);

        Decomposition decomposition{};
        decomposition.processGrid = processGrid;
        decomposition.coords = {cartCoords[2], cartCoords[1], cartCoords[0]};
        decomposition.x = blockForCoordinate(nx, decomposition.coords[0], processGrid[0]);
        decomposition.y = blockForCoordinate(ny, decomposition.coords[1], processGrid[1]);
        decomposition.z = blockForCoordinate(nz, decomposition.coords[2], processGrid[2]);
        MPI_Cart_shift(cartesianCommunicator, 2, 1, &decomposition.neighbors[XMinus],
                       &decomposition.neighbors[XPlus]);
        MPI_Cart_shift(cartesianCommunicator, 1, 1, &decomposition.neighbors[YMinus],
                       &decomposition.neighbors[YPlus]);
        MPI_Cart_shift(cartesianCommunicator, 0, 1, &decomposition.neighbors[ZMinus],
                       &decomposition.neighbors[ZPlus]);

        if (worldRank == 0) {
            printf("3D Stencil Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Iterations: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d, process grid: %d x %d x %d\n", activeProcessCount,
                   processGrid[0], processGrid[1], processGrid[2]);
            printf("Initializing grid...\n");
        }

        const size_t localGridSize = (decomposition.x.size + 2) *
                                     (decomposition.y.size + 2) * (decomposition.z.size + 2);
        std::vector<Real> grid1(localGridSize);
        std::vector<Real> grid2(localGridSize);
        initializeGrid(grid1, decomposition, nx, ny);
        HaloBuffers buffers = makeHaloBuffers(decomposition);

        if (worldRank == 0) {
            printf("Running stencil computation...\n");
        }
        MPI_Barrier(cartesianCommunicator);
        const double start = MPI_Wtime();

        std::vector<Real>* current = &grid1;
        std::vector<Real>* next = &grid2;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stencilIteration(*current, *next, decomposition, nx, ny, nz, buffers,
                             cartesianCommunicator);
            std::swap(current, next);
        }

        const double localDuration = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
                   cartesianCommunicator);

        if (worldRank == 0) {
            const long long milliseconds = static_cast<long long>(duration * 1000.0);
            printf("Computation time: %lld ms\n", milliseconds);
            const double interiorX = nx > 2 ? static_cast<double>(nx - 2) : 0.0;
            const double interiorY = ny > 2 ? static_cast<double>(ny - 2) : 0.0;
            const double interiorZ = nz > 2 ? static_cast<double>(nz - 2) : 0.0;
            const double cellUpdates = interiorX * interiorY * interiorZ * iterations;
            const double mcups = duration > 0.0 ? cellUpdates / duration / 1.0e6 : 0.0;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            const std::vector<Real> finalGrid =
                gatherGlobalGrid(*current, decomposition, nx, ny, nz, cartesianCommunicator);
            if (worldRank == 0) {
                print_results(finalGrid, "Grid");
            }
        }

        if (validate) {
            if (worldRank == 0) {
                printf("Validating result...\n");
            }
            const bool valid = validateResult(*current, decomposition, cartesianCommunicator);
            if (worldRank == 0) {
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                exitStatus = valid ? 0 : 1;
            }
        }

        MPI_Comm_free(&cartesianCommunicator);
        MPI_Comm_free(&activeCommunicator);
    }

    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
