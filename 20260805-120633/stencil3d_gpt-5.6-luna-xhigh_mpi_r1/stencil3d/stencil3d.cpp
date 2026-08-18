#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// The global grid is split into rectangular blocks.  Each local block has
// one ghost cell layer on every side, so all six halo exchanges are one-cell
// exchanges regardless of the process grid shape.
struct LocalGrid {
    const size_t ownedX;
    const size_t ownedY;
    const size_t ownedZ;
    const size_t nx;
    const size_t ny;
    const size_t nz;
    std::vector<Real> values;

    LocalGrid(const size_t x, const size_t y, const size_t z)
        : ownedX(x), ownedY(y), ownedZ(z), nx(x + 2), ny(y + 2), nz(z + 2),
          values(nx * ny * nz) {}

    inline size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return z * (nx * ny) + y * nx + x;
    }
};

struct Block {
    size_t start;
    size_t extent;
};

Block splitBlock(const size_t globalExtent, const int coordinate, const int processExtent) {
    const size_t base = globalExtent / static_cast<size_t>(processExtent);
    const size_t remainder = globalExtent % static_cast<size_t>(processExtent);
    const size_t coord = static_cast<size_t>(coordinate);
    return {
        coord * base + std::min(coord, remainder),
        base + (coord < remainder ? 1 : 0)
    };
}

// Choose a process grid that is both valid for the requested dimensions and
// close to minimizing the communicated surface area of each block.  MPI's
// generic MPI_Dims_create() cannot account for anisotropic grid dimensions
// and can otherwise assign more ranks to a dimension than it contains cells.
bool chooseProcessGrid(const int processCount, const size_t globalX,
                       const size_t globalY, const size_t globalZ, int processGrid[3]) {
    double bestSurface = std::numeric_limits<double>::max();
    processGrid[0] = 1;
    processGrid[1] = 1;
    processGrid[2] = processCount;

    for (int x = 1; x <= processCount; ++x) {
        if (processCount % x != 0 || static_cast<size_t>(x) > globalX) {
            continue;
        }
        const int remaining = processCount / x;
        for (int y = 1; y <= remaining; ++y) {
            if (remaining % y != 0 || static_cast<size_t>(y) > globalY) {
                continue;
            }
            const int z = remaining / y;
            if (static_cast<size_t>(z) > globalZ) {
                continue;
            }

            const double blockX = static_cast<double>(globalX) / x;
            const double blockY = static_cast<double>(globalY) / y;
            const double blockZ = static_cast<double>(globalZ) / z;
            const double surface = blockY * blockZ + blockX * blockZ + blockX * blockY;
            if (surface < bestSurface) {
                bestSurface = surface;
                processGrid[0] = x;
                processGrid[1] = y;
                processGrid[2] = z;
            }
        }
    }
    return bestSurface != std::numeric_limits<double>::max();
}

inline void computeCell(const LocalGrid& input, LocalGrid& output,
                        const size_t x, const size_t y, const size_t z) noexcept {
    const size_t index = input.index(x, y, z);
    const size_t rowStride = input.nx;
    const size_t planeStride = input.nx * input.ny;
    const Real* const source = input.values.data() + index;

    // Keep the operation and operand order identical to the original stencil.
    output.values[index] = (source[0] + source[-1] + source[1] +
                            source[-static_cast<ptrdiff_t>(rowStride)] +
                            source[rowStride] +
                            source[-static_cast<ptrdiff_t>(planeStride)] +
                            source[planeStride]) / 7.0;
}

void initializeGrid(LocalGrid& grid, const size_t globalX, const size_t globalY,
                    const size_t startX, const size_t startY, const size_t startZ) {
    for (size_t z = 1; z <= grid.ownedZ; ++z) {
        const size_t globalZ = startZ + z - 1;
        for (size_t y = 1; y <= grid.ownedY; ++y) {
            const size_t globalYIndex = startY + y - 1;
            for (size_t x = 1; x <= grid.ownedX; ++x) {
                const size_t globalXIndex = startX + x - 1;
                const size_t globalIndex =
                    globalZ * (globalX * globalY) + globalYIndex * globalX + globalXIndex;
                grid.values[grid.index(x, y, z)] = static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

// MPI subarray datatypes let the network layer read/write faces directly in
// the ghost-padded array.  This avoids a pack/unpack pass on every iteration.
struct HaloTypes {
    enum TypeIndex {
        XLowSend, XLowReceive, XHighSend, XHighReceive,
        YLowSend, YLowReceive, YHighSend, YHighReceive,
        ZLowSend, ZLowReceive, ZHighSend, ZHighReceive,
        TypeCount
    };

    MPI_Datatype types[TypeCount]{};

    explicit HaloTypes(const LocalGrid& grid) {
        const int sizes[3] = {
            static_cast<int>(grid.nz), static_cast<int>(grid.ny), static_cast<int>(grid.nx)
        };

        create(sizes, {static_cast<int>(grid.ownedZ), static_cast<int>(grid.ownedY), 1},
               {1, 1, 1}, types[XLowSend]);
        create(sizes, {static_cast<int>(grid.ownedZ), static_cast<int>(grid.ownedY), 1},
               {1, 1, 0}, types[XLowReceive]);
        create(sizes, {static_cast<int>(grid.ownedZ), static_cast<int>(grid.ownedY), 1},
               {1, 1, static_cast<int>(grid.ownedX)}, types[XHighSend]);
        create(sizes, {static_cast<int>(grid.ownedZ), static_cast<int>(grid.ownedY), 1},
               {1, 1, static_cast<int>(grid.ownedX + 1)}, types[XHighReceive]);

        create(sizes, {static_cast<int>(grid.ownedZ), 1, static_cast<int>(grid.ownedX)},
               {1, 1, 1}, types[YLowSend]);
        create(sizes, {static_cast<int>(grid.ownedZ), 1, static_cast<int>(grid.ownedX)},
               {1, 0, 1}, types[YLowReceive]);
        create(sizes, {static_cast<int>(grid.ownedZ), 1, static_cast<int>(grid.ownedX)},
               {1, static_cast<int>(grid.ownedY), 1}, types[YHighSend]);
        create(sizes, {static_cast<int>(grid.ownedZ), 1, static_cast<int>(grid.ownedX)},
               {1, static_cast<int>(grid.ownedY + 1), 1}, types[YHighReceive]);

        create(sizes, {1, static_cast<int>(grid.ownedY), static_cast<int>(grid.ownedX)},
               {1, 1, 1}, types[ZLowSend]);
        create(sizes, {1, static_cast<int>(grid.ownedY), static_cast<int>(grid.ownedX)},
               {0, 1, 1}, types[ZLowReceive]);
        create(sizes, {1, static_cast<int>(grid.ownedY), static_cast<int>(grid.ownedX)},
               {static_cast<int>(grid.ownedZ), 1, 1}, types[ZHighSend]);
        create(sizes, {1, static_cast<int>(grid.ownedY), static_cast<int>(grid.ownedX)},
               {static_cast<int>(grid.ownedZ + 1), 1, 1}, types[ZHighReceive]);
    }

    void release() noexcept {
        for (MPI_Datatype& type : types) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
            }
        }
    }

  private:
    static void create(const int sizes[3], const std::initializer_list<int> subsizes,
                       const std::initializer_list<int> starts, MPI_Datatype& type) {
        int subarraySizes[3];
        int subarrayStarts[3];
        std::copy(subsizes.begin(), subsizes.end(), subarraySizes);
        std::copy(starts.begin(), starts.end(), subarrayStarts);
        MPI_Type_create_subarray(3, sizes, subarraySizes, subarrayStarts,
                                 MPI_ORDER_C, MPI_DOUBLE, &type);
        MPI_Type_commit(&type);
    }
};

void exchangeHalos(LocalGrid& grid, const HaloTypes& halo, const int neighbors[6],
                   MPI_Comm communicator) {
    enum NeighborIndex { XMinus, XPlus, YMinus, YPlus, ZMinus, ZPlus };
    constexpr int xLowTag = 100;
    constexpr int xHighTag = 101;
    constexpr int yLowTag = 110;
    constexpr int yHighTag = 111;
    constexpr int zLowTag = 120;
    constexpr int zHighTag = 121;

    MPI_Request requests[12];
    int requestCount = 0;

    // Post all receives before sends.  The tag is the face being sent by the
    // peer, which makes the two directions match without any ordering stalls.
    MPI_Irecv(grid.values.data(), 1, halo.types[HaloTypes::XLowReceive],
              neighbors[XMinus], xHighTag, communicator, &requests[requestCount++]);
    MPI_Irecv(grid.values.data(), 1, halo.types[HaloTypes::XHighReceive],
              neighbors[XPlus], xLowTag, communicator, &requests[requestCount++]);
    MPI_Irecv(grid.values.data(), 1, halo.types[HaloTypes::YLowReceive],
              neighbors[YMinus], yHighTag, communicator, &requests[requestCount++]);
    MPI_Irecv(grid.values.data(), 1, halo.types[HaloTypes::YHighReceive],
              neighbors[YPlus], yLowTag, communicator, &requests[requestCount++]);
    MPI_Irecv(grid.values.data(), 1, halo.types[HaloTypes::ZLowReceive],
              neighbors[ZMinus], zHighTag, communicator, &requests[requestCount++]);
    MPI_Irecv(grid.values.data(), 1, halo.types[HaloTypes::ZHighReceive],
              neighbors[ZPlus], zLowTag, communicator, &requests[requestCount++]);

    MPI_Isend(grid.values.data(), 1, halo.types[HaloTypes::XLowSend],
              neighbors[XMinus], xLowTag, communicator, &requests[requestCount++]);
    MPI_Isend(grid.values.data(), 1, halo.types[HaloTypes::XHighSend],
              neighbors[XPlus], xHighTag, communicator, &requests[requestCount++]);
    MPI_Isend(grid.values.data(), 1, halo.types[HaloTypes::YLowSend],
              neighbors[YMinus], yLowTag, communicator, &requests[requestCount++]);
    MPI_Isend(grid.values.data(), 1, halo.types[HaloTypes::YHighSend],
              neighbors[YPlus], yHighTag, communicator, &requests[requestCount++]);
    MPI_Isend(grid.values.data(), 1, halo.types[HaloTypes::ZLowSend],
              neighbors[ZMinus], zLowTag, communicator, &requests[requestCount++]);
    MPI_Isend(grid.values.data(), 1, halo.types[HaloTypes::ZHighSend],
              neighbors[ZPlus], zHighTag, communicator, &requests[requestCount++]);

    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
}

void stencilIteration(LocalGrid& input, LocalGrid& output,
                      const size_t globalX, const size_t globalY, const size_t globalZ,
                      const size_t startX, const size_t startY, const size_t startZ,
                      const HaloTypes& halo, const int neighbors[6], MPI_Comm communicator) {
    // Exchange halos and compute the strictly local region concurrently.
    // This region excludes all six faces of the local block and needs no
    // remote values, so it can run while MPI progresses the exchanges.
    exchangeHalos(input, halo, neighbors, communicator);

    for (size_t z = 2; z < input.ownedZ; ++z) {
        for (size_t y = 2; y < input.ownedY; ++y) {
            for (size_t x = 2; x < input.ownedX; ++x) {
                computeCell(input, output, x, y, z);
            }
        }
    }

    // The local shell depends on the completed halo exchange.  Global domain
    // boundaries retain their input value, exactly as in the original code.
    for (size_t z = 1; z <= input.ownedZ; ++z) {
        const size_t globalZIndex = startZ + z - 1;
        for (size_t y = 1; y <= input.ownedY; ++y) {
            const size_t globalYIndex = startY + y - 1;
            for (size_t x = 1; x <= input.ownedX; ++x) {
                if (x > 1 && x < input.ownedX &&
                    y > 1 && y < input.ownedY &&
                    z > 1 && z < input.ownedZ) {
                    continue;
                }

                const size_t globalXIndex = startX + x - 1;
                const size_t index = input.index(x, y, z);
                if (globalXIndex == 0 || globalXIndex + 1 == globalX ||
                    globalYIndex == 0 || globalYIndex + 1 == globalY ||
                    globalZIndex == 0 || globalZIndex + 1 == globalZ) {
                    output.values[index] = input.values[index];
                } else {
                    computeCell(input, output, x, y, z);
                }
            }
        }
    }
}

void packOwned(const LocalGrid& grid, std::vector<Real>& packed) {
    packed.resize(grid.ownedX * grid.ownedY * grid.ownedZ);
    size_t position = 0;
    for (size_t z = 1; z <= grid.ownedZ; ++z) {
        for (size_t y = 1; y <= grid.ownedY; ++y) {
            for (size_t x = 1; x <= grid.ownedX; ++x) {
                packed[position++] = grid.values[grid.index(x, y, z)];
            }
        }
    }
}

void gatherGrid(const LocalGrid& localGrid, const size_t globalX, const size_t globalY,
                const size_t globalZ, MPI_Comm communicator, const int rank,
                const int processCount, const int processGrid[3],
                std::vector<Real>& globalGrid) {
    const size_t localSize = localGrid.ownedX * localGrid.ownedY * localGrid.ownedZ;
    const int localCount = static_cast<int>(localSize);
    std::vector<Real> packed;
    packOwned(localGrid, packed);

    std::vector<int> counts;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(processCount));
    }
    MPI_Gather(&localCount, 1, MPI_INT,
               rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, communicator);

    std::vector<int> displacements;
    std::vector<Real> gathered;
    if (rank == 0) {
        displacements.resize(static_cast<size_t>(processCount));
        int totalCount = 0;
        for (int process = 0; process < processCount; ++process) {
            displacements[static_cast<size_t>(process)] = totalCount;
            totalCount += counts[static_cast<size_t>(process)];
        }
        gathered.resize(static_cast<size_t>(totalCount));
    }

    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, communicator);

    if (rank != 0) {
        return;
    }

    globalGrid.assign(globalX * globalY * globalZ, 0.0);
    for (int process = 0; process < processCount; ++process) {
        int coordinates[3];
        MPI_Cart_coords(communicator, process, 3, coordinates);
        const Block xBlock = splitBlock(globalX, coordinates[0], processGrid[0]);
        const Block yBlock = splitBlock(globalY, coordinates[1], processGrid[1]);
        const Block zBlock = splitBlock(globalZ, coordinates[2], processGrid[2]);
        size_t position = static_cast<size_t>(displacements[static_cast<size_t>(process)]);
        for (size_t z = 0; z < zBlock.extent; ++z) {
            for (size_t y = 0; y < yBlock.extent; ++y) {
                for (size_t x = 0; x < xBlock.extent; ++x) {
                    const size_t globalIndex =
                        (zBlock.start + z) * (globalX * globalY) +
                        (yBlock.start + y) * globalX + xBlock.start + x;
                    globalGrid[globalIndex] = gathered[position++];
                }
            }
        }
    }
}

bool validateDistributed(const LocalGrid& grid, MPI_Comm communicator, const int rank) {
    int localInvalid = 0;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();

    for (size_t z = 1; z <= grid.ownedZ; ++z) {
        for (size_t y = 1; y <= grid.ownedY; ++y) {
            for (size_t x = 1; x <= grid.ownedX; ++x) {
                const Real value = grid.values[grid.index(x, y, z)];
                if (std::isnan(value) || std::isinf(value)) {
                    localInvalid = 1;
                } else {
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
            }
        }
    }

    int invalid = 0;
    MPI_Allreduce(&localInvalid, &invalid, 1, MPI_INT, MPI_MAX, communicator);
    if (invalid != 0) {
        if (rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    Real minValue = 0.0;
    Real maxValue = 0.0;
    MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, communicator);
    if (rank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
    }

    const bool valid = maxValue <= 1e6 && minValue >= -1e6;
    if (!valid && rank == 0) {
        std::printf("Validation failed: values out of expected range\n");
    }
    return valid;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    size_t globalX = 128;
    size_t globalY = 0;
    size_t globalZ = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            globalX = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            globalY = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            globalZ = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
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

    if (globalY == 0) {
        globalY = globalX;
    }
    if (globalZ == 0) {
        globalZ = globalX;
    }

    int processCount = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    if (globalX == 0 || globalY == 0 || globalZ == 0 || iterations < 0) {
        if (worldRank == 0) {
            std::printf("Grid dimensions must be positive and iterations must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    int processGrid[3] = {1, 1, 1};
    if (!chooseProcessGrid(processCount, globalX, globalY, globalZ, processGrid)) {
        if (worldRank == 0) {
            std::printf("MPI process grid is larger than the requested grid\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t maxSize = std::numeric_limits<size_t>::max();
    if ((globalX > maxSize / globalY) ||
        (globalX * globalY > maxSize / globalZ)) {
        if (worldRank == 0) {
            std::printf("Grid is too large\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm cartesian = MPI_COMM_NULL;
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, processGrid, periods, 1, &cartesian);
    int rank = 0;
    MPI_Comm_rank(cartesian, &rank);

    int coordinates[3];
    MPI_Cart_coords(cartesian, rank, 3, coordinates);
    const Block xBlock = splitBlock(globalX, coordinates[0], processGrid[0]);
    const Block yBlock = splitBlock(globalY, coordinates[1], processGrid[1]);
    const Block zBlock = splitBlock(globalZ, coordinates[2], processGrid[2]);

    if (xBlock.extent + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        yBlock.extent + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        zBlock.extent + 2 > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::printf("Local grid dimensions exceed MPI datatype limits\n");
        }
        MPI_Comm_free(&cartesian);
        MPI_Finalize();
        return 1;
    }

    int neighbors[6];
    MPI_Cart_shift(cartesian, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(cartesian, 1, 1, &neighbors[2], &neighbors[3]);
    MPI_Cart_shift(cartesian, 2, 1, &neighbors[4], &neighbors[5]);

    LocalGrid grid1(xBlock.extent, yBlock.extent, zBlock.extent);
    LocalGrid grid2(xBlock.extent, yBlock.extent, zBlock.extent);
    initializeGrid(grid1, globalX, globalY, xBlock.start, yBlock.start, zBlock.start);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", globalX, globalY, globalZ);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
        std::printf("Running stencil computation...\n");
        std::fflush(stdout);
    }

    HaloTypes halo(grid1);
    MPI_Barrier(cartesian);
    const double startTime = MPI_Wtime();

    LocalGrid* input = &grid1;
    LocalGrid* output = &grid2;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        stencilIteration(*input, *output, globalX, globalY, globalZ,
                         xBlock.start, yBlock.start, zBlock.start,
                         halo, neighbors, cartesian);
        std::swap(input, output);
    }

    MPI_Barrier(cartesian);
    const double localSeconds = MPI_Wtime() - startTime;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, cartesian);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);
        const size_t interiorX = globalX > 2 ? globalX - 2 : 0;
        const size_t interiorY = globalY > 2 ? globalY - 2 : 0;
        const size_t interiorZ = globalZ > 2 ? globalZ - 2 : 0;
        const double cellUpdates = static_cast<double>(interiorX) *
                                   static_cast<double>(interiorY) *
                                   static_cast<double>(interiorZ) *
                                   static_cast<double>(iterations);
        const double mcups = cellUpdates / elapsedSeconds / 1e6;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const LocalGrid& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> globalGrid;
    if (printResults) {
        // Gather only on request; normal benchmark runs keep memory and
        // communication proportional to each rank's local subdomain.
        const size_t localSize = finalGrid.ownedX * finalGrid.ownedY * finalGrid.ownedZ;
        if (localSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                std::printf("Result block is too large for MPI_Gatherv\n");
            }
            halo.release();
            MPI_Comm_free(&cartesian);
            MPI_Finalize();
            return 1;
        }
        gatherGrid(finalGrid, globalX, globalY, globalZ,
                   cartesian, rank, processCount, processGrid, globalGrid);
    }

    bool valid = true;
    if (printResults && rank == 0) {
        print_results(globalGrid, "Grid");
    }
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateDistributed(finalGrid, cartesian, rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    halo.release();
    MPI_Comm_free(&cartesian);
    MPI_Finalize();
    return valid ? 0 : 1;
}
