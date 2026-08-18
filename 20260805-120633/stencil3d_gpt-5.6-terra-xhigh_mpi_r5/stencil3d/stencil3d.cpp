#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define STENCIL_RESTRICT __restrict
#else
#define STENCIL_RESTRICT
#endif

// Initialize only owned planes.  The two extra planes in each local grid are
// receive halos and are populated before every iteration.
void initializeGrid(std::vector<Real>& grid, const size_t planeCells,
                    const size_t localNz, const size_t globalZStart) {
    for (size_t localZ = 0; localZ < localNz; ++localZ) {
        Real* const plane = grid.data() + (localZ + 1) * planeCells;
        const size_t globalOffset = (globalZStart + localZ) * planeCells;
        for (size_t i = 0; i < planeCells; ++i) {
            plane[i] = static_cast<Real>((globalOffset + i) % 19);
        }
    }
}

// Copy the global boundary cells, which retain their values at every step.
// Interior z planes only need their four x/y faces copied.
void copyBoundaryValues(const Real* const input, Real* const output,
                        const size_t nx, const size_t ny, const size_t nz,
                        const size_t localNz, const size_t globalZStart) {
    const size_t planeCells = nx * ny;
    const size_t rowBytes = nx * sizeof(Real);

    for (size_t localZ = 1; localZ <= localNz; ++localZ) {
        const size_t globalZ = globalZStart + localZ - 1;
        const Real* const inPlane = input + localZ * planeCells;
        Real* const outPlane = output + localZ * planeCells;

        if (globalZ == 0 || globalZ + 1 == nz) {
            std::memcpy(outPlane, inPlane, planeCells * sizeof(Real));
            continue;
        }

        std::memcpy(outPlane, inPlane, rowBytes);
        std::memcpy(outPlane + (ny - 1) * nx,
                    inPlane + (ny - 1) * nx, rowBytes);
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = y * nx;
            outPlane[row] = inPlane[row];
            outPlane[row + nx - 1] = inPlane[row + nx - 1];
        }
    }
}

// Compute a contiguous set of local z planes.  firstLocalZ and lastLocalZ
// include the local-grid halo offset, so owned planes are [1, localNz].
void stencilInteriorPlanes(const Real* STENCIL_RESTRICT input,
                           Real* STENCIL_RESTRICT output,
                           const size_t nx, const size_t ny,
                           const size_t firstLocalZ,
                           const size_t lastLocalZ) {
    if (firstLocalZ > lastLocalZ) {
        return;
    }

    const size_t planeCells = nx * ny;
    for (size_t localZ = firstLocalZ; localZ <= lastLocalZ; ++localZ) {
        const Real* const bottom = input + (localZ - 1) * planeCells;
        const Real* const center = input + localZ * planeCells;
        const Real* const top = input + (localZ + 1) * planeCells;
        Real* const out = output + localZ * planeCells;

        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = y * nx;
            const Real* const bottomRow = bottom + row;
            const Real* const centerRow = center + row;
            const Real* const topRow = top + row;
            Real* const outRow = out + row;

            for (size_t x = 1; x + 1 < nx; ++x) {
                outRow[x] = (centerRow[x] + centerRow[x - 1] + centerRow[x + 1]
                             + centerRow[x - nx] + centerRow[x + nx]
                             + bottomRow[x] + topRow[x]) / 7.0;
            }
        }
    }
}

int beginHaloExchange(std::vector<Real>& input, const size_t localNz,
                      const int planeCount, const int activeRank,
                      const int activeSize, MPI_Comm communicator,
                      MPI_Request requests[4]) {
    const int lowerRank = activeRank > 0 ? activeRank - 1 : MPI_PROC_NULL;
    const int upperRank = activeRank + 1 < activeSize ? activeRank + 1 : MPI_PROC_NULL;
    Real* const firstOwnedPlane = input.data() + planeCount;
    Real* const lastOwnedPlane = input.data() + static_cast<size_t>(localNz) * planeCount;
    int requestCount = 0;

    // A lower neighbour sends its last plane upward with tag 1, while an
    // upper neighbour sends its first plane downward with tag 0.
    if (lowerRank != MPI_PROC_NULL) {
        MPI_Irecv(input.data(), planeCount, MPI_DOUBLE, lowerRank, 1,
                  communicator, &requests[requestCount++]);
        MPI_Isend(firstOwnedPlane, planeCount, MPI_DOUBLE, lowerRank, 0,
                  communicator, &requests[requestCount++]);
    }
    if (upperRank != MPI_PROC_NULL) {
        MPI_Irecv(input.data() + static_cast<size_t>(localNz + 1) * planeCount,
                  planeCount, MPI_DOUBLE, upperRank, 0, communicator,
                  &requests[requestCount++]);
        MPI_Isend(lastOwnedPlane, planeCount, MPI_DOUBLE, upperRank, 1,
                  communicator, &requests[requestCount++]);
    }
    return requestCount;
}

bool validateResult(const std::vector<Real>& grid, const size_t localNz,
                    const size_t planeCells, MPI_Comm communicator,
                    const int activeRank) {
    const Real* const owned = grid.data() + planeCells;
    const size_t localCells = localNz * planeCells;
    int localInvalid = 0;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();

    for (size_t i = 0; i < localCells; ++i) {
        const Real value = owned[i];
        if (!std::isfinite(value)) {
            localInvalid = 1;
        } else {
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    }

    int invalid = 0;
    Real minValue = 0.0;
    Real maxValue = 0.0;
    MPI_Allreduce(&localInvalid, &invalid, 1, MPI_INT, MPI_MAX, communicator);
    MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, communicator);

    if (activeRank == 0 && invalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
    }
    if (activeRank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
    }
    if (minValue < -1e6 || maxValue > 1e6) {
        if (activeRank == 0) {
            std::printf("Validation failed: values out of expected range\n");
        }
        return false;
    }
    return invalid == 0;
}

void gatherFinalGrid(const std::vector<Real>& localGrid,
                     std::vector<Real>& globalGrid, const size_t nx,
                     const size_t ny, const size_t nz, const size_t localNz,
                     const int activeRank, const int activeSize,
                     MPI_Comm communicator) {
    const size_t planeCells = nx * ny;
    const int sendCount = static_cast<int>(localNz * planeCells);
    std::vector<int> receiveCounts;
    std::vector<int> displacements;

    if (activeRank == 0) {
        globalGrid.resize(nz * planeCells);
        receiveCounts.resize(activeSize);
        displacements.resize(activeSize);

        const size_t basePlanes = nz / static_cast<size_t>(activeSize);
        const size_t remainder = nz % static_cast<size_t>(activeSize);
        size_t displacement = 0;
        for (int rank = 0; rank < activeSize; ++rank) {
            const size_t rankPlanes = basePlanes
                + (static_cast<size_t>(rank) < remainder ? 1 : 0);
            receiveCounts[rank] = static_cast<int>(rankPlanes * planeCells);
            displacements[rank] = static_cast<int>(displacement);
            displacement += rankPlanes * planeCells;
        }
    }

    MPI_Gatherv(localGrid.data() + planeCells, sendCount, MPI_DOUBLE,
                activeRank == 0 ? globalGrid.data() : nullptr,
                activeRank == 0 ? receiveCounts.data() : nullptr,
                activeRank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, communicator);
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
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (worldRank == 0) {
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

    const bool invalidDimensions = nx < 3 || ny < 3 || nz < 3
        || nx > std::numeric_limits<size_t>::max() / ny
        || nx * ny > std::numeric_limits<size_t>::max() / nz;
    if (invalidDimensions || iterations < 0) {
        if (worldRank == 0) {
            std::printf("Grid dimensions must be at least 3 and iterations must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeCells = nx * ny;
    const size_t gridCells = planeCells * nz;
    if (planeCells > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::printf("A grid plane is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (printResults && gridCells > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::printf("Grid is too large to gather for result printing\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int activeSize = nz < static_cast<size_t>(worldSize)
        ? static_cast<int>(nz) : worldSize;
    MPI_Comm activeCommunicator = MPI_COMM_NULL;
    const int color = worldRank < activeSize ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &activeCommunicator);

    int exitCode = 0;
    if (activeCommunicator != MPI_COMM_NULL) {
        int activeRank = 0;
        MPI_Comm_rank(activeCommunicator, &activeRank);

        const size_t basePlanes = nz / static_cast<size_t>(activeSize);
        const size_t remainder = nz % static_cast<size_t>(activeSize);
        const size_t localNz = basePlanes
            + (static_cast<size_t>(activeRank) < remainder ? 1 : 0);
        const size_t globalZStart = static_cast<size_t>(activeRank) * basePlanes
            + std::min(static_cast<size_t>(activeRank), remainder);
        const size_t localCellsWithHalos = (localNz + 2) * planeCells;

        std::vector<Real> grid1(localCellsWithHalos);
        std::vector<Real> grid2(localCellsWithHalos);

        if (activeRank == 0) {
            std::printf("3D Stencil Benchmark\n");
            std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            std::printf("Iterations: %d\n", iterations);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
            std::printf("Initializing grid...\n");
        }
        initializeGrid(grid1, planeCells, localNz, globalZStart);

        if (activeRank == 0) {
            std::printf("Running stencil computation...\n");
        }
        MPI_Barrier(activeCommunicator);
        const double startTime = MPI_Wtime();

        for (int iteration = 0; iteration < iterations; ++iteration) {
            std::vector<Real>& input = iteration % 2 == 0 ? grid1 : grid2;
            std::vector<Real>& output = iteration % 2 == 0 ? grid2 : grid1;
            MPI_Request requests[4];
            const int requestCount = beginHaloExchange(
                input, localNz, static_cast<int>(planeCells), activeRank,
                activeSize, activeCommunicator, requests);

            copyBoundaryValues(input.data(), output.data(), nx, ny, nz,
                               localNz, globalZStart);

            // These planes use only local data, so their work hides halo
            // communication.  The first and last planes are done after wait.
            if (localNz > 2) {
                stencilInteriorPlanes(input.data(), output.data(), nx, ny, 2,
                                      localNz - 1);
            }
            if (requestCount != 0) {
                MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
            }

            if (globalZStart != 0 && globalZStart + 1 != nz) {
                stencilInteriorPlanes(input.data(), output.data(), nx, ny, 1, 1);
            }
            if (localNz > 1 && globalZStart + localNz != nz) {
                stencilInteriorPlanes(input.data(), output.data(), nx, ny,
                                      localNz, localNz);
            }
        }

        const double localElapsed = MPI_Wtime() - startTime;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                   activeCommunicator);

        if (activeRank == 0) {
            const long milliseconds = static_cast<long>(elapsed * 1000.0);
            const double cellUpdates = static_cast<double>(nx - 2)
                * static_cast<double>(ny - 2) * static_cast<double>(nz - 2)
                * static_cast<double>(iterations);
            const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
            std::printf("Computation time: %ld ms\n", milliseconds);
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        const std::vector<Real>& finalGrid = iterations % 2 == 0 ? grid1 : grid2;
        if (printResults) {
            std::vector<Real> gatheredGrid;
            gatherFinalGrid(finalGrid, gatheredGrid, nx, ny, nz, localNz,
                            activeRank, activeSize, activeCommunicator);
            if (activeRank == 0) {
                print_results(gatheredGrid, "Grid");
            }
        }

        if (validate) {
            if (activeRank == 0) {
                std::printf("Validating result...\n");
            }
            if (validateResult(finalGrid, localNz, planeCells,
                               activeCommunicator, activeRank)) {
                if (activeRank == 0) {
                    std::printf("Validation: PASSED\n");
                }
            } else {
                if (activeRank == 0) {
                    std::printf("Validation: FAILED\n");
                }
                exitCode = 1;
            }
        }

        MPI_Comm_free(&activeCommunicator);
    }

    MPI_Finalize();
    return exitCode;
}
