#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

struct Config {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    int validate = 0;
    int printResults = 0;
};

struct LocalDomain {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t localNz;
    size_t zStart;
    size_t planeSize;
    int planeCount;
    int rank;
    int size;
};

bool checkedMultiply(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool parseSize(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
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

// Returns 0 to run, 1 for an error, and 2 after successfully printing help.
int parseArguments(const int argc, char** argv, Config& config) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], config.nx)) {
                printf("Invalid X grid size\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], config.ny)) {
                printf("Invalid Y grid size\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], config.nz)) {
                printf("Invalid Z grid size\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIterations(argv[++i], config.iterations)) {
                printf("Invalid iteration count\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            config.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            config.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 2;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (config.ny == 0) {
        config.ny = config.nx;
    }
    if (config.nz == 0) {
        config.nz = config.nx;
    }
    return 0;
}

void initializeGrid(std::vector<Real>& grid, const LocalDomain& domain) {
    const size_t firstGlobalIndex = domain.zStart * domain.planeSize;
    Real* const localGrid = grid.data() + domain.planeSize;
    const size_t localElements = domain.localNz * domain.planeSize;

    for (size_t localIndex = 0; localIndex < localElements; ++localIndex) {
        localGrid[localIndex] = static_cast<Real>((firstGlobalIndex + localIndex) % 19);
    }
}

// Compute a physical Z plane.  The source and destination include one halo
// plane on each side, so localZ is in [1, localNz].
inline void stencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                         const LocalDomain& domain, const size_t localZ) {
    const size_t planeOffset = localZ * domain.planeSize;
    const Real* const in = input.data() + planeOffset;
    Real* const out = output.data() + planeOffset;
    const size_t globalZ = domain.zStart + localZ - 1;

    // Every point in these planes is a global boundary point.  This also
    // handles thin X/Y domains without special-case loop bounds.
    if (globalZ == 0 || globalZ + 1 == domain.nz || domain.nx <= 2 || domain.ny <= 2) {
        std::copy_n(in, domain.planeSize, out);
        return;
    }

    std::copy_n(in, domain.nx, out);
    for (size_t y = 1; y + 1 < domain.ny; ++y) {
        const size_t row = y * domain.nx;
        out[row] = in[row];
        for (size_t x = 1; x + 1 < domain.nx; ++x) {
            const size_t cell = row + x;
            out[cell] = (in[cell] + in[cell - 1] + in[cell + 1] +
                         in[cell - domain.nx] + in[cell + domain.nx] +
                         in[cell - domain.planeSize] + in[cell + domain.planeSize]) / 7.0;
        }
        out[row + domain.nx - 1] = in[row + domain.nx - 1];
    }
    std::copy_n(in + domain.planeSize - domain.nx, domain.nx,
                out + domain.planeSize - domain.nx);
}

// Exchange Z halos asynchronously, computing planes that do not depend on a
// halo while the transfers are in flight.  A Z-slab makes every message a
// contiguous plane, avoiding packing and minimizing the per-step message count.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const LocalDomain& domain, const MPI_Comm communicator) {
    const int lowerRank = domain.zStart == 0 ? MPI_PROC_NULL : domain.rank - 1;
    const int upperRank = domain.zStart + domain.localNz == domain.nz
                              ? MPI_PROC_NULL
                              : domain.rank + 1;
    MPI_Request requests[4];

    MPI_Irecv(input.data(), domain.planeCount, MPI_DOUBLE, lowerRank, 12, communicator,
              &requests[0]);
    MPI_Irecv(input.data() + (domain.localNz + 1) * domain.planeSize,
              domain.planeCount, MPI_DOUBLE, upperRank, 11, communicator, &requests[1]);
    MPI_Isend(input.data() + domain.planeSize, domain.planeCount, MPI_DOUBLE, lowerRank,
              11, communicator, &requests[2]);
    MPI_Isend(input.data() + domain.localNz * domain.planeSize, domain.planeCount,
              MPI_DOUBLE, upperRank, 12, communicator, &requests[3]);

    // These planes only access local data, hence they can overlap the two halo exchanges.
    for (size_t localZ = 2; localZ < domain.localNz; ++localZ) {
        stencilPlane(input, output, domain, localZ);
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    stencilPlane(input, output, domain, 1);
    if (domain.localNz > 1) {
        stencilPlane(input, output, domain, domain.localNz);
    }
}

bool validateResult(const std::vector<Real>& grid, const LocalDomain& domain,
                    const MPI_Comm communicator) {
    const Real* const localGrid = grid.data() + domain.planeSize;
    const size_t localElements = domain.localNz * domain.planeSize;
    int localHasInvalid = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();

    for (size_t i = 0; i < localElements; ++i) {
        const Real value = localGrid[i];
        if (std::isnan(value) || std::isinf(value)) {
            localHasInvalid = 1;
        }
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }

    int hasInvalid = 0;
    Real minValue = 0.0;
    Real maxValue = 0.0;
    MPI_Reduce(&localHasInvalid, &hasInvalid, 1, MPI_INT, MPI_MAX, 0, communicator);
    MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, communicator);
    MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);

    if (domain.rank != 0) {
        return true;
    }
    if (hasInvalid != 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 1e6 || minValue < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    Config config;
    int startupStatus = 0;
    if (worldRank == 0) {
        startupStatus = parseArguments(argc, argv, config);
    }
    MPI_Bcast(&startupStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (startupStatus != 0) {
        MPI_Finalize();
        return startupStatus == 2 ? 0 : 1;
    }
    MPI_Bcast(&config, static_cast<int>(sizeof(config)), MPI_BYTE, 0, MPI_COMM_WORLD);

    const int activeRanks = config.nz < static_cast<size_t>(worldSize)
                                ? static_cast<int>(config.nz)
                                : worldSize;
    size_t planeSize = 0;
    size_t globalElements = 0;
    size_t maximumLocalPlanes = 0;
    bool configurationValid = checkedMultiply(config.nx, config.ny, planeSize) &&
                              checkedMultiply(planeSize, config.nz, globalElements) &&
                              planeSize <= static_cast<size_t>(INT_MAX);
    if (configurationValid) {
        const size_t basePlanes = config.nz / static_cast<size_t>(activeRanks);
        configurationValid = basePlanes <= std::numeric_limits<size_t>::max() - 3;
        if (configurationValid) {
            maximumLocalPlanes = basePlanes + 3;
            configurationValid = maximumLocalPlanes <= std::numeric_limits<size_t>::max() / planeSize;
        }
    }
    if (configurationValid && config.printResults != 0) {
        // MPI_Gatherv uses int counts and displacements.  The computation itself
        // remains distributed and is not subject to this reporting-only limit.
        configurationValid = globalElements <= static_cast<size_t>(INT_MAX);
    }
    if (!configurationValid) {
        if (worldRank == 0) {
            printf("Grid is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int color = worldRank < activeRanks ? 0 : MPI_UNDEFINED;
    MPI_Comm workCommunicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &workCommunicator);

    int exitCode = 0;
    if (color != MPI_UNDEFINED) {
        int workRank = 0;
        int workSize = 0;
        MPI_Comm_rank(workCommunicator, &workRank);
        MPI_Comm_size(workCommunicator, &workSize);

        const size_t basePlanes = config.nz / static_cast<size_t>(workSize);
        const size_t extraPlanes = config.nz % static_cast<size_t>(workSize);
        const size_t localNz = basePlanes + (static_cast<size_t>(workRank) < extraPlanes ? 1 : 0);
        const size_t zStart = basePlanes * static_cast<size_t>(workRank) +
                              std::min(static_cast<size_t>(workRank), extraPlanes);
        const LocalDomain domain{config.nx, config.ny, config.nz, localNz, zStart, planeSize,
                                 static_cast<int>(planeSize), workRank, workSize};

        std::vector<Real> grid1((localNz + 2) * planeSize);
        std::vector<Real> grid2((localNz + 2) * planeSize);

        if (workRank == 0) {
            printf("3D Stencil Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", config.nx, config.ny, config.nz);
            printf("Iterations: %d\n", config.iterations);
            printf("Validation: %s\n", config.validate != 0 ? "enabled" : "disabled");
            printf("MPI ranks: %d active of %d launched\n", workSize, worldSize);
            printf("Initializing grid...\n");
        }
        initializeGrid(grid1, domain);

        if (workRank == 0) {
            printf("Running stencil computation...\n");
        }
        MPI_Barrier(workCommunicator);
        const double start = MPI_Wtime();
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            if ((iteration & 1) == 0) {
                stencilIteration(grid1, grid2, domain, workCommunicator);
            } else {
                stencilIteration(grid2, grid1, domain, workCommunicator);
            }
        }
        const double localSeconds = MPI_Wtime() - start;
        double elapsedSeconds = 0.0;
        MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, workCommunicator);

        std::vector<Real>& finalGrid = (config.iterations % 2 == 0) ? grid1 : grid2;
        if (workRank == 0) {
            const long long milliseconds = static_cast<long long>(elapsedSeconds * 1000.0);
            printf("Computation time: %lld ms\n", milliseconds);
            const double interiorCells = (config.nx > 2 && config.ny > 2 && config.nz > 2)
                                             ? static_cast<double>(config.nx - 2) *
                                                   static_cast<double>(config.ny - 2) *
                                                   static_cast<double>(config.nz - 2)
                                             : 0.0;
            const double mcups = elapsedSeconds > 0.0
                                     ? interiorCells * static_cast<double>(config.iterations) /
                                           elapsedSeconds / 1.0e6
                                     : 0.0;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (config.printResults != 0) {
            std::vector<int> receiveCounts;
            std::vector<int> displacements;
            std::vector<Real> globalGrid;
            if (workRank == 0) {
                receiveCounts.resize(static_cast<size_t>(worldSize), 0);
                displacements.resize(static_cast<size_t>(worldSize), 0);
                for (int rank = 0; rank < workSize; ++rank) {
                    const size_t rankLocalNz = basePlanes +
                                               (static_cast<size_t>(rank) < extraPlanes ? 1 : 0);
                    const size_t rankZStart = basePlanes * static_cast<size_t>(rank) +
                                              std::min(static_cast<size_t>(rank), extraPlanes);
                    receiveCounts[static_cast<size_t>(rank)] =
                        static_cast<int>(rankLocalNz * planeSize);
                    displacements[static_cast<size_t>(rank)] =
                        static_cast<int>(rankZStart * planeSize);
                }
                globalGrid.resize(globalElements);
            }

            // This collective deliberately uses MPI_COMM_WORLD so ranks with no
            // physical planes still participate with a zero send count.
            MPI_Gatherv(finalGrid.data() + planeSize, static_cast<int>(localNz * planeSize),
                        MPI_DOUBLE, worldRank == 0 ? globalGrid.data() : nullptr,
                        worldRank == 0 ? receiveCounts.data() : nullptr,
                        worldRank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
            if (workRank == 0) {
                print_results(globalGrid, "Grid");
            }
        }

        if (config.validate != 0) {
            if (workRank == 0) {
                printf("Validating result...\n");
            }
            const bool valid = validateResult(finalGrid, domain, workCommunicator);
            if (workRank == 0) {
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }

        MPI_Comm_free(&workCommunicator);
    } else if (config.printResults != 0) {
        // Inactive ranks need a valid (but ignored) send buffer for MPI_Gatherv.
        MPI_Gatherv(nullptr, 0, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
