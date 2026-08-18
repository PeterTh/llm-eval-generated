#include <algorithm>
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

// The local allocation has one halo plane at each end.  Planes 1 through
// localNz contain this rank's contiguous portion of the global Z dimension.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t localNz, const size_t firstGlobalZ) {
    const size_t planeSize = nx * ny;
    for (size_t localZ = 1; localZ <= localNz; ++localZ) {
        const size_t globalOffset = (firstGlobalZ + localZ - 1) * planeSize;
        Real* const plane = grid.data() + localZ * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            plane[i] = static_cast<Real>((globalOffset + i) % 19);
        }
    }
}

// Update a single local plane.  Global boundaries are copied exactly as in
// the serial implementation; all other cells use the 7-point average.
inline void stencilPlane(const Real* const input, Real* const output,
                         const size_t localZ, const size_t firstGlobalZ,
                         const size_t nx, const size_t ny, const size_t nz) {
    const size_t planeSize = nx * ny;
    const size_t globalZ = firstGlobalZ + localZ - 1;
    const size_t planeOffset = localZ * planeSize;
    const Real* const src = input + planeOffset;
    Real* const dst = output + planeOffset;

    // This also handles degenerate X/Y dimensions, where every cell in the
    // plane is a physical boundary.
    if (globalZ == 0 || globalZ + 1 == nz || nx < 3 || ny < 3) {
        std::memcpy(dst, src, planeSize * sizeof(Real));
        return;
    }

    std::memcpy(dst, src, nx * sizeof(Real));
    std::memcpy(dst + (ny - 1) * nx, src + (ny - 1) * nx,
                nx * sizeof(Real));

    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t rowOffset = y * nx;
        dst[rowOffset] = src[rowOffset];
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t index = rowOffset + x;
            dst[index] = (src[index] + src[index - 1] + src[index + 1] +
                          src[index - nx] + src[index + nx] +
                          input[planeOffset - planeSize + index] +
                          input[planeOffset + planeSize + index]) /
                         7.0;
        }
        dst[rowOffset + nx - 1] = src[rowOffset + nx - 1];
    }
}

// Exchange Z halo planes and overlap communication with planes that need no
// remote data.  Neighbouring active ranks own consecutive global Z slabs.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t localNz, const size_t firstGlobalZ,
                      const int planeCount, const int previous,
                      const int next, const MPI_Comm communicator) {
    const size_t planeSize = nx * ny;
    Real* const in = input.data();

    MPI_Request requests[4];
    MPI_Irecv(in, planeCount, MPI_DOUBLE, previous, 1, communicator,
              &requests[0]);
    MPI_Irecv(in + (localNz + 1) * planeSize, planeCount, MPI_DOUBLE, next,
              0, communicator, &requests[1]);
    MPI_Isend(in + planeSize, planeCount, MPI_DOUBLE, previous, 0,
              communicator, &requests[2]);
    MPI_Isend(in + localNz * planeSize, planeCount, MPI_DOUBLE, next, 1,
              communicator, &requests[3]);

    // Planes strictly between the two ends only depend on local data.
    for (size_t localZ = 2; localZ < localNz; ++localZ) {
        stencilPlane(in, output.data(), localZ, firstGlobalZ, nx, ny, nz);
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    stencilPlane(in, output.data(), 1, firstGlobalZ, nx, ny, nz);
    if (localNz > 1) {
        stencilPlane(in, output.data(), localNz, firstGlobalZ, nx, ny, nz);
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t localNz,
                    const size_t nx, const size_t ny,
                    const MPI_Comm communicator, const int rank) {
    const size_t planeSize = nx * ny;
    bool localFinite = true;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();

    for (size_t localZ = 1; localZ <= localNz; ++localZ) {
        const Real* const plane = grid.data() + localZ * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            const Real value = plane[i];
            if (std::isnan(value) || std::isinf(value)) {
                localFinite = false;
            } else {
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
    }

    int finite = localFinite ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &finite, 1, MPI_INT, MPI_LAND, communicator);
    if (!finite) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0,
               communicator);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0,
               communicator);

    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    int valid = 1;
    if (rank == 0 && (globalMax > 1e6 || globalMin < -1e6)) {
        valid = 0;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, communicator);
    return valid != 0;
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

    unsigned long long dimensions[3] = {128, 0, 0};
    int iterations = 10;
    int validate = 0;
    int printResults = 0;
    int earlyExit = 0;

    // Only one rank interprets options and writes diagnostics.  The values
    // are then broadcast so every process has an identical configuration.
    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                dimensions[0] = static_cast<unsigned long long>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                dimensions[1] = static_cast<unsigned long long>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                dimensions[2] = static_cast<unsigned long long>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                earlyExit = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                earlyExit = 2;
                break;
            }
        }
    }

    MPI_Bcast(&earlyExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (earlyExit != 0) {
        MPI_Finalize();
        return earlyExit == 1 ? 0 : 1;
    }

    MPI_Bcast(dimensions, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (dimensions[1] == 0) dimensions[1] = dimensions[0];
    if (dimensions[2] == 0) dimensions[2] = dimensions[0];

    const size_t nx = static_cast<size_t>(dimensions[0]);
    const size_t ny = static_cast<size_t>(dimensions[1]);
    const size_t nz = static_cast<size_t>(dimensions[2]);

    // A zero extent is invalid for the original stencil and no non-empty
    // decomposition exists.  Fail collectively instead of risking overflow.
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) {
            printf("Invalid grid dimensions\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t activeSize = std::min(nz, static_cast<size_t>(worldSize));
    const int active = worldRank < static_cast<int>(activeSize) ? 1 : MPI_UNDEFINED;
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active, worldRank, &activeComm);

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    int exitCode = 0;
    if (active == 1) {
        int rank = 0;
        int size = 0;
        MPI_Comm_rank(activeComm, &rank);
        MPI_Comm_size(activeComm, &size);

        const size_t basePlanes = nz / static_cast<size_t>(size);
        const size_t remainder = nz % static_cast<size_t>(size);
        const size_t localNz = basePlanes + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        const size_t firstGlobalZ = static_cast<size_t>(rank) * basePlanes +
                                    std::min(static_cast<size_t>(rank), remainder);
        const size_t planeSize = nx * ny;
        const size_t localElements = localNz * planeSize;

        std::vector<Real> grid1((localNz + 2) * planeSize);
        std::vector<Real> grid2((localNz + 2) * planeSize);
        initializeGrid(grid1, nx, ny, localNz, firstGlobalZ);

        if (rank == 0) {
            printf("Running stencil computation...\n");
        }

        const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int next = rank + 1 == size ? MPI_PROC_NULL : rank + 1;
        const int planeCount = static_cast<int>(planeSize);

        MPI_Barrier(activeComm);
        const double start = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            if ((iter & 1) == 0) {
                stencilIteration(grid1, grid2, nx, ny, nz, localNz, firstGlobalZ,
                                 planeCount, previous, next, activeComm);
            } else {
                stencilIteration(grid2, grid1, nx, ny, nz, localNz, firstGlobalZ,
                                 planeCount, previous, next, activeComm);
            }
        }
        const double localElapsed = MPI_Wtime() - start;

        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);
        const long durationMs = static_cast<long>(elapsed * 1000.0);

        const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

        if (rank == 0) {
            printf("Computation time: %ld ms\n", durationMs);
            const double cellUpdates =
                static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
            const double mcups = cellUpdates / (static_cast<double>(durationMs) / 1000.0) / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            std::vector<int> counts;
            std::vector<int> displacements;
            std::vector<Real> globalGrid;
            if (rank == 0) {
                counts.resize(static_cast<size_t>(size));
                displacements.resize(static_cast<size_t>(size));
                for (int process = 0; process < size; ++process) {
                    const size_t processPlanes =
                        basePlanes + (static_cast<size_t>(process) < remainder ? 1 : 0);
                    counts[static_cast<size_t>(process)] =
                        static_cast<int>(processPlanes * planeSize);
                    displacements[static_cast<size_t>(process)] =
                        static_cast<int>((static_cast<size_t>(process) * basePlanes +
                                          std::min(static_cast<size_t>(process), remainder)) *
                                         planeSize);
                }
                globalGrid.resize(nx * ny * nz);
            }
            MPI_Gatherv(finalGrid.data() + planeSize, static_cast<int>(localElements),
                        MPI_DOUBLE, rank == 0 ? globalGrid.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                        activeComm);
            if (rank == 0) {
                print_results(globalGrid, "Grid");
            }
        }

        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
            }
            const bool valid = validateResult(finalGrid, localNz, nx, ny, activeComm, rank);
            if (rank == 0) {
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                exitCode = valid ? 0 : 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (activeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&activeComm);
    }
    MPI_Finalize();
    return exitCode;
}
