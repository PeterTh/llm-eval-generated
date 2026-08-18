#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The global domain is split into contiguous Z slabs.  Each local field has a
// halo plane at both ends, so the physical cells are in planes [1, localNz].
inline constexpr size_t idx2(const size_t x, const size_t y, const size_t nx) noexcept {
    return y * nx + x;
}

void beginHaloExchange(std::vector<double>& field, const size_t localNz, const size_t planeSize,
                       const int previousRank, const int nextRank, const int mpiPlaneSize,
                       const int tag, const MPI_Comm communicator,
                       MPI_Request requests[4], int& requestCount) {
    requestCount = 0;

    if (previousRank == MPI_PROC_NULL) {
        // Clamped global lower Z boundary.
        std::copy_n(field.data() + planeSize, planeSize, field.data());
    } else {
        MPI_Irecv(field.data(), mpiPlaneSize, MPI_DOUBLE, previousRank, tag, communicator,
                  &requests[requestCount++]);
        MPI_Isend(field.data() + planeSize, mpiPlaneSize, MPI_DOUBLE, previousRank, tag,
                  communicator, &requests[requestCount++]);
    }

    if (nextRank == MPI_PROC_NULL) {
        // Clamped global upper Z boundary.
        std::copy_n(field.data() + localNz * planeSize, planeSize,
                    field.data() + (localNz + 1) * planeSize);
    } else {
        MPI_Irecv(field.data() + (localNz + 1) * planeSize, mpiPlaneSize, MPI_DOUBLE,
                  nextRank, tag, communicator, &requests[requestCount++]);
        MPI_Isend(field.data() + localNz * planeSize, mpiPlaneSize, MPI_DOUBLE, nextRank, tag,
                  communicator, &requests[requestCount++]);
    }
}

void finishHaloExchange(MPI_Request requests[4], const int requestCount) {
    if (requestCount != 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
}

void computeChemicalPotentialPlanes(const std::vector<double>& concentration,
                                   std::vector<double>& chemicalPotential,
                                   const size_t nx, const size_t ny, const size_t firstPlane,
                                   const size_t lastPlane, const double invDx2, const double invDy2,
                                   const double invDz2, const double gamma, const double eAA,
                                   const double eBB, const double eAB) {
    const size_t planeSize = nx * ny;

    for (size_t z = firstPlane; z < lastPlane; ++z) {
        const double* const lowerPlane = concentration.data() + (z - 1) * planeSize;
        const double* const currentPlane = concentration.data() + z * planeSize;
        const double* const upperPlane = concentration.data() + (z + 1) * planeSize;
        double* const muPlane = chemicalPotential.data() + z * planeSize;

        for (size_t y = 0; y < ny; ++y) {
            const double* const lowerRow = currentPlane + idx2(0, y == 0 ? 0 : y - 1, nx);
            const double* const currentRow = currentPlane + idx2(0, y, nx);
            const double* const upperRow =
                currentPlane + idx2(0, y + 1 == ny ? y : y + 1, nx);
            const double* const lowerZRow = lowerPlane + idx2(0, y, nx);
            const double* const upperZRow = upperPlane + idx2(0, y, nx);
            double* const muRow = muPlane + idx2(0, y, nx);

            for (size_t x = 0; x < nx; ++x) {
                const double cv = currentRow[x];
                const size_t xMinus = x == 0 ? 0 : x - 1;
                const size_t xPlus = x + 1 == nx ? x : x + 1;
                const double cxx = (currentRow[xPlus] + currentRow[xMinus] - 2.0 * cv) * invDx2;
                const double cyy = (upperRow[x] + lowerRow[x] - 2.0 * cv) * invDy2;
                const double czz = (upperZRow[x] + lowerZRow[x] - 2.0 * cv) * invDz2;

                muRow[x] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                           3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
            }
        }
    }
}

void updateConcentrationPlanes(std::vector<double>& concentrationNew,
                               const std::vector<double>& concentrationOld,
                               const std::vector<double>& chemicalPotential, const size_t nx,
                               const size_t ny, const size_t firstPlane, const size_t lastPlane,
                               const double updateScale, const double invDx2, const double invDy2,
                               const double invDz2) {
    const size_t planeSize = nx * ny;

    for (size_t z = firstPlane; z < lastPlane; ++z) {
        const double* const lowerPlane = chemicalPotential.data() + (z - 1) * planeSize;
        const double* const currentPlane = chemicalPotential.data() + z * planeSize;
        const double* const upperPlane = chemicalPotential.data() + (z + 1) * planeSize;
        const double* const oldPlane = concentrationOld.data() + z * planeSize;
        double* const newPlane = concentrationNew.data() + z * planeSize;

        for (size_t y = 0; y < ny; ++y) {
            const double* const lowerRow = currentPlane + idx2(0, y == 0 ? 0 : y - 1, nx);
            const double* const currentRow = currentPlane + idx2(0, y, nx);
            const double* const upperRow =
                currentPlane + idx2(0, y + 1 == ny ? y : y + 1, nx);
            const double* const lowerZRow = lowerPlane + idx2(0, y, nx);
            const double* const upperZRow = upperPlane + idx2(0, y, nx);
            const double* const oldRow = oldPlane + idx2(0, y, nx);
            double* const newRow = newPlane + idx2(0, y, nx);

            for (size_t x = 0; x < nx; ++x) {
                const double mu = currentRow[x];
                const size_t xMinus = x == 0 ? 0 : x - 1;
                const size_t xPlus = x + 1 == nx ? x : x + 1;
                const double mxx = (currentRow[xPlus] + currentRow[xMinus] - 2.0 * mu) * invDx2;
                const double myy = (upperRow[x] + lowerRow[x] - 2.0 * mu) * invDy2;
                const double mzz = (upperZRow[x] + lowerZRow[x] - 2.0 * mu) * invDz2;
                newRow[x] = oldRow[x] + updateScale * (mxx + myy + mzz);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& concentration, const size_t nx, const size_t ny,
                             const size_t localNz, const size_t globalZStart,
                             const size_t globalNz) {
    const size_t planeSize = nx * ny;
    const size_t volume = planeSize * globalNz;

    for (size_t localZ = 0; localZ < localNz; ++localZ) {
        const size_t globalZ = globalZStart + localZ;
        double* const plane = concentration.data() + (localZ + 1) * planeSize;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = globalZ * planeSize + idx2(x, y, nx);
                const double pseudo =
                    (((linearId + 1) * 1299709) % volume) / static_cast<double>(volume);
                plane[idx2(x, y, nx)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& concentration, const size_t localNz,
                    const size_t planeSize, const int rank, const MPI_Comm communicator) {
    bool localHasInvalidValue = false;
    for (size_t z = 1; z <= localNz && !localHasInvalidValue; ++z) {
        const double* const plane = concentration.data() + z * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            if (std::isnan(plane[i]) || std::isinf(plane[i])) {
                localHasInvalidValue = true;
                break;
            }
        }
    }

    int hasInvalidValue = localHasInvalidValue ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &hasInvalidValue, 1, MPI_INT, MPI_MAX, communicator);
    if (hasInvalidValue != 0) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    const double* const firstPlane = concentration.data() + planeSize;
    double localMin = firstPlane[0];
    double localMax = firstPlane[0];
    for (size_t z = 1; z <= localNz; ++z) {
        const double* const plane = concentration.data() + z * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            localMin = std::min(localMin, plane[i]);
            localMax = std::max(localMax, plane[i]);
        }
    }

    double globalMin = localMin;
    double globalMax = localMax;
    MPI_Allreduce(MPI_IN_PLACE, &globalMin, 1, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(MPI_IN_PLACE, &globalMax, 1, MPI_DOUBLE, MPI_MAX, communicator);

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    return globalMax <= 10.0 && globalMin >= -10.0;
}

void printDistributedResults(const std::vector<double>& concentration, const size_t nx, const size_t ny,
                             const size_t nz, const size_t localNz, const int rank,
                             const int processCount, const MPI_Comm communicator) {
    const size_t planeSize = nx * ny;
    const size_t localElementCount = localNz * planeSize;
    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<double> globalConcentration;

    if (rank == 0) {
        receiveCounts.resize(processCount);
        displacements.resize(processCount);
        globalConcentration.resize(planeSize * nz);

        const size_t basePlanes = nz / static_cast<size_t>(processCount);
        const size_t extraPlanes = nz % static_cast<size_t>(processCount);
        for (int process = 0; process < processCount; ++process) {
            const size_t processIndex = static_cast<size_t>(process);
            const size_t processPlanes = basePlanes + (processIndex < extraPlanes ? 1 : 0);
            const size_t processStart = processIndex * basePlanes + std::min(processIndex, extraPlanes);
            receiveCounts[process] = static_cast<int>(processPlanes * planeSize);
            displacements[process] = static_cast<int>(processStart * planeSize);
        }
    }

    MPI_Gatherv(concentration.data() + planeSize, static_cast<int>(localElementCount), MPI_DOUBLE,
                rank == 0 ? globalConcentration.data() : nullptr,
                rank == 0 ? receiveCounts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, communicator);

    if (rank == 0) {
        print_results(globalConcentration, "Concentration");
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
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

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    int argumentStatus = 0;

    // Parse the same arguments on every rank; only rank zero reports messages.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
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
            argumentStatus = 1;
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            break;
        }
    }

    if (argumentStatus != 0) {
        MPI_Finalize();
        return argumentStatus;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0) {
        if (worldRank == 0) {
            printf("Grid dimensions must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Ranks beyond the number of Z planes cannot own a cell.  They leave the
    // active communicator, while all usable ranks still execute the MPI solver.
    const int activeProcessCount =
        std::min(worldSize, static_cast<int>(std::min(nz, static_cast<size_t>(worldSize))));
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeProcessCount ? 0 : MPI_UNDEFINED, worldRank,
                   &communicator);
    if (worldRank >= activeProcessCount) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int processCount = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processCount);

    const size_t planeSize = nx * ny;
    const size_t basePlanes = nz / static_cast<size_t>(processCount);
    const size_t extraPlanes = nz % static_cast<size_t>(processCount);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t localNz = basePlanes + (rankIndex < extraPlanes ? 1 : 0);
    const size_t globalZStart = rankIndex * basePlanes + std::min(rankIndex, extraPlanes);
    const size_t localElementCount = localNz * planeSize;
    const size_t gridSize = planeSize * nz;

    int oversizedMessage =
        (planeSize > static_cast<size_t>(std::numeric_limits<int>::max()) ||
         (printResults &&
          (localElementCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
           gridSize > static_cast<size_t>(std::numeric_limits<int>::max()))))
            ? 1
            : 0;
    MPI_Allreduce(MPI_IN_PLACE, &oversizedMessage, 1, MPI_INT, MPI_MAX, communicator);
    if (oversizedMessage != 0) {
        if (rank == 0) {
            printf("Grid is too large for this MPI implementation's count interface\n");
        }
        MPI_Comm_free(&communicator);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double diffusion = 1.0;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double updateScale = dt * diffusion;

    // Two extra planes per field provide the Z-direction ghost cells.
    std::vector<double> concentration((localNz + 2) * planeSize);
    std::vector<double> concentrationNew((localNz + 2) * planeSize);
    std::vector<double> chemicalPotential((localNz + 2) * planeSize);

    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(concentration, nx, ny, localNz, globalZStart, nz);

    const int previousRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int nextRank = rank + 1 == processCount ? MPI_PROC_NULL : rank + 1;
    const int mpiPlaneSize = static_cast<int>(planeSize);
    MPI_Request requests[4];

    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(communicator);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        int requestCount = 0;
        beginHaloExchange(concentration, localNz, planeSize, previousRank, nextRank, mpiPlaneSize,
                          0, communicator, requests, requestCount);

        // These planes only depend on local data, so communication is hidden by
        // the bulk of both stencil calculations whenever a slab has 3+ planes.
        computeChemicalPotentialPlanes(concentration, chemicalPotential, nx, ny, 2, localNz,
                                       invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
        finishHaloExchange(requests, requestCount);
        computeChemicalPotentialPlanes(concentration, chemicalPotential, nx, ny, 1, 2, invDx2,
                                       invDy2, invDz2, gamma, eAA, eBB, eAB);
        if (localNz > 1) {
            computeChemicalPotentialPlanes(concentration, chemicalPotential, nx, ny, localNz,
                                           localNz + 1, invDx2, invDy2, invDz2, gamma, eAA, eBB,
                                           eAB);
        }

        beginHaloExchange(chemicalPotential, localNz, planeSize, previousRank, nextRank,
                          mpiPlaneSize, 1, communicator, requests, requestCount);
        updateConcentrationPlanes(concentrationNew, concentration, chemicalPotential, nx, ny, 2,
                                  localNz, updateScale, invDx2, invDy2, invDz2);
        finishHaloExchange(requests, requestCount);
        updateConcentrationPlanes(concentrationNew, concentration, chemicalPotential, nx, ny, 1,
                                  2, updateScale, invDx2, invDy2, invDz2);
        if (localNz > 1) {
            updateConcentrationPlanes(concentrationNew, concentration, chemicalPotential, nx, ny,
                                      localNz, localNz + 1, updateScale, invDx2, invDy2, invDz2);
        }

        std::swap(concentration, concentrationNew);
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(duration * 1000.0);
        printf("Computation time: %ld ms\n", durationMilliseconds);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / duration / 1.0e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        printDistributedResults(concentration, nx, ny, nz, localNz, rank, processCount,
                                communicator);
    }

    int exitStatus = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        if (validateResult(concentration, localNz, planeSize, rank, communicator)) {
            if (rank == 0) {
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            exitStatus = 1;
        }
    }

    MPI_Comm_free(&communicator);
    MPI_Finalize();
    return exitStatus;
}
