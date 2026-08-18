#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

struct HaloExchange {
    MPI_Request requests[4];
    int count = 0;
};

// Storage is Z-slab local with one ghost plane on each side.  The active
// cells therefore occupy z = [1, localNz].
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

HaloExchange beginHaloExchange(double* field, const size_t localNz,
                               const size_t planeSize, const int previous,
                               const int next, MPI_Comm communicator) {
    HaloExchange exchange;

    if (previous == MPI_PROC_NULL) {
        std::memcpy(field, field + planeSize, planeSize * sizeof(double));
    } else {
        MPI_Irecv(field, static_cast<int>(planeSize), MPI_DOUBLE, previous, 1,
                  communicator, &exchange.requests[exchange.count++]);
        MPI_Isend(field + planeSize, static_cast<int>(planeSize), MPI_DOUBLE,
                  previous, 0, communicator, &exchange.requests[exchange.count++]);
    }

    if (next == MPI_PROC_NULL) {
        std::memcpy(field + (localNz + 1) * planeSize,
                    field + localNz * planeSize, planeSize * sizeof(double));
    } else {
        MPI_Irecv(field + (localNz + 1) * planeSize, static_cast<int>(planeSize),
                  MPI_DOUBLE, next, 0, communicator,
                  &exchange.requests[exchange.count++]);
        MPI_Isend(field + localNz * planeSize, static_cast<int>(planeSize),
                  MPI_DOUBLE, next, 1, communicator,
                  &exchange.requests[exchange.count++]);
    }

    return exchange;
}

void finishHaloExchange(HaloExchange& exchange) {
    MPI_Waitall(exchange.count, exchange.requests, MPI_STATUSES_IGNORE);
}

void computeChemicalPotential(const double* c, double* mu, const size_t nx,
                              const size_t ny, const size_t zBegin,
                              const size_t zEnd, const double invDx2,
                              const double invDy2, const double invDz2,
                              const double gamma, const double eAA,
                              const double eBB, const double eAB) {
    const size_t planeSize = nx * ny;

    for (size_t z = zBegin; z < zEnd; ++z) {
        const double* const previousPlane = c + (z - 1) * planeSize;
        const double* const currentPlane = c + z * planeSize;
        const double* const nextPlane = c + (z + 1) * planeSize;
        double* const muPlane = mu + z * planeSize;

        for (size_t y = 0; y < ny; ++y) {
            const size_t previousY = (y == 0) ? 0 : y - 1;
            const size_t nextY = (y + 1 < ny) ? y + 1 : y;
            const double* const previousRow = currentPlane + previousY * nx;
            const double* const currentRow = currentPlane + y * nx;
            const double* const nextRow = currentPlane + nextY * nx;
            double* const muRow = muPlane + y * nx;

            for (size_t x = 0; x < nx; ++x) {
                const size_t previousX = (x == 0) ? 0 : x - 1;
                const size_t nextX = (x + 1 < nx) ? x + 1 : x;
                const double cv = currentRow[x];
                const double laplacian =
                    (currentRow[nextX] + currentRow[previousX] - 2.0 * cv) * invDx2 +
                    (nextRow[x] + previousRow[x] - 2.0 * cv) * invDy2 +
                    (nextPlane[y * nx + x] + previousPlane[y * nx + x] - 2.0 * cv) * invDz2;

                muRow[x] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                         + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            }
        }
    }
}

void cahnHilliardUpdate(const double* cold, const double* mu, double* cnew,
                        const size_t nx, const size_t ny, const size_t zBegin,
                        const size_t zEnd, const double scale,
                        const double invDx2, const double invDy2,
                        const double invDz2) {
    const size_t planeSize = nx * ny;

    for (size_t z = zBegin; z < zEnd; ++z) {
        const double* const previousMuPlane = mu + (z - 1) * planeSize;
        const double* const currentMuPlane = mu + z * planeSize;
        const double* const nextMuPlane = mu + (z + 1) * planeSize;
        const double* const coldPlane = cold + z * planeSize;
        double* const cnewPlane = cnew + z * planeSize;

        for (size_t y = 0; y < ny; ++y) {
            const size_t previousY = (y == 0) ? 0 : y - 1;
            const size_t nextY = (y + 1 < ny) ? y + 1 : y;
            const double* const previousMuRow = currentMuPlane + previousY * nx;
            const double* const currentMuRow = currentMuPlane + y * nx;
            const double* const nextMuRow = currentMuPlane + nextY * nx;
            const double* const coldRow = coldPlane + y * nx;
            double* const cnewRow = cnewPlane + y * nx;

            for (size_t x = 0; x < nx; ++x) {
                const size_t previousX = (x == 0) ? 0 : x - 1;
                const size_t nextX = (x + 1 < nx) ? x + 1 : x;
                const double muv = currentMuRow[x];
                const double laplacian =
                    (currentMuRow[nextX] + currentMuRow[previousX] - 2.0 * muv) * invDx2 +
                    (nextMuRow[x] + previousMuRow[x] - 2.0 * muv) * invDy2 +
                    (nextMuPlane[y * nx + x] + previousMuPlane[y * nx + x] - 2.0 * muv) * invDz2;
                cnewRow[x] = coldRow[x] + scale * laplacian;
            }
        }
    }
}

void initializeConcentration(double* c, const size_t nx, const size_t ny,
                             const size_t localNz, const size_t globalZStart,
                             const size_t globalVolume) {
    const size_t planeSize = nx * ny;
    for (size_t localZ = 0; localZ < localNz; ++localZ) {
        const size_t globalZ = globalZStart + localZ;
        double* const plane = c + (localZ + 1) * planeSize;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = globalZ * planeSize + y * nx + x;
                const double pseudo = (((linearId + 1) * 1299709) % globalVolume) /
                                      static_cast<double>(globalVolume);
                plane[y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const double* c, const size_t localNz, const size_t planeSize,
                    MPI_Comm communicator, const int rank) {
    bool localFinite = true;
    double localMinimum = c[planeSize];
    double localMaximum = c[planeSize];
    for (size_t z = 1; z <= localNz; ++z) {
        const double* const plane = c + z * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            const double value = plane[i];
            if (!std::isfinite(value)) {
                localFinite = false;
            }
            localMinimum = std::min(localMinimum, value);
            localMaximum = std::max(localMaximum, value);
        }
    }

    int finite = localFinite ? 1 : 0;
    int allFinite = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    MPI_Reduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, 0, communicator);
    MPI_Reduce(&localMinimum, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, communicator);
    MPI_Reduce(&localMaximum, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);

    if (rank != 0) {
        return true;
    }
    if (!allFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
    if (maximum > 10.0 || minimum < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printDistributedResults(const std::vector<double>& localValues,
                             const size_t localNz, const size_t planeSize,
                             const size_t nz, MPI_Comm communicator,
                             const int rank, const int size) {
    const int localCount = static_cast<int>(localNz * planeSize);
    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<double> globalValues;
    if (rank == 0) {
        receiveCounts.resize(size);
    }

    MPI_Gather(&localCount, 1, MPI_INT,
               rank == 0 ? receiveCounts.data() : nullptr, 1, MPI_INT,
               0, communicator);

    if (rank == 0) {
        displacements.resize(size);
        int offset = 0;
        for (int process = 0; process < size; ++process) {
            displacements[process] = offset;
            offset += receiveCounts[process];
        }
        globalValues.resize(nz * planeSize);
    }

    MPI_Gatherv(localValues.data() + planeSize, localCount, MPI_DOUBLE,
                rank == 0 ? globalValues.data() : nullptr,
                rank == 0 ? receiveCounts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE,
                0, communicator);

    if (rank == 0) {
        print_results(globalValues, "Concentration");
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

} // namespace

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
            printf("Grid dimensions must be greater than zero\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeSize = nx * ny;
    const size_t gridSize = planeSize * nz;
    if (planeSize > static_cast<size_t>(INT_MAX) || gridSize > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) {
            printf("Grid is too large for MPI count arguments\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int activeSize = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm activeCommunicator = MPI_COMM_NULL;
    const int active = worldRank < activeSize;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank,
                   &activeCommunicator);

    int exitStatus = 0;
    if (active) {
        int rank = 0;
        int size = 0;
        MPI_Comm_rank(activeCommunicator, &rank);
        MPI_Comm_size(activeCommunicator, &size);

        const size_t basePlanes = nz / static_cast<size_t>(size);
        const size_t remainder = nz % static_cast<size_t>(size);
        const size_t localNz = basePlanes + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        const size_t globalZStart = static_cast<size_t>(rank) * basePlanes +
                                    std::min(static_cast<size_t>(rank), remainder);
        const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int next = rank + 1 == size ? MPI_PROC_NULL : rank + 1;

        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
        }

        const double dx = 1.0;
        const double dy = 1.0;
        const double dz = 1.0;
        const double dt = 0.01;
        const double eAA = -(2.0 / 9.0);
        const double eBB = -(2.0 / 9.0);
        const double eAB = (2.0 / 9.0);
        const double gamma = 0.5;
        const double diffusivity = 1.0;
        const double invDx2 = 1.0 / (dx * dx);
        const double invDy2 = 1.0 / (dy * dy);
        const double invDz2 = 1.0 / (dz * dz);

        std::vector<double> cold((localNz + 2) * planeSize);
        std::vector<double> cnew((localNz + 2) * planeSize);
        std::vector<double> mu((localNz + 2) * planeSize);

        if (rank == 0) {
            printf("Initializing concentration field...\n");
        }
        initializeConcentration(cold.data(), nx, ny, localNz, globalZStart, gridSize);

        if (rank == 0) {
            printf("Running Cahn-Hilliard simulation...\n");
        }
        MPI_Barrier(activeCommunicator);
        const double start = MPI_Wtime();

        for (int t = 0; t < iterations; ++t) {
            HaloExchange cExchange = beginHaloExchange(cold.data(), localNz, planeSize,
                                                        previous, next, activeCommunicator);
            // MPI may still read its send buffers.  Only calculate planes whose
            // stencil does not touch either outgoing boundary plane until the
            // corresponding communication has completed.
            const bool overlapCommunication = localNz > 4;
            const size_t cInteriorBegin = previous == MPI_PROC_NULL ? 1 : 3;
            const size_t cInteriorEnd = next == MPI_PROC_NULL ? localNz + 1 : localNz - 1;
            if (overlapCommunication && cInteriorBegin < cInteriorEnd) {
                computeChemicalPotential(cold.data(), mu.data(), nx, ny,
                                         cInteriorBegin, cInteriorEnd,
                                         invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
            }
            finishHaloExchange(cExchange);
            if (!overlapCommunication) {
                computeChemicalPotential(cold.data(), mu.data(), nx, ny, 1,
                                         localNz + 1, invDx2, invDy2, invDz2,
                                         gamma, eAA, eBB, eAB);
            } else if (cInteriorBegin > 1) {
                computeChemicalPotential(cold.data(), mu.data(), nx, ny, 1,
                                         cInteriorBegin, invDx2, invDy2, invDz2,
                                         gamma, eAA, eBB, eAB);
            }
            if (overlapCommunication && cInteriorEnd < localNz + 1) {
                computeChemicalPotential(cold.data(), mu.data(), nx, ny,
                                         cInteriorEnd, localNz + 1, invDx2, invDy2,
                                         invDz2, gamma, eAA, eBB, eAB);
            }

            HaloExchange muExchange = beginHaloExchange(mu.data(), localNz, planeSize,
                                                         previous, next, activeCommunicator);
            const size_t muInteriorBegin = previous == MPI_PROC_NULL ? 1 : 3;
            const size_t muInteriorEnd = next == MPI_PROC_NULL ? localNz + 1 : localNz - 1;
            if (overlapCommunication && muInteriorBegin < muInteriorEnd) {
                cahnHilliardUpdate(cold.data(), mu.data(), cnew.data(), nx, ny,
                                    muInteriorBegin, muInteriorEnd,
                                    diffusivity * dt, invDx2, invDy2, invDz2);
            }
            finishHaloExchange(muExchange);
            if (!overlapCommunication) {
                cahnHilliardUpdate(cold.data(), mu.data(), cnew.data(), nx, ny, 1,
                                    localNz + 1, diffusivity * dt,
                                    invDx2, invDy2, invDz2);
            } else if (muInteriorBegin > 1) {
                cahnHilliardUpdate(cold.data(), mu.data(), cnew.data(), nx, ny, 1,
                                    muInteriorBegin, diffusivity * dt,
                                    invDx2, invDy2, invDz2);
            }
            if (overlapCommunication && muInteriorEnd < localNz + 1) {
                cahnHilliardUpdate(cold.data(), mu.data(), cnew.data(), nx, ny,
                                    muInteriorEnd, localNz + 1, diffusivity * dt,
                                    invDx2, invDy2, invDz2);
            }
            std::swap(cold, cnew);
        }

        const double localDuration = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
                   activeCommunicator);

        if (rank == 0) {
            const long durationMs = static_cast<long>(duration * 1000.0);
            printf("Computation time: %ld ms\n", durationMs);
            const double cellUpdates = static_cast<double>(gridSize) * iterations;
            const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            printDistributedResults(cold, localNz, planeSize, nz, activeCommunicator, rank, size);
        }

        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
            }
            const bool valid = validateResult(cold.data(), localNz, planeSize,
                                              activeCommunicator, rank);
            if (rank == 0) {
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitStatus = 1;
                }
            }
        }

        MPI_Comm_free(&activeCommunicator);
    }

    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
