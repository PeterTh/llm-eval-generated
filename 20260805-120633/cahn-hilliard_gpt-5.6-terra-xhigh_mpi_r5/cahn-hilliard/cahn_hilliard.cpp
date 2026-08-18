#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The domain is split into contiguous z-slabs.  Each local array has a
// one-plane halo on both sides; owned planes are in [1, localNz].
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline double computeLaplacian(const std::vector<double>& field, const size_t nx,
                               const size_t ny, const size_t localNz,
                               const bool lowerGlobalBoundary,
                               const bool upperGlobalBoundary, const double dx,
                               const double dy, const double dz, const size_t x,
                               const size_t y, const size_t z) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : y;
    const size_t zp = (z < localNz) ? z + 1 : (upperGlobalBoundary ? z : z + 1);
    const size_t zn = (z > 1) ? z - 1 : (lowerGlobalBoundary ? z : z - 1);

    const size_t center = idx3(x, y, z, nx, ny);
    const double cxx = (field[idx3(xp, y, z, nx, ny)] + field[idx3(xn, y, z, nx, ny)] -
                        2.0 * field[center]) /
                       (dx * dx);
    const double cyy = (field[idx3(x, yp, z, nx, ny)] + field[idx3(x, yn, z, nx, ny)] -
                        2.0 * field[center]) /
                       (dy * dy);
    const double czz = (field[idx3(x, y, zp, nx, ny)] + field[idx3(x, y, zn, nx, ny)] -
                        2.0 * field[center]) /
                       (dz * dz);
    return cxx + cyy + czz;
}

void computeChemicalPotentialRange(const std::vector<double>& c,
                                   std::vector<double>& mu, const size_t nx,
                                   const size_t ny, const size_t localNz,
                                   const size_t zBegin, const size_t zEnd,
                                   const bool lowerGlobalBoundary,
                                   const bool upperGlobalBoundary,
                                   const double dx, const double dy,
                                   const double dz, const double gamma,
                                   const double eAA, const double eBB,
                                   const double eAB) {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = idx3(x, y, z, nx, ny);
                const double cv = c[index];
                mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                            3.0 * cv + cv * cv * cv -
                            gamma * computeLaplacian(c, nx, ny, localNz,
                                                     lowerGlobalBoundary,
                                                     upperGlobalBoundary, dx, dy,
                                                     dz, x, y, z);
            }
        }
    }
}

void cahnHilliardUpdateRange(std::vector<double>& cnew,
                             const std::vector<double>& cold,
                             const std::vector<double>& mu, const size_t nx,
                             const size_t ny, const size_t localNz,
                             const size_t zBegin, const size_t zEnd,
                             const bool lowerGlobalBoundary,
                             const bool upperGlobalBoundary, const double D,
                             const double dt, const double dx, const double dy,
                             const double dz) {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = idx3(x, y, z, nx, ny);
                cnew[index] = cold[index] +
                              dt * D * computeLaplacian(mu, nx, ny, localNz,
                                                         lowerGlobalBoundary,
                                                         upperGlobalBoundary, dx,
                                                         dy, dz, x, y, z);
            }
        }
    }
}

// Begin an exchange before calculating the interior planes.  This overlaps
// communication with useful work whenever a rank owns at least three planes.
void beginHaloExchange(std::vector<double>& field, const size_t planeSize,
                       const size_t localNz, const int previousRank,
                       const int nextRank, MPI_Comm comm,
                       MPI_Request requests[4]) {
    const int count = static_cast<int>(planeSize);
    int requestCount = 0;
    if (previousRank != MPI_PROC_NULL) {
        MPI_Irecv(field.data(), count, MPI_DOUBLE, previousRank, 1, comm,
                  &requests[requestCount++]);
        MPI_Isend(field.data() + planeSize, count, MPI_DOUBLE, previousRank, 0,
                  comm, &requests[requestCount++]);
    }
    if (nextRank != MPI_PROC_NULL) {
        MPI_Irecv(field.data() + (localNz + 1) * planeSize, count, MPI_DOUBLE,
                  nextRank, 0, comm, &requests[requestCount++]);
        MPI_Isend(field.data() + localNz * planeSize, count, MPI_DOUBLE,
                  nextRank, 1, comm, &requests[requestCount++]);
    }
    for (; requestCount < 4; ++requestCount) {
        requests[requestCount] = MPI_REQUEST_NULL;
    }
}

inline void completeHaloExchange(MPI_Request requests[4]) {
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

void initializeConcentration(std::vector<double>& c, const size_t nx,
                             const size_t ny, const size_t localNz,
                             const size_t globalZStart, const size_t globalNz) {
    const size_t planeSize = nx * ny;
    const size_t volume = planeSize * globalNz;
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = globalZStart + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t localIndex = idx3(x, y, z, nx, ny);
                const size_t linearId = globalZ * planeSize + y * nx + x;
                const double pseudo = (((linearId + 1) * 1299709) % volume) /
                                      static_cast<double>(volume);
                c[localIndex] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t planeSize,
                    const size_t localNz, MPI_Comm comm, const int rank) {
    bool localFinite = true;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    for (size_t i = planeSize; i < (localNz + 1) * planeSize; ++i) {
        const double value = c[i];
        if (!std::isfinite(value)) {
            localFinite = false;
            break;
        }
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }

    int finite = localFinite ? 1 : 0;
    int globallyFinite = 0;
    MPI_Allreduce(&finite, &globallyFinite, 1, MPI_INT, MPI_LAND, comm);
    if (!globallyFinite) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int valid = 1;
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            valid = 0;
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    const char* invalidOption = nullptr;

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
            parseStatus = 1;
            invalidOption = argv[i];
            break;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (parseStatus != 0 || nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (worldRank == 0) {
            if (parseStatus != 0) {
                printf("Unknown option: %s\n", invalidOption);
            } else {
                printf("Grid dimensions must be positive and time steps non-negative\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeSize = nx * ny;
    if (planeSize / nx != ny || nz > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        planeSize > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nz > std::numeric_limits<size_t>::max() / planeSize) {
        if (worldRank == 0) {
            printf("Grid is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Empty slabs never communicate.  They remain MPI participants, while the
    // active communicator contains one or more z planes per rank.
    const int activeSize = std::min(worldSize, static_cast<int>(nz));
    const bool active = worldRank < activeSize;
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);

    int exitCode = 0;
    if (active) {
        int rank = 0;
        MPI_Comm_rank(activeComm, &rank);

        const size_t basePlanes = nz / static_cast<size_t>(activeSize);
        const size_t extraPlanes = nz % static_cast<size_t>(activeSize);
        const size_t localNz = basePlanes + (static_cast<size_t>(rank) < extraPlanes ? 1 : 0);
        const size_t globalZStart = static_cast<size_t>(rank) * basePlanes +
                                    std::min(static_cast<size_t>(rank), extraPlanes);
        const bool lowerGlobalBoundary = rank == 0;
        const bool upperGlobalBoundary = rank == activeSize - 1;
        const int previousRank = lowerGlobalBoundary ? MPI_PROC_NULL : rank - 1;
        const int nextRank = upperGlobalBoundary ? MPI_PROC_NULL : rank + 1;
        const size_t localCellCount = localNz * planeSize;

        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d active of %d launched\n", activeSize, worldSize);
            printf("Initializing concentration field...\n");
        }

        std::vector<double> cold((localNz + 2) * planeSize);
        std::vector<double> cnew((localNz + 2) * planeSize);
        std::vector<double> mu((localNz + 2) * planeSize);
        initializeConcentration(cold, nx, ny, localNz, globalZStart, nz);

        if (rank == 0) {
            printf("Running Cahn-Hilliard simulation...\n");
        }
        MPI_Barrier(activeComm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            MPI_Request requests[4];
            beginHaloExchange(cold, planeSize, localNz, previousRank, nextRank,
                              activeComm, requests);
            if (localNz > 2) {
                computeChemicalPotentialRange(cold, mu, nx, ny, localNz, 2,
                                              localNz, lowerGlobalBoundary,
                                              upperGlobalBoundary, 1.0, 1.0, 1.0,
                                              0.5, -(2.0 / 9.0), -(2.0 / 9.0),
                                              2.0 / 9.0);
            }
            completeHaloExchange(requests);
            computeChemicalPotentialRange(cold, mu, nx, ny, localNz, 1,
                                          std::min(localNz, size_t{1}) + 1,
                                          lowerGlobalBoundary,
                                          upperGlobalBoundary, 1.0, 1.0, 1.0, 0.5,
                                          -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
            if (localNz > 1) {
                computeChemicalPotentialRange(cold, mu, nx, ny, localNz, localNz,
                                              localNz + 1, lowerGlobalBoundary,
                                              upperGlobalBoundary, 1.0, 1.0, 1.0,
                                              0.5, -(2.0 / 9.0), -(2.0 / 9.0),
                                              2.0 / 9.0);
            }

            beginHaloExchange(mu, planeSize, localNz, previousRank, nextRank,
                              activeComm, requests);
            if (localNz > 2) {
                cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, localNz, 2,
                                        localNz, lowerGlobalBoundary,
                                        upperGlobalBoundary, 1.0, 0.01, 1.0, 1.0,
                                        1.0);
            }
            completeHaloExchange(requests);
            cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, localNz, 1,
                                    std::min(localNz, size_t{1}) + 1,
                                    lowerGlobalBoundary, upperGlobalBoundary,
                                    1.0, 0.01, 1.0, 1.0, 1.0);
            if (localNz > 1) {
                cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, localNz, localNz,
                                        localNz + 1, lowerGlobalBoundary,
                                        upperGlobalBoundary, 1.0, 0.01, 1.0, 1.0,
                                        1.0);
            }
            std::swap(cold, cnew);
        }

        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);
        if (rank == 0) {
            printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
            const double cellUpdates = static_cast<double>(planeSize) *
                                       static_cast<double>(nz) * iterations;
            const double mcups = maxElapsed > 0.0 ? cellUpdates / maxElapsed / 1.0e6 : 0.0;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            const int sendCount = static_cast<int>(localCellCount);
            std::vector<int> receiveCounts;
            std::vector<int> displacements;
            std::vector<double> globalConcentration;
            if (rank == 0) {
                receiveCounts.resize(activeSize);
                displacements.resize(activeSize);
                int displacement = 0;
                for (int r = 0; r < activeSize; ++r) {
                    const size_t planes = basePlanes +
                        (static_cast<size_t>(r) < extraPlanes ? 1 : 0);
                    receiveCounts[r] = static_cast<int>(planes * planeSize);
                    displacements[r] = displacement;
                    displacement += receiveCounts[r];
                }
                globalConcentration.resize(planeSize * nz);
            }
            MPI_Gatherv(cold.data() + planeSize, sendCount, MPI_DOUBLE,
                        rank == 0 ? globalConcentration.data() : nullptr,
                        rank == 0 ? receiveCounts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                        activeComm);
            if (rank == 0) {
                print_results(globalConcentration, "Concentration");
            }
        }

        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
            }
            const bool valid = validateResult(cold, planeSize, localNz, activeComm, rank);
            if (rank == 0) {
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                exitCode = valid ? 0 : 1;
            }
        }
        MPI_Comm_free(&activeComm);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
