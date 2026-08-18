#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

// A field contains one halo plane on each side of the z slab.  The owned
// planes are [1, localNz], which makes the stencil inner loop contiguous.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Slab {
    size_t firstGlobalZ;
    size_t localNz;
    size_t planeSize;
    int previous;
    int next;
};

// Start the nearest-neighbor exchange needed by the z part of the stencil.
// The two physical boundaries use the same clamped values as the serial code.
void beginHaloExchange(std::vector<double>& field, const Slab& slab,
                       MPI_Comm comm, MPI_Request requests[4], int& requestCount) {
    const size_t plane = slab.planeSize;
    double* const firstOwned = field.data() + plane;
    double* const lastOwned = field.data() + slab.localNz * plane;

    if (slab.previous == MPI_PROC_NULL) {
        std::copy_n(firstOwned, plane, field.data());
    }
    if (slab.next == MPI_PROC_NULL) {
        std::copy_n(lastOwned, plane, field.data() + (slab.localNz + 1) * plane);
    }

    requestCount = 0;
    if (slab.previous != MPI_PROC_NULL) {
        MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, slab.previous,
                  1, comm, &requests[requestCount++]);
        MPI_Isend(firstOwned, static_cast<int>(plane), MPI_DOUBLE, slab.previous,
                  0, comm, &requests[requestCount++]);
    }
    if (slab.next != MPI_PROC_NULL) {
        MPI_Irecv(field.data() + (slab.localNz + 1) * plane,
                  static_cast<int>(plane), MPI_DOUBLE, slab.next, 0, comm,
                  &requests[requestCount++]);
        MPI_Isend(lastOwned, static_cast<int>(plane), MPI_DOUBLE, slab.next, 1,
                  comm, &requests[requestCount++]);
    }
}

void finishHaloExchange(MPI_Request requests[4], const int requestCount) {
    if (requestCount != 0) {
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    }
}

inline void computeChemicalPotentialPlane(const std::vector<double>& c,
                                          std::vector<double>& mu,
                                          const size_t nx, const size_t ny,
                                          const size_t z, const double dx,
                                          const double dy, const double dz,
                                          const double gamma, const double e_AA,
                                          const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double* const previous = c.data() + (z - 1) * plane;
    const double* const current = c.data() + z * plane;
    const double* const next = c.data() + (z + 1) * plane;
    double* const output = mu.data() + z * plane;

    for (size_t y = 0; y < ny; ++y) {
        const size_t row = y * nx;
        const size_t previousRow = (y == 0 ? y : y - 1) * nx;
        const size_t nextRow = (y + 1 < ny ? y + 1 : y) * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t point = row + x;
            const size_t previousX = x == 0 ? x : x - 1;
            const size_t nextX = x + 1 < nx ? x + 1 : x;
            const double cv = current[point];

            const double cxx = (current[row + nextX] + current[row + previousX] -
                                2.0 * cv) * invDx2;
            const double cyy = (current[nextRow + x] + current[previousRow + x] -
                                2.0 * cv) * invDy2;
            const double czz = (next[point] + previous[point] - 2.0 * cv) * invDz2;
            const double laplacian = cxx + cyy + czz;

            output[point] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                                   2.0 * cv * e_AB) +
                            3.0 * cv + cv * cv * cv - gamma * laplacian;
        }
    }
}

inline void updateConcentrationPlane(std::vector<double>& cnew,
                                     const std::vector<double>& cold,
                                     const std::vector<double>& mu,
                                     const size_t nx, const size_t ny,
                                     const size_t z, const double D,
                                     const double dt, const double dx,
                                     const double dy, const double dz) {
    const size_t plane = nx * ny;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double* const previous = mu.data() + (z - 1) * plane;
    const double* const current = mu.data() + z * plane;
    const double* const next = mu.data() + (z + 1) * plane;
    const double* const oldValues = cold.data() + z * plane;
    double* const output = cnew.data() + z * plane;

    for (size_t y = 0; y < ny; ++y) {
        const size_t row = y * nx;
        const size_t previousRow = (y == 0 ? y : y - 1) * nx;
        const size_t nextRow = (y + 1 < ny ? y + 1 : y) * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t point = row + x;
            const size_t previousX = x == 0 ? x : x - 1;
            const size_t nextX = x + 1 < nx ? x + 1 : x;
            const double value = current[point];

            const double mxx = (current[row + nextX] + current[row + previousX] -
                                2.0 * value) * invDx2;
            const double myy = (current[nextRow + x] + current[previousRow + x] -
                                2.0 * value) * invDy2;
            const double mzz = (next[point] + previous[point] - 2.0 * value) * invDz2;
            output[point] = oldValues[point] + dt * D * (mxx + myy + mzz);
        }
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx,
                             const size_t ny, const size_t globalNz,
                             const Slab& slab) {
    const size_t volume = nx * ny * globalNz;
    for (size_t localZ = 1; localZ <= slab.localNz; ++localZ) {
        const size_t globalZ = slab.firstGlobalZ + localZ - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t localIndex = idx3(x, y, localZ, nx, ny);
                const size_t linearId = globalZ * slab.planeSize + y * nx + x;
                const double pseudo = (((linearId + 1) * 1299709) % volume) /
                                      static_cast<double>(volume);
                c[localIndex] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const Slab& slab,
                    MPI_Comm comm, const int rank) {
    const double* const values = c.data() + slab.planeSize;
    const size_t count = slab.localNz * slab.planeSize;
    int localInvalid = 0;
    double localMin = values[0];
    double localMax = values[0];
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            localInvalid = 1;
            break;
        }
        localMin = std::min(localMin, values[i]);
        localMax = std::max(localMax, values[i]);
    }

    int globalInvalid = 0;
    MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
    if (globalInvalid != 0) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int rangeValid = 1;
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            rangeValid = 0;
        }
    }
    MPI_Bcast(&rangeValid, 1, MPI_INT, 0, comm);
    return rangeValid != 0;
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

    int parseStatus = 0;
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
            parseStatus = 1;
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            break;
        }
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return 1;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const bool dimensionsInvalid =
        nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max());
    if (dimensionsInvalid) {
        if (worldRank == 0) {
            printf("Invalid grid dimensions for MPI decomposition\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeSize = nx * ny;
    const size_t gridSize = planeSize * nz;

    // With more processes than z planes, ranks without cells leave the active
    // communicator.  This preserves a real distributed decomposition for all
    // useful ranks without ever allocating or exchanging empty slabs.
    const int activeSize = static_cast<size_t>(worldSize) < nz
                               ? worldSize
                               : static_cast<int>(nz);
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 1 : MPI_UNDEFINED,
                   worldRank, &activeComm);
    if (worldRank >= activeSize) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int processCount = 0;
    MPI_Comm_rank(activeComm, &rank);
    MPI_Comm_size(activeComm, &processCount);

    const size_t basePlanes = nz / static_cast<size_t>(processCount);
    const size_t extraPlanes = nz % static_cast<size_t>(processCount);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < extraPlanes ? 1 : 0);
    const size_t firstGlobalZ = static_cast<size_t>(rank) * basePlanes +
                                std::min(static_cast<size_t>(rank), extraPlanes);
    const Slab slab{firstGlobalZ, localNz, planeSize,
                    rank == 0 ? MPI_PROC_NULL : rank - 1,
                    rank + 1 == processCount ? MPI_PROC_NULL : rank + 1};

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
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t localFieldSize = (localNz + 2) * planeSize;
    std::vector<double> cold(localFieldSize);
    std::vector<double> cnew(localFieldSize);
    std::vector<double> mu(localFieldSize);

    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, nz, slab);

    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(activeComm);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        MPI_Request requests[4];
        int requestCount = 0;
        beginHaloExchange(cold, slab, activeComm, requests, requestCount);

        // Interior planes do not depend on halo data and overlap communication.
        for (size_t z = 2; z < localNz; ++z) {
            computeChemicalPotentialPlane(cold, mu, nx, ny, z, dx, dy, dz,
                                          gamma, e_AA, e_BB, e_AB);
        }
        finishHaloExchange(requests, requestCount);
        computeChemicalPotentialPlane(cold, mu, nx, ny, 1, dx, dy, dz,
                                      gamma, e_AA, e_BB, e_AB);
        if (localNz > 1) {
            computeChemicalPotentialPlane(cold, mu, nx, ny, localNz, dx, dy, dz,
                                          gamma, e_AA, e_BB, e_AB);
        }

        beginHaloExchange(mu, slab, activeComm, requests, requestCount);
        for (size_t z = 2; z < localNz; ++z) {
            updateConcentrationPlane(cnew, cold, mu, nx, ny, z, D, dt, dx, dy, dz);
        }
        finishHaloExchange(requests, requestCount);
        updateConcentrationPlane(cnew, cold, mu, nx, ny, 1, D, dt, dx, dy, dz);
        if (localNz > 1) {
            updateConcentrationPlane(cnew, cold, mu, nx, ny, localNz,
                                     D, dt, dx, dy, dz);
        }
        std::swap(cold, cnew);
    }

    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / elapsedSeconds / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        const size_t localElementCount = localNz * planeSize;
        if (gridSize > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            localElementCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                printf("Cannot print results: grid exceeds MPI_Gatherv count limits\n");
            }
            MPI_Comm_free(&activeComm);
            MPI_Finalize();
            return 1;
        }

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<double> globalConcentration;
        if (rank == 0) {
            receiveCounts.resize(processCount);
            displacements.resize(processCount);
            for (int process = 0; process < processCount; ++process) {
                const size_t count = basePlanes +
                    (static_cast<size_t>(process) < extraPlanes ? 1 : 0);
                receiveCounts[process] = static_cast<int>(count * planeSize);
                displacements[process] = static_cast<int>(
                    (static_cast<size_t>(process) * basePlanes +
                     std::min(static_cast<size_t>(process), extraPlanes)) * planeSize);
            }
            globalConcentration.resize(gridSize);
        }
        MPI_Gatherv(cold.data() + planeSize, static_cast<int>(localElementCount),
                    MPI_DOUBLE, rank == 0 ? globalConcentration.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, activeComm);
        if (rank == 0) {
            print_results(globalConcentration, "Concentration");
        }
    }

    int result = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(cold, slab, activeComm, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        result = valid ? 0 : 1;
    }

    MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return result;
}
