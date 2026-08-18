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

namespace {

#if defined(__GNUC__) || defined(__clang__)
#define CH_RESTRICT __restrict__
#else
#define CH_RESTRICT
#endif

// Each physical plane has a one-cell X/Y halo.  This makes all stencil reads
// branch-free; the Z halos are exchanged with adjacent MPI ranks.
void fillLateralHalos(double* const field, const size_t localNz, const size_t nx,
                      const size_t ny, const size_t rowStride, const size_t planeStride) {
    for (size_t z = 1; z <= localNz; ++z) {
        double* const plane = field + z * planeStride;
        for (size_t y = 1; y <= ny; ++y) {
            double* const row = plane + y * rowStride;
            row[0] = row[1];
            row[nx + 1] = row[nx];
        }
        std::copy_n(plane + rowStride, rowStride, plane);
        std::copy_n(plane + ny * rowStride, rowStride, plane + (ny + 1) * rowStride);
    }
}

struct HaloExchange {
    MPI_Request requests[4];
    int count = 0;
};

HaloExchange startZHaloExchange(double* const field, const size_t localNz,
                                const size_t planeStride, const int rank,
                                const int ranks, const MPI_Comm comm) {
    constexpr int upperPlaneTag = 701;
    constexpr int lowerPlaneTag = 702;
    const int planeCount = static_cast<int>(planeStride);
    HaloExchange exchange;

    if (rank == 0) {
        std::copy_n(field + planeStride, planeStride, field);
    } else {
        MPI_Irecv(field, planeCount, MPI_DOUBLE, rank - 1, upperPlaneTag, comm,
                  &exchange.requests[exchange.count++]);
    }
    if (rank == ranks - 1) {
        std::copy_n(field + localNz * planeStride, planeStride,
                    field + (localNz + 1) * planeStride);
    } else {
        MPI_Irecv(field + (localNz + 1) * planeStride, planeCount, MPI_DOUBLE,
                  rank + 1, lowerPlaneTag, comm, &exchange.requests[exchange.count++]);
    }
    if (rank != 0) {
        MPI_Isend(field + planeStride, planeCount, MPI_DOUBLE, rank - 1, lowerPlaneTag, comm,
                  &exchange.requests[exchange.count++]);
    }
    if (rank != ranks - 1) {
        MPI_Isend(field + localNz * planeStride, planeCount, MPI_DOUBLE, rank + 1, upperPlaneTag,
                  comm, &exchange.requests[exchange.count++]);
    }
    return exchange;
}

void finishHaloExchange(HaloExchange& exchange) {
    if (exchange.count != 0) {
        MPI_Waitall(exchange.count, exchange.requests, MPI_STATUSES_IGNORE);
    }
}

void computeChemicalPotentialRange(const double* CH_RESTRICT const c,
                                   double* CH_RESTRICT const mu, const size_t firstZ,
                                   const size_t lastZ, const size_t nx, const size_t ny,
                                   const size_t rowStride, const size_t planeStride,
                                   const double dx, const double dy, const double dz,
                                   const double gamma, const double eAA, const double eBB,
                                   const double eAB) {
    for (size_t z = firstZ; z <= lastZ; ++z) {
        const double* const previousPlane = c + (z - 1) * planeStride;
        const double* const currentPlane = c + z * planeStride;
        const double* const nextPlane = c + (z + 1) * planeStride;
        double* const outputPlane = mu + z * planeStride;

        for (size_t y = 1; y <= ny; ++y) {
            const double* const previousRow = currentPlane + (y - 1) * rowStride;
            const double* const currentRow = currentPlane + y * rowStride;
            const double* const nextRow = currentPlane + (y + 1) * rowStride;
            double* const outputRow = outputPlane + y * rowStride;
            const double* const previousZRow = previousPlane + y * rowStride;
            const double* const nextZRow = nextPlane + y * rowStride;

            for (size_t x = 1; x <= nx; ++x) {
                const double cv = currentRow[x];
                const double cxx = (currentRow[x + 1] + currentRow[x - 1] - 2.0 * cv) / (dx * dx);
                const double cyy = (nextRow[x] + previousRow[x] - 2.0 * cv) / (dy * dy);
                const double czz = (nextZRow[x] + previousZRow[x] - 2.0 * cv) / (dz * dz);
                outputRow[x] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                               3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
            }
        }
    }
}

void cahnHilliardUpdateRange(double* CH_RESTRICT const cnew, const double* CH_RESTRICT const cold,
                             const double* CH_RESTRICT const mu, const size_t firstZ,
                             const size_t lastZ, const size_t nx, const size_t ny,
                             const size_t rowStride, const size_t planeStride, const double D,
                             const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = firstZ; z <= lastZ; ++z) {
        const double* const previousPlane = mu + (z - 1) * planeStride;
        const double* const currentPlane = mu + z * planeStride;
        const double* const nextPlane = mu + (z + 1) * planeStride;
        const double* const oldPlane = cold + z * planeStride;
        double* const newPlane = cnew + z * planeStride;

        for (size_t y = 1; y <= ny; ++y) {
            const double* const previousRow = currentPlane + (y - 1) * rowStride;
            const double* const currentRow = currentPlane + y * rowStride;
            const double* const nextRow = currentPlane + (y + 1) * rowStride;
            const double* const oldRow = oldPlane + y * rowStride;
            double* const newRow = newPlane + y * rowStride;
            const double* const previousZRow = previousPlane + y * rowStride;
            const double* const nextZRow = nextPlane + y * rowStride;

            for (size_t x = 1; x <= nx; ++x) {
                const double muv = currentRow[x];
                const double mxx = (currentRow[x + 1] + currentRow[x - 1] - 2.0 * muv) / (dx * dx);
                const double myy = (nextRow[x] + previousRow[x] - 2.0 * muv) / (dy * dy);
                const double mzz = (nextZRow[x] + previousZRow[x] - 2.0 * muv) / (dz * dz);
                newRow[x] = oldRow[x] + dt * D * (mxx + myy + mzz);
            }
        }
    }
}

void initializeConcentration(double* const c, const size_t nx, const size_t ny,
                             const size_t localNz, const size_t globalZStart,
                             const size_t rowStride, const size_t planeStride,
                             const size_t globalVolume) {
    const size_t xySize = nx * ny;
    for (size_t localZ = 0; localZ < localNz; ++localZ) {
        const size_t globalZ = globalZStart + localZ;
        double* const plane = c + (localZ + 1) * planeStride;
        for (size_t y = 0; y < ny; ++y) {
            double* const row = plane + (y + 1) * rowStride + 1;
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = globalZ * xySize + y * nx + x;
                const double pseudo = ((((linearId + 1) * 1299709) % globalVolume) /
                                       static_cast<double>(globalVolume));
                row[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

struct LocalValidation {
    int finite;
    double minValue;
    double maxValue;
};

LocalValidation validateLocal(const double* const c, const size_t localNz, const size_t nx,
                              const size_t ny, const size_t rowStride,
                              const size_t planeStride) {
    LocalValidation result{1, std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
    for (size_t z = 1; z <= localNz; ++z) {
        const double* const plane = c + z * planeStride;
        for (size_t y = 1; y <= ny; ++y) {
            const double* const row = plane + y * rowStride;
            for (size_t x = 1; x <= nx; ++x) {
                const double value = row[x];
                if (!std::isfinite(value)) {
                    result.finite = 0;
                    return result;
                }
                result.minValue = std::min(result.minValue, value);
                result.maxValue = std::max(result.maxValue, value);
            }
        }
    }
    return result;
}

void packInterior(const double* const field, std::vector<double>& packed, const size_t localNz,
                  const size_t nx, const size_t ny, const size_t rowStride,
                  const size_t planeStride) {
    size_t destination = 0;
    for (size_t z = 1; z <= localNz; ++z) {
        const double* const plane = field + z * planeStride;
        for (size_t y = 1; y <= ny; ++y) {
            std::copy_n(plane + y * rowStride + 1, nx, packed.data() + destination);
            destination += nx;
        }
    }
}

bool multiplyWouldOverflow(const size_t left, const size_t right) {
    return left != 0 && right > std::numeric_limits<size_t>::max() / left;
}

bool parseSize(const char* const text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* const text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

}  // namespace

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

    struct Configuration {
        size_t nx = 64;
        size_t ny = 0;
        size_t nz = 0;
        int iterations = 20;
        int validate = 0;
        int printResults = 0;
        int status = 0;  // 0: run, 1: help, 2: invalid arguments
    } configuration;

    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                if (!parseSize(argv[++i], configuration.nx)) configuration.status = 2;
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                if (!parseSize(argv[++i], configuration.ny)) configuration.status = 2;
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                if (!parseSize(argv[++i], configuration.nz)) configuration.status = 2;
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                if (!parseIterations(argv[++i], configuration.iterations)) configuration.status = 2;
            } else if (strcmp(argv[i], "-v") == 0) {
                configuration.validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                configuration.printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                configuration.status = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                configuration.status = 2;
            }
        }
        if (configuration.status == 1 || configuration.status == 2) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&configuration, static_cast<int>(sizeof(configuration)), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (configuration.status != 0) {
        MPI_Finalize();
        return configuration.status == 1 ? 0 : 1;
    }

    if (configuration.ny == 0) configuration.ny = configuration.nx;
    if (configuration.nz == 0) configuration.nz = configuration.nx;
    const size_t nx = configuration.nx;
    const size_t ny = configuration.ny;
    const size_t nz = configuration.nz;
    const int iterations = configuration.iterations;

    if (multiplyWouldOverflow(nx, ny) || multiplyWouldOverflow(nx * ny, nz) ||
        nx > std::numeric_limits<size_t>::max() - 2 ||
        ny > std::numeric_limits<size_t>::max() - 2 ||
        multiplyWouldOverflow(nx + 2, ny + 2)) {
        if (worldRank == 0) printf("Invalid grid size: dimensions are too large\n");
        MPI_Finalize();
        return 1;
    }

    const size_t physicalPlaneSize = nx * ny;
    const size_t gridSize = physicalPlaneSize * nz;
    const size_t rowStride = nx + 2;
    const size_t planeStride = rowStride * (ny + 2);
    if (planeStride > static_cast<size_t>(INT_MAX) ||
        (configuration.printResults && gridSize > static_cast<size_t>(INT_MAX))) {
        if (worldRank == 0) printf("Grid size exceeds the supported MPI message count\n");
        MPI_Finalize();
        return 1;
    }

    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    if (worldRank >= activeRanks) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(activeComm, &rank);
    MPI_Comm_size(activeComm, &ranks);
    const size_t basePlanes = nz / static_cast<size_t>(ranks);
    const size_t remainderPlanes = nz % static_cast<size_t>(ranks);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < remainderPlanes ? 1 : 0);
    const size_t globalZStart = static_cast<size_t>(rank) * basePlanes +
                                std::min(static_cast<size_t>(rank), remainderPlanes);
    if (multiplyWouldOverflow(localNz + 2, planeStride)) {
        if (rank == 0) printf("Grid size exceeds local addressable memory\n");
        MPI_Comm_free(&activeComm);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", configuration.validate ? "enabled" : "disabled");
    }

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = 2.0 / 9.0;
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t localStorageSize = (localNz + 2) * planeStride;
    std::vector<double> cold(localStorageSize);
    std::vector<double> cnew(localStorageSize);
    std::vector<double> mu(localStorageSize);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), nx, ny, localNz, globalZStart, rowStride, planeStride, gridSize);
    fillLateralHalos(cold.data(), localNz, nx, ny, rowStride, planeStride);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(activeComm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        HaloExchange coldExchange = startZHaloExchange(cold.data(), localNz, planeStride, rank, ranks, activeComm);
        if (localNz > 2) {
            computeChemicalPotentialRange(cold.data(), mu.data(), 2, localNz - 1, nx, ny, rowStride,
                                          planeStride, dx, dy, dz, gamma, eAA, eBB, eAB);
        }
        finishHaloExchange(coldExchange);
        computeChemicalPotentialRange(cold.data(), mu.data(), 1, 1, nx, ny, rowStride, planeStride,
                                      dx, dy, dz, gamma, eAA, eBB, eAB);
        if (localNz > 1) {
            computeChemicalPotentialRange(cold.data(), mu.data(), localNz, localNz, nx, ny, rowStride,
                                          planeStride, dx, dy, dz, gamma, eAA, eBB, eAB);
        }
        fillLateralHalos(mu.data(), localNz, nx, ny, rowStride, planeStride);

        HaloExchange muExchange = startZHaloExchange(mu.data(), localNz, planeStride, rank, ranks, activeComm);
        if (localNz > 2) {
            cahnHilliardUpdateRange(cnew.data(), cold.data(), mu.data(), 2, localNz - 1, nx, ny,
                                    rowStride, planeStride, D, dt, dx, dy, dz);
        }
        finishHaloExchange(muExchange);
        cahnHilliardUpdateRange(cnew.data(), cold.data(), mu.data(), 1, 1, nx, ny, rowStride,
                                planeStride, D, dt, dx, dy, dz);
        if (localNz > 1) {
            cahnHilliardUpdateRange(cnew.data(), cold.data(), mu.data(), localNz, localNz, nx, ny,
                                    rowStride, planeStride, D, dt, dx, dy, dz);
        }
        fillLateralHalos(cnew.data(), localNz, nx, ny, rowStride, planeStride);
        std::swap(cold, cnew);
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (configuration.printResults) {
        const size_t localElementCount = localNz * physicalPlaneSize;
        std::vector<double> packed(localElementCount);
        packInterior(cold.data(), packed, localNz, nx, ny, rowStride, planeStride);

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        std::vector<double> globalResult;
        if (rank == 0) {
            receiveCounts.resize(ranks);
            displacements.resize(ranks);
            globalResult.resize(gridSize);
            for (int process = 0; process < ranks; ++process) {
                const size_t processPlanes = basePlanes +
                    (static_cast<size_t>(process) < remainderPlanes ? 1 : 0);
                const size_t processStart = static_cast<size_t>(process) * basePlanes +
                    std::min(static_cast<size_t>(process), remainderPlanes);
                receiveCounts[process] = static_cast<int>(processPlanes * physicalPlaneSize);
                displacements[process] = static_cast<int>(processStart * physicalPlaneSize);
            }
        }
        MPI_Gatherv(packed.data(), static_cast<int>(localElementCount), MPI_DOUBLE,
                    rank == 0 ? globalResult.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, activeComm);
        if (rank == 0) print_results(globalResult, "Concentration");
    }

    int returnCode = 0;
    if (configuration.validate) {
        if (rank == 0) printf("Validating result...\n");
        const LocalValidation localValidation =
            validateLocal(cold.data(), localNz, nx, ny, rowStride, planeStride);
        int allFinite = 0;
        double minValue = 0.0;
        double maxValue = 0.0;
        MPI_Allreduce(&localValidation.finite, &allFinite, 1, MPI_INT, MPI_MIN, activeComm);
        MPI_Reduce(&localValidation.minValue, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, activeComm);
        MPI_Reduce(&localValidation.maxValue, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);
        if (rank == 0) {
            if (!allFinite) {
                printf("Validation failed: found NaN or Inf value\n");
                printf("Validation: FAILED\n");
                returnCode = 1;
            } else {
                printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
                if (maxValue > 10.0 || minValue < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    printf("Validation: FAILED\n");
                    returnCode = 1;
                } else {
                    printf("Validation: PASSED\n");
                }
            }
        }
        MPI_Bcast(&returnCode, 1, MPI_INT, 0, activeComm);
    }

    MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return returnCode;
}
