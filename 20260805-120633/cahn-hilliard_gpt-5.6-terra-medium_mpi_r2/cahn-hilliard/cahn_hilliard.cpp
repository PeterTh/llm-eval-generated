#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

// The local arrays contain a ghost plane at z=0 and z=localNz+1.  X and Y
// boundaries remain clamped exactly as in the original single-process code.
inline double laplacian(const std::vector<double>& field, const size_t x,
                        const size_t y, const size_t z, const size_t nx,
                        const size_t ny, const double invDx2,
                        const double invDy2, const double invDz2) {
    const size_t xp = x + (x + 1 < nx);
    const size_t xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t ym = y - (y > 0);
    const size_t center = idx3(x, y, z, nx, ny);
    return (field[idx3(xp, y, z, nx, ny)] + field[idx3(xm, y, z, nx, ny)] - 2.0 * field[center]) * invDx2 +
           (field[idx3(x, yp, z, nx, ny)] + field[idx3(x, ym, z, nx, ny)] - 2.0 * field[center]) * invDy2 +
           (field[idx3(x, y, z + 1, nx, ny)] + field[idx3(x, y, z - 1, nx, ny)] - 2.0 * field[center]) * invDz2;
}

// Exchange Z faces.  At physical boundaries, mirror the adjacent interior
// plane to implement the original clamped boundary condition.
void exchangeHalos(std::vector<double>& field, const size_t planeSize,
                   const size_t localNz, const int previous, const int next,
                   MPI_Comm comm) {
    if (previous == MPI_PROC_NULL) {
        std::copy_n(field.data() + planeSize, planeSize, field.data());
    }
    if (next == MPI_PROC_NULL) {
        std::copy_n(field.data() + localNz * planeSize, planeSize,
                    field.data() + (localNz + 1) * planeSize);
    }

    MPI_Request requests[4];
    int count = 0;
    if (previous != MPI_PROC_NULL) {
        MPI_Irecv(field.data(), static_cast<int>(planeSize), MPI_DOUBLE, previous, 1, comm, &requests[count++]);
        MPI_Isend(field.data() + planeSize, static_cast<int>(planeSize), MPI_DOUBLE, previous, 2, comm, &requests[count++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(field.data() + (localNz + 1) * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, next, 2, comm, &requests[count++]);
        MPI_Isend(field.data() + localNz * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, next, 1, comm, &requests[count++]);
    }
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t localNz,
                              const double gamma, const double invDx2,
                              const double invDy2, const double invDz2,
                              const double eAA, const double eBB, const double eAB) {
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double value = c[i];
                mu[i] = 4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB)
                      + 3.0 * value + value * value * value
                      - gamma * laplacian(c, x, y, z, nx, ny, invDx2, invDy2, invDz2);
            }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, const size_t nx,
                        const size_t ny, const size_t localNz, const double scale,
                        const double invDx2, const double invDy2, const double invDz2) {
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                cnew[i] = cold[i] + scale * laplacian(mu, x, y, z, nx, ny, invDx2, invDy2, invDz2);
            }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    bool parseError = false, showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) showHelp = true;
        else parseError = true;
    }
    if (showHelp || parseError) {
        if (worldRank == 0) {
            if (parseError) printf("Unknown or incomplete option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) printf("Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    const int activeRanks = std::min<int>(worldSize, static_cast<int>(nz));
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    if (worldRank >= activeRanks) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    int rank;
    MPI_Comm_rank(activeComm, &rank);
    const size_t baseNz = nz / activeRanks;
    const size_t remainder = nz % activeRanks;
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder);
    const size_t zStart = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);
    const size_t planeSize = nx * ny;
    const size_t localSize = localNz * planeSize;
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == activeRanks ? MPI_PROC_NULL : rank + 1;

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\nMPI ranks: %d\nInitializing concentration field...\n", iterations, validate ? "enabled" : "disabled", activeRanks);
    }

    std::vector<double> cold((localNz + 2) * planeSize);
    std::vector<double> cnew((localNz + 2) * planeSize);
    std::vector<double> mu((localNz + 2) * planeSize);
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalId = (zStart + z) * planeSize + y * nx + x;
                cold[idx3(x, y, z + 1, nx, ny)] = -1.0 + 2.0 * (((globalId + 1) * 1299709 % volume) / static_cast<double>(volume));
            }

    const double invDx2 = 1.0, invDy2 = 1.0, invDz2 = 1.0;
    const double gamma = 0.5, eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(activeComm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, planeSize, localNz, previous, next, activeComm);
        computeChemicalPotential(cold, mu, nx, ny, localNz, gamma, invDx2, invDy2, invDz2, eAA, eBB, eAB);
        exchangeHalos(mu, planeSize, localNz, previous, next, activeComm);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, localNz, 0.01, invDx2, invDy2, invDz2);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);
    if (worldRank == 0) {
        const double milliseconds = maxElapsed * 1000.0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(volume) * iterations / maxElapsed / 1e6);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> global;
        if (rank == 0) {
            counts.resize(activeRanks); displacements.resize(activeRanks); global.resize(volume);
            for (int r = 0; r < activeRanks; ++r) {
                const size_t slabs = baseNz + (static_cast<size_t>(r) < remainder);
                counts[r] = static_cast<int>(slabs * planeSize);
                displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseNz + std::min(static_cast<size_t>(r), remainder)) * planeSize);
            }
        }
        MPI_Gatherv(cold.data() + planeSize, static_cast<int>(localSize), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, activeComm);
        if (rank == 0) print_results(global, "Concentration");
    }

    int localBad = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    if (validate) {
        for (size_t i = planeSize; i < planeSize + localSize; ++i) {
            localBad |= !std::isfinite(cold[i]);
            localMin = std::min(localMin, cold[i]);
            localMax = std::max(localMax, cold[i]);
        }
        int anyBad;
        double minValue, maxValue;
        MPI_Reduce(&localBad, &anyBad, 1, MPI_INT, MPI_LOR, 0, activeComm);
        MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, activeComm);
        MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);
        if (rank == 0) {
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", minValue, maxValue);
            if (anyBad || maxValue > 10.0 || minValue < -10.0) {
                printf("Validation: FAILED\n");
                MPI_Comm_free(&activeComm); MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); return 1;
            }
            printf("Validation: PASSED\n");
        }
    }
    MPI_Comm_free(&activeComm);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
