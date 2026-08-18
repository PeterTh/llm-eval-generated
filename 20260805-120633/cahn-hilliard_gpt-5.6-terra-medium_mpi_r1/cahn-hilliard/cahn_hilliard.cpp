#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The domain is distributed as contiguous Z slabs.  Each field has a halo
// plane at either end, allowing the original clamped seven-point stencil to
// be evaluated without any special treatment at rank boundaries.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

inline void exchangeHalos(std::vector<double>& field, const size_t plane,
                          const size_t localNz, const int previous,
                          const int next, MPI_Comm comm) {
    double* const first = field.data() + plane;
    double* const last = field.data() + localNz * plane;
    double* const lowerHalo = field.data();
    double* const upperHalo = field.data() + (localNz + 1) * plane;

    if (previous == MPI_PROC_NULL) std::copy_n(first, plane, lowerHalo);
    if (next == MPI_PROC_NULL) std::copy_n(last, plane, upperHalo);

    // Post both directions together so the two nearest-neighbour transfers
    // can progress concurrently on the network.
    MPI_Request requests[4];
    MPI_Irecv(upperHalo, static_cast<int>(plane), MPI_DOUBLE, next, 0, comm, &requests[0]);
    MPI_Irecv(lowerHalo, static_cast<int>(plane), MPI_DOUBLE, previous, 1, comm, &requests[1]);
    MPI_Isend(first, static_cast<int>(plane), MPI_DOUBLE, previous, 0, comm, &requests[2]);
    MPI_Isend(last, static_cast<int>(plane), MPI_DOUBLE, next, 1, comm, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

inline void computeChemicalPotential(const std::vector<double>& c,
                                     std::vector<double>& mu, const size_t nx,
                                     const size_t ny, const size_t localNz,
                                     const double gamma, const double eAA,
                                     const double eBB, const double eAB) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t zbase = z * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t ybase = zbase + y * nx;
            const size_t ypbase = zbase + (y + (y + 1 < ny)) * nx;
            const size_t ynbase = zbase + (y - (y > 0)) * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = ybase + x;
                const size_t xp = x + (x + 1 < nx);
                const size_t xn = x - (x > 0);
                const double cv = c[i];
                const double lap = c[ybase + xp] + c[ybase + xn]
                                 + c[ypbase + x] + c[ynbase + x]
                                 + c[i + plane] + c[i - plane] - 6.0 * cv;
                mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                      + 3.0 * cv + cv * cv * cv - gamma * lap;
            }
        }
    }
}

inline void cahnHilliardUpdate(std::vector<double>& cnew,
                               const std::vector<double>& cold,
                               const std::vector<double>& mu,
                               const size_t nx, const size_t ny,
                               const size_t localNz, const double dtD) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t zbase = z * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t ybase = zbase + y * nx;
            const size_t ypbase = zbase + (y + (y + 1 < ny)) * nx;
            const size_t ynbase = zbase + (y - (y > 0)) * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = ybase + x;
                const size_t xp = x + (x + 1 < nx);
                const size_t xn = x - (x > 0);
                cnew[i] = cold[i] + dtD * (mu[ybase + xp] + mu[ybase + xn]
                          + mu[ypbase + x] + mu[ynbase + x]
                          + mu[i + plane] + mu[i - plane] - 6.0 * mu[i]);
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n"
           "  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of time steps (default: 20)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    bool help = false, badOption = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) help = true;
        else badOption = true;
    }
    if (help || badOption) {
        if (worldRank == 0) { if (badOption) printf("Unknown or incomplete option\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return badOption ? 1 : 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nx > std::numeric_limits<int>::max() / ny) {
        if (worldRank == 0) printf("Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    const int activeCount = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeCount ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeCount) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    const size_t plane = nx * ny;
    const size_t baseNz = nz / ranks, remainder = nz % ranks;
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder);
    const size_t globalZ = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const size_t localCells = localNz * plane;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n"
               "MPI ranks: %d\nInitializing concentration field...\n", nx, ny, nz, iterations,
               validate ? "enabled" : "disabled", ranks);
    }
    std::vector<double> cold((localNz + 2) * plane), cnew((localNz + 2) * plane), mu((localNz + 2) * plane);
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalId = (globalZ + z) * plane + y * nx + x;
                cold[idx3(x, y, z + 1, nx, ny)] = -1.0 + 2.0 * (((globalId + 1) * 1299709 % volume) / static_cast<double>(volume));
            }

    constexpr double gamma = 0.5, eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0, dtD = 0.01;
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, plane, localNz, previous, next, comm);
        computeChemicalPotential(cold, mu, nx, ny, localNz, gamma, eAA, eBB, eAB);
        exchangeHalos(mu, plane, localNz, previous, next, comm);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, localNz, dtD);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",
               maxElapsed * 1000.0, static_cast<double>(volume) * iterations / maxElapsed / 1.0e6);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> global;
        if (rank == 0) {
            counts.resize(ranks); displacements.resize(ranks); global.resize(volume);
            for (int r = 0; r < ranks; ++r) {
                const size_t rNz = baseNz + (static_cast<size_t>(r) < remainder);
                counts[r] = static_cast<int>(rNz * plane);
                displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseNz + std::min(static_cast<size_t>(r), remainder)) * plane);
            }
        }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(localCells), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(global, "Concentration");
    }

    int localFinite = 1;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    if (validate) for (size_t i = plane; i < plane + localCells; ++i) {
        localFinite &= std::isfinite(cold[i]);
        localMin = std::min(localMin, cold[i]); localMax = std::max(localMax, cold[i]);
    }
    int allFinite; double minValue, maxValue;
    if (validate) {
        MPI_Reduce(&localFinite, &allFinite, 1, MPI_INT, MPI_LAND, 0, comm);
        MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    }
    int result = 0;
    if (validate && rank == 0) {
        printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", minValue, maxValue);
        result = (!allFinite || maxValue > 10.0 || minValue < -10.0);
        printf("Validation: %s\n", result ? "FAILED" : "PASSED");
    }
    if (validate) MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return result;
}
