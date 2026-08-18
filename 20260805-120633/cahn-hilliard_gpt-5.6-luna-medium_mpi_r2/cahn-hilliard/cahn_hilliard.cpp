#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// The domain is decomposed into contiguous z-slabs.  Each local array has one
// plane of halos on either side, so all stencil accesses remain local after a
// single nearest-neighbor exchange.
inline constexpr std::size_t idx3(const std::size_t x, const std::size_t y,
                                  const std::size_t z, const std::size_t nx,
                                  const std::size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

double computeLaplacian(const std::vector<double>& field, const std::size_t nx,
                        const std::size_t ny, const std::size_t x,
                        const std::size_t y, const std::size_t z,
                        const double dx, const double dy, const double dz) {
    const std::size_t xp = (x + 1 < nx) ? x + 1 : x;
    const std::size_t yp = (y + 1 < ny) ? y + 1 : y;
    const std::size_t xn = (x > 0) ? x - 1 : 0;
    const std::size_t yn = (y > 0) ? y - 1 : 0;
    const std::size_t plane = nx * ny;
    const std::size_t center = idx3(x, y, z, nx, ny);

    return (field[idx3(xp, y, z, nx, ny)] + field[idx3(xn, y, z, nx, ny)] -
            2.0 * field[center]) / (dx * dx) +
           (field[idx3(x, yp, z, nx, ny)] + field[idx3(x, yn, z, nx, ny)] -
            2.0 * field[center]) / (dy * dy) +
           (field[center + plane] + field[center - plane] - 2.0 * field[center]) /
               (dz * dz);
}

void exchangeHalos(std::vector<double>& field, const std::size_t plane,
                   const std::size_t localNz, const int rank, const int size,
                   MPI_Comm comm) {
    const int lower = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    double* first = field.data() + plane;
    double* last = field.data() + localNz * plane;
    double* lowerHalo = field.data();
    double* upperHalo = field.data() + (localNz + 1) * plane;
    const int count = static_cast<int>(plane);

    // Send the first plane down while receiving the upper neighbor's first
    // plane, then send the last plane up while receiving the lower neighbor's
    // last plane.  MPI_PROC_NULL handles physical boundaries.
    MPI_Sendrecv(first, count, MPI_DOUBLE, lower, 0,
                 upperHalo, count, MPI_DOUBLE, upper, 0, comm,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(last, count, MPI_DOUBLE, upper, 1,
                 lowerHalo, count, MPI_DOUBLE, lower, 1, comm,
                 MPI_STATUS_IGNORE);

    if (lower == MPI_PROC_NULL) {
        std::copy_n(first, plane, lowerHalo); // clamped z=0 boundary
    }
    if (upper == MPI_PROC_NULL) {
        std::copy_n(last, plane, upperHalo); // clamped z=nz-1 boundary
    }
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const std::size_t nx, const std::size_t ny,
                              const std::size_t localNz, const double dx,
                              const double dy, const double dz, const double gamma,
                              const double eAA, const double eBB, const double eAB) {
    for (std::size_t z = 1; z <= localNz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const std::size_t i = idx3(x, y, z, nx, ny);
                const double cv = c[i];
                mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                        3.0 * cv + cv * cv * cv -
                        gamma * computeLaplacian(c, nx, ny, x, y, z, dx, dy, dz);
            }
        }
    }
}

void updateConcentration(std::vector<double>& cnew, const std::vector<double>& cold,
                         const std::vector<double>& mu, const std::size_t nx,
                         const std::size_t ny, const std::size_t localNz,
                         const double D, const double dt, const double dx,
                         const double dy, const double dz) {
    for (std::size_t z = 1; z <= localNz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const std::size_t i = idx3(x, y, z, nx, ny);
                cnew[i] = cold[i] + dt * D *
                          computeLaplacian(mu, nx, ny, x, y, z, dx, dy, dz);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const std::size_t nx,
                             const std::size_t ny, const std::size_t localNz,
                             const std::size_t globalZ, const std::size_t volume) {
    const std::size_t plane = nx * ny;
    for (std::size_t z = 1; z <= localNz; ++z) {
        const std::size_t globalPlane = (globalZ + z - 1) * plane;
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const std::size_t i = idx3(x, y, z, nx, ny);
                const std::size_t linearId = globalPlane + y * nx + x;
                const std::size_t pseudoInt = ((linearId + 1) * 1299709) % volume;
                c[i] = -1.0 + 2.0 * pseudoInt / static_cast<double>(volume);
            }
        }
    }
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 < argc && std::strcmp(argv[i], "-x") == 0) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (i + 1 < argc && std::strcmp(argv[i], "-y") == 0) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (i + 1 < argc && std::strcmp(argv[i], "-z") == 0) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (i + 1 < argc && std::strcmp(argv[i], "-i") == 0) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else parseError = 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || parseError) {
        if (rank == 0) { std::printf("Invalid command line arguments\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }
    const std::size_t plane = nx * ny;
    if (plane > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        size > static_cast<int>(nz)) {
        if (rank == 0) std::printf("MPI requires grid plane <= INT_MAX and no more ranks than z planes\n");
        MPI_Finalize(); return 1;
    }

    const std::size_t base = nz / static_cast<std::size_t>(size);
    const std::size_t remainder = nz % static_cast<std::size_t>(size);
    const std::size_t localNz = base + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
    const std::size_t globalZ = static_cast<std::size_t>(rank) * base +
                                std::min(static_cast<std::size_t>(rank), remainder);
    const std::size_t volume = nx * ny * nz;
    std::vector<double> cold((localNz + 2) * plane), cnew((localNz + 2) * plane),
        mu((localNz + 2) * plane);
    initializeConcentration(cold, nx, ny, localNz, globalZ, volume);

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n", size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, plane, localNz, rank, size, MPI_COMM_WORLD);
        computeChemicalPotential(cold, mu, nx, ny, localNz, 1.0, 1.0, 1.0, 0.5,
                                 -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        exchangeHalos(mu, plane, localNz, rank, size, MPI_COMM_WORLD);
        updateConcentration(cnew, cold, mu, nx, ny, localNz, 1.0, 0.01, 1.0, 1.0, 1.0);
        cold.swap(cnew);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    seconds > 0.0 ? static_cast<double>(volume) * iterations / seconds / 1.0e6 : 0.0);
    }

    std::vector<int> counts, displacements;
    std::vector<double> global;
    if (printResults || validate) {
        if (rank == 0) {
            counts.resize(size); displacements.resize(size); global.resize(volume);
            for (int r = 0; r < size; ++r) {
                const std::size_t planes = base + (static_cast<std::size_t>(r) < remainder ? 1 : 0);
                counts[r] = static_cast<int>(planes * plane);
                displacements[r] = static_cast<int>((static_cast<std::size_t>(r) * base +
                    std::min(static_cast<std::size_t>(r), remainder)) * plane);
            }
        }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (printResults && rank == 0) print_results(global, "Concentration");
    if (validate) {
        int localBad = 0;
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        for (std::size_t i = plane; i < (localNz + 1) * plane; ++i) {
            localBad |= !std::isfinite(cold[i]);
            localMin = std::min(localMin, cold[i]); localMax = std::max(localMax, cold[i]);
        }
        int bad = 0; double minVal = 0.0, maxVal = 0.0;
        MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", minVal, maxVal);
            const bool valid = !bad && maxVal <= 10.0 && minVal >= -10.0;
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) { MPI_Finalize(); return 1; }
        }
    }
    MPI_Finalize();
    return 0;
}
