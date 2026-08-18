#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

// Exchange the first and last owned Z planes.  Physical Z boundaries use the
// same plane as their ghost, exactly matching the original clamped stencil.
struct HaloExchange {
    MPI_Request requests[4];
    int count = 0;
};

HaloExchange beginHaloExchange(std::vector<double>& field, size_t localNz, size_t planeSize,
                               int lower, int upper, MPI_Comm comm) {
    if (lower == MPI_PROC_NULL)
        std::copy_n(field.data() + planeSize, planeSize, field.data());
    if (upper == MPI_PROC_NULL)
        std::copy_n(field.data() + localNz * planeSize, planeSize,
                    field.data() + (localNz + 1) * planeSize);

    HaloExchange exchange;
    if (lower != MPI_PROC_NULL) {
        MPI_Irecv(field.data(), static_cast<int>(planeSize), MPI_DOUBLE, lower, 1, comm, &exchange.requests[exchange.count++]);
        MPI_Isend(field.data() + planeSize, static_cast<int>(planeSize), MPI_DOUBLE, lower, 0, comm, &exchange.requests[exchange.count++]);
    }
    if (upper != MPI_PROC_NULL) {
        MPI_Irecv(field.data() + (localNz + 1) * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, upper, 0, comm, &exchange.requests[exchange.count++]);
        MPI_Isend(field.data() + localNz * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, upper, 1, comm, &exchange.requests[exchange.count++]);
    }
    return exchange;
}

void finishHaloExchange(HaloExchange& exchange) {
    MPI_Waitall(exchange.count, exchange.requests, MPI_STATUSES_IGNORE);
}

inline double laplacian(const std::vector<double>& a, size_t x, size_t y, size_t z,
                        size_t nx, size_t ny, double invDx2, double invDy2, double invDz2) {
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t center = idx3(x, y, z, nx, ny);
    return (a[idx3(xp, y, z, nx, ny)] + a[idx3(xn, y, z, nx, ny)] - 2.0 * a[center]) * invDx2
         + (a[idx3(x, yp, z, nx, ny)] + a[idx3(x, yn, z, nx, ny)] - 2.0 * a[center]) * invDy2
         + (a[idx3(x, y, z + 1, nx, ny)] + a[idx3(x, y, z - 1, nx, ny)] - 2.0 * a[center]) * invDz2;
}

void chemicalPotentialPlanes(const std::vector<double>& c, std::vector<double>& mu,
                             size_t firstZ, size_t lastZ, size_t nx, size_t ny,
                             double invDx2, double invDy2, double invDz2,
                             double gamma, double eAA, double eBB, double eAB) {
    for (size_t z = firstZ; z <= lastZ; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double v = c[i];
                mu[i] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
                      + 3.0 * v + v * v * v
                      - gamma * laplacian(c, x, y, z, nx, ny, invDx2, invDy2, invDz2);
            }
}

void updatePlanes(std::vector<double>& out, const std::vector<double>& in, const std::vector<double>& mu,
                  size_t firstZ, size_t lastZ, size_t nx, size_t ny,
                  double invDx2, double invDy2, double invDz2, double Ddt) {
    for (size_t z = firstZ; z <= lastZ; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                out[i] = in[i] + Ddt * laplacian(mu, x, y, z, nx, ny, invDx2, invDy2, invDz2);
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int parseOk = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseOk = 0; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nz < static_cast<size_t>(ranks) || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Invalid grid or MPI size: each rank requires at least one Z plane.\n");
        parseOk = 0;
    }
    int allParseOk;
    MPI_Allreduce(&parseOk, &allParseOk, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!allParseOk) { MPI_Finalize(); return 1; }

    const size_t baseNz = nz / ranks, remainder = nz % ranks;
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder);
    const size_t zOffset = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);
    const size_t planeSize = nx * ny;
    const size_t localSize = (localNz + 2) * planeSize;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI, %d ranks)\n", ranks);
        printf("Grid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }

    std::vector<double> cold(localSize), cnew(localSize), mu(localSize);
    const size_t globalSize = nx * ny * nz;
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalId = (zOffset + z - 1) * planeSize + y * nx + x;
                cold[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * (((globalId + 1) * 1299709 % globalSize) / static_cast<double>(globalSize));
            }

    constexpr double gamma = 0.5, eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    const double invDx2 = 1.0, invDy2 = 1.0, invDz2 = 1.0, Ddt = 0.01;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        auto coldExchange = beginHaloExchange(cold, localNz, planeSize, lower, upper, MPI_COMM_WORLD);
        if (localNz > 2)
            chemicalPotentialPlanes(cold, mu, 2, localNz - 1, nx, ny, invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
        finishHaloExchange(coldExchange);
        chemicalPotentialPlanes(cold, mu, 1, 1, nx, ny, invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
        if (localNz > 1)
            chemicalPotentialPlanes(cold, mu, localNz, localNz, nx, ny, invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);

        auto muExchange = beginHaloExchange(mu, localNz, planeSize, lower, upper, MPI_COMM_WORLD);
        if (localNz > 2)
            updatePlanes(cnew, cold, mu, 2, localNz - 1, nx, ny, invDx2, invDy2, invDz2, Ddt);
        finishHaloExchange(muExchange);
        updatePlanes(cnew, cold, mu, 1, 1, nx, ny, invDx2, invDy2, invDz2, Ddt);
        if (localNz > 1)
            updatePlanes(cnew, cold, mu, localNz, localNz, nx, ny, invDx2, invDy2, invDz2, Ddt);
        std::swap(cold, cnew);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double mcups = seconds > 0.0 ? static_cast<double>(globalSize) * iterations / seconds / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> global;
        if (rank == 0) {
            counts.resize(ranks); displacements.resize(ranks); global.resize(globalSize);
            for (int r = 0; r < ranks; ++r) {
                const size_t count = (baseNz + (static_cast<size_t>(r) < remainder)) * planeSize;
                counts[r] = static_cast<int>(count);
                displacements[r] = r == 0 ? 0 : displacements[r - 1] + counts[r - 1];
            }
        }
        MPI_Gatherv(cold.data() + planeSize, static_cast<int>(localNz * planeSize), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "Concentration");
    }

    int exitCode = 0;
    if (validate) {
        int localBad = 0;
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        for (size_t i = planeSize; i < (localNz + 1) * planeSize; ++i) {
            localBad |= !std::isfinite(cold[i]);
            localMin = std::min(localMin, cold[i]); localMax = std::max(localMax, cold[i]);
        }
        int bad; double minValue, maxValue;
        MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", minValue, maxValue);
            exitCode = bad || maxValue > 10.0 || minValue < -10.0;
            printf("Validation: %s\n", exitCode ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
