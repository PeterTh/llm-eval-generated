#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local arrays have one halo plane on each side of their owned Z planes.
struct HaloExchange {
    MPI_Request requests[4];
    int count = 0;
};

HaloExchange startHaloExchange(std::vector<double>& field, size_t plane, size_t localNz,
                               int rank, int ranks, MPI_Comm comm) {
    HaloExchange exchange;
    if (rank == 0) {
        std::memcpy(field.data(), field.data() + plane, plane * sizeof(double));
    } else {
        MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1,
                  comm, &exchange.requests[exchange.count++]);
        MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                  comm, &exchange.requests[exchange.count++]);
    }
    if (rank == ranks - 1) {
        std::memcpy(field.data() + (localNz + 1) * plane,
                    field.data() + localNz * plane, plane * sizeof(double));
    } else {
        MPI_Irecv(field.data() + (localNz + 1) * plane, static_cast<int>(plane),
                  MPI_DOUBLE, rank + 1, 0, comm, &exchange.requests[exchange.count++]);
        MPI_Isend(field.data() + localNz * plane, static_cast<int>(plane),
                  MPI_DOUBLE, rank + 1, 1, comm, &exchange.requests[exchange.count++]);
    }
    return exchange;
}

[[gnu::always_inline]] inline void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              size_t nx, size_t ny, size_t firstZ, size_t lastZ,
                              double dx, double dy, double dz, double gamma,
                              double e_AA, double e_BB, double e_AB) {
    for (size_t z = firstZ; z < lastZ; ++z) {
        const size_t planeOffset = z * nx * ny;
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowOffset = planeOffset + y * nx;
            const double* row = c.data() + rowOffset;
            const double* rowUp = c.data() + planeOffset + ((y < ny - 1) ? y + 1 : y) * nx;
            const double* rowDown = c.data() + planeOffset + ((y > 0) ? y - 1 : y) * nx;
            const double* zUp = c.data() + rowOffset + nx * ny;
            const double* zDown = c.data() + rowOffset - nx * ny;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : x;
                const double cv = row[x];
                const double cxx = (row[xp] + row[xn] - 2.0 * cv) / (dx * dx);
                const double cyy = (rowUp[x] + rowDown[x] - 2.0 * cv) / (dy * dy);
                const double czz = (zUp[x] + zDown[x] - 2.0 * cv) / (dz * dz);
                mu[rowOffset + x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                          + 3.0 * cv + cv * cv * cv
                          - gamma * (cxx + cyy + czz);
            }
        }
    }
}

[[gnu::always_inline]] inline void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, size_t nx, size_t ny,
                        size_t firstZ, size_t lastZ, double D, double dt,
                        double dx, double dy, double dz) {
    for (size_t z = firstZ; z < lastZ; ++z) {
        const size_t planeOffset = z * nx * ny;
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowOffset = planeOffset + y * nx;
            const double* row = mu.data() + rowOffset;
            const double* rowUp = mu.data() + planeOffset + ((y < ny - 1) ? y + 1 : y) * nx;
            const double* rowDown = mu.data() + planeOffset + ((y > 0) ? y - 1 : y) * nx;
            const double* zUp = mu.data() + rowOffset + nx * ny;
            const double* zDown = mu.data() + rowOffset - nx * ny;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : x;
                const double cv = row[x];
                const double cxx = (row[xp] + row[xn] - 2.0 * cv) / (dx * dx);
                const double cyy = (rowUp[x] + rowDown[x] - 2.0 * cv) / (dy * dy);
                const double czz = (zUp[x] + zDown[x] - 2.0 * cv) / (dz * dz);
                cnew[rowOffset + x] = cold[rowOffset + x] + dt * D * (cxx + cyy + czz);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, size_t nx, size_t ny, size_t nz,
                             size_t zStart, size_t localNz) {
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < localNz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = (zStart + z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linearId + 1) * 1299709) % volume) /
                                       static_cast<double>(volume));
                c[idx3(x, y, z + 1, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny > INT_MAX ||
        (printResults && nx * ny * nz > INT_MAX)) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size or MPI count exceeds INT_MAX\n");
        MPI_Finalize();
        return 1;
    }

    // Exclude ranks with no Z plane from the stencil communicator.
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const size_t base = nz / activeRanks;
    const size_t remainder = nz % activeRanks;
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t zStart = static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    std::vector<double> cold((localNz + 2) * plane);
    std::vector<double> cnew((localNz + 2) * plane);
    std::vector<double> mu((localNz + 2) * plane);
    initializeConcentration(cold, nx, ny, nz, zStart, localNz);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        auto cExchange = startHaloExchange(cold, plane, localNz, rank, activeRanks, comm);
        computeChemicalPotential(cold, mu, nx, ny, 2, localNz, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB);
        MPI_Waitall(cExchange.count, cExchange.requests, MPI_STATUSES_IGNORE);
        computeChemicalPotential(cold, mu, nx, ny, 1, 2, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB);
        if (localNz > 1) {
            computeChemicalPotential(cold, mu, nx, ny, localNz, localNz + 1,
                                     dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        }

        auto muExchange = startHaloExchange(mu, plane, localNz, rank, activeRanks, comm);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, 2, localNz,
                           D, dt, dx, dy, dz);
        MPI_Waitall(muExchange.count, muExchange.requests, MPI_STATUSES_IGNORE);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, 1, 2, D, dt, dx, dy, dz);
        if (localNz > 1) {
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, localNz, localNz + 1,
                               D, dt, dx, dy, dz);
        }
        std::swap(cold, cnew);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);
    }

    if (printResults) {
        std::vector<int> counts, offsets;
        std::vector<double> result;
        if (rank == 0) {
            result.resize(gridSize);
            counts.resize(activeRanks);
            offsets.resize(activeRanks);
            for (int r = 0; r < activeRanks; ++r) {
                const size_t owned = base + (static_cast<size_t>(r) < remainder);
                const size_t first = static_cast<size_t>(r) * base +
                                     std::min(static_cast<size_t>(r), remainder);
                counts[r] = static_cast<int>(owned * plane);
                offsets[r] = static_cast<int>(first * plane);
            }
        }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? offsets.data() : nullptr, MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(result, "Concentration");
    }

    if (validate) {
        int finite = 1;
        double localMin = cold[plane], localMax = cold[plane];
        for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
            const double value = cold[i];
            if (!std::isfinite(value)) finite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int allFinite = 0;
        double minValue = 0.0, maxValue = 0.0;
        MPI_Reduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, 0, comm);
        MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
                if (maxValue > 10.0 || minValue < -10.0)
                    printf("Validation failed: values out of expected range\n");
            }
            exitCode = allFinite && maxValue <= 10.0 && minValue >= -10.0 ? 0 : 1;
            printf("Validation: %s\n", exitCode == 0 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return exitCode;
}
