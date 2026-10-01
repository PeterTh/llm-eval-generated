#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local planes are numbered 1 through localNz; planes 0 and localNz+1 are halos.
inline double computeLaplacian(const std::vector<double>& field, size_t nx, size_t ny,
                               size_t nz, size_t x, size_t y, size_t z, size_t globalZ) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (globalZ < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : x;
    const size_t yn = (y > 0) ? y - 1 : y;
    const size_t zn = (globalZ > 0) ? z - 1 : z;
    const size_t center = idx3(x, y, z, nx, ny);
    const double cxx = field[idx3(xp, y, z, nx, ny)] + field[idx3(xn, y, z, nx, ny)] - 2.0 * field[center];
    const double cyy = field[idx3(x, yp, z, nx, ny)] + field[idx3(x, yn, z, nx, ny)] - 2.0 * field[center];
    const double czz = field[idx3(x, y, zp, nx, ny)] + field[idx3(x, y, zn, nx, ny)] - 2.0 * field[center];
    return cxx + cyy + czz;
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              size_t nx, size_t ny, size_t nz, size_t zBegin,
                              size_t first, size_t last) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    for (size_t z = first; z < last; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = idx3(x, y, z, nx, ny);
                const double cv = c[index];
                mu[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                          + 3.0 * cv + cv * cv * cv
                          - gamma * computeLaplacian(c, nx, ny, nz, x, y, z, zBegin + z - 1);
            }
        }
    }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, size_t nx, size_t ny, size_t nz,
                        size_t zBegin, size_t first, size_t last) {
    constexpr double dt = 0.01;
    constexpr double D = 1.0;
    for (size_t z = first; z < last; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = idx3(x, y, z, nx, ny);
                cnew[index] = cold[index] + dt * D *
                              computeLaplacian(mu, nx, ny, nz, x, y, z, zBegin + z - 1);
            }
        }
    }
}

// Exchange only the two contiguous boundary planes. The caller computes interior
// planes while the transfers are in flight, then waits before using the halos.
int exchangeHalos(std::vector<double>& field, size_t localNz, size_t plane,
                  int rank, int ranks, MPI_Comm comm, MPI_Request requests[4]) {
    int count = 0;
    if (rank > 0) {
        MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[count++]);
        MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[count++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(field.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 0, comm, &requests[count++]);
        MPI_Isend(field.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 1, comm, &requests[count++]);
    }
    return count;
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else {
            if (worldRank == 0) printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
        }
    }
    if (parseStatus) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size or MPI plane count exceeds INT_MAX\n");
        MPI_Finalize();
        return 1;
    }

    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const size_t plane = nx * ny;
    const size_t localNz = nz / activeRanks + (static_cast<size_t>(rank) < nz % activeRanks);
    const size_t zBegin = static_cast<size_t>(rank) * (nz / activeRanks) +
                          std::min(static_cast<size_t>(rank), nz % activeRanks);
    const size_t gridSize = plane * nz;
    std::vector<double> cold((localNz + 2) * plane);
    std::vector<double> cnew((localNz + 2) * plane);
    std::vector<double> mu((localNz + 2) * plane);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    for (size_t z = 1; z <= localNz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = (zBegin + z - 1) * plane + y * nx + x;
                const double pseudo = ((((linearId + 1) * 1299709) % gridSize) /
                                       static_cast<double>(gridSize));
                cold[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        MPI_Request requests[4];
        int count = exchangeHalos(cold, localNz, plane, rank, activeRanks, comm, requests);
        if (localNz > 2) computeChemicalPotential(cold, mu, nx, ny, nz, zBegin, 2, localNz);
        if (count) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        computeChemicalPotential(cold, mu, nx, ny, nz, zBegin, 1, 2);
        if (localNz > 1) computeChemicalPotential(cold, mu, nx, ny, nz, zBegin, localNz, localNz + 1);

        count = exchangeHalos(mu, localNz, plane, rank, activeRanks, comm, requests);
        if (localNz > 2) cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, zBegin, 2, localNz);
        if (count) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, zBegin, 1, 2);
        if (localNz > 1) cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, zBegin, localNz, localNz + 1);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double mcups = maxElapsed > 0.0 ?
            static_cast<double>(gridSize) * iterations / maxElapsed / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        // Gather only for external reporting; the simulation itself stays distributed.
        // Chunked transfers avoid MPI's int count and displacement limits.
        constexpr size_t chunk = static_cast<size_t>(INT_MAX);
        if (rank == 0) {
            std::vector<double> all(gridSize);
            std::copy_n(cold.data() + plane, localNz * plane, all.data());
            for (int source = 1; source < activeRanks; ++source) {
                const size_t sourceNz = nz / activeRanks +
                                        (static_cast<size_t>(source) < nz % activeRanks);
                const size_t sourceBegin = static_cast<size_t>(source) * (nz / activeRanks) +
                                           std::min(static_cast<size_t>(source), nz % activeRanks);
                const size_t total = sourceNz * plane;
                for (size_t offset = 0; offset < total; offset += chunk) {
                    MPI_Recv(all.data() + sourceBegin * plane + offset,
                             static_cast<int>(std::min(chunk, total - offset)), MPI_DOUBLE,
                             source, 2, comm, MPI_STATUS_IGNORE);
                }
            }
            print_results(all, "Concentration");
        } else {
            const size_t total = localNz * plane;
            for (size_t offset = 0; offset < total; offset += chunk) {
                MPI_Send(cold.data() + plane + offset,
                         static_cast<int>(std::min(chunk, total - offset)), MPI_DOUBLE, 0, 2, comm);
            }
        }
    }

    int status = 0;
    if (validate) {
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        int localFinite = 1;
        for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
            const double value = cold[i];
            if (!std::isfinite(value)) localFinite = 0;
            else {
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
        double globalMin, globalMax;
        int allFinite;
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&localFinite, &allFinite, 1, MPI_INT, MPI_MIN, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 10.0 || globalMin < -10.0)
                    printf("Validation failed: values out of expected range\n");
            }
            status = allFinite && globalMax <= 10.0 && globalMin >= -10.0 ? 0 : 1;
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
