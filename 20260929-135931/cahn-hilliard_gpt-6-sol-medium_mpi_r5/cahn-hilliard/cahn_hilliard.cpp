#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Contiguous Z slabs with one ghost plane on either side.
struct Slab {
    size_t nx, ny, nz, plane, localNz, firstZ;
    int rank, ranks;
    MPI_Comm comm;
};

static size_t startZ(size_t nz, int rank, int ranks) {
    return (nz / static_cast<size_t>(ranks)) * static_cast<size_t>(rank)
         + std::min(static_cast<size_t>(rank), nz % static_cast<size_t>(ranks));
}

static size_t ownedZ(size_t nz, int rank, int ranks) {
    return nz / static_cast<size_t>(ranks)
         + (static_cast<size_t>(rank) < nz % static_cast<size_t>(ranks));
}

// Post the two plane transfers before evaluating cells independent of halos.
static int exchangeHalos(std::vector<double>& field, const Slab& s, MPI_Request requests[4]) {
    int count = 0;
    const int plane = static_cast<int>(s.plane);
    if (s.rank > 0) {
        MPI_Irecv(field.data(), plane, MPI_DOUBLE, s.rank - 1, 2, s.comm, &requests[count++]);
        MPI_Isend(field.data() + s.plane, plane, MPI_DOUBLE, s.rank - 1, 1, s.comm, &requests[count++]);
    } else {
        std::copy_n(field.data() + s.plane, s.plane, field.data());
    }
    if (s.rank + 1 < s.ranks) {
        MPI_Irecv(field.data() + (s.localNz + 1) * s.plane, plane, MPI_DOUBLE,
                  s.rank + 1, 1, s.comm, &requests[count++]);
        MPI_Isend(field.data() + s.localNz * s.plane, plane, MPI_DOUBLE,
                  s.rank + 1, 2, s.comm, &requests[count++]);
    } else {
        std::copy_n(field.data() + s.localNz * s.plane, s.plane,
                    field.data() + (s.localNz + 1) * s.plane);
    }
    return count;
}

static inline double laplacian(const double* f, size_t i, size_t x, size_t y,
                               size_t nx, size_t ny, size_t plane) {
    const double center = 2.0 * f[i];
    const double xx = f[x + 1 < nx ? i + 1 : i] + f[x ? i - 1 : i] - center;
    const double yy = f[y + 1 < ny ? i + nx : i] + f[y ? i - nx : i] - center;
    const double zz = f[i + plane] + f[i - plane] - center;
    return xx + yy + zz;
}

static void chemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const Slab& s, size_t zBegin, size_t zEnd) {
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    const double* in = c.data();
    double* out = mu.data();
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < s.ny; ++y) {
            for (size_t x = 0; x < s.nx; ++x) {
                const size_t i = z * s.plane + y * s.nx + x;
                const double cv = in[i];
                out[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * laplacian(in, i, x, y, s.nx, s.ny, s.plane);
            }
        }
    }
}

static void updateConcentration(const std::vector<double>& old,
                                const std::vector<double>& mu, std::vector<double>& next,
                                const Slab& s, size_t zBegin, size_t zEnd) {
    constexpr double dt = 0.01;
    constexpr double D = 1.0;
    const double* in = mu.data();
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < s.ny; ++y) {
            for (size_t x = 0; x < s.nx; ++x) {
                const size_t i = z * s.plane + y * s.nx + x;
                next[i] = old[i] + dt * D * laplacian(in, i, x, y, s.nx, s.ny, s.plane);
            }
        }
    }
}

static void initializeConcentration(std::vector<double>& c, const Slab& s) {
    const size_t volume = s.plane * s.nz;
    for (size_t z = 0; z < s.localNz; ++z) {
        const size_t globalBase = (s.firstZ + z) * s.plane;
        const size_t localBase = (z + 1) * s.plane;
        for (size_t p = 0; p < s.plane; ++p) {
            const size_t linearId = globalBase + p;
            const double pseudo = (((linearId + 1) * 1299709) % volume) / static_cast<double>(volume);
            c[localBase + p] = -1.0 + 2.0 * pseudo;
        }
    }
}

static void printUsage(const char* program) {
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

// Materialize the full field only for external result reporting.
static void printDistributedResults(const std::vector<double>& c, const Slab& s) {
    const size_t localCount = s.localNz * s.plane;
    if (s.rank == 0) {
        std::vector<double> full(s.plane * s.nz);
        std::copy_n(c.data() + s.plane, localCount, full.data());
        for (int rank = 1; rank < s.ranks; ++rank) {
            size_t count = ownedZ(s.nz, rank, s.ranks) * s.plane;
            double* target = full.data() + startZ(s.nz, rank, s.ranks) * s.plane;
            while (count) {
                const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                MPI_Recv(target, chunk, MPI_DOUBLE, rank, 3, s.comm, MPI_STATUS_IGNORE);
                target += chunk;
                count -= chunk;
            }
        }
        print_results(full, "Concentration");
    } else {
        const double* source = c.data() + s.plane;
        size_t count = localCount;
        while (count) {
            const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
            MPI_Send(source, chunk, MPI_DOUBLE, 0, 3, s.comm);
            source += chunk;
            count -= chunk;
        }
    }
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
        else if (strcmp(argv[i], "-h") == 0) { exitCode = 2; break; }
        else {
            if (worldRank == 0) printf("Unknown option: %s\n", argv[i]);
            exitCode = 1;
            break;
        }
    }
    if (exitCode) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return exitCode == 2 ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size or MPI halo plane exceeds INT_MAX elements\n");
        MPI_Finalize();
        return 1;
    }

    // There cannot be more nonempty slabs than Z planes.
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    Slab s{nx, ny, nz, nx * ny, ownedZ(nz, worldRank, activeRanks),
           startZ(nz, worldRank, activeRanks), worldRank, activeRanks, comm};
    if (s.localNz > SIZE_MAX / s.plane - 2) {
        if (worldRank == 0) fprintf(stderr, "Grid too large for local allocation\n");
        MPI_Abort(comm, 1);
    }
    std::vector<double> cold((s.localNz + 2) * s.plane);
    std::vector<double> cnew((s.localNz + 2) * s.plane);
    std::vector<double> mu((s.localNz + 2) * s.plane);

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, s);
    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        MPI_Request requests[4];
        int count = exchangeHalos(cold, s, requests);
        if (s.localNz > 2) chemicalPotential(cold, mu, s, 2, s.localNz);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        chemicalPotential(cold, mu, s, 1, 2);
        if (s.localNz > 1) chemicalPotential(cold, mu, s, s.localNz, s.localNz + 1);

        count = exchangeHalos(mu, s, requests);
        if (s.localNz > 2) updateConcentration(cold, mu, cnew, s, 2, s.localNz);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        updateConcentration(cold, mu, cnew, s, 1, 2);
        if (s.localNz > 1) updateConcentration(cold, mu, cnew, s, s.localNz, s.localNz + 1);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double cellUpdates = static_cast<double>(s.plane * s.nz) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / maxElapsed / 1e6);
    }

    if (printResults) printDistributedResults(cold, s);
    if (validate) {
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        int localFinite = 1;
        for (size_t i = s.plane; i < (s.localNz + 1) * s.plane; ++i) {
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
        if (worldRank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 10.0 || globalMin < -10.0)
                    printf("Validation failed: values out of expected range\n");
            }
            exitCode = allFinite && globalMin >= -10.0 && globalMax <= 10.0 ? 0 : 1;
            printf("Validation: %s\n", exitCode ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return exitCode;
}
