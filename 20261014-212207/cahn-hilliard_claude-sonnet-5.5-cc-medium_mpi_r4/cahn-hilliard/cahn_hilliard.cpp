#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Parallelization: 1D slab decomposition along z (slowest dimension) using MPI.
// Each rank stores its owned planes plus 2 ghost planes on each side. One
// width-2 halo exchange of the concentration per step lets each rank compute
// the chemical potential on its ghost-adjacent planes redundantly, so no second
// exchange is needed for the update. Communication is overlapped with the
// computation of planes that do not depend on ghost data.
static constexpr int HALO = 2;

// Chemical potential on one local plane lz (global plane gz)
static void muPlane(const double* c, double* mu, const size_t lz, const long gz, const size_t nx, const size_t ny,
                    const long nz, const double dx, const double dy, const double dz,
                    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t pl = nx * ny;
    const double* cz = c + lz * pl;
    const double* czp = c + ((gz < nz - 1) ? lz + 1 : lz) * pl;
    const double* czn = c + ((gz > 0) ? lz - 1 : lz) * pl;
    double* m = mu + lz * pl;
    for (size_t y = 0; y < ny; ++y) {
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;
        const double* c0 = cz + y * nx;
        const double* cyp = cz + yp * nx;
        const double* cyn = cz + yn * nx;
        const double* zp = czp + y * nx;
        const double* zn = czn + y * nx;
        double* mr = m + y * nx;
        auto edge = [&](size_t x) {
            const size_t xp = (x < nx - 1) ? x + 1 : x;
            const size_t xn = (x > 0) ? x - 1 : 0;
            const double cv = c0[x];
            const double cxx = (c0[xp] + c0[xn] - 2.0 * cv) / (dx * dx);
            const double cyy = (cyp[x] + cyn[x] - 2.0 * cv) / (dy * dy);
            const double czz = (zp[x] + zn[x] - 2.0 * cv) / (dz * dz);
            mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
        };
        edge(0);
        for (size_t x = 1; x + 1 < nx; ++x) {
            const double cv = c0[x];
            const double cxx = (c0[x + 1] + c0[x - 1] - 2.0 * cv) / (dx * dx);
            const double cyy = (cyp[x] + cyn[x] - 2.0 * cv) / (dy * dy);
            const double czz = (zp[x] + zn[x] - 2.0 * cv) / (dz * dz);
            mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
        }
        if (nx > 1) edge(nx - 1);
    }
}

// Cahn-Hilliard update on one local plane
static void updatePlane(double* cnew, const double* cold, const double* mu, const size_t lz, const long gz,
                        const size_t nx, const size_t ny, const long nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t pl = nx * ny;
    const double* mz = mu + lz * pl;
    const double* mzp = mu + ((gz < nz - 1) ? lz + 1 : lz) * pl;
    const double* mzn = mu + ((gz > 0) ? lz - 1 : lz) * pl;
    const double* co = cold + lz * pl;
    double* cn = cnew + lz * pl;
    for (size_t y = 0; y < ny; ++y) {
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;
        const double* m0 = mz + y * nx;
        const double* myp = mz + yp * nx;
        const double* myn = mz + yn * nx;
        const double* zp = mzp + y * nx;
        const double* zn = mzn + y * nx;
        const double* c0 = co + y * nx;
        double* out = cn + y * nx;
        auto edge = [&](size_t x) {
            const size_t xp = (x < nx - 1) ? x + 1 : x;
            const size_t xn = (x > 0) ? x - 1 : 0;
            const double mv = m0[x];
            const double cxx = (m0[xp] + m0[xn] - 2.0 * mv) / (dx * dx);
            const double cyy = (myp[x] + myn[x] - 2.0 * mv) / (dy * dy);
            const double czz = (zp[x] + zn[x] - 2.0 * mv) / (dz * dz);
            out[x] = c0[x] + dt * D * (cxx + cyy + czz);
        };
        edge(0);
        for (size_t x = 1; x + 1 < nx; ++x) {
            const double mv = m0[x];
            const double cxx = (m0[x + 1] + m0[x - 1] - 2.0 * mv) / (dx * dx);
            const double cyy = (myp[x] + myn[x] - 2.0 * mv) / (dy * dy);
            const double czz = (zp[x] + zn[x] - 2.0 * mv) / (dz * dz);
            out[x] = c0[x] + dt * D * (cxx + cyy + czz);
        }
        if (nx > 1) edge(nx - 1);
    }
}

// Initialize concentration field for owned planes [z0, z0+nzl) into local array (offset by HALO planes)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzl) {
    const size_t vol = nx * ny * nz;
    for (size_t zl = 0; zl < nzl; ++zl) {
        const size_t z = z0 + zl;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = (zl + HALO) * (nx * ny) + y * nx + x;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validation on distributed data: each rank reduces its owned values, then globally combined.
bool validateResult(const double* c, const size_t n, MPI_Comm comm, const int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (bad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
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
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    
    if (worldRank == 0) {
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

    const size_t gridSize = nx * ny * nz;
    const size_t pl = nx * ny;

    // Use at most nz/HALO ranks so every active rank owns >= HALO planes
    const int active = (int)std::max<size_t>(1, std::min<size_t>((size_t)worldSize, nz / HALO));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    const bool isActive = comm != MPI_COMM_NULL;
    const int rank = worldRank;

    size_t z0 = 0, nzl = 0;
    if (isActive) {
        const size_t base = nz / active, rem = nz % active;
        nzl = base + ((size_t)rank < rem ? 1 : 0);
        z0 = (size_t)rank * base + std::min((size_t)rank, rem);
    }
    const long lnz = (long)nz;
    const int down = (isActive && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (isActive && rank < active - 1) ? rank + 1 : MPI_PROC_NULL;

    std::vector<double> cold, cnew, mu;
    if (isActive) {
        const size_t localSize = (nzl + 2 * HALO) * pl;
        cold.assign(localSize, 0.0);
        cnew.assign(localSize, 0.0);
        mu.assign(localSize, 0.0);
    }

    if (worldRank == 0) printf("Initializing concentration field...\n");
    if (isActive) initializeConcentration(cold, nx, ny, nz, z0, nzl);

    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (isActive) {
        const size_t H = HALO;
        const long gbase = (long)z0 - (long)H;  // global plane of local plane 0
        auto mup = [&](size_t lz) {
            const long gz = gbase + (long)lz;
            if (gz < 0 || gz >= lnz) return;
            muPlane(cold.data(), mu.data(), lz, gz, nx, ny, lnz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        };
        for (int t = 0; t < iterations; ++t) {
            MPI_Request req[4];
            double* c = cold.data();
            MPI_Irecv(c, (int)(H * pl), MPI_DOUBLE, down, 0, comm, &req[0]);
            MPI_Irecv(c + (nzl + H) * pl, (int)(H * pl), MPI_DOUBLE, up, 1, comm, &req[1]);
            MPI_Isend(c + H * pl, (int)(H * pl), MPI_DOUBLE, down, 1, comm, &req[2]);
            MPI_Isend(c + nzl * pl, (int)(H * pl), MPI_DOUBLE, up, 0, comm, &req[3]);

            // Planes independent of ghost data
            for (size_t lz = H + 1; lz <= nzl; ++lz) mup(lz);
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            mup(1);
            mup(2);
            mup(nzl + 1);
            mup(nzl + 2);

            for (size_t lz = H; lz < H + nzl; ++lz)
                updatePlane(cnew.data(), cold.data(), mu.data(), lz, gbase + (long)lz, nx, ny, lnz, D, dt, dx, dy, dz);
            std::swap(cold, cnew);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", max_ms);
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        std::vector<int> counts(worldSize, 0), displs(worldSize, 0);
        std::vector<double> full;
        if (worldRank == 0) {
            full.resize(gridSize);
            const size_t base = nz / active, rem = nz % active;
            size_t off = 0;
            for (int r = 0; r < active; ++r) {
                const size_t n = (base + ((size_t)r < rem ? 1 : 0)) * pl;
                counts[r] = (int)n;
                displs[r] = (int)off;
                off += n;
            }
        }
        MPI_Gatherv(isActive ? cold.data() + HALO * pl : nullptr, (int)(nzl * pl), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (worldRank == 0) print_results(full, "Concentration");
    }

    int ret = 0;
    if (validate) {
        if (worldRank == 0) printf("Validating result...\n");
        bool valid = validateResult(isActive ? cold.data() + HALO * pl : nullptr, nzl * pl, MPI_COMM_WORLD, worldRank);
        if (worldRank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        ret = valid ? 0 : 1;
    }

    if (isActive) MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
