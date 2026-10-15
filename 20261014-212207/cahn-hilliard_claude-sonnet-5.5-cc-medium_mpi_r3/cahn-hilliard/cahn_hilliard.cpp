#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Laplacian value for one point given its neighbours (clamped boundaries are
// handled by the caller choosing the neighbour pointers)
static inline double lap1(const double xp, const double xn, const double yp, const double yn,
                          const double zp, const double zn, const double c,
                          const double dx2, const double dy2, const double dz2) noexcept {
    const double cxx = (xp + xn - 2.0 * c) / dx2;
    const double cyy = (yp + yn - 2.0 * c) / dy2;
    const double czz = (zp + zn - 2.0 * c) / dz2;
    return cxx + cyy + czz;
}

// Apply op(i, laplacian(in)[i]) to all points of local plane p (planes include ghost layers)
template <typename Op>
static inline void stencilPlane(const double* __restrict in, const size_t p, const size_t nx, const size_t ny,
                                const double dx2, const double dy2, const double dz2, Op op) {
    const size_t nxy = nx * ny;
    for (size_t y = 0; y < ny; ++y) {
        const size_t base = p * nxy + y * nx;
        const double* __restrict row = in + base;
        const double* __restrict rn = (y > 0) ? row - nx : row;
        const double* __restrict rp = (y < ny - 1) ? row + nx : row;
        const double* __restrict rzp = row + nxy;
        const double* __restrict rzn = row - nxy;
        // x = 0 and x = nx-1 (clamped)
        {
            const size_t xs[2] = {0, nx - 1};
            for (int k = 0; k < (nx > 1 ? 2 : 1); ++k) {
                const size_t x = xs[k];
                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : 0;
                op(base + x, lap1(row[xp], row[xn], rp[x], rn[x], rzp[x], rzn[x], row[x], dx2, dy2, dz2));
            }
        }
        for (size_t x = 1; x + 1 < nx; ++x) {
            op(base + x, lap1(row[x + 1], row[x - 1], rp[x], rn[x], rzp[x], rzn[x], row[x], dx2, dy2, dz2));
        }
    }
}

// Initialize concentration field for owned planes [z0, z0+nzl) (local plane index 1..nzl)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzl) {
    const size_t vol = nx * ny * nz;
    for (size_t zl = 0; zl < nzl; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = (z0 + zl) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[(zl + 1) * nx * ny + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Distributed validation: c holds the owned planes (no ghosts) of this rank
bool validateResult(const double* c, const size_t n, MPI_Comm comm, const int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    double gmin, gmax;
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 10.0 || gmin < -10.0) {
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
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

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
            if (wrank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (wrank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (wrank == 0) {
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
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;

    const size_t gridSize = nx * ny * nz;
    const size_t nxy = nx * ny;

    // 1D slab decomposition along z; only min(P, nz) ranks take part
    const int nact = (int)std::min<size_t>((size_t)wsize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < nact ? 0 : MPI_UNDEFINED, wrank, &comm);

    int result = 0;  // 1 = failed validation
    if (comm != MPI_COMM_NULL) {
        const int rank = wrank;
        const size_t base = nz / nact, rem = nz % nact;
        const size_t nzl = base + ((size_t)rank < rem ? 1 : 0);
        const size_t z0 = (size_t)rank * base + std::min<size_t>((size_t)rank, rem);
        const int down = rank > 0 ? rank - 1 : MPI_PROC_NULL;
        const int up = rank < nact - 1 ? rank + 1 : MPI_PROC_NULL;

        // local arrays with one ghost plane on each side
        const size_t localSize = (nzl + 2) * nxy;
        std::vector<double> cold(localSize, 0.0), cnew(localSize, 0.0), mu(localSize, 0.0);

        if (rank == 0) printf("Initializing concentration field...\n");
        initializeConcentration(cold, nx, ny, nz, z0, nzl);

        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        const double k1 = 4.5, dtD = dt * D;
        auto muOp = [&](const double* __restrict cp, double* __restrict mp) {
            return [=](size_t i, double lap) {
                const double cv = cp[i];
                mp[i] = k1 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv - gamma * lap;
            };
        };

        // Exchange ghost planes of arr; clamped copy at global boundaries is done separately
        auto postHalo = [&](std::vector<double>& arr, MPI_Request* req) {
            double* a = arr.data();
            MPI_Irecv(a, (int)nxy, MPI_DOUBLE, down, 0, comm, &req[0]);
            MPI_Irecv(a + (nzl + 1) * nxy, (int)nxy, MPI_DOUBLE, up, 1, comm, &req[1]);
            MPI_Isend(a + nzl * nxy, (int)nxy, MPI_DOUBLE, up, 0, comm, &req[2]);
            MPI_Isend(a + nxy, (int)nxy, MPI_DOUBLE, down, 1, comm, &req[3]);
        };
        auto clampGhost = [&](std::vector<double>& arr) {
            double* a = arr.data();
            if (down == MPI_PROC_NULL) memcpy(a, a + nxy, nxy * sizeof(double));
            if (up == MPI_PROC_NULL) memcpy(a + (nzl + 1) * nxy, a + nzl * nxy, nxy * sizeof(double));
        };

        for (int t = 0; t < iterations; ++t) {
            MPI_Request req[4];
            // --- chemical potential ---
            postHalo(cold, req);
            clampGhost(cold);
            {
                auto op = muOp(cold.data(), mu.data());
                for (size_t p = 2; p + 1 <= nzl; ++p)
                    stencilPlane(cold.data(), p, nx, ny, dx2, dy2, dz2, op);
                MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
                stencilPlane(cold.data(), 1, nx, ny, dx2, dy2, dz2, op);
                if (nzl > 1) stencilPlane(cold.data(), nzl, nx, ny, dx2, dy2, dz2, op);
            }
            // --- update ---
            postHalo(mu, req);
            clampGhost(mu);
            {
                const double* mp = mu.data();
                const double* cp = cold.data();
                double* np = cnew.data();
                auto op = [=](size_t i, double lap) { np[i] = cp[i] + dtD * lap; };
                for (size_t p = 2; p + 1 <= nzl; ++p)
                    stencilPlane(mp, p, nx, ny, dx2, dy2, dz2, op);
                MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
                stencilPlane(mp, 1, nx, ny, dx2, dy2, dz2, op);
                if (nzl > 1) stencilPlane(mp, nzl, nx, ny, dx2, dy2, dz2, op);
            }
            std::swap(cold, cnew);
        }

        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        long local_ms = duration.count();
        long max_ms;
        MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, comm);

        if (rank == 0) {
            printf("Computation time: %ld ms\n", max_ms);
            double cellUpdates = (double)gridSize * iterations;
            double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // Print results for external validation (gather full field on rank 0)
        if (printResults) {
            std::vector<double> full;
            std::vector<int> counts(nact), displs(nact);
            if (rank == 0) {
                full.resize(gridSize);
                for (int r = 0; r < nact; ++r) {
                    const size_t nr = base + ((size_t)r < rem ? 1 : 0);
                    const size_t zr = (size_t)r * base + std::min<size_t>((size_t)r, rem);
                    counts[r] = (int)(nr * nxy);
                    displs[r] = (int)(zr * nxy);
                }
            }
            MPI_Gatherv(cold.data() + nxy, (int)(nzl * nxy), MPI_DOUBLE, full.data(), counts.data(),
                        displs.data(), MPI_DOUBLE, 0, comm);
            if (rank == 0) print_results(full, "Concentration");
        }

        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateResult(cold.data() + nxy, nzl * nxy, comm, rank);
            if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            result = valid ? 0 : 1;
        }
        MPI_Comm_free(&comm);
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
