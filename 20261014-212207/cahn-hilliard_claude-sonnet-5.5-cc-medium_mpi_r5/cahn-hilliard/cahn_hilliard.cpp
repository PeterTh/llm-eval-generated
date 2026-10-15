#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Physical constants (identical to the serial benchmark)
static constexpr double DX = 1.0, DY = 1.0, DZ = 1.0;
static constexpr double DT = 0.01;
static constexpr double E_AA = -(2.0 / 9.0);
static constexpr double E_BB = -(2.0 / 9.0);
static constexpr double E_AB = (2.0 / 9.0);
static constexpr double GAMMA = 0.5;
static constexpr double DIFF = 1.0;

// Laplacian at one point with clamped boundaries, given row pointers (already clamped in y and z)
static inline double lap(const double* __restrict r0, const double* __restrict ryn, const double* __restrict ryp,
                         const double* __restrict rzn, const double* __restrict rzp,
                         const size_t x, const size_t xn, const size_t xp) noexcept {
    const double cxx = (r0[xp] + r0[xn] - 2.0 * r0[x]) / (DX * DX);
    const double cyy = (ryp[x] + ryn[x] - 2.0 * r0[x]) / (DY * DY);
    const double czz = (rzp[x] + rzn[x] - 2.0 * r0[x]) / (DZ * DZ);
    return cxx + cyy + czz;
}

static inline double muAt(const double* __restrict r0, const double* __restrict ryn, const double* __restrict ryp,
                          const double* __restrict rzn, const double* __restrict rzp,
                          const size_t x, const size_t xn, const size_t xp) noexcept {
    const double cv = r0[x];
    return 4.5 * ((cv + 1.0) * E_AA + (cv - 1.0) * E_BB - 2.0 * cv * E_AB)
           + 3.0 * cv + cv * cv * cv
           - GAMMA * lap(r0, ryn, ryp, rzn, rzp, x, xn, xp);
}

// Chemical potential on one plane (local plane arrays)
static void muPlane(double* __restrict mu, const double* __restrict c0, const double* __restrict czn,
                    const double* __restrict czp, const size_t nx, const size_t ny) {
    for (size_t y = 0; y < ny; ++y) {
        const double* r0 = c0 + y * nx;
        const double* ryn = c0 + (y > 0 ? y - 1 : 0) * nx;
        const double* ryp = c0 + (y < ny - 1 ? y + 1 : y) * nx;
        const double* rzn = czn + y * nx;
        const double* rzp = czp + y * nx;
        double* o = mu + y * nx;
        o[0] = muAt(r0, ryn, ryp, rzn, rzp, 0, 0, nx > 1 ? 1 : 0);
        if (nx > 1) o[nx - 1] = muAt(r0, ryn, ryp, rzn, rzp, nx - 1, nx - 2, nx - 1);
        for (size_t x = 1; x + 1 < nx; ++x) {
            o[x] = muAt(r0, ryn, ryp, rzn, rzp, x, x - 1, x + 1);
        }
    }
}

static void updatePlane(double* __restrict cn, const double* __restrict co, const double* __restrict m0,
                        const double* __restrict mzn, const double* __restrict mzp, const size_t nx, const size_t ny) {
    for (size_t y = 0; y < ny; ++y) {
        const double* r0 = m0 + y * nx;
        const double* ryn = m0 + (y > 0 ? y - 1 : 0) * nx;
        const double* ryp = m0 + (y < ny - 1 ? y + 1 : y) * nx;
        const double* rzn = mzn + y * nx;
        const double* rzp = mzp + y * nx;
        const double* cr = co + y * nx;
        double* o = cn + y * nx;
        o[0] = cr[0] + DT * DIFF * lap(r0, ryn, ryp, rzn, rzp, 0, 0, nx > 1 ? 1 : 0);
        if (nx > 1) o[nx - 1] = cr[nx - 1] + DT * DIFF * lap(r0, ryn, ryp, rzn, rzp, nx - 1, nx - 2, nx - 1);
        for (size_t x = 1; x + 1 < nx; ++x) {
            o[x] = cr[x] + DT * DIFF * lap(r0, ryn, ryp, rzn, rzp, x, x - 1, x + 1);
        }
    }
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

    // Parse command line arguments (every rank parses identically)
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

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // 1D slab decomposition along z; every active rank owns at least 2 planes so a
    // halo of width 2 always comes from the immediate neighbours.
    int P = (int)std::min<size_t>((size_t)wsize, std::max<size_t>(1, nz / 2));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < P ? 0 : MPI_UNDEFINED, wrank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = wrank;

    const size_t base = nz / P, rem = nz % P;
    const size_t nl = base + ((size_t)rank < rem ? 1 : 0);
    const size_t z0 = (size_t)rank * base + std::min<size_t>((size_t)rank, rem);
    const bool hasLo = rank > 0, hasHi = rank < P - 1;
    const int lo = rank - 1, hi = rank + 1;

    // Local arrays: 2 halo planes on each side. Local plane lp <-> global z = z0 + lp - 2
    const size_t H = 2;
    const size_t nlp = nl + 2 * H;
    std::vector<double> cold(nlp * plane, 0.0), cnew(nlp * plane, 0.0), mu(nlp * plane, 0.0);

    if (rank == 0) printf("Initializing concentration field...\n");
    for (size_t lz = 0; lz < nl; ++lz) {
        double* dst = &cold[(lz + H) * plane];
        const size_t zoff = (z0 + lz) * plane;
        for (size_t i = 0; i < plane; ++i) {
            const size_t linear_id = zoff + i;
            const double pseudo = ((((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize));
            dst[i] = -1.0 + 2.0 * pseudo;
        }
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double tstart = MPI_Wtime();

    const int hcount = (int)(H * plane);
    for (int t = 0; t < iterations; ++t) {
        // Halo exchange of 2 planes with each neighbour
        MPI_Request req[4];
        int nreq = 0;
        if (hasLo) {
            MPI_Irecv(&cold[0], hcount, MPI_DOUBLE, lo, 0, comm, &req[nreq++]);
            MPI_Isend(&cold[H * plane], hcount, MPI_DOUBLE, lo, 1, comm, &req[nreq++]);
        }
        if (hasHi) {
            MPI_Irecv(&cold[(H + nl) * plane], hcount, MPI_DOUBLE, hi, 1, comm, &req[nreq++]);
            MPI_Isend(&cold[nl * plane], hcount, MPI_DOUBLE, hi, 0, comm, &req[nreq++]);
        }

        // Interior planes (not needing halo data) first: mu on lp in [H, H+nl) needs c on lp±1.
        // mu on lp=2..nl-1 (well inside) needs only owned data.
        auto doMu = [&](size_t lp) {
            // clamp at global domain boundaries
            const bool gLo = (!hasLo && lp == H);
            const bool gHi = (!hasHi && lp == H + nl - 1);
            const size_t pn = gLo ? lp : lp - 1;
            const size_t pp = gHi ? lp : lp + 1;
            muPlane(&mu[lp * plane], &cold[lp * plane], &cold[pn * plane], &cold[pp * plane], nx, ny);
        };
        for (size_t lp = H + 1; lp + 1 < H + nl; ++lp) doMu(lp);

        if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

        // Owned boundary planes
        doMu(H);
        doMu(H + nl - 1);
        // Extra mu planes in halo region (redundant compute replaces a second exchange)
        if (hasLo) doMu(H - 1);
        if (hasHi) doMu(H + nl);

        // Update owned planes
        for (size_t lp = H; lp < H + nl; ++lp) {
            const bool gLo = (!hasLo && lp == H);
            const bool gHi = (!hasHi && lp == H + nl - 1);
            const size_t pn = gLo ? lp : lp - 1;
            const size_t pp = gHi ? lp : lp + 1;
            updatePlane(&cnew[lp * plane], &cold[lp * plane], &mu[lp * plane], &mu[pn * plane], &mu[pp * plane], nx, ny);
        }
        std::swap(cold, cnew);
    }

    double elapsed = MPI_Wtime() - tstart;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    long ms = (long)(maxElapsed * 1000.0);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", ms);
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 for external output
    if (printResults) {
        std::vector<double> full;
        std::vector<int> counts(P), displs(P);
        if (rank == 0) full.resize(gridSize);
        for (int r = 0; r < P; ++r) {
            const size_t n = base + ((size_t)r < rem ? 1 : 0);
            const size_t s = (size_t)r * base + std::min<size_t>((size_t)r, rem);
            counts[r] = (int)(n * plane);
            displs[r] = (int)(s * plane);
        }
        MPI_Gatherv(&cold[H * plane], (int)(nl * plane), MPI_DOUBLE, full.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(full, "Concentration");
    }

    int ret = 0;
    if (validate) {
        double lmin = cold[H * plane], lmax = lmin;
        int bad = 0;
        for (size_t i = H * plane; i < (H + nl) * plane; ++i) {
            const double v = cold[i];
            if (std::isnan(v) || std::isinf(v)) bad = 1;
            lmin = std::min(lmin, v);
            lmax = std::max(lmax, v);
        }
        int gbad;
        double gmin, gmax;
        MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
        bool valid = true;
        if (rank == 0) {
            printf("Validating result...\n");
            if (gbad) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = false;
            } else {
                printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
                if (gmax > 10.0 || gmin < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        } else {
            valid = !gbad && !(gmax > 10.0 || gmin < -10.0);
        }
        ret = valid ? 0 : 1;
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
