#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// MPI parallelization: 1D slab decomposition along the (slowest-varying) z
// axis. Every rank owns nzl contiguous z-planes and stores them together with
// one ghost plane below and one above (local plane index 0 and nzl + 1).
// The global clamped boundary condition is reproduced exactly by filling the
// ghost plane with a copy of the adjacent owned plane on the ranks at the
// global ends of the domain, so that c[zn] == c[z] there as in the original.
// Halo exchange uses non-blocking point-to-point communication overlapped
// with the computation of the interior planes.
// ---------------------------------------------------------------------------

// 3D index calculation (local, including ghost planes)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Laplacian from explicit neighbour values (same expression as the original).
inline double laplacianVals(const double cxp, const double cxn, const double cc,
                            const double cyp, const double cyn, const double czp, const double czn,
                            const double dx, const double dy, const double dz) noexcept {
    const double cxx = (cxp + cxn - 2.0 * cc) / (dx * dx);
    const double cyy = (cyp + cyn - 2.0 * cc) / (dy * dy);
    const double czz = (czp + czn - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

// Apply f(x, laplacian) to every cell of one x-row. The z and y neighbour rows
// are passed already clamped / halo-filled; clamping in x is handled by
// peeling the first and last cell so the inner loop is branch-free and
// vectorizes.
template <typename F>
inline void forEachRowLaplacian(const double* __restrict cm, const double* __restrict czm, const double* __restrict czp,
                                const double* __restrict cym, const double* __restrict cyp,
                                const size_t nx, const double dx, const double dy, const double dz, F&& f) {
    if (nx == 1) {
        f(0, laplacianVals(cm[0], cm[0], cm[0], cyp[0], cym[0], czp[0], czm[0], dx, dy, dz));
        return;
    }
    f(0, laplacianVals(cm[1], cm[0], cm[0], cyp[0], cym[0], czp[0], czm[0], dx, dy, dz));
    for (size_t x = 1; x < nx - 1; ++x) {
        f(x, laplacianVals(cm[x + 1], cm[x - 1], cm[x], cyp[x], cym[x], czp[x], czm[x], dx, dy, dz));
    }
    const size_t e = nx - 1;
    f(e, laplacianVals(cm[e], cm[e - 1], cm[e], cyp[e], cym[e], czp[e], czm[e], dx, dy, dz));
}

// Compute chemical potential for local planes [zBegin, zEnd) (local indices,
// 1 <= zBegin, zEnd <= nzl + 1). Requires valid ghost planes for neighbours.
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t zBegin, const size_t zEnd,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    for (size_t z = zBegin; z < zEnd; ++z) {
        const double* __restrict cz  = c.data() + z * plane;
        const double* __restrict czm = cz - plane;
        const double* __restrict czp = cz + plane;
        double* __restrict muz = mu.data() + z * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const double* __restrict cm  = cz + y * nx;
            const double* __restrict cym = cz + yn * nx;
            const double* __restrict cyp = cz + yp * nx;
            const double* __restrict zm  = czm + y * nx;
            const double* __restrict zp  = czp + y * nx;
            double* __restrict mrow = muz + y * nx;
            forEachRowLaplacian(cm, zm, zp, cym, cyp, nx, dx, dy, dz, [&](const size_t x, const double lap) {
                const double cv = cm[x];
                mrow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * lap;
            });
        }
    }
}

// Cahn-Hilliard update step for local planes [zBegin, zEnd).
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t zBegin, const size_t zEnd,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    for (size_t z = zBegin; z < zEnd; ++z) {
        const double* __restrict muz  = mu.data() + z * plane;
        const double* __restrict muzm = muz - plane;
        const double* __restrict muzp = muz + plane;
        const double* __restrict coz  = cold.data() + z * plane;
        double* __restrict cnz = cnew.data() + z * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const double* __restrict mm  = muz + y * nx;
            const double* __restrict mym = muz + yn * nx;
            const double* __restrict myp = muz + yp * nx;
            const double* __restrict zm  = muzm + y * nx;
            const double* __restrict zp  = muzp + y * nx;
            const double* __restrict crow = coz + y * nx;
            double* __restrict nrow = cnz + y * nx;
            forEachRowLaplacian(mm, zm, zp, mym, myp, nx, dx, dy, dz, [&](const size_t x, const double lap) {
                nrow[x] = crow[x] + dt * D * lap;
            });
        }
    }
}

// Initialize the locally owned part of the concentration field (global planes
// [z0, z0 + nzl) stored at local planes [1, nzl + 1)).
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzl) {
    const size_t vol = nx * ny * nz;

    for (size_t zl = 0; zl < nzl; ++zl) {
        const size_t z = z0 + zl;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zl + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Halo exchange context
struct Halo {
    MPI_Comm comm;
    int lower;      // rank owning the plane below, or MPI_PROC_NULL at the global boundary
    int upper;      // rank owning the plane above, or MPI_PROC_NULL at the global boundary
    size_t plane;   // nx * ny
    size_t nzl;     // number of owned planes
};

// Start the non-blocking exchange of the ghost planes of `f`. Where there is
// no neighbour (global boundary), the ghost plane is filled with a copy of the
// adjacent owned plane to reproduce the clamped boundary condition.
inline void haloStart(std::vector<double>& f, const Halo& h, MPI_Request req[4]) {
    double* data = f.data();
    const size_t p = h.plane;
    const int cnt = static_cast<int>(p);
    // receive into ghost planes
    MPI_Irecv(data, cnt, MPI_DOUBLE, h.lower, 0, h.comm, &req[0]);
    MPI_Irecv(data + (h.nzl + 1) * p, cnt, MPI_DOUBLE, h.upper, 1, h.comm, &req[1]);
    // send boundary owned planes
    MPI_Isend(data + p, cnt, MPI_DOUBLE, h.lower, 1, h.comm, &req[2]);
    MPI_Isend(data + h.nzl * p, cnt, MPI_DOUBLE, h.upper, 0, h.comm, &req[3]);
    if (h.lower == MPI_PROC_NULL) std::memcpy(data, data + p, p * sizeof(double));
    if (h.upper == MPI_PROC_NULL) std::memcpy(data + (h.nzl + 1) * p, data + h.nzl * p, p * sizeof(double));
}

inline void haloWait(MPI_Request req[4]) {
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (root) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
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
    const size_t plane = nx * ny;

    // Domain decomposition along z: block distribution, first (nz % nprocs)
    // ranks get one extra plane.
    auto planesOf = [&](int r) -> size_t {
        return nz / nprocs + (static_cast<size_t>(r) < nz % nprocs ? 1 : 0);
    };
    auto startOf = [&](int r) -> size_t {
        const size_t base = nz / nprocs;
        const size_t rem = nz % nprocs;
        const size_t ur = static_cast<size_t>(r);
        return ur * base + std::min(ur, rem);
    };
    const size_t nzl = planesOf(rank);
    const size_t z0 = startOf(rank);

    // Neighbours: nearest rank in each direction that owns at least one plane
    // (ranks with zero planes are skipped; they take no part in the exchange).
    Halo halo{MPI_COMM_WORLD, MPI_PROC_NULL, MPI_PROC_NULL, plane, nzl};
    for (int r = rank - 1; r >= 0; --r) {
        if (planesOf(r) > 0) { halo.lower = r; break; }
    }
    for (int r = rank + 1; r < nprocs; ++r) {
        if (planesOf(r) > 0) { halo.upper = r; break; }
    }
    const bool active = (nzl > 0);

    // Allocate arrays (owned planes + 2 ghost planes)
    const size_t localSize = (nzl + 2) * plane;
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z0, nzl);

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    MPI_Request req[4];
    // Interior planes do not depend on ghost data and can be computed while
    // the halo exchange is in flight; boundary planes are done after the wait.
    const bool hasInterior = (nzl > 2);
    const size_t iBegin = 2;                    // first interior plane (local index)
    const size_t iEnd = hasInterior ? nzl : 2;  // one past last interior plane

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            // Compute chemical potential
            haloStart(cold, halo, req);
            computeChemicalPotential(cold, mu, nx, ny, iBegin, iEnd, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            haloWait(req);
            computeChemicalPotential(cold, mu, nx, ny, 1, std::min<size_t>(2, nzl + 1), dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (nzl > 1)
                computeChemicalPotential(cold, mu, nx, ny, nzl, nzl + 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

            // Update concentration
            haloStart(mu, halo, req);
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, iBegin, iEnd, D, dt, dx, dy, dz);
            haloWait(req);
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, 1, std::min<size_t>(2, nzl + 1), D, dt, dx, dy, dz);
            if (nzl > 1)
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, nzl, nzl + 1, D, dt, dx, dy, dz);
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxMs = localMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %ld ms\n", maxMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (maxMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;

    if (printResults || validate) {
        // Gather the full field on the root rank so that the output statistics
        // (ordering-dependent sum and hash) are identical to the serial program.
        std::vector<double> full;
        std::vector<int> counts, displs;
        if (root) {
            full.resize(gridSize);
            counts.resize(nprocs);
            displs.resize(nprocs);
            for (int r = 0; r < nprocs; ++r) {
                counts[r] = static_cast<int>(planesOf(r) * plane);
                displs[r] = static_cast<int>(startOf(r) * plane);
            }
        }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(nzl * plane), MPI_DOUBLE,
                    root ? full.data() : nullptr, root ? counts.data() : nullptr,
                    root ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(full, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
