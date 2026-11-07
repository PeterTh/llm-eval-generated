#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// MPI parallelization strategy
//
// The global nx x ny x nz grid is decomposed into contiguous slabs along the
// slowest (z) dimension.  Each rank owns nz_loc planes and keeps a halo of
// two planes on either side for the concentration field c, and one plane on
// either side for the chemical potential mu.  A two-plane halo of c lets each
// rank compute mu on one ghost plane per side itself, so that only ONE halo
// exchange is needed per time step (instead of one for c and one for mu).
// The halo exchange is non-blocking and overlapped with the computation of
// mu on the interior planes.
//
// Clamped (zero-gradient-like index clamp) boundary conditions of the
// original code are reproduced exactly: at the global z boundaries the ghost
// plane is a copy of the boundary plane, which is identical to the clamped
// index access in the original, and the x/y clamps are applied in-plane.
// The arithmetic order per cell is unchanged, so results are bit-identical
// to the serial version regardless of the number of ranks.
// ---------------------------------------------------------------------------

// Plane layout helpers: plane p (may be negative for ghosts) of an array with
// `halo` ghost planes on each side.
inline double* planePtr(std::vector<double>& a, const long p, const long halo, const size_t planeSize) noexcept {
    return a.data() + static_cast<size_t>(p + halo) * planeSize;
}
inline const double* planePtr(const std::vector<double>& a, const long p, const long halo, const size_t planeSize) noexcept {
    return a.data() + static_cast<size_t>(p + halo) * planeSize;
}

// Compute the chemical potential for a single plane.
// cm, c0, cp: concentration on the previous, current, and next z-plane.
static inline void computeChemicalPotentialPlane(const double* __restrict cm, const double* __restrict c0,
                                                 const double* __restrict cp, double* __restrict mu,
                                                 const size_t nx, const size_t ny,
                                                 const double dx, const double dy, const double dz,
                                                 const double gamma, const double e_AA, const double e_BB,
                                                 const double e_AB) {
    for (size_t y = 0; y < ny; ++y) {
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;
        const double* __restrict row = c0 + y * nx;
        const double* __restrict rowp = c0 + yp * nx;
        const double* __restrict rown = c0 + yn * nx;
        const double* __restrict rowzp = cp + y * nx;
        const double* __restrict rowzn = cm + y * nx;
        double* __restrict murow = mu + y * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t xp = (x < nx - 1) ? x + 1 : x;
            const size_t xn = (x > 0) ? x - 1 : 0;
            const double cv = row[x];
            const double cxx = (row[xp] + row[xn] - 2.0 * cv) / (dx * dx);
            const double cyy = (rowp[x] + rown[x] - 2.0 * cv) / (dy * dy);
            const double czz = (rowzp[x] + rowzn[x] - 2.0 * cv) / (dz * dz);
            const double lap = cxx + cyy + czz;
            murow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * lap;
        }
    }
}

// Cahn-Hilliard update for a single plane.
static inline void cahnHilliardUpdatePlane(const double* __restrict mum, const double* __restrict mu0,
                                           const double* __restrict mup, const double* __restrict cold,
                                           double* __restrict cnew, const size_t nx, const size_t ny,
                                           const double D, const double dt,
                                           const double dx, const double dy, const double dz) {
    for (size_t y = 0; y < ny; ++y) {
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;
        const double* __restrict row = mu0 + y * nx;
        const double* __restrict rowp = mu0 + yp * nx;
        const double* __restrict rown = mu0 + yn * nx;
        const double* __restrict rowzp = mup + y * nx;
        const double* __restrict rowzn = mum + y * nx;
        const double* __restrict crow = cold + y * nx;
        double* __restrict nrow = cnew + y * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t xp = (x < nx - 1) ? x + 1 : x;
            const size_t xn = (x > 0) ? x - 1 : 0;
            const double mv = row[x];
            const double mxx = (row[xp] + row[xn] - 2.0 * mv) / (dx * dx);
            const double myy = (rowp[x] + rown[x] - 2.0 * mv) / (dy * dy);
            const double mzz = (rowzp[x] + rowzn[x] - 2.0 * mv) / (dz * dz);
            nrow[x] = crow[x] + dt * D * (mxx + myy + mzz);
        }
    }
}

// Initialize the local slab of the concentration field (global planes
// [z0, z0 + nz_loc)), using the same pseudo-random generator as the original.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nz_loc, const long halo) {
    const size_t vol = nx * ny * nz;
    const size_t planeSize = nx * ny;

    for (size_t zl = 0; zl < nz_loc; ++zl) {
        const size_t z = z0 + zl;
        double* plane = planePtr(c, static_cast<long>(zl), halo, planeSize);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                plane[y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
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

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank receives the same argv)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
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
    const size_t planeSize = nx * ny;

    // ---- Domain decomposition along z --------------------------------------
    // Every active rank must own at least two planes so that the two-plane
    // halo it sends consists purely of its own data.  Ranks beyond `nactive`
    // own no planes and only take part in collectives.
    const int nactive = static_cast<int>(std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(nprocs), nz / 2)));
    size_t z0 = 0, nz_loc = 0;
    if (rank < nactive) {
        const size_t base = nz / nactive;
        const size_t rem = nz % nactive;
        const size_t r = static_cast<size_t>(rank);
        nz_loc = base + (r < rem ? 1 : 0);
        z0 = r * base + std::min(r, rem);
    }
    const bool active = nz_loc > 0;
    const int prev = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (active && rank + 1 < nactive) ? rank + 1 : MPI_PROC_NULL;
    const bool atLow = active && prev == MPI_PROC_NULL;    // owns global plane 0
    const bool atHigh = active && next == MPI_PROC_NULL;   // owns global plane nz-1

    constexpr long CH = 2;  // halo planes for c
    constexpr long MH = 1;  // halo planes for mu
    const long nzl = static_cast<long>(nz_loc);

    // Allocate arrays (local slab + halos)
    std::vector<double> cold(active ? (nz_loc + 2 * CH) * planeSize : 0);
    std::vector<double> cnew(active ? (nz_loc + 2 * CH) * planeSize : 0);
    std::vector<double> mu(active ? (nz_loc + 2 * MH) * planeSize : 0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    if (active) initializeConcentration(cold, nx, ny, nz, z0, nz_loc, CH);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    const int planeCount = static_cast<int>(planeSize);
    const int twoPlanes = 2 * planeCount;

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            // ---- Start halo exchange of c (two planes each direction) ------
            MPI_Request reqs[4];
            // Receive planes -2,-1 from prev; planes nz_loc, nz_loc+1 from next.
            MPI_Irecv(planePtr(cold, -2, CH, planeSize), twoPlanes, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(planePtr(cold, nzl, CH, planeSize), twoPlanes, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[1]);
            // Send planes 0,1 to prev; planes nz_loc-2, nz_loc-1 to next.
            MPI_Isend(planePtr(cold, 0, CH, planeSize), twoPlanes, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Isend(planePtr(cold, nzl - 2, CH, planeSize), twoPlanes, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[3]);

            // ---- Chemical potential on interior planes [1, nz_loc-2] -------
            // These need only c planes [0, nz_loc-1], which are local.
            for (long p = 1; p + 1 < nzl; ++p) {
                computeChemicalPotentialPlane(planePtr(cold, p - 1, CH, planeSize), planePtr(cold, p, CH, planeSize),
                                              planePtr(cold, p + 1, CH, planeSize), planePtr(mu, p, MH, planeSize),
                                              nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            // Clamped global boundaries: ghost c plane is a copy of the edge plane.
            if (atLow) std::memcpy(planePtr(cold, -1, CH, planeSize), planePtr(cold, 0, CH, planeSize), planeSize * sizeof(double));
            if (atHigh) std::memcpy(planePtr(cold, nzl, CH, planeSize), planePtr(cold, nzl - 1, CH, planeSize), planeSize * sizeof(double));

            // ---- Chemical potential on boundary planes ----------------------
            // Local edge planes 0 and nz_loc-1 (nz_loc >= 2 for active ranks,
            // except when the whole grid has a single plane).
            computeChemicalPotentialPlane(planePtr(cold, -1, CH, planeSize), planePtr(cold, 0, CH, planeSize),
                                          planePtr(cold, 1, CH, planeSize), planePtr(mu, 0, MH, planeSize),
                                          nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (nzl > 1) {
                computeChemicalPotentialPlane(planePtr(cold, nzl - 2, CH, planeSize), planePtr(cold, nzl - 1, CH, planeSize),
                                              planePtr(cold, nzl, CH, planeSize), planePtr(mu, nzl - 1, MH, planeSize),
                                              nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
            // Ghost mu planes: computed locally from the two-plane c halo for
            // interior ranks (bit-identical to the neighbour's own value),
            // or a copy of the edge plane at the clamped global boundary.
            if (atLow) {
                std::memcpy(planePtr(mu, -1, MH, planeSize), planePtr(mu, 0, MH, planeSize), planeSize * sizeof(double));
            } else {
                computeChemicalPotentialPlane(planePtr(cold, -2, CH, planeSize), planePtr(cold, -1, CH, planeSize),
                                              planePtr(cold, 0, CH, planeSize), planePtr(mu, -1, MH, planeSize),
                                              nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
            if (atHigh) {
                std::memcpy(planePtr(mu, nzl, MH, planeSize), planePtr(mu, nzl - 1, MH, planeSize), planeSize * sizeof(double));
            } else {
                computeChemicalPotentialPlane(planePtr(cold, nzl - 1, CH, planeSize), planePtr(cold, nzl, CH, planeSize),
                                              planePtr(cold, nzl + 1, CH, planeSize), planePtr(mu, nzl, MH, planeSize),
                                              nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // ---- Update concentration on all owned planes ------------------
            for (long p = 0; p < nzl; ++p) {
                cahnHilliardUpdatePlane(planePtr(mu, p - 1, MH, planeSize), planePtr(mu, p, MH, planeSize),
                                        planePtr(mu, p + 1, MH, planeSize), planePtr(cold, p, CH, planeSize),
                                        planePtr(cnew, p, CH, planeSize), nx, ny, D, dt, dx, dy, dz);
            }
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    double tLocal = MPI_Wtime() - tStart;
    double tMax = tLocal;
    MPI_Reduce(&tLocal, &tMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(tMax * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 only when it is needed for output.
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<int> counts, displs;
        if (rank == 0) {
            counts.resize(nprocs);
            displs.resize(nprocs);
            for (int r = 0; r < nprocs; ++r) {
                size_t cnt = 0, off = 0;
                if (r < nactive) {
                    const size_t base = nz / nactive;
                    const size_t rem = nz % nactive;
                    const size_t rr = static_cast<size_t>(r);
                    cnt = (base + (rr < rem ? 1 : 0)) * planeSize;
                    off = (rr * base + std::min(rr, rem)) * planeSize;
                }
                counts[r] = static_cast<int>(cnt);
                displs[r] = static_cast<int>(off);
            }
        }
        std::vector<double> full(rank == 0 ? gridSize : 0);
        MPI_Gatherv(active ? planePtr(cold, 0, CH, planeSize) : nullptr, static_cast<int>(nz_loc * planeSize), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
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
