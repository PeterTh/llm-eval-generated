#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory layout
// -------------------------
// The global grid (nx x ny x nz, z-major) is split into contiguous slabs of z-planes,
// one (or zero) slab per MPI rank. Each rank stores its slab in a padded array with
// one ghost plane below and one above:
//
//   plane 0          : ghost (copy of lower neighbour's top plane, or clamped copy)
//   planes 1..nzl    : owned planes
//   plane nzl+1      : ghost (copy of upper neighbour's bottom plane, or clamped copy)
//
// Clamped boundary conditions in z are reproduced exactly by filling the physical
// boundary ghost plane with a copy of the adjacent owned plane, so the stencil code
// never needs special cases in z.
//
// Per time step:
//   1. start halo exchange of c, compute mu on interior planes (overlaps comm)
//   2. wait for c halo, compute mu on boundary planes, start halo exchange of mu
//   3. compute cnew on interior planes (overlaps comm)
//   4. wait for mu halo, compute cnew on boundary planes, swap buffers
//
// Floating-point expressions are kept identical to the serial formulation so results
// are bit-for-bit reproducible independently of the number of ranks.

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

namespace {

constexpr int TAG_C_UP = 10;   // c message moving towards higher rank
constexpr int TAG_C_DN = 11;   // c message moving towards lower rank
constexpr int TAG_M_UP = 20;   // mu message moving towards higher rank
constexpr int TAG_M_DN = 21;   // mu message moving towards lower rank

struct Slab {
    size_t z0;   // first global z-plane owned by this rank
    size_t nzl;  // number of owned z-planes (0 for idle ranks)
    int up;      // neighbour with higher z, or MPI_PROC_NULL
    int down;    // neighbour with lower z, or MPI_PROC_NULL
};

// Block decomposition of nz planes over `active` ranks (active = min(size, nz)).
Slab makeSlab(int rank, int size, size_t nz) {
    const size_t active = std::min(static_cast<size_t>(size), nz);
    const size_t r = static_cast<size_t>(rank);
    Slab s{0, 0, MPI_PROC_NULL, MPI_PROC_NULL};
    if (r < active) {
        const size_t base = nz / active;
        const size_t rem = nz % active;
        s.nzl = base + (r < rem ? 1 : 0);
        s.z0 = r * base + std::min(r, rem);
        s.up = (r + 1 < active) ? rank + 1 : MPI_PROC_NULL;
        s.down = (r > 0) ? rank - 1 : MPI_PROC_NULL;
    }
    return s;
}

// Apply f(local_index, laplacian) to every point of one nx*ny plane.
// cur  : plane being evaluated (x/y clamped at edges as in the serial code)
// dn   : plane below (z-ghost, already clamped at physical boundaries)
// up   : plane above (z-ghost, already clamped at physical boundaries)
template <typename F>
inline void forEachLaplacianInPlane(const double* cur, const double* dn, const double* up,
                                    const size_t nx, const size_t ny,
                                    const double dx, const double dy, const double dz, F&& f) {
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    for (size_t y = 0; y < ny; ++y) {
        const size_t yn = (y > 0) ? y - 1 : 0;
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t base = y * nx;
        const double* row = cur + base;
        const double* rowN = cur + yn * nx;
        const double* rowP = cur + yp * nx;
        const double* dnRow = dn + base;
        const double* upRow = up + base;

        auto point = [&](const size_t x, const size_t xn, const size_t xp) {
            const double c0 = row[x];
            const double cxx = (row[xp] + row[xn] - 2.0 * c0) / dx2;
            const double cyy = (rowP[x] + rowN[x] - 2.0 * c0) / dy2;
            const double czz = (upRow[x] + dnRow[x] - 2.0 * c0) / dz2;
            f(base + x, cxx + cyy + czz);
        };

        if (nx == 1) {
            point(0, 0, 0);
            continue;
        }
        // Edge points use clamped x-neighbours; interior uses unit-stride offsets.
        point(0, 0, 1);
        for (size_t x = 1; x + 1 < nx; ++x) {
            point(x, x - 1, x + 1);
        }
        point(nx - 1, nx - 2, nx - 1);
    }
}

// Chemical potential for one padded plane (cp/mup point at the plane's first element).
void computeChemicalPotentialPlane(const double* cp, double* mup,
                                   const size_t nx, const size_t ny,
                                   const double dx, const double dy, const double dz,
                                   const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nxy = nx * ny;
    forEachLaplacianInPlane(cp, cp - nxy, cp + nxy, nx, ny, dx, dy, dz,
        [&](const size_t i, const double lap) {
            const double cv = cp[i];
            mup[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                     + 3.0 * cv + cv * cv * cv
                     - gamma * lap;
        });
}

// Cahn-Hilliard update for one padded plane.
void cahnHilliardUpdatePlane(const double* coldp, double* cnewp, const double* mup,
                             const size_t nx, const size_t ny,
                             const double D, const double dt,
                             const double dx, const double dy, const double dz) {
    const size_t nxy = nx * ny;
    forEachLaplacianInPlane(mup, mup - nxy, mup + nxy, nx, ny, dx, dy, dz,
        [&](const size_t i, const double lap) {
            cnewp[i] = coldp[i] + dt * D * lap;
        });
}

// Non-blocking exchange of the boundary planes of a padded slab array.
void startHaloExchange(double* a, const size_t nxy, const size_t nzl, const int up, const int down,
                       const int tagUp, const int tagDn, MPI_Comm comm, MPI_Request req[4]) {
    MPI_Irecv(a, static_cast<int>(nxy), MPI_DOUBLE, down, tagUp, comm, &req[0]);
    MPI_Irecv(a + (nzl + 1) * nxy, static_cast<int>(nxy), MPI_DOUBLE, up, tagDn, comm, &req[1]);
    MPI_Isend(a + nzl * nxy, static_cast<int>(nxy), MPI_DOUBLE, up, tagUp, comm, &req[2]);
    MPI_Isend(a + nxy, static_cast<int>(nxy), MPI_DOUBLE, down, tagDn, comm, &req[3]);
}

// Fill ghost planes at physical (non-neighbour) domain boundaries with clamped copies.
void fillPhysicalGhosts(double* a, const size_t nxy, const size_t nzl, const int up, const int down) {
    if (down == MPI_PROC_NULL) {
        std::copy_n(a + nxy, nxy, a);
    }
    if (up == MPI_PROC_NULL) {
        std::copy_n(a + nzl * nxy, nxy, a + (nzl + 1) * nxy);
    }
}

// Initialize owned planes of the padded array with the global pseudo-random field.
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzl) {
    const size_t nxy = nx * ny;
    const size_t vol = nx * ny * nz;
    for (size_t k = 1; k <= nzl; ++k) {
        const size_t z = z0 + k - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1] from the global linear index
                const size_t linear_id = z * nxy + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[k * nxy + y * nx + x] = -1.0 + 2.0 * pseudo;
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const bool root = (rank == 0);

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

    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) printf("Error: grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
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

    const size_t nxy = nx * ny;
    const size_t gridSize = nxy * nz;
    const Slab slab = makeSlab(rank, size, nz);
    const size_t nzl = slab.nzl;
    const size_t paddedSize = (nzl + 2) * nxy;

    // Local padded arrays (owned planes plus one ghost plane on each side)
    std::vector<double> cold(paddedSize, 0.0);
    std::vector<double> cnew(paddedSize, 0.0);
    std::vector<double> mu(paddedSize, 0.0);

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), nx, ny, nz, slab.z0, nzl);

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double tStart = MPI_Wtime();

    if (nzl > 0) {
        MPI_Request reqC[4];
        MPI_Request reqM[4];
        double* cold_p = cold.data();
        double* cnew_p = cnew.data();
        double* mu_p = mu.data();

        for (int t = 0; t < iterations; ++t) {
            // (1) Start halo exchange of c; compute mu on interior planes meanwhile.
            startHaloExchange(cold_p, nxy, nzl, slab.up, slab.down, TAG_C_UP, TAG_C_DN, comm, reqC);
            fillPhysicalGhosts(cold_p, nxy, nzl, slab.up, slab.down);
            for (size_t k = 2; k < nzl; ++k) {
                computeChemicalPotentialPlane(cold_p + k * nxy, mu_p + k * nxy, nx, ny,
                                              dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // (2) Boundary mu planes need c ghosts; then exchange mu ghosts.
            MPI_Waitall(4, reqC, MPI_STATUSES_IGNORE);
            computeChemicalPotentialPlane(cold_p + nxy, mu_p + nxy, nx, ny,
                                          dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (nzl > 1) {
                computeChemicalPotentialPlane(cold_p + nzl * nxy, mu_p + nzl * nxy, nx, ny,
                                              dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
            fillPhysicalGhosts(mu_p, nxy, nzl, slab.up, slab.down);
            startHaloExchange(mu_p, nxy, nzl, slab.up, slab.down, TAG_M_UP, TAG_M_DN, comm, reqM);

            // (3) Update interior planes (no mu ghosts needed); overlaps mu exchange.
            for (size_t k = 2; k < nzl; ++k) {
                cahnHilliardUpdatePlane(cold_p + k * nxy, cnew_p + k * nxy, mu_p + k * nxy,
                                        nx, ny, D, dt, dx, dy, dz);
            }

            // (4) Wait for mu ghosts; update boundary planes.
            MPI_Waitall(4, reqM, MPI_STATUSES_IGNORE);
            cahnHilliardUpdatePlane(cold_p + nxy, cnew_p + nxy, mu_p + nxy,
                                    nx, ny, D, dt, dx, dy, dz);
            if (nzl > 1) {
                cahnHilliardUpdatePlane(cold_p + nzl * nxy, cnew_p + nzl * nxy, mu_p + nzl * nxy,
                                        nx, ny, D, dt, dx, dy, dz);
            }

            // Swap buffers
            std::swap(cold, cnew);
            cold_p = cold.data();
            cnew_p = cnew.data();
        }
    }

    const double tLocal = MPI_Wtime() - tStart;
    double tMax = 0.0;
    MPI_Reduce(&tLocal, &tMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (root) {
        const long durationMs = static_cast<long>(tMax * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int rc = 0;
    if (printResults || validate) {
        // Gather the full field (z-major, identical ordering to the serial code) on root.
        std::vector<double> full;
        std::vector<int> counts(size, 0);
        std::vector<int> displs(size, 0);
        for (int r = 0; r < size; ++r) {
            const Slab sr = makeSlab(r, size, nz);
            counts[r] = static_cast<int>(sr.nzl * nxy);
            displs[r] = static_cast<int>(sr.z0 * nxy);
        }
        if (root) full.resize(gridSize);
        MPI_Gatherv(cold.data() + nxy, static_cast<int>(nzl * nxy), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(full, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }
    }

    // Make the exit status consistent across all ranks.
    MPI_Bcast(&rc, 1, MPI_INT, 0, comm);
    MPI_Finalize();
    return rc;
}
