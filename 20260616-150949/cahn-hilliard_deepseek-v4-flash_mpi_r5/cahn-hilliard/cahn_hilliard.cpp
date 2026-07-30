#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for local grid (includes ghost cells in z)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ------------------------------------------------------------------------
// Compute Laplacian with clamped boundary conditions in x and y.
// Ghost cells in z handle z-boundary conditions via MPI exchange.
// 'z' is the local z-index in the ghost-extended grid.
// This function is only called for interior cells (z in [1, nz_local]),
// so z-1 and z+1 always index valid ghost-or-interior memory.
// ------------------------------------------------------------------------
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    // z-direction: ghosts are always valid (pre-filled by exchange or clamping)
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)]
                       - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)]
                       - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)]
                       - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// ------------------------------------------------------------------------
// Exchange ghost cells in the z-direction for a 3D field.
// Local grid layout: [0] = bottom ghost, [1..nz_local] = interior,
//                    [nz_local+1] = top ghost.
// At physical boundaries (rank 0 / rank nprocs-1) the ghost is set to the
// boundary value, implementing zero-gradient (clamped) conditions.
// ------------------------------------------------------------------------
void exchangeGhostsZ(std::vector<double>& data, size_t nx, size_t ny,
                     size_t nz_local, int rank, int nprocs) {
    const size_t layer_size = nx * ny;

    // Bottom ghost (local_z = 0)
    if (rank > 0) {
        MPI_Sendrecv(&data[idx3(0, 0, 1, nx, ny)],       layer_size, MPI_DOUBLE, rank - 1, 0,
                     &data[idx3(0, 0, 0, nx, ny)],       layer_size, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Physical bottom: zero-gradient (copy interior[1] into ghost[0])
        std::copy_n(&data[idx3(0, 0, 1, nx, ny)], layer_size,
                    &data[idx3(0, 0, 0, nx, ny)]);
    }

    // Top ghost (local_z = nz_local + 1)
    if (rank < nprocs - 1) {
        MPI_Sendrecv(&data[idx3(0, 0, nz_local, nx, ny)], layer_size, MPI_DOUBLE, rank + 1, 1,
                     &data[idx3(0, 0, nz_local + 1, nx, ny)], layer_size, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Physical top: zero-gradient (copy interior[nz_local] into ghost[nz_local+1])
        std::copy_n(&data[idx3(0, 0, nz_local, nx, ny)], layer_size,
                    &data[idx3(0, 0, nz_local + 1, nx, ny)]);
    }
}

// ------------------------------------------------------------------------
// Compute chemical potential on the local subdomain.
// ------------------------------------------------------------------------
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                             const size_t nx, const size_t ny, const size_t nz_local,
                             const double dx, const double dy, const double dz,
                             const double gamma, const double e_AA,
                             const double e_BB, const double e_AB) {
    for (size_t local_z = 1; local_z <= nz_local; ++local_z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, local_z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, local_z);
            }
        }
    }
}

// ------------------------------------------------------------------------
// Cahn-Hilliard update step on the local subdomain.
// ------------------------------------------------------------------------
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt,
                        const double dx, const double dy, const double dz) {
    for (size_t local_z = 1; local_z <= nz_local; ++local_z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, local_z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, local_z);
            }
        }
    }
}

// ------------------------------------------------------------------------
// Initialize local portion of concentration field.
// Uses the same deterministic pseudo-random formula as the serial code
// but only for the local range of global z-planes.
// ------------------------------------------------------------------------
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz_local, const size_t z_start,
                             const size_t vol) {
    for (size_t local_z = 1; local_z <= nz_local; ++local_z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx      = idx3(x, y, local_z, nx, ny);
                const size_t global_z = z_start + local_z - 1;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo =
                    ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// ------------------------------------------------------------------------
// Distributed validation using MPI reductions.
// Only rank 0 prints diagnostic messages, but all ranks return the same
// boolean result.
// ------------------------------------------------------------------------
bool validateResult(const std::vector<double>& c, size_t nx, size_t ny,
                    size_t nz_local, int rank) {
    int local_has_nan = 0;
    double local_min = c[idx3(0, 0, 1, nx, ny)];
    double local_max = local_min;

    for (size_t local_z = 1; local_z <= nz_local; ++local_z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, local_z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_has_nan = 1;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    int global_has_nan;
    double global_min, global_max;
    MPI_Allreduce(&local_has_nan, &global_has_nan, 1, MPI_INT,    MPI_LOR,  MPI_COMM_WORLD);
    MPI_Allreduce(&local_min,     &global_min,     1, MPI_DOUBLE, MPI_MIN,  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max,     &global_max,     1, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_has_nan) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    // All ranks agree because Allreduce results are identical
    return (global_has_nan == 0 && global_max <= 10.0 && global_min >= -10.0);
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // ---- default parameters ------------------------------------------------
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // ---- parse arguments (all ranks parse identically) ---------------------
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
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

    // ---- domain decomposition (1D along Z) ---------------------------------
    if (static_cast<size_t>(nprocs) > nz) {
        if (rank == 0) {
            printf("Error: number of MPI processes (%d) exceeds "
                   "Z dimension (%zu)\n", nprocs, nz);
        }
        MPI_Finalize();
        return 1;
    }

    size_t nz_base  = nz / static_cast<size_t>(nprocs);
    size_t nz_rem   = nz % static_cast<size_t>(nprocs);
    size_t z_start;
    size_t nz_local;

    if (static_cast<size_t>(rank) < nz_rem) {
        nz_local = nz_base + 1;
        z_start  = static_cast<size_t>(rank) * nz_local;
    } else {
        nz_local = nz_base;
        z_start  = nz_rem * (nz_base + 1)
                 + (static_cast<size_t>(rank) - nz_rem) * nz_base;
    }

    const size_t vol            = nx * ny * nz;
    const size_t local_grid_sz  = nx * ny * (nz_local + 2);   // +2 for ghost layers

    // ---- physical parameters -----------------------------------------------
    const double dx    = 1.0;
    const double dy    = 1.0;
    const double dz    = 1.0;
    const double dt    = 0.01;
    const double e_AA  = -(2.0 / 9.0);
    const double e_BB  = -(2.0 / 9.0);
    const double e_AB  =  (2.0 / 9.0);
    const double gamma =  0.5;
    const double D     =  1.0;

    // ---- allocate local arrays (with ghost cells) --------------------------
    std::vector<double> cold(local_grid_sz);
    std::vector<double> cnew(local_grid_sz);
    std::vector<double> mu(local_grid_sz);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d\n", nprocs);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
        fflush(stdout);
    }

    initializeConcentration(cold, nx, ny, nz_local, z_start, vol);

    // Fill initial ghost cells for cold
    exchangeGhostsZ(cold, nx, ny, nz_local, rank, nprocs);

    // ---- main simulation loop ----------------------------------------------
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // 1. Chemical potential (needs ghosts of cold)
        computeChemicalPotential(cold, mu, nx, ny, nz_local,
                                 dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // 2. Exchange ghosts for mu
        exchangeGhostsZ(mu, nx, ny, nz_local, rank, nprocs);

        // 3. Update concentration (needs ghosts of mu)
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local,
                           D, dt, dx, dy, dz);

        // 4. Swap buffers
        std::swap(cold, cnew);

        // 5. Exchange ghosts for cold for next iteration (skip last)
        if (t < iterations - 1) {
            exchangeGhostsZ(cold, nx, ny, nz_local, rank, nprocs);
        }
    }

    const double t_end        = MPI_Wtime();
    const double elapsed_ms   = (t_end - t_start) * 1000.0;

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", elapsed_ms);

        const double cell_updates = static_cast<double>(vol) * iterations;
        const double mcups = cell_updates / (t_end - t_start) / 1.0e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- print results (gather to rank 0) ----------------------------------
    if (printResults) {
        // Each rank contributes its interior (nz_local * nx * ny) values.
        // Since the decomposition is 1D along Z, the interior slab for each
        // rank is contiguous in memory, so we can gather them directly.
        std::vector<int> recv_counts(nprocs);
        std::vector<int> displs(nprocs);

        const int local_count = static_cast<int>(nx * ny * nz_local);
        MPI_Gather(&local_count, 1, MPI_INT,
                   recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i) {
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            }
        }

        std::vector<double> full_cold;
        if (rank == 0) full_cold.resize(vol);

        MPI_Gatherv(&cold[idx3(0, 0, 1, nx, ny)], local_count, MPI_DOUBLE,
                    full_cold.data(), recv_counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(full_cold, "Concentration");
        }
    }

    // ---- validation --------------------------------------------------------
    if (validate) {
        if (rank == 0) printf("Validating result...\n");

        const bool valid = validateResult(cold, nx, ny, nz_local, rank);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
