#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for ghost-layered arrays.
// z_ghost = 0 is lower ghost, z_ghost = nz_local+1 is upper ghost,
// z_ghost = 1..nz_local is the local domain.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z_ghost,
                             const size_t nx, const size_t ny) noexcept {
    return z_ghost * (nx * ny) + y * nx + x;
}

// Compute Laplacian with ghost-cell boundary handling.
// Ghost cells for the Z direction are filled by exchangeGhostCells before calling,
// so no special boundary logic is needed in Z.
static double computeLaplacian(const std::vector<double>& c,
                               const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y,
                               const size_t z_ghost) {
    // Clamped boundary in X and Y (unchanged from serial version)
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    // Z direction uses ghost cells (already filled by exchange or clamping)
    const size_t zp = z_ghost + 1;
    const size_t zn = z_ghost - 1;

    const double cxx = (c[idx3(xp, y, z_ghost, nx, ny)] +
                        c[idx3(xn, y, z_ghost, nx, ny)] -
                        2.0 * c[idx3(x, y, z_ghost, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z_ghost, nx, ny)] +
                        c[idx3(x, yn, z_ghost, nx, ny)] -
                        2.0 * c[idx3(x, y, z_ghost, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] +
                        c[idx3(x, y, zn, nx, ny)] -
                        2.0 * c[idx3(x, y, z_ghost, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Exchange Z-direction ghost cells with neighboring MPI ranks.
// After this call, c[0..slice_size-1] = lower ghost (clamped or from rank-1)
// and c[(nz_local+1)*slice_size .. (nz_local+2)*slice_size-1] = upper ghost.
static void exchangeGhostCells(std::vector<double>& c,
                               const size_t nx, const size_t ny,
                               const size_t nz_local,
                               const int rank, const int num_procs) {
    const size_t slice_size = nx * ny;

    if (num_procs == 1) {
        // With a single rank, clamp both boundaries
        std::copy(c.begin() + 1 * slice_size,
                  c.begin() + 2 * slice_size,
                  c.begin() + 0 * slice_size);
        std::copy(c.begin() + nz_local * slice_size,
                  c.begin() + (nz_local + 1) * slice_size,
                  c.begin() + (nz_local + 1) * slice_size);
        return;
    }

    MPI_Status status;
    const int tag_lo = 0;  // tag for "sending first slice downward"
    const int tag_hi = 1;  // tag for "sending last slice upward"

    // Exchange lower ghost: receive from rank-1, send first local slice to rank-1
    if (rank > 0) {
        MPI_Sendrecv(c.data() + 1 * slice_size, slice_size, MPI_DOUBLE, rank - 1, tag_lo,
                     c.data() + 0 * slice_size, slice_size, MPI_DOUBLE, rank - 1, tag_hi,
                     MPI_COMM_WORLD, &status);
    } else {
        // Global lower boundary: clamp (ghost = first local slice)
        std::copy(c.begin() + 1 * slice_size,
                  c.begin() + 2 * slice_size,
                  c.begin() + 0 * slice_size);
    }

    // Exchange upper ghost: receive from rank+1, send last local slice to rank+1
    if (rank < num_procs - 1) {
        MPI_Sendrecv(c.data() + nz_local * slice_size, slice_size, MPI_DOUBLE, rank + 1, tag_hi,
                     c.data() + (nz_local + 1) * slice_size, slice_size, MPI_DOUBLE, rank + 1, tag_lo,
                     MPI_COMM_WORLD, &status);
    } else {
        // Global upper boundary: clamp (ghost = last local slice)
        std::copy(c.begin() + nz_local * slice_size,
                  c.begin() + (nz_local + 1) * slice_size,
                  c.begin() + (nz_local + 1) * slice_size);
    }
}

// Compute chemical potential (local domain only, uses ghost cells for c)
static void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                                     const size_t nx, const size_t ny, const size_t nz_local,
                                     const double dx, const double dy, const double dz,
                                     const double gamma,
                                     const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z_local = 0; z_local < nz_local; ++z_local) {
        const size_t z_ghost = z_local + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_ghost, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z_ghost);
            }
        }
    }
}

// Cahn-Hilliard update step (local domain only, uses ghost cells for mu)
static void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                               const std::vector<double>& mu,
                               const size_t nx, const size_t ny, const size_t nz_local,
                               const double D, const double dt,
                               const double dx, const double dy, const double dz) {
    for (size_t z_local = 0; z_local < nz_local; ++z_local) {
        const size_t z_ghost = z_local + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_ghost, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                            computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z_ghost);
            }
        }
    }
}

// Initialize concentration field (local portion only, uses global coordinates
// to produce the same pseudo-random sequence as the serial version)
static void initializeConcentration(std::vector<double>& c,
                                    const size_t nx, const size_t ny,
                                    const size_t nz_local,
                                    const size_t z_start,
                                    const size_t global_vol) {
    for (size_t z_local = 0; z_local < nz_local; ++z_local) {
        const size_t z_ghost = z_local + 1;
        const size_t global_z = z_start + z_local;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_ghost, nx, ny);
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo =
                    ((((linear_id + 1) * 1299709) % global_vol) /
                     static_cast<double>(global_vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

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

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse the same args)
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

    // Domain decomposition along Z (1D slab decomposition)
    const size_t nz_per_rank = nz / num_procs;
    const size_t nz_rem = nz % num_procs;
    const size_t nz_local = nz_per_rank + (static_cast<size_t>(rank) < nz_rem ? 1 : 0);

    // Compute global Z start for this rank
    size_t z_start = 0;
    for (int r = 0; r < rank; ++r) {
        z_start += nz_per_rank + (static_cast<size_t>(r) < nz_rem ? 1 : 0);
    }

    size_t gridSize = nx * ny * nz;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI processes: %d\n", num_procs);
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

    // Allocate arrays with ghost cells (one extra slice at each Z boundary)
    const size_t ghost_size = (nz_local + 2) * nx * ny;
    std::vector<double> cold(ghost_size);
    std::vector<double> cnew(ghost_size);
    std::vector<double> mu(ghost_size);

    // Initialize local portion of concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz_local, z_start, gridSize);

    // Initial ghost cell exchange for cold
    exchangeGhostCells(cold, nx, ny, nz_local, rank, num_procs);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // 1. Compute chemical potential (needs cold ghost cells, which are valid)
        computeChemicalPotential(cold, mu, nx, ny, nz_local, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // 2. Exchange ghost cells for mu (needed by cahnHilliardUpdate)
        exchangeGhostCells(mu, nx, ny, nz_local, rank, num_procs);

        // 3. Update concentration (uses mu ghost cells)
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local, D, dt, dx, dy, dz);

        // 4. Swap buffers: cold now holds the newly computed values
        std::swap(cold, cnew);

        // 5. Exchange ghost cells for cold (needed in next iteration)
        if (t < iterations - 1) {
            exchangeGhostCells(cold, nx, ny, nz_local, rank, num_procs);
        }
    }

    double end = MPI_Wtime();
    double local_duration = (end - start) * 1000.0;  // milliseconds

    // Compute max time across all ranks
    double max_duration = 0.0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_duration);

        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / (max_duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the local portions to rank 0 for post-processing
    if (printResults || validate) {
        // Pack local data excluding ghost cells
        const size_t local_elements = nz_local * nx * ny;
        std::vector<double> local_cold(local_elements);
        for (size_t z = 0; z < nz_local; ++z) {
            std::copy(cold.begin() + (z + 1) * nx * ny,
                      cold.begin() + (z + 2) * nx * ny,
                      local_cold.begin() + z * nx * ny);
        }

        // Gather sizes from all ranks
        const int local_size_int = static_cast<int>(local_elements);
        std::vector<int> recv_counts(num_procs);
        std::vector<int> recv_offsets(num_procs);
        MPI_Gather(&local_size_int, 1, MPI_INT,
                   recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> global_cold;
        if (rank == 0) {
            global_cold.resize(gridSize);
            recv_offsets[0] = 0;
            for (int i = 1; i < num_procs; ++i) {
                recv_offsets[i] = recv_offsets[i - 1] + recv_counts[i - 1];
            }
        }

        MPI_Gatherv(local_cold.data(), local_size_int, MPI_DOUBLE,
                    rank == 0 ? global_cold.data() : nullptr,
                    recv_counts.data(), recv_offsets.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // All ranks must participate in the gather, but only rank 0 performs output
        if (printResults && rank == 0) {
            print_results(global_cold, "Concentration");
        }

        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
                bool valid = validateResult(global_cold, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
                MPI_Finalize();
                return valid ? 0 : 1;
            }
            // Non-root ranks: after gather is done, just clean up
            MPI_Finalize();
            return 0;
        }
    }

    MPI_Finalize();
    return 0;
}
