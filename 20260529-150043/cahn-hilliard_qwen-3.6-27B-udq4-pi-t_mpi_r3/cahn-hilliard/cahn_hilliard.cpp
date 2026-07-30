#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Local index helpers  –  z runs from 0 .. nz_local+1  (includes ghosts)
// Interior cells: z = 1 .. nz_local
// Ghost bottom: z = 0
// Ghost top: z = nz_local + 1
// ---------------------------------------------------------------------------
inline constexpr size_t lidx3(size_t x, size_t y, size_t z,
                              size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Compute Laplacian with clamped boundary conditions
// ---------------------------------------------------------------------------
double computeLaplacian(const std::vector<double>& c,
                        size_t nx, size_t ny, size_t nz,
                        double dx, double dy, double dz,
                        size_t x, size_t y, size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[lidx3(xp, y, z, nx, ny)] +
                        c[lidx3(xn, y, z, nx, ny)] -
                        2.0 * c[lidx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[lidx3(x, yp, z, nx, ny)] +
                        c[lidx3(x, yn, z, nx, ny)] -
                        2.0 * c[lidx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[lidx3(x, y, zp, nx, ny)] +
                        c[lidx3(x, y, zn, nx, ny)] -
                        2.0 * c[lidx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// ---------------------------------------------------------------------------
// Compute chemical potential over interior cells only
// ---------------------------------------------------------------------------
void computeChemicalPotential(const std::vector<double>& c,
                              std::vector<double>& mu,
                              size_t nx, size_t ny, size_t nz_local,
                              size_t nz_total,
                              double dx, double dy, double dz,
                              double gamma, double e_AA, double e_BB,
                              double e_AB) {
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lidx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                                  2.0 * cv * e_AB) +
                          3.0 * cv + cv * cv * cv -
                          gamma * computeLaplacian(c, nx, ny, nz_total, dx, dy,
                                                   dz, x, y, z);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cahn-Hilliard update step over interior cells only
// ---------------------------------------------------------------------------
void cahnHilliardUpdate(std::vector<double>& cnew,
                        const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        size_t nx, size_t ny, size_t nz_local,
                        size_t nz_total,
                        double D, double dt, double dx, double dy, double dz) {
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lidx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] +
                            dt * D * computeLaplacian(mu, nx, ny, nz_total,
                                                      dx, dy, dz, x, y, z);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Initialize interior concentration field (ghosts stay 0)
// Uses global (x,y,z) for deterministic pseudo-random init.
// ---------------------------------------------------------------------------
void initializeConcentration(std::vector<double>& c,
                             size_t nx, size_t ny, size_t nz_global,
                             size_t nz_local, size_t z_start_global) {
    const size_t vol = nx * ny * nz_global;

    for (size_t z_local = 1; z_local <= nz_local; ++z_local) {
        const size_t z_global = z_start_global + (z_local - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lidx3(x, y, z_local, nx, ny);
                const size_t linear_id = z_global * (nx * ny) + y * nx + x;
                const double pseudo =
                    ((((linear_id + 1) * 1299709) % vol) /
                     static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validate local interior cells
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<double>& c, size_t nx, size_t ny,
                    size_t nz_local, double& local_min, double& local_max) {
    bool valid = true;
    local_min = 0.0;
    local_max = 0.0;
    bool first = true;

    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[lidx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    valid = false;
                }
                if (first) {
                    local_min = val;
                    local_max = val;
                    first = false;
                } else {
                    local_min = std::min(local_min, val);
                    local_max = std::max(local_max, val);
                }
            }
        }
    }
    return valid;
}

// ---------------------------------------------------------------------------
// Print usage
// ---------------------------------------------------------------------------
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
    // ------------------------------------------------------------------
    // MPI initialisation
    // ------------------------------------------------------------------
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(&argc, &argv);
    }
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // ------------------------------------------------------------------
    // Parse command line arguments
    // ------------------------------------------------------------------
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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

    // ------------------------------------------------------------------
    // Domain decomposition along Z-axis
    // Local layout:  [ ghost_bottom ][ interior ][ ghost_top ]
    //   z_local = 0  -> ghost bottom
    //   z_local = 1 .. nz_local -> interior
    //   z_local = nz_local+1 -> ghost top
    // ------------------------------------------------------------------
    const size_t base_z = nz / numRanks;
    const size_t remainder = nz % numRanks;
    const size_t nz_local =
        base_z + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Global starting z-index for this rank's interior
    size_t z_start_global = 0;
    for (int r = 0; r < rank; ++r) {
        z_start_global += base_z + (static_cast<size_t>(r) < remainder ? 1 : 0);
    }

    // Total local z (interior + 2 ghosts)
    const size_t nz_total = (nz_local > 0) ? nz_local + 2 : 2;
    const size_t localSize = nx * ny * nz_total;
    const size_t xy = nx * ny;

    // ------------------------------------------------------------------
    // Physical parameters
    // ------------------------------------------------------------------
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    // ------------------------------------------------------------------
    // Allocate local arrays (zero-initialized)
    // ------------------------------------------------------------------
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // ------------------------------------------------------------------
    // Initialise interior cells (ghosts stay 0)
    // ------------------------------------------------------------------
    if (nz_local > 0) {
        initializeConcentration(cold, nx, ny, nz, nz_local, z_start_global);
        // cnew interior = cold interior (ghosts already 0)
        std::memcpy(cnew.data() + lidx3(0, 0, 1, nx, ny),
                    cold.data() + lidx3(0, 0, 1, nx, ny),
                    nz_local * xy * sizeof(double));
    }

    // ------------------------------------------------------------------
    // Print header (rank 0 only)
    // ------------------------------------------------------------------
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Simulation loop
    // ------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // ---- Halo exchange for cold (z-direction) ----
        if (nz_local > 0) {
            // Send top interior -> rank above; receive into ghost top
            if (rank + 1 < numRanks) {
                MPI_Sendrecv(
                    cold.data() + lidx3(0, 0, nz_local, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank + 1, 100,
                    cold.data() + lidx3(0, 0, nz_local + 1, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank + 1, 200,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // Boundary rank: clamp ghost top to interior top (Neumann BC)
                std::memcpy(
                    cold.data() + lidx3(0, 0, nz_local + 1, nx, ny),
                    cold.data() + lidx3(0, 0, nz_local, nx, ny),
                    xy * sizeof(double));
            }
            // Send bottom interior -> rank below; receive into ghost bottom
            if (rank - 1 >= 0) {
                MPI_Sendrecv(
                    cold.data() + lidx3(0, 0, 1, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank - 1, 300,
                    cold.data() + lidx3(0, 0, 0, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank - 1, 400,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // Boundary rank: clamp ghost bottom to interior bottom
                std::memcpy(
                    cold.data() + lidx3(0, 0, 0, nx, ny),
                    cold.data() + lidx3(0, 0, 1, nx, ny),
                    xy * sizeof(double));
            }
        }

        // ---- Compute chemical potential (interior only) ----
        if (nz_local > 0) {
            computeChemicalPotential(cold, mu, nx, ny, nz_local, nz_total,
                                     dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        }

        // ---- Halo exchange for mu (z-direction) ----
        if (nz_local > 0) {
            if (rank + 1 < numRanks) {
                MPI_Sendrecv(
                    mu.data() + lidx3(0, 0, nz_local, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank + 1, 101,
                    mu.data() + lidx3(0, 0, nz_local + 1, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank + 1, 201,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // Boundary rank: clamp ghost top to interior top
                std::memcpy(
                    mu.data() + lidx3(0, 0, nz_local + 1, nx, ny),
                    mu.data() + lidx3(0, 0, nz_local, nx, ny),
                    xy * sizeof(double));
            }
            if (rank - 1 >= 0) {
                MPI_Sendrecv(
                    mu.data() + lidx3(0, 0, 1, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank - 1, 301,
                    mu.data() + lidx3(0, 0, 0, nx, ny),
                    static_cast<int>(xy), MPI_DOUBLE, rank - 1, 401,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // Boundary rank: clamp ghost bottom to interior bottom
                std::memcpy(
                    mu.data() + lidx3(0, 0, 0, nx, ny),
                    mu.data() + lidx3(0, 0, 1, nx, ny),
                    xy * sizeof(double));
            }
        }

        // ---- Update concentration (interior only) ----
        if (nz_local > 0) {
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local, nz_total,
                               D, dt, dx, dy, dz);
        }

        // ---- Copy interior boundary to ghost cells in cnew ----
        // This ensures ghost cells have valid data after swap.
        // Ghost top of cnew = interior top of cnew
        // Ghost bottom of cnew = interior bottom of cnew
        if (nz_local > 0) {
            std::memcpy(cnew.data() + lidx3(0, 0, nz_local + 1, nx, ny),
                        cnew.data() + lidx3(0, 0, nz_local, nx, ny),
                        xy * sizeof(double));
            std::memcpy(cnew.data() + lidx3(0, 0, 0, nx, ny),
                        cnew.data() + lidx3(0, 0, 1, nx, ny),
                        xy * sizeof(double));
        }

        // ---- Swap buffers ----
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ------------------------------------------------------------------
    // Performance reporting
    // ------------------------------------------------------------------
    const size_t globalGridSize = nx * ny * nz;
    double cellUpdates = static_cast<double>(globalGridSize) * iterations;
    double mcups = 0.0;
    if (duration.count() > 0) {
        mcups = cellUpdates / (static_cast<double>(duration.count()) / 1000.0) /
                1e6;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ------------------------------------------------------------------
    // Gather interior cells to rank 0 for results / validation
    // ------------------------------------------------------------------
    const size_t myInteriorCount = nz_local * xy;
    std::vector<double> myInterior(myInteriorCount);

    // Extract interior cells from local buffer (skip ghosts)
    if (myInteriorCount > 0) {
        std::memcpy(myInterior.data(),
                    cold.data() + lidx3(0, 0, 1, nx, ny),
                    myInteriorCount * sizeof(double));
    }

    // Build recvcounts and displs for MPI_Gatherv
    std::vector<int> recvcounts(numRanks, 0);
    std::vector<int> displs(numRanks, 0);
    {
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t r_base = nz / numRanks;
            const size_t r_rem =
                r_base + (static_cast<size_t>(r) < (nz % numRanks) ? 1 : 0);
            recvcounts[r] = static_cast<int>(r_rem * xy);
            displs[r] = static_cast<int>(offset);
            offset += r_rem * xy;
        }
    }

    // Gather to rank 0
    std::vector<double> globalC(globalGridSize);
    MPI_Gatherv(myInterior.data(),
                static_cast<int>(myInteriorCount), MPI_DOUBLE,
                globalC.data(), recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Print results (rank 0 only)
    // ------------------------------------------------------------------
    if (rank == 0 && printResults) {
        print_results(globalC, "Concentration");
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------
    if (validate) {
        bool localValid = true;
        double localMin = 0.0, localMax = 0.0;

        if (nz_local > 0) {
            localValid = validateResult(cold, nx, ny, nz_local, localMin,
                                        localMax);
        }

        // Reduce validation status and min/max across ranks
        bool globalValid = localValid;
        double globalMin = localMin, globalMax = localMax;

        MPI_Allreduce(MPI_IN_PLACE, &globalValid, 1, MPI_C_BOOL, MPI_LAND,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);

            if (globalValid && globalMax <= 10.0 && globalMin >= -10.0) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
