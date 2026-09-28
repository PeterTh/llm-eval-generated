#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions in x/y.
// z indexes into the local array including ghost planes, so z-1 and z+1 are
// always valid; global clamping in z is handled by filling ghost planes.
inline double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cc = c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * cc) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * cc) / (dy * dy);
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] - 2.0 * cc) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential for local z planes [zlo, zhi) (ghost-inclusive indices)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny,
                              const size_t zlo, const size_t zhi,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = zlo; z < zhi; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c.data(), nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step for local z planes [zlo, zhi) (ghost-inclusive indices)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny,
                        const size_t zlo, const size_t zhi,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = zlo; z < zhi; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu.data(), nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize the local slab of the concentration field (global planes [z0, z0+lnz))
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t lnz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < lnz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Local array has one ghost plane below the first interior plane
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1] from the global linear id
                const size_t linear_id = (z0 + z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
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

// Exchange ghost planes with z-neighbors and apply clamped boundaries at the
// global domain edges. Non-blocking; returns outstanding requests in reqs.
static void startHaloExchange(std::vector<double>& f, const size_t planeSize, const size_t lnz,
                              const int prevRank, const int nextRank, MPI_Request reqs[4]) {
    // Interior planes occupy [1, lnz]; ghosts are plane 0 and plane lnz+1
    MPI_Irecv(f.data(), (int)planeSize, MPI_DOUBLE, prevRank, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(f.data() + (lnz + 1) * planeSize, (int)planeSize, MPI_DOUBLE, nextRank, 1, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(f.data() + 1 * planeSize, (int)planeSize, MPI_DOUBLE, prevRank, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(f.data() + lnz * planeSize, (int)planeSize, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD, &reqs[3]);
}

static void finishHaloExchange(std::vector<double>& f, const size_t planeSize, const size_t lnz,
                               const int prevRank, const int nextRank, MPI_Request reqs[4]) {
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    // Clamped boundary conditions at the global z edges: ghost = edge plane
    if (prevRank == MPI_PROC_NULL) {
        std::memcpy(f.data(), f.data() + planeSize, planeSize * sizeof(double));
    }
    if (nextRank == MPI_PROC_NULL) {
        std::memcpy(f.data() + (lnz + 1) * planeSize, f.data() + lnz * planeSize, planeSize * sizeof(double));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // 1D slab decomposition along z: block distribution, first (nz % nprocs)
    // ranks get one extra plane, so occupied ranks are contiguous from 0.
    const size_t base = nz / nprocs;
    const size_t rem = nz % nprocs;
    const size_t urank = (size_t)rank;
    const size_t lnz = base + (urank < rem ? 1 : 0);
    const size_t z0 = urank * base + std::min(urank, rem);

    const int prevRank = (lnz > 0 && z0 > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (lnz > 0 && z0 + lnz < nz) ? rank + 1 : MPI_PROC_NULL;

    const size_t planeSize = nx * ny;
    const size_t localSize = (lnz + 2) * planeSize;  // interior planes + 2 ghost planes

    // Allocate arrays
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z0, lnz);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    MPI_Request reqs[4];
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost planes of the concentration field, overlapping the
        // communication with the interior chemical-potential computation.
        startHaloExchange(cold, planeSize, lnz, prevRank, nextRank, reqs);
        if (lnz > 2) {
            computeChemicalPotential(cold, mu, nx, ny, 2, lnz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        }
        finishHaloExchange(cold, planeSize, lnz, prevRank, nextRank, reqs);
        if (lnz > 0) {
            computeChemicalPotential(cold, mu, nx, ny, 1, std::min<size_t>(2, lnz + 1), dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (lnz > 1) {
                computeChemicalPotential(cold, mu, nx, ny, lnz, lnz + 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
        }

        // Exchange ghost planes of the chemical potential, overlapping the
        // communication with the interior concentration update.
        startHaloExchange(mu, planeSize, lnz, prevRank, nextRank, reqs);
        if (lnz > 2) {
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, 2, lnz, D, dt, dx, dy, dz);
        }
        finishHaloExchange(mu, planeSize, lnz, prevRank, nextRank, reqs);
        if (lnz > 0) {
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, 1, std::min<size_t>(2, lnz + 1), D, dt, dx, dy, dz);
            if (lnz > 1) {
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, lnz, lnz + 1, D, dt, dx, dy, dz);
            }
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double endTime = MPI_Wtime();
    const long durationMs = (long)((endTime - startTime) * 1000.0);
    long maxDurationMs = 0;
    MPI_Reduce(&durationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    const size_t gridSize = nx * ny * nz;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (maxDurationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;

    // Gather the full field on rank 0 for result output and validation
    if (printResults || validate) {
        std::vector<double> full;
        std::vector<int> counts(nprocs);
        std::vector<int> displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            const size_t ur = (size_t)r;
            const size_t rlnz = base + (ur < rem ? 1 : 0);
            const size_t rz0 = ur * base + std::min(ur, rem);
            counts[r] = (int)(rlnz * planeSize);
            displs[r] = (int)(rz0 * planeSize);
        }
        if (rank == 0) full.resize(gridSize);
        MPI_Gatherv(cold.data() + planeSize, (int)(lnz * planeSize), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

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
