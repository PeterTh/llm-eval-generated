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
// The z direction indexes into ghost planes (z-1 and z+1 always valid),
// which hold either neighbor-rank data or a copy of the boundary plane
// (implementing the clamped boundary condition at global domain edges).
static inline double computeLaplacian(const double* c, const size_t nx, const size_t ny,
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

// Compute chemical potential for local planes [zBegin, zEnd) (ghost-offset coordinates)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny,
                              const size_t zBegin, const size_t zEnd,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double* cp = c.data();
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = cp[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(cp, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step for local planes [zBegin, zEnd) (ghost-offset coordinates)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny,
                        const size_t zBegin, const size_t zEnd,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double* mup = mu.data();
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mup, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field for the local slab (global planes [zStart, zStart+localNz))
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t zStart, const size_t localNz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < localNz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (zStart + z) * (nx * ny) + y * nx + x;
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

// Exchange ghost planes of `field` with z-neighbors (nonblocking); requests must be
// completed with MPI_Waitall before the ghost planes / sent boundary planes are used.
static void startHaloExchange(std::vector<double>& field, const size_t planeSize, const size_t localNz,
                              const int prevRank, const int nextRank, MPI_Request req[4]) {
    double* f = field.data();
    // Receive into ghost plane 0 from prev, ghost plane localNz+1 from next
    MPI_Irecv(f, (int)planeSize, MPI_DOUBLE, prevRank, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(f + (localNz + 1) * planeSize, (int)planeSize, MPI_DOUBLE, nextRank, 1, MPI_COMM_WORLD, &req[1]);
    // Send boundary planes
    MPI_Isend(f + 1 * planeSize, (int)planeSize, MPI_DOUBLE, prevRank, 1, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(f + localNz * planeSize, (int)planeSize, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD, &req[3]);
}

// Fill ghost planes at global domain edges with a copy of the adjacent boundary
// plane, implementing the clamped boundary condition of the original code.
static void fillClampedGhosts(std::vector<double>& field, const size_t planeSize, const size_t localNz,
                              const int prevRank, const int nextRank) {
    double* f = field.data();
    if (prevRank == MPI_PROC_NULL) {
        std::memcpy(f, f + planeSize, planeSize * sizeof(double));
    }
    if (nextRank == MPI_PROC_NULL) {
        std::memcpy(f + (localNz + 1) * planeSize, f + localNz * planeSize, planeSize * sizeof(double));
    }
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

    const size_t gridSize = nx * ny * nz;
    const size_t planeSize = nx * ny;

    // 1D slab decomposition along z: rank r owns global planes [zStart, zStart+localNz)
    const size_t base = nz / nprocs;
    const size_t rem = nz % nprocs;
    const size_t localNz = base + ((size_t)rank < rem ? 1 : 0);
    const size_t zStart = (size_t)rank * base + std::min((size_t)rank, rem);

    // z-neighbor ranks; MPI_PROC_NULL past domain edges and around empty ranks
    // (with this block distribution, empty ranks only occur at the high end)
    const int prevRank = (localNz > 0 && zStart > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (localNz > 0 && zStart + localNz < nz) ? rank + 1 : MPI_PROC_NULL;

    // Allocate local arrays with one ghost plane on each side
    const size_t localSize = planeSize * (localNz + 2);
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, zStart, localNz);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    MPI_Request req[4];
    for (int t = 0; t < iterations; ++t) {
        if (localNz > 0) {
            // Exchange c halos; overlap with interior chemical potential computation
            startHaloExchange(cold, planeSize, localNz, prevRank, nextRank, req);
            fillClampedGhosts(cold, planeSize, localNz, prevRank, nextRank);
            if (localNz > 2) {
                computeChemicalPotential(cold, mu, nx, ny, 2, localNz, dx, dy, dz,
                                         gamma, e_AA, e_BB, e_AB);
            }
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            // Boundary planes (1 and localNz) need the freshly received halos
            computeChemicalPotential(cold, mu, nx, ny, 1, std::min<size_t>(2, localNz + 1),
                                     dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (localNz > 1) {
                computeChemicalPotential(cold, mu, nx, ny, localNz, localNz + 1,
                                         dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // Exchange mu halos; overlap with interior concentration update
            startHaloExchange(mu, planeSize, localNz, prevRank, nextRank, req);
            fillClampedGhosts(mu, planeSize, localNz, prevRank, nextRank);
            if (localNz > 2) {
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, 2, localNz, D, dt, dx, dy, dz);
            }
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, 1, std::min<size_t>(2, localNz + 1),
                               D, dt, dx, dy, dz);
            if (localNz > 1) {
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, localNz, localNz + 1,
                                   D, dt, dx, dy, dz);
            }
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - tStart;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)(elapsed * 1000.0));

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;

    if (printResults || validate) {
        // Gather the full field on rank 0 (interior planes only, in global z order)
        std::vector<double> full;
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            const size_t rNz = base + ((size_t)r < rem ? 1 : 0);
            const size_t rStart = (size_t)r * base + std::min((size_t)r, rem);
            counts[r] = (int)(rNz * planeSize);
            displs[r] = (int)(rStart * planeSize);
        }
        if (rank == 0) full.resize(gridSize);
        MPI_Gatherv(cold.data() + planeSize, (int)(localNz * planeSize), MPI_DOUBLE,
                    rank == 0 ? full.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
