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

// Domain decomposition: 1D slab decomposition along z.
// Local arrays hold nzl owned planes at local z = 1..nzl plus ghost planes
// at local z = 0 and z = nzl + 1.

// Compute Laplacian with clamped boundary conditions (global domain edges).
// zLoc is the local plane index (1-based within owned planes), zGlob the
// corresponding global z coordinate.
inline double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                               const size_t nzGlob, const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t zLoc, const size_t zGlob) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    // Clamp at the global domain boundaries; otherwise neighbor planes are
    // either owned planes or halo planes received from adjacent ranks.
    const size_t zp = (zGlob < nzGlob - 1) ? zLoc + 1 : zLoc;
    const size_t zn = (zGlob > 0) ? zLoc - 1 : zLoc;

    const double cxx = (c[idx3(xp, y, zLoc, nx, ny)] + c[idx3(xn, y, zLoc, nx, ny)] -
                  2.0 * c[idx3(x, y, zLoc, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, zLoc, nx, ny)] + c[idx3(x, yn, zLoc, nx, ny)] -
                  2.0 * c[idx3(x, y, zLoc, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, zLoc, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Exchange halo planes of a field with the z-neighbors.
void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny, const size_t nzl,
                   const int rankDown, const int rankUp, MPI_Comm comm) {
    const size_t plane = nx * ny;
    // Send bottom owned plane down, receive top ghost from up.
    MPI_Sendrecv(field.data() + plane, (int)plane, MPI_DOUBLE, rankDown, 0,
                 field.data() + (nzl + 1) * plane, (int)plane, MPI_DOUBLE, rankUp, 0,
                 comm, MPI_STATUS_IGNORE);
    // Send top owned plane up, receive bottom ghost from down.
    MPI_Sendrecv(field.data() + nzl * plane, (int)plane, MPI_DOUBLE, rankUp, 1,
                 field.data(), (int)plane, MPI_DOUBLE, rankDown, 1,
                 comm, MPI_STATUS_IGNORE);
}

// Compute chemical potential on the owned planes
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nzl,
                              const size_t nzGlob, const size_t zOffset,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= nzl; ++z) {
        const size_t zGlob = zOffset + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nzGlob, dx, dy, dz, x, y, z, zGlob);
            }
        }
    }
}

// Cahn-Hilliard update step on the owned planes
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nzl,
                        const size_t nzGlob, const size_t zOffset,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= nzl; ++z) {
        const size_t zGlob = zOffset + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, nzGlob, dx, dy, dz, x, y, z, zGlob);
            }
        }
    }
}

// Initialize concentration field on the owned planes
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nzl, const size_t nzGlob, const size_t zOffset) {
    const size_t vol = nx * ny * nzGlob;

    for (size_t z = 1; z <= nzl; ++z) {
        const size_t zGlob = zOffset + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = zGlob * (nx * ny) + y * nx + x;
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
    const size_t plane = nx * ny;

    // Slab decomposition along z: balanced distribution of planes.
    std::vector<size_t> counts(nprocs), offsets(nprocs);
    {
        const size_t base = nz / nprocs;
        const size_t rem = nz % nprocs;
        size_t off = 0;
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = base + ((size_t)r < rem ? 1 : 0);
            offsets[r] = off;
            off += counts[r];
        }
    }
    const size_t nzl = counts[rank];
    const size_t zOffset = offsets[rank];

    // z-neighbors: nearest ranks with a non-empty slab (empty slabs can occur
    // when nz < nprocs); global boundaries use MPI_PROC_NULL.
    int rankDown = MPI_PROC_NULL;
    int rankUp = MPI_PROC_NULL;
    for (int r = rank - 1; r >= 0; --r) {
        if (counts[r] > 0) { rankDown = r; break; }
    }
    for (int r = rank + 1; r < nprocs; ++r) {
        if (counts[r] > 0) { rankUp = r; break; }
    }
    if (nzl == 0) {
        rankDown = MPI_PROC_NULL;
        rankUp = MPI_PROC_NULL;
    }

    // Allocate local arrays with two ghost planes
    const size_t localSize = plane * (nzl + 2);
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nzl, nz, zOffset);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration halos, then compute chemical potential
        exchangeHalos(cold, nx, ny, nzl, rankDown, rankUp, MPI_COMM_WORLD);
        computeChemicalPotential(cold, mu, nx, ny, nzl, nz, zOffset, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange potential halos, then update concentration
        exchangeHalos(mu, nx, ny, nzl, rankDown, rankUp, MPI_COMM_WORLD);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nzl, nz, zOffset, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long durationMs = (long)((tEnd - tStart) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;

    if (printResults || validate) {
        // Gather the full concentration field on rank 0
        std::vector<double> cfull;
        std::vector<int> recvCounts(nprocs), recvDispls(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            recvCounts[r] = (int)(counts[r] * plane);
            recvDispls[r] = (int)(offsets[r] * plane);
        }
        if (rank == 0) cfull.resize(gridSize);
        MPI_Gatherv(cold.data() + plane, (int)(nzl * plane), MPI_DOUBLE,
                    cfull.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Print results for external validation
        if (printResults && rank == 0) {
            print_results(cfull, "Concentration");
        }

        // Validation
        if (validate) {
            int valid = 1;
            if (rank == 0) {
                printf("Validating result...\n");
                valid = validateResult(cfull, nx, ny, nz) ? 1 : 0;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            exitCode = valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
