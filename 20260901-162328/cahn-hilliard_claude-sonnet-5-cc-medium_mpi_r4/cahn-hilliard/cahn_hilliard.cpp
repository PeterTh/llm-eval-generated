#include <algorithm>
#include <chrono>
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

// Compute Laplacian with clamped boundary conditions.
// Note: the z-direction neighbors (zp/zn) are expected to always be valid
// indices into `c` (either real cells or halo cells populated by the caller),
// so no clamping is required in z; clamping is only needed for x and y since
// those dimensions are never domain-decomposed.
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1;
    (void)nz;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential for local z-slab [zStart, zEnd) (padded local indices)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nzPadded,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t zStart, const size_t zEnd) {
    for (size_t z = zStart; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nzPadded, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step for local z-slab [zStart, zEnd) (padded local indices)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nzPadded,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t zStart, const size_t zEnd) {
    for (size_t z = zStart; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, nzPadded, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field for the local z-slab (padded local indices [1, localNz+1)),
// using global coordinates so the result is identical to the non-decomposed reference.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t zGlobalOffset, const size_t localNz) {
    const size_t vol = nx * ny * nz;

    for (size_t zl = 0; zl < localNz; ++zl) {
        const size_t zGlobal = zGlobalOffset + zl;
        const size_t zLocalPadded = zl + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zLocalPadded, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = zGlobal * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Exchange halo (ghost) layers with neighboring ranks along the z dimension.
// Ranks at the global domain boundary clamp by copying their own edge slab,
// matching the clamped boundary condition of the original single-rank code.
void exchangeGhosts(std::vector<double>& c, const size_t nx, const size_t ny, const size_t localNz,
                    const int rank, const int size) {
    const size_t plane = nx * ny;
    const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    double* base = c.data();

    // Exchange with "down" neighbor: send our first real slab (z=1), receive into our
    // bottom ghost (z=0) from the neighbor's last real slab.
    MPI_Sendrecv(base + 1 * plane, static_cast<int>(plane), MPI_DOUBLE, down, 0,
                 base + 0 * plane, static_cast<int>(plane), MPI_DOUBLE, down, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Exchange with "up" neighbor: send our last real slab (z=localNz), receive into our
    // top ghost (z=localNz+1) from the neighbor's first real slab.
    MPI_Sendrecv(base + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, up, 1,
                 base + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, up, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (down == MPI_PROC_NULL) {
        std::memcpy(base + 0 * plane, base + 1 * plane, plane * sizeof(double));
    }
    if (up == MPI_PROC_NULL) {
        std::memcpy(base + (localNz + 1) * plane, base + localNz * plane, plane * sizeof(double));
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (nz < static_cast<size_t>(size)) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds grid Z dimension (%zu)\n", size, nz);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
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

    const size_t gridSize = nx * ny * nz;

    // Block decomposition of the z dimension across ranks
    const size_t baseNz = nz / static_cast<size_t>(size);
    const size_t remainder = nz % static_cast<size_t>(size);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t zGlobalOffset = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);

    const size_t plane = nx * ny;
    const size_t localNzPadded = localNz + 2; // 1 ghost layer on each side
    const size_t localGridSize = localNzPadded * plane;

    // Allocate local arrays (with halo/ghost layers)
    std::vector<double> cold(localGridSize);
    std::vector<double> cnew(localGridSize);
    std::vector<double> mu(localGridSize);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, zGlobalOffset, localNz);
    exchangeGhosts(cold, nx, ny, localNz, rank, size);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, localNzPadded, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB, 1, localNz + 1);

        // Halo exchange for mu before computing its Laplacian
        exchangeGhosts(mu, nx, ny, localNz, rank, size);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, localNzPadded, D, dt, dx, dy, dz, 1, localNz + 1);

        // Swap buffers
        std::swap(cold, cnew);

        // Halo exchange for cold before the next iteration's Laplacian
        exchangeGhosts(cold, nx, ny, localNz, rank, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (maxDuration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full concentration field on rank 0 when needed for
    // result printing/validation, to preserve semantics identical to the
    // single-rank reference implementation.
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<int> recvCounts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t rNz = baseNz + (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t rOffset = static_cast<size_t>(r) * baseNz + std::min(static_cast<size_t>(r), remainder);
            recvCounts[r] = static_cast<int>(rNz * plane);
            displs[r] = static_cast<int>(rOffset * plane);
        }

        std::vector<double> globalC;
        if (rank == 0) globalC.resize(gridSize);

        MPI_Gatherv(cold.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(globalC, "Concentration");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(globalC, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
