#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (z is the outermost/slowest dimension, matching the
// domain decomposition axis, so per-rank slabs are contiguous blocks).
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions in x/y, and explicit
// (pre-clamped) neighbor indices in z so that MPI halo exchange can supply
// true neighbor values at interior rank boundaries while still clamping at
// the global domain boundary.
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z,
                        const size_t zn, const size_t zp) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential over the local (owned) slab, given the local
// buffer has one ghost plane of padding on each side of z (local z index
// runs 0..nzLocal+1, with 1..nzLocal being the owned cells).
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nzLocal,
                              const size_t zStart, const size_t nzGlobal,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        const size_t gz = zStart + lz - 1;
        const size_t zn = (gz > 0) ? lz - 1 : lz;
        const size_t zp = (gz < nzGlobal - 1) ? lz + 1 : lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, lz, zn, zp);
            }
        }
    }
}

// Cahn-Hilliard update step over the local (owned) slab.
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nzLocal,
                        const size_t zStart, const size_t nzGlobal,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        const size_t gz = zStart + lz - 1;
        const size_t zn = (gz > 0) ? lz - 1 : lz;
        const size_t zp = (gz < nzGlobal - 1) ? lz + 1 : lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, lz, zn, zp);
            }
        }
    }
}

// Initialize concentration field for the local slab. The pseudo-random value
// is derived from the global linear index so results are independent of the
// domain decomposition / rank count.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nzLocal,
                             const size_t zStart, const size_t nzGlobal) {
    const size_t vol = nx * ny * nzGlobal;

    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        const size_t gz = zStart + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
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

// Exchange z-boundary ghost planes with the lower/upper MPI neighbors.
// The buffer layout is [ghost(lz=0)] [owned(lz=1..nzLocal)] [ghost(lz=nzLocal+1)],
// each plane being nx*ny contiguous doubles.
void haloExchange(std::vector<double>& field, const size_t nx, const size_t ny, const size_t nzLocal,
                  const int rank, const int numRanks, MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    const int lower = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;

    // Send our first owned plane down / receive into our lower ghost plane.
    MPI_Sendrecv(field.data() + planeSize, planeSize, MPI_DOUBLE, lower, 0,
                 field.data(), planeSize, MPI_DOUBLE, lower, 1,
                 comm, MPI_STATUS_IGNORE);

    // Send our last owned plane up / receive into our upper ghost plane.
    MPI_Sendrecv(field.data() + nzLocal * planeSize, planeSize, MPI_DOUBLE, upper, 1,
                 field.data() + (nzLocal + 1) * planeSize, planeSize, MPI_DOUBLE, upper, 0,
                 comm, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    if (numRanks > static_cast<int>(nz)) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds grid size in Z (%zu)\n", numRanks, nz);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
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

    // Decompose the Z dimension into contiguous slabs across ranks (a block
    // distribution: the first `rem` ranks get one extra plane).
    const size_t baseLocal = nz / static_cast<size_t>(numRanks);
    const size_t rem = nz % static_cast<size_t>(numRanks);
    const size_t nzLocal = baseLocal + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * baseLocal + std::min(static_cast<size_t>(rank), rem);

    const size_t planeSize = nx * ny;
    const size_t localPaddedSize = planeSize * (nzLocal + 2);

    // Allocate arrays (padded with one ghost plane on each side in z)
    std::vector<double> cold(localPaddedSize);
    std::vector<double> cnew(localPaddedSize);
    std::vector<double> mu(localPaddedSize);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nzLocal, zStart, nz);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost planes for concentration before computing mu
        haloExchange(cold, nx, ny, nzLocal, rank, numRanks, MPI_COMM_WORLD);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nzLocal, zStart, nz, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange ghost planes for chemical potential before updating concentration
        haloExchange(mu, nx, ny, nzLocal, rank, numRanks, MPI_COMM_WORLD);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nzLocal, zStart, nz, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const long durationMs = static_cast<long>((end - start) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full global field on rank 0 if needed for printing/validation
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> globalC;

        if (rank == 0) {
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const size_t rLocal = baseLocal + (static_cast<size_t>(r) < rem ? 1 : 0);
                const size_t rStart = static_cast<size_t>(r) * baseLocal + std::min(static_cast<size_t>(r), rem);
                counts[r] = static_cast<int>(rLocal * planeSize);
                displs[r] = static_cast<int>(rStart * planeSize);
            }
            globalC.resize(gridSize);
        }

        MPI_Gatherv(cold.data() + planeSize, static_cast<int>(nzLocal * planeSize), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(globalC, "Concentration");
            }

            // Validation
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
