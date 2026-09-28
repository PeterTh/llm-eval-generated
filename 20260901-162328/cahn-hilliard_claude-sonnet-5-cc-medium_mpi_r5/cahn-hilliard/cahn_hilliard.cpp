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

// Compute Laplacian with clamped boundary conditions in x/y.
// The z direction is decomposed across MPI ranks; the local array carries one
// ghost plane on each side (lz == 0 and lz == nzLocal + 1) that is kept in
// sync with neighboring ranks (or self-clamped at the global z boundary), so
// no special-casing of z is needed here.
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nzExt,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t lz) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (c[idx3(xp, y, lz, nx, ny)] + c[idx3(xn, y, lz, nx, ny)] -
                  2.0 * c[idx3(x, y, lz, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, lz, nx, ny)] + c[idx3(x, yn, lz, nx, ny)] -
                  2.0 * c[idx3(x, y, lz, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, lz + 1, nx, ny)] + c[idx3(x, y, lz - 1, nx, ny)] -
                  2.0 * c[idx3(x, y, lz, nx, ny)]) / (dz * dz);
    (void)nzExt;

    return cxx + cyy + czz;
}

// Exchange the single ghost plane on each side with the z-neighbors. Ranks at
// the global domain boundary have no neighbor (MPI_PROC_NULL) and instead
// self-clamp, matching the original serial code's clamped boundary condition.
void exchangeGhosts(std::vector<double>& local, const size_t nx, const size_t ny, const size_t nzLocal,
                    const int left, const int right, MPI_Comm comm) {
    const size_t plane = nx * ny;

    // send my top real plane to "right", receive into my bottom ghost from "left"
    MPI_Sendrecv(&local[nzLocal * plane], static_cast<int>(plane), MPI_DOUBLE, right, 0,
                 &local[0], static_cast<int>(plane), MPI_DOUBLE, left, 0,
                 comm, MPI_STATUS_IGNORE);

    // send my bottom real plane to "left", receive into my top ghost from "right"
    MPI_Sendrecv(&local[1 * plane], static_cast<int>(plane), MPI_DOUBLE, left, 1,
                 &local[(nzLocal + 1) * plane], static_cast<int>(plane), MPI_DOUBLE, right, 1,
                 comm, MPI_STATUS_IGNORE);

    if (left == MPI_PROC_NULL) {
        std::copy(local.begin() + 1 * plane, local.begin() + 2 * plane, local.begin() + 0 * plane);
    }
    if (right == MPI_PROC_NULL) {
        std::copy(local.begin() + nzLocal * plane, local.begin() + (nzLocal + 1) * plane,
                   local.begin() + (nzLocal + 1) * plane);
    }
}

// Compute chemical potential for the local z-slab (lz in [1, nzLocal])
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nzLocal,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nzExt = nzLocal + 2;
    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nzExt, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Cahn-Hilliard update step for the local z-slab (lz in [1, nzLocal])
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nzLocal,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nzExt = nzLocal + 2;
    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, nzExt, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Initialize concentration field for the local z-slab, using global indices
// so results are bit-identical to the original single-rank computation.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t zStart, const size_t nzLocal) {
    const size_t vol = nx * ny * nz;

    for (size_t lz = 0; lz < nzLocal; ++lz) {
        const size_t gz = zStart + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz + 1, nx, ny);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (static_cast<size_t>(nranks) > nz) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) must not exceed the z grid size (%zu)\n", nranks, nz);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
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

    // Decompose the z dimension across ranks (contiguous, near-even split).
    const size_t baseNz = nz / static_cast<size_t>(nranks);
    const size_t remainder = nz % static_cast<size_t>(nranks);
    const size_t nzLocal = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);

    const int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int right = (rank == nranks - 1) ? MPI_PROC_NULL : rank + 1;

    const size_t plane = nx * ny;
    const size_t localExtSize = plane * (nzLocal + 2);

    // Allocate local arrays (with one ghost plane on each side in z)
    std::vector<double> cold(localExtSize, 0.0);
    std::vector<double> cnew(localExtSize, 0.0);
    std::vector<double> mu(localExtSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, zStart, nzLocal);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Halo exchange for the field used by the Laplacian in this step
        exchangeGhosts(cold, nx, ny, nzLocal, left, right, MPI_COMM_WORLD);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nzLocal, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Halo exchange for mu before it's used in the update's Laplacian
        exchangeGhosts(mu, nx, ny, nzLocal, left, right, MPI_COMM_WORLD);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nzLocal, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long localMs = duration.count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full concentration field on rank 0 for reporting/validation
    std::vector<double> global;
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        global.resize(gridSize);
        counts.resize(nranks);
        displs.resize(nranks);
        size_t offset = 0;
        for (int r = 0; r < nranks; ++r) {
            const size_t rNzLocal = baseNz + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[r] = static_cast<int>(plane * rNzLocal);
            displs[r] = static_cast<int>(offset);
            offset += plane * rNzLocal;
        }
    }
    MPI_Gatherv(&cold[plane], static_cast<int>(plane * nzLocal), MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate performance
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / (maxMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
        if (printResults) {
            print_results(global, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
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
