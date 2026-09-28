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

// Compute Laplacian on a z-ghosted slab: local z index lz is in [1, local_nz],
// ghost planes at lz-1 / lz+1 are always valid (filled by halo exchange or by
// copying the boundary plane, which reproduces the clamped boundary condition).
// x and y are clamped as in the original code.
inline double computeLaplacian(const double* c, const size_t nx, const size_t ny,
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

    return cxx + cyy + czz;
}

// Compute chemical potential for local planes [lzBegin, lzEnd)
void computeChemicalPotential(const double* c, double* mu,
                              const size_t nx, const size_t ny,
                              const size_t lzBegin, const size_t lzEnd,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t lz = lzBegin; lz < lzEnd; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Cahn-Hilliard update step for local planes [lzBegin, lzEnd)
void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu,
                        const size_t nx, const size_t ny,
                        const size_t lzBegin, const size_t lzEnd,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t lz = lzBegin; lz < lzEnd; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Initialize the local slab of the concentration field (planes [1, local_nz])
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t local_nz) {
    const size_t vol = nx * ny * nz;

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t z = z0 + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Start nonblocking exchange of the z ghost planes of a slab field
void haloExchangeStart(double* f, const size_t planeSize, const size_t local_nz,
                       const int prev, const int next, MPI_Request req[4]) {
    MPI_Irecv(f, static_cast<int>(planeSize), MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(f + (local_nz + 1) * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(f + planeSize, static_cast<int>(planeSize), MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(f + local_nz * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &req[3]);
}

// Finish the exchange; at physical domain boundaries, clamp by copying the
// boundary plane into the ghost plane (equivalent to the original clamped stencil)
void haloExchangeFinish(double* f, const size_t planeSize, const size_t local_nz,
                        const int prev, const int next, MPI_Request req[4]) {
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    if (prev == MPI_PROC_NULL) {
        memcpy(f, f + planeSize, planeSize * sizeof(double));
    }
    if (next == MPI_PROC_NULL) {
        memcpy(f + (local_nz + 1) * planeSize, f + local_nz * planeSize, planeSize * sizeof(double));
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
    const size_t planeSize = nx * ny;

    // 1D slab decomposition along z: rank r owns local_nz contiguous planes
    // starting at global plane z0; trailing ranks may own zero planes.
    const size_t base = nz / static_cast<size_t>(nprocs);
    const size_t rem = nz % static_cast<size_t>(nprocs);
    const size_t urank = static_cast<size_t>(rank);
    const size_t local_nz = base + (urank < rem ? 1 : 0);
    const size_t z0 = urank * base + std::min(urank, rem);

    auto planesOf = [&](size_t r) { return base + (r < rem ? 1 : 0); };
    const int prev = (rank > 0 && local_nz > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (urank + 1 < static_cast<size_t>(nprocs) && planesOf(urank + 1) > 0 && local_nz > 0)
                         ? rank + 1 : MPI_PROC_NULL;

    // Allocate local slabs with one ghost plane on each side (planes 0 and local_nz+1)
    const size_t localSize = (local_nz + 2) * planeSize;
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z0, local_nz);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    MPI_Request req[4];
    for (int t = 0; t < iterations; ++t) {
        if (local_nz > 0) {
            // Compute chemical potential: exchange c halos, overlapping with
            // the interior planes that do not touch the ghost layers
            haloExchangeStart(cold.data(), planeSize, local_nz, prev, next, req);
            if (local_nz > 2) {
                computeChemicalPotential(cold.data(), mu.data(), nx, ny, 2, local_nz,
                                         dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
            haloExchangeFinish(cold.data(), planeSize, local_nz, prev, next, req);
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, 1, 2,
                                     dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (local_nz > 1) {
                computeChemicalPotential(cold.data(), mu.data(), nx, ny, local_nz, local_nz + 1,
                                         dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // Update concentration: exchange mu halos, again overlapping
            haloExchangeStart(mu.data(), planeSize, local_nz, prev, next, req);
            if (local_nz > 2) {
                cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, 2, local_nz,
                                   D, dt, dx, dy, dz);
            }
            haloExchangeFinish(mu.data(), planeSize, local_nz, prev, next, req);
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, 1, 2,
                               D, dt, dx, dy, dz);
            if (local_nz > 1) {
                cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, local_nz, local_nz + 1,
                                   D, dt, dx, dy, dz);
            }
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const double localDuration = tEnd - tStart;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxDuration * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / maxDuration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 for result output and validation
    std::vector<double> cfull;
    if (printResults || validate) {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = static_cast<int>(planesOf(static_cast<size_t>(r)) * planeSize);
            displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
        }
        if (rank == 0) cfull.resize(gridSize);
        MPI_Gatherv(cold.data() + planeSize, static_cast<int>(local_nz * planeSize), MPI_DOUBLE,
                    cfull.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(cfull, "Concentration");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(cfull, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
