#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (z includes ghost planes: local plane p lives at z = p + 1)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian at a point of the local slab. x/y boundaries are clamped
// locally; in z the ghost planes (filled by halo exchange or clamp copies)
// always provide valid z-1 / z+1 neighbors.
inline double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
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

// Compute chemical potential for local planes [zBegin, zEnd) (ghost-inclusive z indices)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny,
                              const size_t zBegin, const size_t zEnd,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step for local planes [zBegin, zEnd)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny,
                        const size_t zBegin, const size_t zEnd,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize the local slab of the concentration field (identical values to the
// serial code: derived from the global linear index)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t zOffset, const size_t localNz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < localNz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z + zOffset) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Exchange ghost planes with z neighbors; global z boundaries are clamped by
// copying the owned boundary plane into the ghost plane.
struct HaloExchange {
    MPI_Request reqs[4];

    void start(std::vector<double>& f, const size_t planeSize, const size_t localNz,
               const int rankDown, const int rankUp, MPI_Comm comm) {
        double* bottomGhost = f.data();
        double* bottomPlane = f.data() + planeSize;
        double* topPlane = f.data() + localNz * planeSize;
        double* topGhost = f.data() + (localNz + 1) * planeSize;

        MPI_Irecv(bottomGhost, (int)planeSize, MPI_DOUBLE, rankDown, 0, comm, &reqs[0]);
        MPI_Irecv(topGhost, (int)planeSize, MPI_DOUBLE, rankUp, 1, comm, &reqs[1]);
        MPI_Isend(bottomPlane, (int)planeSize, MPI_DOUBLE, rankDown, 1, comm, &reqs[2]);
        MPI_Isend(topPlane, (int)planeSize, MPI_DOUBLE, rankUp, 0, comm, &reqs[3]);

        // Clamped global boundaries: neighbor == self
        if (rankDown == MPI_PROC_NULL) {
            std::memcpy(bottomGhost, bottomPlane, planeSize * sizeof(double));
        }
        if (rankUp == MPI_PROC_NULL) {
            std::memcpy(topGhost, topPlane, planeSize * sizeof(double));
        }
    }

    void finish() {
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    }
};

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
        printf("MPI ranks: %d\n", nprocs);
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

    // 1D slab decomposition along z: rank r owns planes [zOffset, zOffset + localNz)
    const size_t baseNz = nz / (size_t)nprocs;
    const size_t remNz = nz % (size_t)nprocs;
    const size_t localNz = baseNz + ((size_t)rank < remNz ? 1 : 0);
    const size_t zOffset = (size_t)rank * baseNz + std::min((size_t)rank, remNz);

    const size_t planeSize = nx * ny;
    const size_t localSize = planeSize * (localNz + 2); // + 2 ghost planes

    // z neighbors; MPI_PROC_NULL at global boundaries and toward empty ranks
    const bool hasPlanes = localNz > 0;
    int rankDown = MPI_PROC_NULL;
    int rankUp = MPI_PROC_NULL;
    if (hasPlanes) {
        if (zOffset > 0) rankDown = rank - 1;
        if (zOffset + localNz < nz) rankUp = rank + 1;
    }

    // Allocate arrays
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, zOffset, localNz);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    HaloExchange halo;
    for (int t = 0; t < iterations; ++t) {
        if (hasPlanes) {
            // Exchange c ghost planes, overlapping with interior mu computation
            halo.start(cold, planeSize, localNz, rankDown, rankUp, MPI_COMM_WORLD);
            if (localNz > 2) {
                computeChemicalPotential(cold, mu, nx, ny, 2, localNz, dx, dy, dz,
                                         gamma, e_AA, e_BB, e_AB);
            }
            halo.finish();
            computeChemicalPotential(cold, mu, nx, ny, 1, 2, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
            if (localNz > 1) {
                computeChemicalPotential(cold, mu, nx, ny, localNz, localNz + 1, dx, dy, dz,
                                         gamma, e_AA, e_BB, e_AB);
            }

            // Exchange mu ghost planes, overlapping with interior update
            halo.start(mu, planeSize, localNz, rankDown, rankUp, MPI_COMM_WORLD);
            if (localNz > 2) {
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, 2, localNz, D, dt, dx, dy, dz);
            }
            halo.finish();
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, 1, 2, D, dt, dx, dy, dz);
            if (localNz > 1) {
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, localNz, localNz + 1, D, dt, dx, dy, dz);
            }
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    const size_t gridSize = nx * ny * nz;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 for result printing / validation
    std::vector<double> cfull;
    if (printResults || validate) {
        std::vector<int> counts(nprocs);
        std::vector<int> displs(nprocs);
        if (rank == 0) {
            for (int r = 0; r < nprocs; ++r) {
                const size_t rNz = baseNz + ((size_t)r < remNz ? 1 : 0);
                const size_t rOff = (size_t)r * baseNz + std::min((size_t)r, remNz);
                counts[r] = (int)(rNz * planeSize);
                displs[r] = (int)(rOff * planeSize);
            }
            cfull.resize(gridSize);
        }
        MPI_Gatherv(cold.data() + planeSize, (int)(localNz * planeSize), MPI_DOUBLE,
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
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
