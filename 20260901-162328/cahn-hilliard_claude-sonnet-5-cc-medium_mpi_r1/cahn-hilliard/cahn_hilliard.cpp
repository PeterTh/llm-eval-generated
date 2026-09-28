#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (local buffers include one ghost layer on each side
// of the z-axis; z here is always a *local* z-index)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions in x/y and ghost-cell
// based neighbor access in z (ghost planes already hold the correct
// neighbor-rank data, or a self-clamped copy at the global domain boundary)
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t zLocal) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = zLocal + 1;
    const size_t zn = zLocal - 1;

    const double cxx = (c[idx3(xp, y, zLocal, nx, ny)] + c[idx3(xn, y, zLocal, nx, ny)] -
                  2.0 * c[idx3(x, y, zLocal, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, zLocal, nx, ny)] + c[idx3(x, yn, zLocal, nx, ny)] -
                  2.0 * c[idx3(x, y, zLocal, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, zLocal, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential over the locally owned z-slab (local z-indices
// [1, nzOwned]); requires the ghost planes of c to be up to date
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nzOwned,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= nzOwned; ++z) {
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

// Cahn-Hilliard update step over the locally owned z-slab; requires the
// ghost planes of mu to be up to date
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nzOwned,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= nzOwned; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Exchange ghost (halo) planes between neighboring ranks along z. At the
// global domain boundaries (rank 0 bottom, last rank top) the ghost plane is
// self-clamped to the boundary-owned plane, matching the original serial
// clamped-boundary semantics.
void haloExchange(std::vector<double>& field, const size_t nx, const size_t ny, const size_t nzOwned,
                  const int rank, const int size, MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    const size_t bottomGhostOff = 0;
    const size_t bottomOwnedOff = planeSize;
    const size_t topOwnedOff = nzOwned * planeSize;
    const size_t topGhostOff = (nzOwned + 1) * planeSize;

    MPI_Request requests[4];
    int nreq = 0;

    if (rank > 0) {
        MPI_Irecv(&field[bottomGhostOff], planeSize, MPI_DOUBLE, rank - 1, 0, comm, &requests[nreq++]);
        MPI_Isend(&field[bottomOwnedOff], planeSize, MPI_DOUBLE, rank - 1, 1, comm, &requests[nreq++]);
    }
    if (rank < size - 1) {
        MPI_Irecv(&field[topGhostOff], planeSize, MPI_DOUBLE, rank + 1, 1, comm, &requests[nreq++]);
        MPI_Isend(&field[topOwnedOff], planeSize, MPI_DOUBLE, rank + 1, 0, comm, &requests[nreq++]);
    }

    if (rank == 0) {
        std::memcpy(&field[bottomGhostOff], &field[bottomOwnedOff], planeSize * sizeof(double));
    }
    if (rank == size - 1) {
        std::memcpy(&field[topGhostOff], &field[topOwnedOff], planeSize * sizeof(double));
    }

    if (nreq > 0) {
        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    }
}

// Initialize concentration field for the locally owned z-slab. Uses the
// global grid dimensions/offset so results are bit-identical to the
// original serial pseudo-random initialization.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const size_t zOffset, const size_t nzOwned) {
    const size_t vol = nx * ny * nz;

    for (size_t zLocal = 1; zLocal <= nzOwned; ++zLocal) {
        const size_t zGlobal = zOffset + (zLocal - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zLocal, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = zGlobal * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nzOwned, MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    const size_t ownedBegin = planeSize;
    const size_t ownedEnd = (nzOwned + 1) * planeSize;

    int localHasBad = 0;
    for (size_t i = ownedBegin; i < ownedEnd; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            localHasBad = 1;
            break;
        }
    }

    int globalHasBad = 0;
    MPI_Allreduce(&localHasBad, &globalHasBad, 1, MPI_INT, MPI_LOR, comm);
    if (globalHasBad) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double minVal = c[ownedBegin];
    double maxVal = c[ownedBegin];
    for (size_t i = ownedBegin; i < ownedEnd; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }

    double globalMin, globalMax;
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
    }

    // Values should generally stay within reasonable bounds
    if (globalMax > 10.0 || globalMin < -10.0) {
        if (rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
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

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, numRanks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numRanks);

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

    if (static_cast<size_t>(numRanks) > nz) {
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

    // 1D domain decomposition along the z-axis
    const size_t base = nz / static_cast<size_t>(numRanks);
    const size_t rem = nz % static_cast<size_t>(numRanks);
    const size_t nzOwned = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t zOffset = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    const size_t localAllocSize = (nzOwned + 2) * nx * ny;

    // Allocate local arrays (with one ghost plane on each side of z)
    std::vector<double> cold(localAllocSize);
    std::vector<double> cnew(localAllocSize);
    std::vector<double> mu(localAllocSize);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, zOffset, nzOwned);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost planes for concentration
        haloExchange(cold, nx, ny, nzOwned, rank, numRanks, comm);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nzOwned, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange ghost planes for chemical potential
        haloExchange(mu, nx, ny, nzOwned, rank, numRanks, comm);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nzOwned, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(comm);
    const double end = MPI_Wtime();
    double localElapsed = end - start;
    double elapsed;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        const size_t planeSize = nx * ny;
        const size_t localOwnedElems = nzOwned * planeSize;

        std::vector<int> recvCounts, displs;
        if (rank == 0) {
            recvCounts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const size_t rNzOwned = base + (static_cast<size_t>(r) < rem ? 1 : 0);
                const size_t rZOffset = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
                recvCounts[r] = static_cast<int>(rNzOwned * planeSize);
                displs[r] = static_cast<int>(rZOffset * planeSize);
            }
        }

        std::vector<double> globalC;
        if (rank == 0) globalC.resize(gridSize);

        MPI_Gatherv(&cold[planeSize], static_cast<int>(localOwnedElems), MPI_DOUBLE,
                    rank == 0 ? globalC.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, comm);

        if (rank == 0) {
            print_results(globalC, "Concentration");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold, nx, ny, nzOwned, comm);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
