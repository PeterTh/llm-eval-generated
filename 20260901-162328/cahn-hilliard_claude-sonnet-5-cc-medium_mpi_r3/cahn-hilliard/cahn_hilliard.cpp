#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (z is the local, ghost-inclusive index)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Number of owned z-planes for a given rank under a contiguous block decomposition of nz planes over size ranks
inline size_t localNzFor(const size_t nz, const int size, const int rank) noexcept {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

// Global starting z-plane index for a given rank under the same decomposition
inline size_t zStartFor(const size_t nz, const int size, const int rank) noexcept {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

// Exchange one ghost plane with each z-neighbor, then replicate the edge plane into
// the ghost slot at true global boundaries (clamped boundary condition).
void exchangeHalo(std::vector<double>& field, const size_t nx, const size_t ny, const size_t local_nz,
                  const int up, const int down, MPI_Comm comm, const int rank, const int size) {
    const size_t planeSize = nx * ny;
    double* firstOwned = field.data() + 1 * planeSize;
    double* lastOwned = field.data() + local_nz * planeSize;
    double* ghostLow = field.data() + 0 * planeSize;
    double* ghostHigh = field.data() + (local_nz + 1) * planeSize;

    // Pass 1: send my first owned plane up, receive into my high ghost from down neighbor
    MPI_Sendrecv(firstOwned, static_cast<int>(planeSize), MPI_DOUBLE, up, 1,
                 ghostHigh, static_cast<int>(planeSize), MPI_DOUBLE, down, 1,
                 comm, MPI_STATUS_IGNORE);

    // Pass 2: send my last owned plane down, receive into my low ghost from up neighbor
    MPI_Sendrecv(lastOwned, static_cast<int>(planeSize), MPI_DOUBLE, down, 2,
                 ghostLow, static_cast<int>(planeSize), MPI_DOUBLE, up, 2,
                 comm, MPI_STATUS_IGNORE);

    // At true global boundaries there is no neighbor: replicate the edge plane (clamped BC)
    if (rank == 0) {
        std::memcpy(ghostLow, firstOwned, planeSize * sizeof(double));
    }
    if (rank == size - 1) {
        std::memcpy(ghostHigh, lastOwned, planeSize * sizeof(double));
    }
}

// Compute Laplacian with clamped boundary conditions.
// x,y are clamped locally (not decomposed); z is clamp-free since halo ghosts already
// hold either the neighbor's data or a replicated edge (clamped) value.
inline double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) noexcept {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential for the locally-owned z-planes (local_nz planes, ghost-indexed 1..local_nz)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= local_nz; ++z) {
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

// Cahn-Hilliard update step for the locally-owned z-planes
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field for the locally-owned z-planes, using global indices
// so results are identical to a non-decomposed run.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t local_nz, const size_t z_start) {
    const size_t vol = nx * ny * nz;

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t z = z_start + (lz - 1);
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

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Restrict participation to at most one rank per z-plane; extra ranks exit immediately.
    const int active_size = static_cast<int>(std::min(static_cast<size_t>(world_size), nz));
    const int color = (world_rank < active_size) ? 0 : 1;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm);
    if (color == 1) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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

    // Domain decomposition along z
    const size_t local_nz = localNzFor(nz, size, rank);
    const size_t z_start = zStartFor(nz, size, rank);
    const size_t planeSize = nx * ny;
    const size_t localVol = planeSize * (local_nz + 2); // +2 ghost planes

    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Allocate arrays (local, with ghost planes)
    std::vector<double> cold(localVol);
    std::vector<double> cnew(localVol);
    std::vector<double> mu(localVol);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, local_nz, z_start);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration halos, then compute chemical potential
        exchangeHalo(cold, nx, ny, local_nz, up, down, comm, rank, size);
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange chemical potential halos, then update concentration
        exchangeHalo(mu, nx, ny, local_nz, up, down, comm, rank, size);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double local_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_ms);
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field to rank 0 for result printing / validation
    std::vector<double> full;
    std::vector<int> recvcounts;
    std::vector<int> displs;
    if (printResults || validate) {
        if (rank == 0) {
            full.resize(gridSize);
            recvcounts.resize(size);
            displs.resize(size);
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                const int count = static_cast<int>(localNzFor(nz, size, r) * planeSize);
                recvcounts[r] = count;
                displs[r] = offset;
                offset += count;
            }
        }
        MPI_Gatherv(cold.data() + planeSize, static_cast<int>(local_nz * planeSize), MPI_DOUBLE,
                    full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    }

    int exitCode = 0;

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
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm);
    MPI_Finalize();
    return exitCode;
}
