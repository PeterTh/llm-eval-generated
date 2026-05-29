#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation (local indices; z includes ghost cells)
inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange halo (ghost) cells with neighbours using non-blocking MPI.
// Local layout: z=0 is bottom ghost, z=1..local_nz are physical, z=local_nz+1 is top ghost.
static void exchangeHalo(std::vector<double>& data, size_t nx, size_t ny, size_t local_nz,
                         int lower, int upper) {
    if (local_nz == 0) return;

    const size_t slice = nx * ny;

    // Apply clamped boundary conditions for ranks at global z-boundaries
    if (lower == MPI_PROC_NULL) {
        // Owns z=0: bottom ghost mirrors bottom physical (clamped)
        std::memcpy(&data[0], &data[slice], slice * sizeof(double));
    }
    if (upper == MPI_PROC_NULL) {
        // Owns z=nz-1: top ghost mirrors top physical (clamped)
        std::memcpy(&data[(local_nz + 1) * slice], &data[local_nz * slice], slice * sizeof(double));
    }

    // Non-blocking halo exchange
    MPI_Request requests[4];
    int nreq = 0;

    // Post receives first (into ghost cells)
    if (lower != MPI_PROC_NULL) {
        MPI_Irecv(&data[0], slice, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &requests[nreq++]);
    }
    if (upper != MPI_PROC_NULL) {
        MPI_Irecv(&data[(local_nz + 1) * slice], slice, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &requests[nreq++]);
    }

    // Post sends (from boundary physical cells)
    if (upper != MPI_PROC_NULL) {
        MPI_Isend(&data[local_nz * slice], slice, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &requests[nreq++]);
    }
    if (lower != MPI_PROC_NULL) {
        MPI_Isend(&data[slice], slice, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &requests[nreq++]);
    }

    if (nreq > 0) {
        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    }
}

// Compute chemical potential on local physical cells (lz = 1..local_nz).
// The Laplacian in z uses ghost cells directly.
static void computeChemicalPotentialLocal(const std::vector<double>& c, std::vector<double>& mu,
                                          size_t nx, size_t ny, size_t local_nz,
                                          double inv_dx2, double inv_dy2, double inv_dz2,
                                          double gamma, double e_AA, double e_BB, double e_AB) {
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                const double cv = c[idx];

                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t xn = (x > 0) ? x - 1 : 0;
                const size_t yn = (y > 0) ? y - 1 : 0;

                const double lap = (c[idx3(xp, y, lz, nx, ny)] + c[idx3(xn, y, lz, nx, ny)] - 2.0 * cv) * inv_dx2
                                 + (c[idx3(x, yp, lz, nx, ny)] + c[idx3(x, yn, lz, nx, ny)] - 2.0 * cv) * inv_dy2
                                 + (c[idx3(x, y, lz + 1, nx, ny)] + c[idx3(x, y, lz - 1, nx, ny)] - 2.0 * cv) * inv_dz2;

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update on local physical cells (lz = 1..local_nz).
static void cahnHilliardUpdateLocal(std::vector<double>& cnew, const std::vector<double>& cold,
                                    const std::vector<double>& mu,
                                    size_t nx, size_t ny, size_t local_nz,
                                    double factor, double inv_dx2, double inv_dy2, double inv_dz2) {
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                const double muv = mu[idx];

                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t xn = (x > 0) ? x - 1 : 0;
                const size_t yn = (y > 0) ? y - 1 : 0;

                const double lap = (mu[idx3(xp, y, lz, nx, ny)] + mu[idx3(xn, y, lz, nx, ny)] - 2.0 * muv) * inv_dx2
                                 + (mu[idx3(x, yp, lz, nx, ny)] + mu[idx3(x, yn, lz, nx, ny)] - 2.0 * muv) * inv_dy2
                                 + (mu[idx3(x, y, lz + 1, nx, ny)] + mu[idx3(x, y, lz - 1, nx, ny)] - 2.0 * muv) * inv_dz2;

                cnew[idx] = cold[idx] + factor * lap;
            }
        }
    }
}

// Initialize concentration field on local physical cells.
static void initializeConcentrationLocal(std::vector<double>& c, size_t nx, size_t ny, size_t local_nz,
                                         size_t global_z_start, size_t vol) {
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t global_z = global_z_start + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0
    if (rank == 0) {
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
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -x <num>     Grid size in X dimension (default: 64)\n");
                printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
                printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
                printf("  -i <num>     Number of time steps (default: 20)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int iter_int = iterations;
    MPI_Bcast(&iter_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    iterations = iter_int;
    int validate_flag = validate ? 1 : 0;
    int printResults_flag = printResults ? 1 : 0;
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validate_flag != 0;
    MPI_Bcast(&printResults_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printResults = printResults_flag != 0;

    // --- Domain decomposition (1D block along Z) ---
    const size_t base_nz = nz / size;
    const size_t remainder = nz % size;
    const size_t local_nz = base_nz + (rank < static_cast<int>(remainder) ? 1 : 0);

    size_t global_z_start = 0;
    for (int r = 0; r < rank; ++r) {
        global_z_start += base_nz + (r < static_cast<int>(remainder) ? 1 : 0);
    }

    const int lower = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Local array: physical cells (local_nz) + 1 ghost on each Z side
    const size_t local_total_z = local_nz + 2;
    const size_t local_vol = nx * ny * local_total_z;

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double factor = dt * D;

    const size_t gridSize = nx * ny * nz;

    // Allocate local arrays
    std::vector<double> cold(local_vol);
    std::vector<double> cnew(local_vol);
    std::vector<double> mu(local_vol);

    // Initialize
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    initializeConcentrationLocal(cold, nx, ny, local_nz, global_z_start, gridSize);

    // Fill ghost cells before simulation
    if (local_nz > 0) {
        exchangeHalo(cold, nx, ny, local_nz, lower, upper);
    }

    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // 1. Exchange cold halos
        exchangeHalo(cold, nx, ny, local_nz, lower, upper);

        // 2. Chemical potential on physical cells
        computeChemicalPotentialLocal(cold, mu, nx, ny, local_nz,
                                       inv_dx2, inv_dy2, inv_dz2,
                                       gamma, e_AA, e_BB, e_AB);

        // 3. Exchange mu halos
        exchangeHalo(mu, nx, ny, local_nz, lower, upper);

        // 4. Update concentration on physical cells
        cahnHilliardUpdateLocal(cnew, cold, mu, nx, ny, local_nz,
                                 factor, inv_dx2, inv_dy2, inv_dz2);

        // 5. Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Report max wall time across all ranks
    long long local_ms = static_cast<long long>(duration.count());
    long long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // --- Gather full concentration field to rank 0 ---
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    size_t z_offset = 0;
    for (int r = 0; r < size; ++r) {
        size_t r_local_nz = base_nz + (r < static_cast<int>(remainder) ? 1 : 0);
        recvcounts[r] = static_cast<int>(r_local_nz * nx * ny);
        displs[r] = static_cast<int>(z_offset * nx * ny);
        z_offset += r_local_nz;
    }

    std::vector<double> full_cold;
    if (rank == 0) {
        full_cold.resize(gridSize);
    }

    const size_t slice = nx * ny;
    MPI_Gatherv(
        local_nz > 0 ? &cold[slice] : nullptr,
        static_cast<int>(local_nz * nx * ny),
        MPI_DOUBLE,
        rank == 0 ? full_cold.data() : nullptr,
        recvcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Results / validation on rank 0
    if (rank == 0) {
        if (printResults) {
            print_results(full_cold, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            double minVal = full_cold[0];
            double maxVal = full_cold[0];
            for (const auto& val : full_cold) {
                if (std::isnan(val) || std::isinf(val)) {
                    printf("Validation failed: found NaN or Inf value\n");
                    valid = false;
                    break;
                }
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }

            if (valid) {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }

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
