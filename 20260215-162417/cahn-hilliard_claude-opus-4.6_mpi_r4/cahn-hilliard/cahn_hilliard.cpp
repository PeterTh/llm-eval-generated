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

// Exchange ghost layers between neighboring ranks (Z-decomposition)
void exchangeGhosts(std::vector<double>& c, size_t nx, size_t ny, size_t local_nz,
                    int rank, int nprocs) {
    if (local_nz == 0) return;
    const int layer_size = static_cast<int>(nx * ny);
    MPI_Status status;

    int below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int above = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Send bottom real layer (z=1) down, receive bottom ghost (z=0) from below
    MPI_Sendrecv(&c[idx3(0, 0, 1, nx, ny)], layer_size, MPI_DOUBLE, below, 0,
                 &c[idx3(0, 0, 0, nx, ny)], layer_size, MPI_DOUBLE, below, 1,
                 MPI_COMM_WORLD, &status);

    // Send top real layer (z=local_nz) up, receive top ghost (z=local_nz+1) from above
    MPI_Sendrecv(&c[idx3(0, 0, local_nz, nx, ny)], layer_size, MPI_DOUBLE, above, 1,
                 &c[idx3(0, 0, local_nz + 1, nx, ny)], layer_size, MPI_DOUBLE, above, 0,
                 MPI_COMM_WORLD, &status);

    // Clamped boundary conditions at global domain edges
    if (rank == 0) {
        std::copy_n(&c[idx3(0, 0, 1, nx, ny)], layer_size,
                    &c[idx3(0, 0, 0, nx, ny)]);
    }
    if (rank == nprocs - 1) {
        std::copy_n(&c[idx3(0, 0, local_nz, nx, ny)], layer_size,
                    &c[idx3(0, 0, local_nz + 1, nx, ny)]);
    }
}

// Compute Laplacian with clamped BC in x,y; ghost layers handle z
inline double computeLaplacian(const double* __restrict__ c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z) {
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

// Compute chemical potential (local real cells z=1..local_nz)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double* cp = c.data();
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double cv = cp[i];

                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(cp, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step (local real cells z=1..local_nz)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double* mp = mu.data();
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                cnew[i] = cold[i] + dt * D *
                           computeLaplacian(mp, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field using global coordinates
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t local_nz, const size_t global_nz, const size_t z_start) {
    const size_t vol = nx * ny * global_nz;

    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t gz = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[i] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny,
                    const size_t local_nz, int rank) {
    int local_valid = 1;
    double local_min = 1e300, local_max = -1e300;

    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) local_valid = 0;
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    double global_min, global_max;
    int all_valid;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        if (!all_valid) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    return all_valid && (global_max <= 10.0) && (global_min >= -10.0);
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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

    // Z-decomposition across MPI ranks
    const size_t local_nz_base = nz / static_cast<size_t>(nprocs);
    const size_t remainder = nz % static_cast<size_t>(nprocs);
    const size_t local_nz = local_nz_base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t z_start = static_cast<size_t>(rank) * local_nz_base + std::min(static_cast<size_t>(rank), remainder);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", nprocs);
        printf("Time steps: %d\n", iterations);
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

    const size_t gridSize = nx * ny * nz;
    // Local arrays include 2 ghost layers in Z (one at each end)
    const size_t localSize = nx * ny * (local_nz + 2);

    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, local_nz, nz, z_start);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost layers for concentration, then compute chemical potential
        exchangeGhosts(cold, nx, ny, local_nz, rank, nprocs);
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange ghost layers for mu, then update concentration
        exchangeGhosts(mu, nx, ny, local_nz, rank, nprocs);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);

        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (global_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather full array to rank 0 for output
    if (printResults) {
        const int local_count = static_cast<int>(nx * ny * local_nz);
        std::vector<double> sendbuf(local_count);
        if (local_nz > 0) {
            std::copy_n(&cold[idx3(0, 0, 1, nx, ny)], local_count, sendbuf.begin());
        }

        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> global_data;
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
            global_data.resize(gridSize);
        }

        MPI_Gatherv(sendbuf.data(), local_count, MPI_DOUBLE,
                     global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global_data, "Concentration");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, local_nz, rank);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
