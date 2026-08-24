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

// Compute Laplacian with clamped boundary in x,y; ghost cells handle z
inline double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential for a range of local z planes [z_begin, z_end]
void computeChemicalPotentialRange(const std::vector<double>& c, std::vector<double>& mu,
                                   const size_t nx, const size_t ny,
                                   const size_t z_begin, const size_t z_end,
                                   const double dx, const double dy, const double dz,
                                   const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = z_begin; z <= z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double cv = c[i];
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step for a range of local z planes [z_begin, z_end]
void cahnHilliardUpdateRange(std::vector<double>& cnew, const std::vector<double>& cold,
                             const std::vector<double>& mu,
                             const size_t nx, const size_t ny,
                             const size_t z_begin, const size_t z_end,
                             const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = z_begin; z <= z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                cnew[i] = cold[i] + dt * D *
                          computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field using global coordinates
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t local_nz, const size_t nz_global, const size_t z_start) {
    const size_t vol = nx * ny * nz_global;
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        const size_t zg = z_start + (zl - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, zl, nx, ny);
                const size_t linear_id = zg * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[i] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

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

    int rank, nprocs;
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

    // Domain decomposition along z
    const size_t base_nz = nz / (size_t)nprocs;
    const size_t rem = nz % (size_t)nprocs;
    size_t local_nz, z_start;
    if ((size_t)rank < rem) {
        local_nz = base_nz + 1;
        z_start = (size_t)rank * (base_nz + 1);
    } else {
        local_nz = base_nz;
        z_start = rem * (base_nz + 1) + ((size_t)rank - rem) * base_nz;
    }

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

    const size_t plane_size = nx * ny;
    // Local arrays: owned data at z_local=[1..local_nz], ghosts at z_local=0 and local_nz+1
    const size_t local_size = plane_size * (local_nz + 2);

    std::vector<double> cold(local_size, 0.0);
    std::vector<double> cnew(local_size, 0.0);
    std::vector<double> mu(local_size, 0.0);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, local_nz, nz, z_start);

    const int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Start async ghost exchange
    auto startGhostExchange = [&](std::vector<double>& arr, MPI_Request reqs[4]) {
        MPI_Irecv(arr.data(), (int)plane_size, MPI_DOUBLE, prev_rank, 1,
                  MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(arr.data() + (local_nz + 1) * plane_size, (int)plane_size, MPI_DOUBLE, next_rank, 0,
                  MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(arr.data() + plane_size, (int)plane_size, MPI_DOUBLE, prev_rank, 0,
                  MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(arr.data() + local_nz * plane_size, (int)plane_size, MPI_DOUBLE, next_rank, 1,
                  MPI_COMM_WORLD, &reqs[3]);
    };

    // Complete ghost exchange and apply clamped boundaries at global edges
    auto finishGhostExchange = [&](std::vector<double>& arr, MPI_Request reqs[4]) {
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        if (rank == 0) {
            std::memcpy(arr.data(), arr.data() + plane_size, plane_size * sizeof(double));
        }
        if (rank == nprocs - 1) {
            std::memcpy(arr.data() + (local_nz + 1) * plane_size,
                        arr.data() + local_nz * plane_size, plane_size * sizeof(double));
        }
    };

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // --- Chemical potential: overlap interior computation with ghost exchange ---
        MPI_Request reqs_c[4];
        startGhostExchange(cold, reqs_c);

        // Compute interior (z=2..local_nz-1) which doesn't depend on ghost cells
        if (local_nz > 2) {
            computeChemicalPotentialRange(cold, mu, nx, ny, 2, local_nz - 1,
                                          dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        }

        finishGhostExchange(cold, reqs_c);

        // Compute boundary planes that depend on ghost cells
        if (local_nz >= 1) {
            computeChemicalPotentialRange(cold, mu, nx, ny, 1, 1,
                                          dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        }
        if (local_nz >= 2) {
            computeChemicalPotentialRange(cold, mu, nx, ny, local_nz, local_nz,
                                          dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        }

        // --- Update: overlap interior computation with mu ghost exchange ---
        MPI_Request reqs_mu[4];
        startGhostExchange(mu, reqs_mu);

        if (local_nz > 2) {
            cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, 2, local_nz - 1,
                                    D, dt, dx, dy, dz);
        }

        finishGhostExchange(mu, reqs_mu);

        if (local_nz >= 1) {
            cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, 1, 1,
                                    D, dt, dx, dy, dz);
        }
        if (local_nz >= 2) {
            cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, local_nz, local_nz,
                                    D, dt, dx, dy, dz);
        }

        std::swap(cold, cnew);
    }

    double local_elapsed = MPI_Wtime() - t_start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long elapsed_ms = (long)(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", elapsed_ms);
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results on rank 0 for output / validation
    if (printResults || validate) {
        int local_count = (int)(local_nz * plane_size);
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }

        // Extract owned data (skip bottom ghost at z_local=0)
        std::vector<double> owned(local_count);
        std::memcpy(owned.data(), cold.data() + plane_size, (size_t)local_count * sizeof(double));

        std::vector<double> global_data;
        if (rank == 0) {
            global_data.resize(nx * ny * nz);
        }

        MPI_Gatherv(owned.data(), local_count, MPI_DOUBLE,
                     global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(global_data, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_data, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
