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

// Start non-blocking ghost layer exchange; clamped BCs filled locally
void startGhostExchange(std::vector<double>& data, const size_t nx, const size_t ny,
                        const size_t local_nz, const int rank, const int num_procs,
                        MPI_Request reqs[], int& nreqs) {
    const size_t ps = nx * ny;
    nreqs = 0;

    if (rank > 0) {
        MPI_Irecv(&data[0], static_cast<int>(ps), MPI_DOUBLE,
                  rank - 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Isend(&data[ps], static_cast<int>(ps), MPI_DOUBLE,
                  rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    } else {
        std::memcpy(&data[0], &data[ps], ps * sizeof(double));
    }

    if (rank < num_procs - 1) {
        MPI_Irecv(&data[(local_nz + 1) * ps], static_cast<int>(ps), MPI_DOUBLE,
                  rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Isend(&data[local_nz * ps], static_cast<int>(ps), MPI_DOUBLE,
                  rank + 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
    } else {
        std::memcpy(&data[(local_nz + 1) * ps], &data[local_nz * ps], ps * sizeof(double));
    }
}

void finishGhostExchange(MPI_Request reqs[], const int nreqs) {
    if (nreqs > 0) MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
}

// Compute Laplacian using local coordinates; ghost layers handle z boundaries
inline double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double center = c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * center) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * center) / (dy * dy);
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] -
                  2.0 * center) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential for z range [z_begin, z_end] in local coordinates
void computeChemicalPotentialRange(const std::vector<double>& c, std::vector<double>& mu,
                                   const size_t nx, const size_t ny,
                                   const size_t z_begin, const size_t z_end,
                                   const double dx, const double dy, const double dz,
                                   const double gamma, const double e_AA,
                                   const double e_BB, const double e_AB) {
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

// Cahn-Hilliard update for z range [z_begin, z_end] in local coordinates
void cahnHilliardUpdateRange(std::vector<double>& cnew, const std::vector<double>& cold,
                             const std::vector<double>& mu,
                             const size_t nx, const size_t ny,
                             const size_t z_begin, const size_t z_end,
                             const double D, const double dt,
                             const double dx, const double dy, const double dz) {
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

// Initialize local concentration field using global coordinates
void initializeConcentrationLocal(std::vector<double>& c, const size_t nx, const size_t ny,
                                  const size_t nz, const size_t local_nz, const size_t z_start) {
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < local_nz; ++z) {
        const size_t gz = z + z_start;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z + 1, nx, ny);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[local_idx] = -1.0 + 2.0 * pseudo;
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
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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

    // Domain decomposition along Z axis
    const size_t base_nz = nz / static_cast<size_t>(num_procs);
    const size_t rem = nz % static_cast<size_t>(num_procs);
    const size_t local_nz = base_nz + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_start = (static_cast<size_t>(rank) < rem)
                           ? static_cast<size_t>(rank) * (base_nz + 1)
                           : rem * (base_nz + 1) + (static_cast<size_t>(rank) - rem) * base_nz;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", num_procs);
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

    // Allocate arrays with ghost layers: (local_nz + 2) xy-planes
    const size_t plane_size = nx * ny;
    const size_t local_alloc = (local_nz + 2) * plane_size;
    std::vector<double> cold(local_alloc, 0.0);
    std::vector<double> cnew(local_alloc, 0.0);
    std::vector<double> mu(local_alloc, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentrationLocal(cold, nx, ny, nz, local_nz, z_start);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    MPI_Request reqs[4];
    int nreqs;

    for (int t = 0; t < iterations; ++t) {
        if (local_nz > 0) {
            // Ghost exchange for c, overlap with interior mu computation
            startGhostExchange(cold, nx, ny, local_nz, rank, num_procs, reqs, nreqs);

            // Compute mu for interior planes (no ghost dependency)
            if (local_nz >= 3) {
                computeChemicalPotentialRange(cold, mu, nx, ny, 2, local_nz - 1,
                                             dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            finishGhostExchange(reqs, nreqs);

            // Compute mu for boundary planes (depend on ghost data)
            computeChemicalPotentialRange(cold, mu, nx, ny, 1, 1,
                                         dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (local_nz >= 2) {
                computeChemicalPotentialRange(cold, mu, nx, ny, local_nz, local_nz,
                                             dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // Ghost exchange for mu, overlap with interior cnew computation
            startGhostExchange(mu, nx, ny, local_nz, rank, num_procs, reqs, nreqs);

            if (local_nz >= 3) {
                cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, 2, local_nz - 1,
                                        D, dt, dx, dy, dz);
            }

            finishGhostExchange(reqs, nreqs);

            cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, 1, 1,
                                    D, dt, dx, dy, dz);
            if (local_nz >= 2) {
                cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, local_nz, local_nz,
                                        D, dt, dx, dy, dz);
            }
        }

        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const long long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        const double cellUpdates = static_cast<double>(nx * ny * nz) * iterations;
        const double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for validation/output
    int result = 0;
    if (printResults || validate) {
        std::vector<int> recvcounts(num_procs);
        std::vector<int> displs(num_procs);
        for (int r = 0; r < num_procs; ++r) {
            const size_t r_nz = base_nz + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t r_start = (static_cast<size_t>(r) < rem)
                                   ? static_cast<size_t>(r) * (base_nz + 1)
                                   : rem * (base_nz + 1) + (static_cast<size_t>(r) - rem) * base_nz;
            recvcounts[r] = static_cast<int>(r_nz * plane_size);
            displs[r] = static_cast<int>(r_start * plane_size);
        }

        std::vector<double> global_c;
        if (rank == 0) global_c.resize(nx * ny * nz);

        MPI_Gatherv(&cold[plane_size], static_cast<int>(local_nz * plane_size), MPI_DOUBLE,
                    global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(global_c, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_c, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    result = 1;
                }
            }
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return result;
}
