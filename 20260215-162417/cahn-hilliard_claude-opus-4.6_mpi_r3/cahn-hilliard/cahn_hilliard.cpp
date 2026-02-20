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

// Exchange ghost layers between neighboring MPI ranks
void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int nprocs) {
    if (local_nz == 0) return;

    const size_t slice_size = nx * ny;
    const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Send top owned layer up, receive bottom ghost from below
    MPI_Sendrecv(&field[local_nz * slice_size], slice_size, MPI_DOUBLE, up, 0,
                 &field[0], slice_size, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send bottom owned layer down, receive top ghost from above
    MPI_Sendrecv(&field[1 * slice_size], slice_size, MPI_DOUBLE, down, 1,
                 &field[(local_nz + 1) * slice_size], slice_size, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Clamped boundary conditions at global edges
    if (rank == 0) {
        std::copy_n(&field[1 * slice_size], slice_size, &field[0]);
    }
    if (rank == nprocs - 1) {
        std::copy_n(&field[local_nz * slice_size], slice_size, &field[(local_nz + 1) * slice_size]);
    }
}

// Compute Laplacian; clamped BCs in x,y; ghost layers handle z
inline double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const double center = c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * center) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local slab (z=1..local_nz are owned)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= local_nz; ++z) {
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

// Cahn-Hilliard update step on local slab
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= local_nz; ++z) {
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
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t local_nz, const size_t z_start) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < local_nz; ++z) {
        const size_t global_z = z_start + z;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z + 1, nx, ny); // z+1 to skip bottom ghost
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[local_idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather local owned data to rank 0 in global order
std::vector<double> gatherField(const std::vector<double>& local_field, const size_t nx, const size_t ny,
                                const size_t local_nz, const size_t nz, const int rank, const int nprocs) {
    const size_t slice_size = nx * ny;
    int local_count = static_cast<int>(local_nz * slice_size);

    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    // Extract owned layers (skip ghost layers)
    std::vector<double> send_buf(local_count);
    for (size_t z = 0; z < local_nz; ++z) {
        std::copy_n(&local_field[(z + 1) * slice_size], slice_size, &send_buf[z * slice_size]);
    }

    std::vector<double> global_field;
    if (rank == 0) {
        global_field.resize(nx * ny * nz);
    }

    MPI_Gatherv(send_buf.data(), local_count, MPI_DOUBLE,
                global_field.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    return global_field;
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
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

    // Domain decomposition along Z axis
    const size_t base_nz = nz / static_cast<size_t>(nprocs);
    const size_t remainder = nz % static_cast<size_t>(nprocs);
    size_t local_nz, z_start;
    if (static_cast<size_t>(rank) < remainder) {
        local_nz = base_nz + 1;
        z_start = static_cast<size_t>(rank) * local_nz;
    } else {
        local_nz = base_nz;
        z_start = remainder * (base_nz + 1) + (static_cast<size_t>(rank) - remainder) * base_nz;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
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

    // Local arrays: local_nz owned layers + 2 ghost layers
    const size_t localSize = nx * ny * (local_nz + 2);

    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, local_nz, z_start);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for concentration
        exchangeHalos(cold, nx, ny, local_nz, rank, nprocs);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange halos for chemical potential
        exchangeHalos(mu, nx, ny, local_nz, rank, nprocs);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    long duration_ms = static_cast<long>((t_end - t_start) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        double cellUpdates = static_cast<double>(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather and output/validate on rank 0
    if (printResults || validate) {
        std::vector<double> global_field = gatherField(cold, nx, ny, local_nz, nz, rank, nprocs);

        if (rank == 0) {
            if (printResults) {
                print_results(global_field, "Concentration");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_field, nx, ny, nz);

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
