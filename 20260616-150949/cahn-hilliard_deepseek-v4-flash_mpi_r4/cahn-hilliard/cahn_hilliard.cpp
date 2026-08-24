#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for local storage with ghost cells.
// z ranges: 0 = bottom ghost, 1..nz_local = real data, nz_local+1 = top ghost.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange ghost cells in the Z direction using odd-even coloring.
static void exchangeHalos(std::vector<double>& arr, const size_t nx, const size_t ny,
                          const size_t nz_local, const int rank, const int size) {
    const size_t plane_size = nx * ny;

    // Phase 1: even ranks exchange with upper neighbor, odd ranks with lower neighbor
    if (rank % 2 == 0 && rank < size - 1) {
        MPI_Sendrecv(&arr[idx3(0, 0, nz_local, nx, ny)], plane_size, MPI_DOUBLE, rank + 1, 0,
                     &arr[idx3(0, 0, nz_local + 1, nx, ny)], plane_size, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank % 2 == 1) {
        MPI_Sendrecv(&arr[idx3(0, 0, 1, nx, ny)], plane_size, MPI_DOUBLE, rank - 1, 0,
                     &arr[idx3(0, 0, 0, nx, ny)], plane_size, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Phase 2: odd ranks exchange with upper neighbor, even ranks with lower neighbor
    if (rank % 2 == 1 && rank < size - 1) {
        MPI_Sendrecv(&arr[idx3(0, 0, nz_local, nx, ny)], plane_size, MPI_DOUBLE, rank + 1, 0,
                     &arr[idx3(0, 0, nz_local + 1, nx, ny)], plane_size, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank % 2 == 0 && rank > 0) {
        MPI_Sendrecv(&arr[idx3(0, 0, 1, nx, ny)], plane_size, MPI_DOUBLE, rank - 1, 0,
                     &arr[idx3(0, 0, 0, nx, ny)], plane_size, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Compute Laplacian with clamped boundary conditions on local storage.
// z_local is the local index in the ghost-cell array (real data at 1..nz_local).
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const size_t z_start,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z_local) {
    const size_t global_z = z_start + z_local - 1;

    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (global_z < nz - 1) ? z_local + 1 : z_local;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (global_z > 0) ? z_local - 1 : z_local;

    const double cxx = (c[idx3(xp, y, z_local, nx, ny)] + c[idx3(xn, y, z_local, nx, ny)] - 
                   2.0 * c[idx3(x, y, z_local, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z_local, nx, ny)] + c[idx3(x, yn, z_local, nx, ny)] - 
                   2.0 * c[idx3(x, y, z_local, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                   2.0 * c[idx3(x, y, z_local, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local data with ghost cells.
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t nz_local, const size_t z_start,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, z_start, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step on local data with ghost cells.
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const size_t nz_local, const size_t z_start,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, z_start, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field on the local sub-domain using global indices
// for deterministic pseudo-random values identical to the serial run.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const size_t nz_local, const size_t z_start) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t global_z = z_start + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validate result across all ranks using MPI reductions.
bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny,
                    const size_t nz_local, const int rank) {
    int local_nan = 0;
    double local_min = c[idx3(0, 0, 1, nx, ny)];
    double local_max = local_min;

    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double val = c[idx];
                if (std::isnan(val) || std::isinf(val)) {
                    local_nan = 1;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    int global_nan = 0;
    double global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_nan, &global_nan, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        if (global_nan) {
            printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        }
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return valid != 0;
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool doValidate = false;
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
            doValidate = true;
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

    // 1D decomposition along the Z dimension
    const size_t nz_local_base = nz / size;
    const size_t nz_extra = nz % size;
    const size_t nz_local = nz_local_base + (static_cast<size_t>(rank) < nz_extra ? 1 : 0);

    size_t z_start;
    if (static_cast<size_t>(rank) < nz_extra) {
        z_start = rank * (nz_local_base + 1);
    } else {
        z_start = rank * nz_local_base + nz_extra;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d\n", size);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", doValidate ? "enabled" : "disabled");
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
    const size_t local_grid_size = (nz_local + 2) * ny * nx; // +2 for ghost cells

    // Allocate arrays with ghost-cell padding
    std::vector<double> cold(local_grid_size);
    std::vector<double> cnew(local_grid_size);
    std::vector<double> mu(local_grid_size);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, nz_local, z_start);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost cells for concentration field
        exchangeHalos(cold, nx, ny, nz_local, rank, size);

        // Compute chemical potential (needs ghost cells of c for Laplacian)
        computeChemicalPotential(cold, mu, nx, ny, nz, nz_local, z_start,
                                dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Exchange ghost cells for chemical potential
        exchangeHalos(mu, nx, ny, nz_local, rank, size);

        // Update concentration (needs ghost cells of mu for Laplacian)
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, nz_local, z_start,
                           D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long local_duration_ms = static_cast<long>(duration.count());
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        const int my_count = static_cast<int>(nz_local * ny * nx);
        std::vector<int> all_counts(size);
        MPI_Gather(&my_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Pack local real data (skip ghost cells)
        std::vector<double> local_packed(my_count);
        for (size_t z = 1; z <= nz_local; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t dst = (z - 1) * (nx * ny) + y * nx + x;
                    const size_t src = idx3(x, y, z, nx, ny);
                    local_packed[dst] = cold[src];
                }
            }
        }

        std::vector<double> global_data;
        std::vector<int> displs;
        if (rank == 0) {
            global_data.resize(gridSize);
            displs.resize(size);
            displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                displs[r] = displs[r - 1] + all_counts[r - 1];
            }
        }

        MPI_Gatherv(local_packed.data(), my_count, MPI_DOUBLE,
                    rank == 0 ? global_data.data() : nullptr,
                    rank == 0 ? all_counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global_data, "Concentration");
        }
    }

    // Validation
    if (doValidate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz_local, rank);

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
