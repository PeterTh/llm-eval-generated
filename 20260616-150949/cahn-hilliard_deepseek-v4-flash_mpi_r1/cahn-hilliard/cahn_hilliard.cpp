#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for local arrays (with ghost cells in Z)
// z is the local z index: 0 = bottom ghost, 1..nz_local = interior, nz_local+1 = top ghost
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions in X and Y.
// Z ghost cells are pre-filled so no Z boundary check is needed.
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1;  // ghost cell always valid
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1;  // ghost cell always valid

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                   2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                   2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                   2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local interior (z=1..nz_local)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= nz_local; ++z) {
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

// Cahn-Hilliard update step on local interior (z=1..nz_local)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Exchange ghost layers in Z direction using MPI Sendrecv.
// At global boundaries, applies Neumann (zero-gradient) condition.
void exchangeGhostsZ(std::vector<double>& data, const size_t nx, const size_t ny,
                     const size_t nz_local, const int rank, const int nprocs) {
    const size_t slice_size = nx * ny;

    // Exchange with upper neighbor (rank+1):
    // send my top interior layer, receive neighbor's bottom interior into my top ghost
    if (rank < nprocs - 1) {
        MPI_Sendrecv(
            &data[idx3(0, 0, nz_local, nx, ny)],  // my top interior
            static_cast<int>(slice_size), MPI_DOUBLE, rank + 1, 0,
            &data[idx3(0, 0, nz_local + 1, nx, ny)],  // my top ghost
            static_cast<int>(slice_size), MPI_DOUBLE, rank + 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Global Z upper boundary: Neumann condition (copy last interior to ghost)
        std::copy_n(&data[idx3(0, 0, nz_local, nx, ny)], slice_size,
                     &data[idx3(0, 0, nz_local + 1, nx, ny)]);
    }

    // Exchange with lower neighbor (rank-1):
    // send my bottom interior layer, receive neighbor's top interior into my bottom ghost
    if (rank > 0) {
        MPI_Sendrecv(
            &data[idx3(0, 0, 1, nx, ny)],  // my bottom interior
            static_cast<int>(slice_size), MPI_DOUBLE, rank - 1, 0,
            &data[idx3(0, 0, 0, nx, ny)],  // my bottom ghost
            static_cast<int>(slice_size), MPI_DOUBLE, rank - 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Global Z lower boundary: Neumann condition (copy first interior to ghost)
        std::copy_n(&data[idx3(0, 0, 1, nx, ny)], slice_size,
                     &data[idx3(0, 0, 0, nx, ny)]);
    }
}

// Initialize local concentration field portion using the same pseudo-random
// sequence as the serial code by computing the global linear index.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                              const size_t nz, const size_t nz_local, const size_t z_offset) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t global_z = z_offset + z - 1;
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

    // Parse command line arguments on all ranks
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

    // Domain decomposition along Z
    size_t nz_local = nz / nprocs;
    size_t remainder = nz % nprocs;
    if ((size_t)rank < remainder) nz_local++;
    const size_t z_offset = rank * (nz / nprocs) + std::min((size_t)rank, remainder);

    const size_t slice_size = nx * ny;
    const size_t local_grid_size = slice_size * (nz_local + 2);  // +2 for ghost cells

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d\n", nprocs);
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

    // Allocate local arrays with ghost cells
    std::vector<double> cold(local_grid_size);
    std::vector<double> cnew(local_grid_size);
    std::vector<double> mu(local_grid_size);

    // Initialize local concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, nz_local, z_offset);

    // Fill initial ghost cells
    exchangeGhostsZ(cold, nx, ny, nz_local, rank, nprocs);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential (uses cold with valid ghost cells)
        computeChemicalPotential(cold, mu, nx, ny, nz_local, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange ghost cells for mu before update step
        exchangeGhostsZ(mu, nx, ny, nz_local, rank, nprocs);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);

        // Exchange ghost cells for next iteration (skip after last)
        if (t < iterations - 1) {
            exchangeGhostsZ(cold, nx, ny, nz_local, rank, nprocs);
        }
    }

    const double t_end = MPI_Wtime();
    const double local_time = t_end - t_start;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const size_t global_grid_size = nx * ny * nz;
        const double time_sec = max_time;
        printf("Computation time: %.0f ms\n", time_sec * 1000.0);
        double cellUpdates = static_cast<double>(global_grid_size) * iterations;
        double mcups = cellUpdates / time_sec / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gather full data to rank 0)
    if (printResults) {
        std::vector<double> full_cold;
        if (rank == 0) {
            full_cold.resize(nx * ny * nz);
        }

        // Build gather counts and displacements for uneven decomposition
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            size_t p_nz = nz / nprocs;
            size_t p_rem = nz % nprocs;
            if ((size_t)p < p_rem) p_nz++;
            recvcounts[p] = static_cast<int>(p_nz * slice_size);
            displs[p] = static_cast<int>((p * (nz / nprocs) + std::min((size_t)p, p_rem)) * slice_size);
        }

        MPI_Gatherv(cold.data() + slice_size,                      // skip bottom ghost
                    static_cast<int>(nz_local * slice_size), MPI_DOUBLE,
                    rank == 0 ? full_cold.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(full_cold, "Concentration");
        }
    }

    // Validation
    if (validate) {
        // Local NaN/Inf check on interior points
        int local_valid = 1;
        for (size_t i = slice_size; i < cold.size() - slice_size; ++i) {
            if (std::isnan(cold[i]) || std::isinf(cold[i])) {
                local_valid = 0;
                break;
            }
        }

        int global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        // Local min/max on interior points
        double local_min = cold[slice_size];
        double local_max = cold[slice_size];
        for (size_t i = slice_size; i < cold.size() - slice_size; ++i) {
            const double val = cold[i];
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }

        double global_min, global_max;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);

            if (!global_valid) {
                printf("Validation failed: found NaN or Inf value\n");
                MPI_Finalize();
                return 1;
            }

            if (global_max > 10.0 || global_min < -10.0) {
                printf("Validation failed: values out of expected range\n");
                MPI_Finalize();
                return 1;
            }

            printf("Validation: PASSED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
