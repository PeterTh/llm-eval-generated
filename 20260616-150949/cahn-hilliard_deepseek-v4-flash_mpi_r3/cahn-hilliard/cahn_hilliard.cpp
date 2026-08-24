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

// Compute Laplacian with clamped boundary conditions (x,y) and ghost cells (z)
// z_stored includes the ghost cell offset (z_stored = local_z + 1)
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z_stored) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    // Z-boundary uses pre-filled ghost cells instead of clamping
    const size_t zp = z_stored + 1;
    const size_t zn = z_stored - 1;

    const double cxx = (c[idx3(xp, y, z_stored, nx, ny)] + c[idx3(xn, y, z_stored, nx, ny)] -
                   2.0 * c[idx3(x, y, z_stored, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z_stored, nx, ny)] + c[idx3(x, yn, z_stored, nx, ny)] -
                   2.0 * c[idx3(x, y, z_stored, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                   2.0 * c[idx3(x, y, z_stored, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Exchange ghost cells in Z direction using blocking send/receive
void exchangeGhostsZ(std::vector<double>& arr, const size_t nx, const size_t ny, const size_t nz_local,
                     MPI_Comm comm, const int rank, const int size) {
    const size_t plane_size = nx * ny;

    // Initialize ghost cells with clamped values (self-boundary fallback)
    double* first_plane = &arr[plane_size];
    double* ghost_bottom = &arr[0];
    std::copy(first_plane, first_plane + plane_size, ghost_bottom);
    double* last_plane = &arr[nz_local * plane_size];
    double* ghost_top = &arr[(nz_local + 1) * plane_size];
    std::copy(last_plane, last_plane + plane_size, ghost_top);

    if (size > 1) {
        MPI_Status status;
        const int top = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
        const int bot = (rank > 0) ? rank - 1 : MPI_PROC_NULL;

        // Send bottom data plane to bottom neighbor, receive top ghost
        MPI_Sendrecv(&arr[plane_size],          static_cast<int>(plane_size), MPI_DOUBLE, bot, 0,
                     &arr[(nz_local + 1) * plane_size], static_cast<int>(plane_size), MPI_DOUBLE, top, 0,
                     comm, &status);

        // Send top data plane to top neighbor, receive bottom ghost
        MPI_Sendrecv(&arr[nz_local * plane_size], static_cast<int>(plane_size), MPI_DOUBLE, top, 1,
                     &arr[0],                     static_cast<int>(plane_size), MPI_DOUBLE, bot, 1,
                     comm, &status);
    }
}

// Compute chemical potential (subdomain with ghost cells)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 0; z < nz_local; ++z) {
        const size_t z_stored = z + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_stored, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z_stored);
            }
        }
    }
}

// Cahn-Hilliard update step (subdomain with ghost cells)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 0; z < nz_local; ++z) {
        const size_t z_stored = z + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_stored, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z_stored);
            }
        }
    }
}

// Initialize concentration field (subdomain with global z index for reproducibility)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz_local, const size_t z_offset, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    const size_t plane_size = nx * ny;

    for (size_t z = 0; z < nz_local; ++z) {
        const size_t z_stored = z + 1;
        const size_t global_z = z_offset + z;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_stored, nx, ny);
                // Generate pseudo-random value in [-1, 1] using global index
                const size_t linear_id = global_z * plane_size + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validate result across all MPI ranks using reductions
bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny,
                    const size_t nz_local, MPI_Comm comm, const int rank) {
    const size_t plane_size = nx * ny;
    int local_has_nan = 0;
    double local_min = c[plane_size];
    double local_max = c[plane_size];

    for (size_t z = 0; z < nz_local; ++z) {
        const size_t z_stored = z + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z_stored, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_has_nan = 1;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    // Global reductions
    int global_has_nan = 0;
    MPI_Allreduce(&local_has_nan, &global_has_nan, 1, MPI_INT, MPI_LOR, comm);
    double global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);

        if (global_has_nan) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }

        return true;
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Check that we have enough Z-planes for the number of MPI ranks
    if (nz < static_cast<size_t>(size)) {
        if (rank == 0) {
            printf("Error: Z grid size (%zu) must be >= number of MPI ranks (%d)\n", nz, size);
        }
        MPI_Finalize();
        return 1;
    }

    // Distribute nz across ranks (1D slab decomposition in Z)
    const int nz_int = static_cast<int>(nz);
    const int rem = nz_int % size;
    const int nz_per_rank = nz_int / size;
    std::vector<int> counts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        counts[i] = nz_per_rank + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }
    const size_t nz_local = static_cast<size_t>(counts[rank]);
    const size_t z_offset = static_cast<size_t>(displs[rank]);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", size);
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

    const size_t plane_size = nx * ny;
    const size_t local_size = plane_size * (nz_local + 2);  // +2 for ghost cells

    // Allocate arrays with ghost cells
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);

    // Initialize concentration field (each rank initializes its subdomain)
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz_local, z_offset, nz);

    // Initial ghost cell exchange for cold
    exchangeGhostsZ(cold, nx, ny, nz_local, MPI_COMM_WORLD, rank, size);

    // Start timing
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential (uses cold ghost cells from previous step)
        computeChemicalPotential(cold, mu, nx, ny, nz_local, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Exchange ghost cells for mu
        exchangeGhostsZ(mu, nx, ny, nz_local, MPI_COMM_WORLD, rank, size);

        // Update concentration (uses mu ghost cells)
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);

        // Exchange ghost cells for cold (ready for next iteration)
        exchangeGhostsZ(cold, nx, ny, nz_local, MPI_COMM_WORLD, rank, size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);

        // Calculate performance using full grid cell count
        const size_t gridSize = nx * ny * nz;
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        std::vector<double> full_grid;
        if (rank == 0) {
            full_grid.resize(nx * ny * nz);
        }

        std::vector<int> recvcounts(size), rdispls(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = counts[i] * static_cast<int>(plane_size);
            rdispls[i] = displs[i] * static_cast<int>(plane_size);
        }

        MPI_Gatherv(&cold[plane_size], static_cast<int>(nz_local * plane_size), MPI_DOUBLE,
                    rank == 0 ? full_grid.data() : nullptr, recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(full_grid, "Concentration");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz_local, MPI_COMM_WORLD, rank);

        int result = 0;
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
                result = 0;
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return result;
    }

    MPI_Finalize();
    return 0;
}
