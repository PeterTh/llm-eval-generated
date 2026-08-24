#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Local 3D index: lz includes ghost layers (0 = bottom ghost, 1..local_nz = owned, local_nz+1 = top ghost)
inline constexpr size_t local_idx3(const size_t x, const size_t y, const size_t lz,
                                    const size_t nx, const size_t ny) noexcept {
    return lz * (nx * ny) + y * nx + x;
}

// Exchange ghost cell slices between neighboring ranks along Z
void exchangeGhostCells(std::vector<double>& data, const size_t nx, const size_t ny,
                        const size_t local_nz, const int rank_below, const int rank_above,
                        MPI_Comm comm) {
    const int slice_size = static_cast<int>(nx * ny);
    double* buf = data.data();

    // Send bottom owned (lz=1) to rank_below, receive top ghost (lz=local_nz+1) from rank_above
    MPI_Sendrecv(buf + (size_t)slice_size, slice_size, MPI_DOUBLE, rank_below, 0,
                 buf + (local_nz + 1) * (size_t)slice_size, slice_size, MPI_DOUBLE, rank_above, 0,
                 comm, MPI_STATUS_IGNORE);
    // Send top owned (lz=local_nz) to rank_above, receive bottom ghost (lz=0) from rank_below
    MPI_Sendrecv(buf + local_nz * (size_t)slice_size, slice_size, MPI_DOUBLE, rank_above, 1,
                 buf, slice_size, MPI_DOUBLE, rank_below, 1,
                 comm, MPI_STATUS_IGNORE);

    // Clamped (Neumann-zero) physical boundary conditions
    if (rank_below == MPI_PROC_NULL) {
        std::memcpy(buf, buf + slice_size, (size_t)slice_size * sizeof(double));
    }
    if (rank_above == MPI_PROC_NULL) {
        std::memcpy(buf + (local_nz + 1) * (size_t)slice_size,
                    buf + local_nz * (size_t)slice_size, (size_t)slice_size * sizeof(double));
    }
}

// Compute Laplacian using local indexing; ghost cells must already be populated
inline double computeLaplacian(const std::vector<double>& c,
                               const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t lz) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t idx_c = local_idx3(x, y, lz, nx, ny);
    const double cv = c[idx_c];

    const double cxx = (c[local_idx3(xp, y, lz, nx, ny)] + c[local_idx3(xn, y, lz, nx, ny)]
                        - 2.0 * cv) / (dx * dx);
    const double cyy = (c[local_idx3(x, yp, lz, nx, ny)] + c[local_idx3(x, yn, lz, nx, ny)]
                        - 2.0 * cv) / (dy * dy);
    const double czz = (c[local_idx3(x, y, lz + 1, nx, ny)] + c[local_idx3(x, y, lz - 1, nx, ny)]
                        - 2.0 * cv) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential over owned region
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA,
                              const double e_BB, const double e_AB) {
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = local_idx3(x, y, lz, nx, ny);
                const double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Cahn-Hilliard update over owned region
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt,
                        const double dx, const double dy, const double dz) {
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = local_idx3(x, y, lz, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Initialize concentration using global indices for reproducibility
void initializeConcentration(std::vector<double>& c,
                             const size_t nx, const size_t ny, const size_t local_nz,
                             const size_t z_start, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol)
                                       / static_cast<double>(vol));
                c[local_idx3(x, y, lz, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minVal = c[0], maxVal = c[0];
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

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

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
            printUsage(argv[0]);
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

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D domain decomposition along Z
    const size_t base_nz = nz / nprocs;
    const size_t remainder = nz % nprocs;
    const size_t local_nz = base_nz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t z_start = static_cast<size_t>(rank) * base_nz
                         + std::min(static_cast<size_t>(rank), remainder);

    const int rank_below = (rank > 0)          ? rank - 1     : MPI_PROC_NULL;
    const int rank_above = (rank < nprocs - 1) ? rank + 1     : MPI_PROC_NULL;

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    // Allocate local arrays with one ghost layer on each Z face
    const size_t local_vol = nx * ny * (local_nz + 2);
    std::vector<double> cold(local_vol, 0.0);
    std::vector<double> cnew(local_vol, 0.0);
    std::vector<double> mu(local_vol, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, local_nz, z_start, nz);

    // Fill initial ghost cells
    exchangeGhostCells(cold, nx, ny, local_nz, rank_below, rank_above, MPI_COMM_WORLD);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB);
        exchangeGhostCells(mu, nx, ny, local_nz, rank_below, rank_above, MPI_COMM_WORLD);

        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
        exchangeGhostCells(cold, nx, ny, local_nz, rank_below, rank_above, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_dur_ms = static_cast<long>(duration.count());
    long dur_ms = 0;
    MPI_Reduce(&local_dur_ms, &dur_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather full result to rank 0 for output / validation
    const size_t global_size = nx * ny * nz;
    std::vector<double> global_c;
    if (rank == 0) global_c.resize(global_size);

    std::vector<int> recv_counts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t r_nz = base_nz + (static_cast<size_t>(r) < remainder ? 1 : 0);
        size_t r_zs = static_cast<size_t>(r) * base_nz
                    + std::min(static_cast<size_t>(r), remainder);
        recv_counts[r] = static_cast<int>(r_nz * nx * ny);
        displs[r]      = static_cast<int>(r_zs * nx * ny);
    }

    const int send_count = static_cast<int>(local_nz * nx * ny);
    double* send_buf = cold.data() + nx * ny; // skip bottom ghost (lz=0)

    MPI_Gatherv(send_buf, send_count, MPI_DOUBLE,
                rank == 0 ? global_c.data() : nullptr,
                recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", dur_ms);

        double cellUpdates = (double)global_size * iterations;
        double mcups = cellUpdates / (dur_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(global_c, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_c);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
