#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (local grid with ghost layers)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize local portion of grid (including ghost layers from formula)
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t nz_local, const size_t z_start) {
    for (size_t z_local = 1; z_local <= nz_local; ++z_local) {
        const size_t z_global = z_start + z_local - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_local, nx, ny);
                const size_t idx_global = z_global * (nx * ny) + y * nx + x;
                grid[idx] = static_cast<Real>((idx_global % 19) * 1.0);
            }
        }
    }
    // Initialize ghost layers from formula as well (will be overwritten by halo exchange)
    if (z_start > 0) {
        const size_t z_ghost = z_start - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, 0, nx, ny);
                const size_t idx_global = z_ghost * (nx * ny) + y * nx + x;
                grid[idx] = static_cast<Real>((idx_global % 19) * 1.0);
            }
        }
    }
    // Ghost above is initialized too, but we don't know the global z index
    // without knowing nz_global. Set to zero; will be overwritten by halo exchange.
}

// Exchange ghost layers using non-blocking MPI
void exchangeGhostLayers(std::vector<Real>& grid, const size_t nx, const size_t ny,
                          const size_t nz_local, const int rank, const int size) {
    if (nz_local == 0) return;
    const int count = static_cast<int>(nx * ny);

    MPI_Request reqs[4];
    int nreqs = 0;

    if (rank > 0) {
        MPI_Irecv(&grid[idx3(0, 0, 0, nx, ny)], count, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (rank < size - 1) {
        MPI_Irecv(&grid[idx3(0, 0, nz_local + 1, nx, ny)], count, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (rank > 0) {
        MPI_Isend(&grid[idx3(0, 0, 1, nx, ny)], count, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (rank < size - 1) {
        MPI_Isend(&grid[idx3(0, 0, nz_local, nx, ny)], count, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }

    if (nreqs > 0) {
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    }
}

// 7-point stencil computation on local slab (includes ghost layers)
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz_local,
                      const size_t z_start, const size_t nz_global) {
    // Copy X and Y boundary cells for all local Z layers
    for (size_t z = 1; z <= nz_local; ++z) {
        // x = 0 and x = nx - 1
        for (size_t y = 0; y < ny; ++y) {
            const size_t idx_x0 = idx3(0, y, z, nx, ny);
            const size_t idx_xn = idx3(nx - 1, y, z, nx, ny);
            output[idx_x0] = input[idx_x0];
            output[idx_xn] = input[idx_xn];
        }
        // y = 0 and y = ny - 1 (excluding corners, already handled)
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t idx_y0 = idx3(x, 0, z, nx, ny);
            const size_t idx_yn = idx3(x, ny - 1, z, nx, ny);
            output[idx_y0] = input[idx_y0];
            output[idx_yn] = input[idx_yn];
        }
    }

    // Copy Z boundary cells (only for ranks at global Z boundaries)
    if (z_start == 0) {
        // Global z = 0 boundary
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, 1, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
    if (z_start + nz_local == nz_global) {
        // Global z = nz_global - 1 boundary
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, nz_local, nx, ny);
                output[idx] = input[idx];
            }
        }
    }

    // Determine iteration range skipping global Z boundaries
    size_t z_start_local = 1;
    size_t z_end_local = nz_local;
    if (z_start == 0)          z_start_local = 2;       // skip z=0 face
    if (z_start + nz_local == nz_global) z_end_local = nz_local - 1; // skip z=nz-1 face

    // Interior stencil computation
    for (size_t z = z_start_local; z <= z_end_local; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);

                const Real center = input[idx];
                const Real left   = input[idx3(x - 1, y,     z,     nx, ny)];
                const Real right  = input[idx3(x + 1, y,     z,     nx, ny)];
                const Real front  = input[idx3(x,     y - 1, z,     nx, ny)];
                const Real back   = input[idx3(x,     y + 1, z,     nx, ny)];
                const Real bottom = input[idx3(x,     y,     z - 1, nx, ny)];
                const Real top    = input[idx3(x,     y,     z + 1, nx, ny)];

                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / static_cast<Real>(7.0);
            }
        }
    }
}

// Validate results across all MPI ranks
bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t nz_local, [[maybe_unused]] const size_t z_start,
                    [[maybe_unused]] const size_t nz_global, const int rank) {
    // 1. Check local portion for NaN or Inf
    int local_nan_inf = 0;
    const size_t local_total = nx * ny * (nz_local + 2);
    for (size_t i = 0; i < local_total; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            local_nan_inf = 1;
            break;
        }
    }

    // 2. Compute local min/max (non-ghost portion only)
    Real local_min = grid[idx3(0, 0, 1, nx, ny)];
    Real local_max = grid[idx3(0, 0, 1, nx, ny)];
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                local_min = std::min(local_min, grid[idx]);
                local_max = std::max(local_max, grid[idx]);
            }
        }
    }

    // Reduce across all ranks
    int global_nan_inf;
    MPI_Allreduce(&local_nan_inf, &global_nan_inf, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    Real global_min, global_max;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_nan_inf) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);

        if (global_max > 1e6 || global_min < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
        if (global_nan_inf) {
            return false;
        }
        return true;
    }
    return true; // non-root ranks always return true; only rank 0 matters
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse the same input)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
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

    // Compute local Z slab sizes
    const size_t nz_per_rank = nz / static_cast<size_t>(size);
    const size_t extra = nz % static_cast<size_t>(size);
    const size_t nz_local = nz_per_rank + (static_cast<size_t>(rank) < extra ? 1 : 0);

    // Compute global Z start offset for this rank
    size_t z_start = 0;
    for (int r = 0; r < rank; ++r) {
        z_start += nz_per_rank + (static_cast<size_t>(r) < extra ? 1 : 0);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local grids with ghost layers
    // Local Z dimension = nz_local + 2 (ghost above and below)
    const size_t local_slab_size = nz_local + 2;
    const size_t local_grid_size = nx * ny * local_slab_size;

    std::vector<Real> grid1(local_grid_size, 0.0);
    std::vector<Real> grid2(local_grid_size, 0.0);

    // Initialize each rank's local portion
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz_local, z_start);
    // grid2 will be filled by stencil; initialize ghost area properly for first exchange
    initializeGrid(grid2, nx, ny, nz_local, z_start);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchangeGhostLayers(grid1, nx, ny, nz_local, rank, size);
            stencilIteration(grid1, grid2, nx, ny, nz_local, z_start, nz);
        } else {
            exchangeGhostLayers(grid2, nx, ny, nz_local, rank, size);
            stencilIteration(grid2, grid1, nx, ny, nz_local, z_start, nz);
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = max_time * 1000.0;
        printf("Computation time: %.0f ms\n", duration_ms);

        // Calculate performance metrics
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * static_cast<double>(iterations);
        const double mcups = cellUpdates / max_time / 1.0e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Determine final grid
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    // Print results for external validation (gather full grid on rank 0)
    if (printResults) {
        // Gather counts from all ranks
        std::vector<int> recvcounts(static_cast<size_t>(size), 0);
        std::vector<int> displs(static_cast<size_t>(size), 0);
        const int my_count = static_cast<int>(nx * ny * nz_local);

        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                displs[r] = displs[r - 1] + recvcounts[r - 1];
            }
        }

        // Pack non-ghost data into contiguous send buffer
        std::vector<Real> send_buf(static_cast<size_t>(my_count));
        for (size_t z = 0; z < nz_local; ++z) {
            const Real* src = &finalGrid[idx3(0, 0, z + 1, nx, ny)];
            Real* dst = &send_buf[z * nx * ny];
            std::copy(src, src + nx * ny, dst);
        }

        std::vector<Real> full_grid;
        if (rank == 0) {
            full_grid.resize(nx * ny * nz);
        }

        MPI_Gatherv(send_buf.data(), my_count, MPI_DOUBLE,
                    full_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(full_grid, "Grid");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz_local, z_start, nz, rank);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        if (rank == 0) return valid ? 0 : 1;
        return 0;
    }

    MPI_Finalize();
    return 0;
}
