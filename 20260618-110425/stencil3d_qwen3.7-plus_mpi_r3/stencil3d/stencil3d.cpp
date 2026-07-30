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

// 3D index calculation for local grid (with ghost planes)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize owned planes using global indices; zero ghost planes
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t local_nz, const size_t z_start) {
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = gz * (nx * ny) + y * nx + x;
                grid[idx3(x, y, lz, nx, ny)] = (global_idx % 19) * 1.0;
            }
        }
    }
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            grid[idx3(x, y, 0, nx, ny)] = 0.0;
            grid[idx3(x, y, local_nz + 1, nx, ny)] = 0.0;
        }
    }
}

// Halo exchange along Z-axis using MPI_Sendrecv
// Convention: rank_below has lower global Z, rank_above has higher global Z.
// Ghost at local_z=0 needs data from rank_below (lower Z neighbor).
// Ghost at local_z=local_nz+1 needs data from rank_above (higher Z neighbor).
//
// Tag protocol for exchange between rank r and rank r+1:
//   r:   Sendrecv(send local_nz to r+1 tag=0, recv local_nz+1 from r+1 tag=1)
//   r+1: Sendrecv(send 1 to r tag=1, recv 0 from r tag=0)
void exchangeHalos(std::vector<Real>& grid,
                   const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank_below, const int rank_above, MPI_Comm comm) {
    const int plane_size = static_cast<int>(nx * ny);

    if (rank_above >= 0) {
        // Exchange with rank_above: send top owned, receive top ghost
        MPI_Sendrecv(grid.data() + idx3(0, 0, local_nz, nx, ny), plane_size, MPI_DOUBLE, rank_above, 0,
                     grid.data() + idx3(0, 0, local_nz + 1, nx, ny), plane_size, MPI_DOUBLE, rank_above, 1,
                     comm, MPI_STATUS_IGNORE);
    }
    if (rank_below >= 0) {
        // Exchange with rank_below: send bottom owned, receive bottom ghost
        MPI_Sendrecv(grid.data() + idx3(0, 0, 1, nx, ny), plane_size, MPI_DOUBLE, rank_below, 1,
                     grid.data() + idx3(0, 0, 0, nx, ny), plane_size, MPI_DOUBLE, rank_below, 0,
                     comm, MPI_STATUS_IGNORE);
    }
}

// 7-point stencil computation with halo exchange
void stencilIteration(std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      const size_t z_start, const size_t global_nz,
                      const int rank_below, const int rank_above, MPI_Comm comm) {
    // Exchange halos on input before computation
    exchangeHalos(input, nx, ny, local_nz, rank_below, rank_above, comm);

    // Process each local Z plane
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + lz - 1;

        // Global Z boundary planes: just copy (they are boundary in the original code)
        if (gz == 0 || gz == global_nz - 1) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    output[idx3(x, y, lz, nx, ny)] = input[idx3(x, y, lz, nx, ny)];
                }
            }
            continue;
        }

        // Y boundary rows (y=0 and y=ny-1): copy
        for (size_t x = 0; x < nx; ++x) {
            output[idx3(x, 0, lz, nx, ny)] = input[idx3(x, 0, lz, nx, ny)];
            output[idx3(x, ny - 1, lz, nx, ny)] = input[idx3(x, ny - 1, lz, nx, ny)];
        }

        // Interior Y: compute stencil for interior X
        for (size_t y = 1; y < ny - 1; ++y) {
            // X boundary: copy
            output[idx3(0, y, lz, nx, ny)] = input[idx3(0, y, lz, nx, ny)];
            output[idx3(nx - 1, y, lz, nx, ny)] = input[idx3(nx - 1, y, lz, nx, ny)];

            // Interior X: stencil
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);

                const Real center = input[idx];
                const Real left   = input[idx3(x - 1, y, lz, nx, ny)];
                const Real right  = input[idx3(x + 1, y, lz, nx, ny)];
                const Real front  = input[idx3(x, y - 1, lz, nx, ny)];
                const Real back   = input[idx3(x, y + 1, lz, nx, ny)];
                const Real bottom = input[idx3(x, y, lz - 1, nx, ny)];
                const Real top    = input[idx3(x, y, lz + 1, nx, ny)];

                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
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
        printf("3D Stencil Benchmark (MPI, %d ranks)\n", nprocs);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Z-axis domain decomposition
    const size_t base_nz = nz / nprocs;
    const size_t rem = nz % nprocs;
    const size_t local_nz = (static_cast<size_t>(rank) < rem) ? base_nz + 1 : base_nz;
    size_t z_start;
    if (static_cast<size_t>(rank) < rem) {
        z_start = static_cast<size_t>(rank) * (base_nz + 1);
    } else {
        z_start = rem * (base_nz + 1) + (static_cast<size_t>(rank) - rem) * base_nz;
    }

    const int rank_below = (rank > 0) ? rank - 1 : -1;
    const int rank_above = (rank < nprocs - 1) ? rank + 1 : -1;

    // Local grid: nx * ny * (local_nz + 2) -- ghost planes at z=0 and z=local_nz+1
    const size_t local_grid_size = nx * ny * (local_nz + 2);
    std::vector<Real> grid1(local_grid_size, 0.0);
    std::vector<Real> grid2(local_grid_size, 0.0);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_nz, z_start);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, local_nz, z_start, nz, rank_below, rank_above, MPI_COMM_WORLD);
        } else {
            stencilIteration(grid2, grid1, nx, ny, local_nz, z_start, nz, rank_below, rank_above, MPI_COMM_WORLD);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long duration_ms_local = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms;
    MPI_Reduce(&duration_ms_local, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for validation/printing
    std::vector<Real> fullGrid;
    if (printResults || validate) {
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
        }

        // Determine which local buffer holds the final result
        // After iterations: if iterations is even, result is in grid1; if odd, in grid2
        const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

        // Build recvcounts and displs on rank 0
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            const size_t r_local_nz = (static_cast<size_t>(r) < rem) ? base_nz + 1 : base_nz;
            recvcounts[r] = static_cast<int>(r_local_nz * nx * ny);
            size_t r_z_start;
            if (static_cast<size_t>(r) < rem) {
                r_z_start = static_cast<size_t>(r) * (base_nz + 1);
            } else {
                r_z_start = rem * (base_nz + 1) + (static_cast<size_t>(r) - rem) * base_nz;
            }
            displs[r] = static_cast<int>(r_z_start * nx * ny);
        }

        // Each rank sends its owned planes (local_z 1..local_nz), which are contiguous
        const int send_count = static_cast<int>(local_nz * nx * ny);
        MPI_Gatherv(finalGrid.data() + nx * ny, // skip ghost plane at local_z=0
                    send_count, MPI_DOUBLE,
                    fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results and validate on rank 0
    if (rank == 0) {
        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
