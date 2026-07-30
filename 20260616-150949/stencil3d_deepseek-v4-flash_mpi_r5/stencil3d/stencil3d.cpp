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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize owned cells of the local grid (ghost cells are filled via MPI)
// Uses global Z index for deterministic initial values matching serial version
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t local_nz, const size_t z_start) {
    for (size_t local_z = 1; local_z <= local_nz; ++local_z) {
        const size_t global_z = z_start + local_z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                const size_t local_idx = idx3(x, y, local_z, nx, ny);
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation with MPI ghost cell exchange.
// Uses 1D domain decomposition along Z with non-blocking communication
// overlapped with computation for maximum performance.
void stencilIteration(std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny,
                      const size_t local_nz, const size_t z_start,
                      const size_t nz, const int rank, const int nprocs) {
    // Start non-blocking halo exchange for the input grid
    MPI_Request requests[4];
    int nreq = 0;

    // Receive ghost cell for bottom plane (local_z = 0) from rank-1
    if (rank > 0) {
        MPI_Irecv(&input[idx3(0, 0, 0, nx, ny)], nx * ny, MPI_DOUBLE,
                  rank - 1, 0, MPI_COMM_WORLD, &requests[nreq++]);
    }
    // Receive ghost cell for top plane (local_z = local_nz + 1) from rank+1
    if (rank < nprocs - 1) {
        MPI_Irecv(&input[idx3(0, 0, local_nz + 1, nx, ny)], nx * ny, MPI_DOUBLE,
                  rank + 1, 1, MPI_COMM_WORLD, &requests[nreq++]);
    }
    // Send owned bottom plane (local_z = 1) to rank-1
    if (rank > 0) {
        MPI_Isend(&input[idx3(0, 0, 1, nx, ny)], nx * ny, MPI_DOUBLE,
                  rank - 1, 1, MPI_COMM_WORLD, &requests[nreq++]);
    }
    // Send owned top plane (local_z = local_nz) to rank+1
    if (rank < nprocs - 1) {
        MPI_Isend(&input[idx3(0, 0, local_nz, nx, ny)], nx * ny, MPI_DOUBLE,
                  rank + 1, 0, MPI_COMM_WORLD, &requests[nreq++]);
    }

    // Compute interior Z planes (local_z = 2 .. local_nz-1) which do NOT
    // depend on ghost cells — overlapped with MPI communication
    for (size_t local_z = 2; local_z <= local_nz - 1; ++local_z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, local_z, nx, ny);
                const Real center = input[idx];
                const Real left   = input[idx3(x - 1, y,     local_z,     nx, ny)];
                const Real right  = input[idx3(x + 1, y,     local_z,     nx, ny)];
                const Real front  = input[idx3(x,     y - 1, local_z,     nx, ny)];
                const Real back   = input[idx3(x,     y + 1, local_z,     nx, ny)];
                const Real bottom = input[idx3(x,     y,     local_z - 1, nx, ny)];
                const Real top    = input[idx3(x,     y,     local_z + 1, nx, ny)];
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Wait for halo exchange to complete
    MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

    // Compute first owned plane (local_z = 1) if not a global Z boundary
    if (z_start != 0) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, 1, nx, ny);
                const Real center = input[idx];
                const Real left   = input[idx3(x - 1, y,     1, nx, ny)];
                const Real right  = input[idx3(x + 1, y,     1, nx, ny)];
                const Real front  = input[idx3(x,     y - 1, 1, nx, ny)];
                const Real back   = input[idx3(x,     y + 1, 1, nx, ny)];
                const Real bottom = input[idx3(x,     y,     0, nx, ny)]; // ghost cell
                const Real top    = input[idx3(x,     y,     2, nx, ny)];
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Compute last owned plane (local_z = local_nz) if not a global Z boundary
    if (z_start + local_nz - 1 != nz - 1) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, local_nz, nx, ny);
                const Real center = input[idx];
                const Real left   = input[idx3(x - 1, y,         local_nz,     nx, ny)];
                const Real right  = input[idx3(x + 1, y,         local_nz,     nx, ny)];
                const Real front  = input[idx3(x,     y - 1,     local_nz,     nx, ny)];
                const Real back   = input[idx3(x,     y + 1,     local_nz,     nx, ny)];
                const Real bottom = input[idx3(x,     y,         local_nz - 1, nx, ny)];
                const Real top    = input[idx3(x,     y,         local_nz + 1, nx, ny)]; // ghost cell
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Copy boundary values (preserves original semantics: Dirichlet BCs)
    for (size_t local_z = 1; local_z <= local_nz; ++local_z) {
        const size_t global_z = z_start + local_z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                    global_z == 0 || global_z == nz - 1) {
                    const size_t idx = idx3(x, y, local_z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
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

    // Parse command line arguments (all ranks participate)
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

    // 1D domain decomposition along Z with load balancing
    const size_t base_nz = nz / nprocs;
    const size_t remainder_nz = nz % nprocs;
    const size_t local_nz = base_nz + (rank < static_cast<int>(remainder_nz) ? 1 : 0);
    const size_t z_start = rank * base_nz + std::min(rank, static_cast<int>(remainder_nz));

    if (local_nz == 0) {
        if (rank == 0) {
            printf("Error: Z dimension (%zu) too small for %d processes\n", nz, nprocs);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d\n", nprocs);
        printf("Local Z planes per rank: %zu", local_nz);
        for (int p = 1; p < nprocs; ++p) {
            const size_t p_local_nz = base_nz + (p < static_cast<int>(remainder_nz) ? 1 : 0);
            printf("/%zu", p_local_nz);
        }
        printf("\n");
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local grids with ghost cells (+2 for halo exchange)
    const size_t local_grid_size = nx * ny * (local_nz + 2);
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);

    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_nz, z_start);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, local_nz, z_start, nz, rank, nprocs);
        } else {
            stencilIteration(grid2, grid1, nx, ny, local_nz, z_start, nz, rank, nprocs);
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long duration_ms = duration.count();

    // Use the maximum wall time across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Prepare gathers for result output / validation
    const std::vector<Real>& finalGridLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // Compute gather counts and displacements for all ranks
    std::vector<int> recvcounts(static_cast<size_t>(nprocs));
    std::vector<int> displs(static_cast<size_t>(nprocs));
    for (int p = 0; p < nprocs; ++p) {
        const size_t p_local_nz = base_nz + (p < static_cast<int>(remainder_nz) ? 1 : 0);
        recvcounts[p] = static_cast<int>(nx * ny * p_local_nz);
        displs[p] = static_cast<int>(nx * ny * (p * base_nz + std::min(p, static_cast<int>(remainder_nz))));
    }

    std::vector<Real> fullGrid;
    if (rank == 0) {
        fullGrid.resize(nx * ny * nz);
    }

    MPI_Gatherv(&finalGridLocal[idx3(0, 0, 1, nx, ny)],
                static_cast<int>(nx * ny * local_nz), MPI_DOUBLE,
                rank == 0 ? fullGrid.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullGrid, nx, ny, nz);
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
