#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline size_t localNzForRank(const size_t nz, const int size, const int rank) noexcept {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return base + ((static_cast<size_t>(rank) < rem) ? 1 : 0);
}

inline size_t zStartForRank(const size_t nz, const int size, const int rank) noexcept {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

void initializeGrid(std::vector<Real>& grid,
                    const size_t nx,
                    const size_t ny,
                    const size_t local_nz,
                    const size_t z_start) {
    std::fill(grid.begin(), grid.end(), 0.0);
    for (size_t z = 0; z < local_nz; ++z) {
        const size_t global_z = z_start + z;
        const size_t local_z = z + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid[idx3(x, y, local_z, nx, ny)] = static_cast<Real>(global_idx % 19);
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& grid,
                   const size_t nx,
                   const size_t ny,
                   const size_t local_nz,
                   const int prev_rank,
                   const int next_rank,
                   MPI_Comm comm) {
    if (local_nz == 0) {
        return;
    }
    const int plane_count = static_cast<int>(nx * ny);
    MPI_Status status;
    MPI_Sendrecv(grid.data() + idx3(0, 0, local_nz, nx, ny), plane_count, MPI_DOUBLE, next_rank, 0,
                 grid.data() + idx3(0, 0, 0, nx, ny), plane_count, MPI_DOUBLE, prev_rank, 0,
                 comm, &status);
    MPI_Sendrecv(grid.data() + idx3(0, 0, 1, nx, ny), plane_count, MPI_DOUBLE, prev_rank, 1,
                 grid.data() + idx3(0, 0, local_nz + 1, nx, ny), plane_count, MPI_DOUBLE, next_rank, 1,
                 comm, &status);
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx,
                      const size_t ny,
                      const size_t local_nz,
                      const size_t z_start,
                      const size_t global_nz) {
    // Process interior points (not on boundaries)
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_start + (z - 1);
        if (global_z == 0 || global_z + 1 == global_nz) {
            continue;
        }
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);

                const Real center = input[idx];
                const Real left = input[idx3(x - 1, y, z, nx, ny)];
                const Real right = input[idx3(x + 1, y, z, nx, ny)];
                const Real front = input[idx3(x, y - 1, z, nx, ny)];
                const Real back = input[idx3(x, y + 1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z - 1, nx, ny)];
                const Real top = input[idx3(x, y, z + 1, nx, ny)];

                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Copy boundary values
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || global_z == 0 || global_z + 1 == global_nz) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid,
                    const size_t nx,
                    const size_t ny,
                    const size_t local_nz,
                    MPI_Comm comm,
                    const int rank) {
    // Simple sanity checks
    bool local_ok = true;
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();

    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const Real val = grid[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_ok = false;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    int local_ok_int = local_ok ? 1 : 0;
    int global_ok_int = 0;
    MPI_Allreduce(&local_ok_int, &global_ok_int, 1, MPI_INT, MPI_LAND, comm);
    if (!global_ok_int) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    Real global_min = 0.0;
    Real global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 1e6 || global_min < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
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
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argError = false;
    const char* errorOpt = nullptr;
    
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
            showHelp = true;
        } else {
            argError = true;
            errorOpt = argv[i];
            break;
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (argError) {
        if (rank == 0) {
            printf("Unknown option: %s\n", errorOpt);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t local_nz = localNzForRank(nz, size, rank);
    const size_t z_start = zStartForRank(nz, size, rank);
    const int prev_rank = (local_nz == 0 || z_start == 0) ? MPI_PROC_NULL : rank - 1;
    const int next_rank = (local_nz == 0 || (z_start + local_nz) >= nz) ? MPI_PROC_NULL : rank + 1;

    const size_t local_alloc_nz = local_nz + 2;
    const size_t local_size = local_alloc_nz * nx * ny;

    // Allocate grids (double buffering) with ghost layers
    std::vector<Real> grid1(local_size);
    std::vector<Real> grid2(local_size);

    // Initialize
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, nx, ny, local_nz, z_start);

    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchangeHalos(grid1, nx, ny, local_nz, prev_rank, next_rank, MPI_COMM_WORLD);
            stencilIteration(grid1, grid2, nx, ny, local_nz, z_start, nz);
        } else {
            exchangeHalos(grid2, nx, ny, local_nz, prev_rank, next_rank, MPI_COMM_WORLD);
            stencilIteration(grid2, grid1, nx, ny, local_nz, z_start, nz);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = cellUpdates / max_time / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        const size_t local_count = local_nz * nx * ny;
        std::vector<int> recv_counts;
        std::vector<int> displs;
        std::vector<Real> fullGrid;
        if (rank == 0) {
            recv_counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t r_local_nz = localNzForRank(nz, size, r);
                const size_t r_z_start = zStartForRank(nz, size, r);
                recv_counts[r] = static_cast<int>(r_local_nz * nx * ny);
                displs[r] = static_cast<int>(r_z_start * nx * ny);
            }
            fullGrid.resize(nx * ny * nz);
        }
        MPI_Gatherv(finalGrid.data() + idx3(0, 0, 1, nx, ny),
                    static_cast<int>(local_count),
                    MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr,
                    rank == 0 ? recv_counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(fullGrid, "Grid");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(finalGrid, nx, ny, local_nz, MPI_COMM_WORLD, rank);

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
