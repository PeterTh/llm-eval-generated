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
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize the local portion of the grid (with ghost cell layering)
void initializeGridLocal(std::vector<Real>& grid, const size_t nx, const size_t ny,
                          const size_t my_nz, const size_t z_start) {
    const size_t slice = nx * ny;
    for (size_t z_local = 1; z_local <= my_nz; ++z_local) {
        const size_t global_z = z_start + z_local - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                const size_t local_idx = z_local * slice + y * nx + x;
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation with MPI ghost cell exchange
// domain decomposition along Z, with overlapping communication and computation
void stencilIterationMPI(std::vector<Real>& input,
                         std::vector<Real>& output,
                         const size_t nx, const size_t ny,
                         const size_t my_nz, const size_t z_start,
                         const size_t nz_global,
                         const int rank, const int size) {
    const size_t slice = nx * ny;
    
    // --- Phase 1: Start non-blocking ghost cell exchange ---
    MPI_Request reqs[4];
    int nreqs = 0;
    if (rank > 0) {
        // Receive into bottom ghost (z_local = 0) from rank-1
        MPI_Irecv(&input[0],               slice, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        // Send bottom interior layer (z_local = 1) to rank-1
        MPI_Isend(&input[slice],            slice, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (rank < size - 1) {
        // Receive into top ghost (z_local = my_nz + 1) from rank+1
        MPI_Irecv(&input[(my_nz + 1) * slice], slice, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        // Send top interior layer (z_local = my_nz) to rank+1
        MPI_Isend(&input[my_nz * slice],       slice, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    
    // --- Phase 2: Compute interior bulk (doesn't need ghost cells from other ranks) ---
    // z_local in [2, my_nz-1] — these layers have all neighbors locally
    for (size_t z_local = 2; z_local < my_nz; ++z_local) {
        const size_t global_z = z_start + z_local - 1;
        if (global_z == 0 || global_z == nz_global - 1) continue;
        
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = z_local * slice + y * nx + x;
                output[idx] = (input[idx] +
                              input[idx - slice] + input[idx + slice] +
                              input[idx - nx]    + input[idx + nx] +
                              input[idx - 1]     + input[idx + 1]) / 7.0;
            }
        }
    }
    
    // --- Phase 3: Wait for ghost exchange ---
    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    
    // --- Phase 4: Compute near-ghost layers (need ghost cells) ---
    // Bottom near-ghost layer (z_local = 1)
    if (my_nz >= 1) {
        const size_t global_z = z_start;  // z_start + 1 - 1 = z_start
        if (global_z != 0 && global_z != nz_global - 1) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = slice + y * nx + x;
                    output[idx] = (input[idx] +
                                  input[idx - slice] + input[idx + slice] +
                                  input[idx - nx]    + input[idx + nx] +
                                  input[idx - 1]     + input[idx + 1]) / 7.0;
                }
            }
        }
    }
    // Top near-ghost layer (z_local = my_nz), only if distinct from bottom
    if (my_nz > 1) {
        const size_t global_z = z_start + my_nz - 1;
        if (global_z != 0 && global_z != nz_global - 1) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = my_nz * slice + y * nx + x;
                    output[idx] = (input[idx] +
                                  input[idx - slice] + input[idx + slice] +
                                  input[idx - nx]    + input[idx + nx] +
                                  input[idx - 1]     + input[idx + 1]) / 7.0;
                }
            }
        }
    }
    
    // --- Phase 5: Copy boundary values ---
    for (size_t z_local = 1; z_local <= my_nz; ++z_local) {
        const size_t global_z = z_start + z_local - 1;
        const bool z_boundary = (global_z == 0 || global_z == nz_global - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z_boundary) {
                    const size_t idx = z_local * slice + y * nx + x;
                    output[idx] = input[idx];
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
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
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse independently)
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
    
    // Each rank must have at least one Z layer for valid ghost exchange
    if ((size_t)size > nz) {
        if (rank == 0) {
            printf("Error: Z dimension (%zu) must be >= number of MPI processes (%d)\n", nz, size);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Domain decomposition along Z dimension
    const size_t local_nz = nz / (size_t)size;
    const size_t remainder = nz % (size_t)size;
    const size_t my_nz = local_nz + ((size_t)rank < remainder ? 1 : 0);
    
    // Compute global Z start offset for this rank
    size_t z_start = 0;
    for (int i = 0; i < rank; ++i) {
        z_start += local_nz + ((size_t)i < remainder ? 1 : 0);
    }
    
    const size_t local_grid_size = nx * ny * (my_nz + 2);  // +2 for ghost cells
    
    // Allocate local grids (double buffering) with ghost cells
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);
    
    // Initialize only the owned portion of grid1 (grid2 will be fully overwritten)
    if (rank == 0) printf("Initializing grid...\n");
    initializeGridLocal(grid1, nx, ny, my_nz, z_start);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationMPI(grid1, grid2, nx, ny, my_nz, z_start, nz, rank, size);
        } else {
            stencilIterationMPI(grid2, grid1, nx, ny, my_nz, z_start, nz, rank, size);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcpus = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcpus);
    }
    
    // Gather results to rank 0 for validation and output
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    const int my_nz_int = (int)my_nz;
    const int slice_int = (int)(nx * ny);
    
    std::vector<int> all_nz(size);
    MPI_Gather(&my_nz_int, 1, MPI_INT, all_nz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<Real> global_grid;
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        global_grid.resize(nx * ny * nz);
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = all_nz[i] * slice_int;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }
    
    // Send buffer starts at z_local = 1 (skip bottom ghost cell)
    MPI_Gatherv(&finalGrid[nx * ny], my_nz_int * slice_int, MPI_DOUBLE,
                rank == 0 ? global_grid.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        print_results(global_grid, "Grid");
    }
    
    // Validation (rank 0 validates, then broadcasts result)
    int global_ret = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(global_grid, nx, ny, nz);
            global_ret = valid ? 0 : 1;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&global_ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return global_ret;
}
