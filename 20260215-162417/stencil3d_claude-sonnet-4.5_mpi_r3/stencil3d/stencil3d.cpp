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

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, 
                    const size_t z_offset, const size_t global_nz) {
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z, nx, ny);
                const size_t global_z = z_offset + z;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Halo exchange for ghost cells
void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz,
                   int rank, int size) {
    const size_t plane_size = nx * ny;
    
    // Exchange with upper neighbor (rank + 1)
    if (rank < size - 1) {
        // Send top interior plane, receive into top ghost
        size_t send_z = local_nz - 2;  // Last interior plane
        size_t recv_z = local_nz - 1;  // Top ghost plane
        MPI_Sendrecv(&grid[send_z * plane_size], plane_size, MPI_DOUBLE, rank + 1, 0,
                     &grid[recv_z * plane_size], plane_size, MPI_DOUBLE, rank + 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Exchange with lower neighbor (rank - 1)
    if (rank > 0) {
        // Send bottom interior plane, receive into bottom ghost
        size_t send_z = 1;  // First interior plane
        size_t recv_z = 0;  // Bottom ghost plane
        MPI_Sendrecv(&grid[send_z * plane_size], plane_size, MPI_DOUBLE, rank - 1, 1,
                     &grid[recv_z * plane_size], plane_size, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      int rank, int size, const size_t global_nz) {
    // Determine Z boundaries for this process
    size_t z_start = (rank == 0) ? 1 : 1;  // Always start at 1 (ghost or boundary)
    size_t z_end = (rank == size - 1) ? local_nz - 1 : local_nz - 1;  // Always end at local_nz - 1
    
    // Process interior points (not on boundaries)
    for (size_t z = z_start; z < z_end; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values (X, Y boundaries and global Z boundaries)
    for (size_t z = 0; z < local_nz; ++z) {
        // Determine if this is a global Z boundary
        bool is_global_z_boundary = (rank == 0 && z == 0) || (rank == size - 1 && z == local_nz - 1);
        
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || is_global_z_boundary) {
                    const size_t idx = idx3(x, y, z, nx, ny);
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
    // Initialize MPI
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition along Z axis
    size_t base_local_nz = nz / size;
    size_t remainder = nz % size;
    
    // Calculate local Z size for this rank (distribute remainder to first processes)
    size_t local_nz_interior = base_local_nz + (rank < (int)remainder ? 1 : 0);
    
    // Add ghost layers
    size_t local_nz = local_nz_interior;
    if (rank > 0) local_nz++;  // Add bottom ghost
    if (rank < size - 1) local_nz++;  // Add top ghost
    
    // Calculate Z offset for this rank
    size_t z_offset = rank * base_local_nz + std::min((size_t)rank, remainder);
    
    // Adjust z_offset for ghost layer
    size_t init_z_offset = z_offset;
    if (rank > 0) {
        z_offset--;  // Ghost layer starts one before actual domain
    }
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("MPI Processes: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t local_gridSize = nx * ny * local_nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(local_gridSize);
    std::vector<Real> grid2(local_gridSize);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_nz, z_offset, nz);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchangeHalos(grid1, nx, ny, local_nz, rank, size);
            stencilIteration(grid1, grid2, nx, ny, local_nz, rank, size, nz);
        } else {
            exchangeHalos(grid2, nx, ny, local_nz, rank, size);
            stencilIteration(grid2, grid1, nx, ny, local_nz, rank, size, nz);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for validation and output
    const std::vector<Real>& localFinalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> globalGrid;
    
    if (printResults || validate) {
        // Prepare data for gathering - extract interior points only
        size_t local_interior_start_z = (rank == 0) ? 0 : 1;
        size_t local_interior_end_z = (rank == size - 1) ? local_nz : local_nz - 1;
        size_t local_interior_nz = local_interior_end_z - local_interior_start_z;
        
        std::vector<Real> localInterior(nx * ny * local_interior_nz);
        for (size_t z = 0; z < local_interior_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t local_idx = idx3(x, y, z + local_interior_start_z, nx, ny);
                    size_t interior_idx = idx3(x, y, z, nx, ny);
                    localInterior[interior_idx] = localFinalGrid[local_idx];
                }
            }
        }
        
        // Gather sizes
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        int local_size = nx * ny * local_interior_nz;
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            globalGrid.resize(nx * ny * nz);
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(localInterior.data(), local_size, MPI_DOUBLE,
                    globalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(globalGrid, "Grid");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(globalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
