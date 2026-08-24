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

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz, const size_t z_offset) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t global_idx = idx3(x, y, z + z_offset, nx, ny);
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation with halo exchange
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      const int rank, const int size) {
    // Allocate temporary buffers for halo zones
    std::vector<Real> halo_bottom(nx * ny, 0.0);
    std::vector<Real> halo_top(nx * ny, 0.0);
    
    // Halo exchange with neighboring processes
    MPI_Request requests[4];
    int req_count = 0;
    
    // Exchange with rank+1 (top neighbor)
    if (rank < size - 1) {
        // Send top plane to rank+1
        MPI_Isend(&input[idx3(0, 0, local_nz - 1, nx, ny)], nx * ny, MPI_DOUBLE,
                  rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        // Receive halo from rank+1
        MPI_Irecv(halo_top.data(), nx * ny, MPI_DOUBLE,
                  rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Exchange with rank-1 (bottom neighbor)
    if (rank > 0) {
        // Send bottom plane to rank-1
        MPI_Isend(&input[idx3(0, 0, 0, nx, ny)], nx * ny, MPI_DOUBLE,
                  rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        // Receive halo from rank-1
        MPI_Irecv(halo_bottom.data(), nx * ny, MPI_DOUBLE,
                  rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Wait for all communications to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }
    
    // Process interior points (not on z boundaries)
    for (size_t z = 1; z < local_nz - 1; ++z) {
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
    
    // Handle z=0 plane (bottom boundary of local domain)
    if (rank > 0) {
        // Interior of bottom plane
        size_t z = 0;
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t halo_idx = y * nx + x;
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = halo_bottom[halo_idx];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Handle z=local_nz-1 plane (top boundary of local domain)
    if (rank < size - 1) {
        // Interior of top plane
        size_t z = local_nz - 1;
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t halo_idx = y * nx + x;
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = halo_top[halo_idx];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values for x,y boundaries and global z boundaries
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                bool is_x_boundary = (x == 0 || x == nx-1);
                bool is_y_boundary = (y == 0 || y == ny-1);
                bool is_global_z_boundary = (rank == 0 && z == 0) || (rank == size - 1 && z == local_nz - 1);
                
                if (is_x_boundary || is_y_boundary || is_global_z_boundary) {
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    
    // Domain decomposition along Z dimension
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    
    // Distribute remainder among first ranks
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        local_nz++;
    }
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("MPI Processes: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate local grids (no halo zones needed in the main grid)
    size_t local_gridSize = nx * ny * local_nz;
    std::vector<Real> grid1(local_gridSize);
    std::vector<Real> grid2(local_gridSize);
    
    // Initialize local portion
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, nx, ny, local_nz, z_start);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, local_nz, rank, size);
        } else {
            stencilIteration(grid2, grid1, nx, ny, local_nz, rank, size);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (global_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results from all ranks for validation/output
    const std::vector<Real>& localFinalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> finalGrid;
    
    if (printResults || validate) {
        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
        }
        
        // Gather sizes from all ranks
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        int local_count = nx * ny * local_nz;
        
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        // Gather grid data (excluding halo zones)
        MPI_Gatherv(localFinalGrid.data(), local_count, MPI_DOUBLE,
                    finalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
