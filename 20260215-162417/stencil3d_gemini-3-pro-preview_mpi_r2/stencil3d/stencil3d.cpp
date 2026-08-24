#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t start_z) {
    // Fill local grid including ghosts with 0 or similar to avoid uninitialized access
    // But we mainly care about the interior [1, local_nz]
    // The ghosts will be filled by exchange or boundary conditions
    std::fill(grid.begin(), grid.end(), 0.0);

    for (size_t z = 1; z <= local_nz; ++z) {
        size_t global_z = start_z + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z, nx, ny);
                const size_t global_idx = global_z * (nx * ny) + y * nx + x;
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      int rank, int size) {
    
    // Exchange ghost layers
    MPI_Request reqs[4];
    MPI_Status stats[4];
    int nreqs = 0;
    
    size_t plane_size = nx * ny;
    
    // Only exchange if we have at least one plane of data
    if (local_nz > 0) {
        // Send to top (rank+1), receive from bottom (rank-1)
        if (rank < size - 1) {
            MPI_Isend(&input[idx3(0, 0, local_nz, nx, ny)], plane_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank > 0) {
            MPI_Irecv(&input[idx3(0, 0, 0, nx, ny)], plane_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        
        // Send to bottom (rank-1), receive from top (rank+1)
        if (rank > 0) {
            MPI_Isend(&input[idx3(0, 0, 1, nx, ny)], plane_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank < size - 1) {
            MPI_Irecv(&input[idx3(0, 0, local_nz + 1, nx, ny)], plane_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
    }
    
    // Compute interior points that do not depend on ghost layers
    // z range [2, local_nz-1]
    if (local_nz >= 3) {
        for (size_t z = 2; z < local_nz; ++z) {
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
                    
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }

    MPI_Waitall(nreqs, reqs, stats);

    // Compute boundary points that depend on ghost layers
    // z=1 and z=local_nz (if local_nz >= 1)
    
    std::vector<size_t> boundary_z;
    if (local_nz >= 1) boundary_z.push_back(1);
    if (local_nz >= 2) boundary_z.push_back(local_nz);
    
    for (size_t z : boundary_z) {
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
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
    // Global boundaries are at rank 0 (z=1 is global 0) and rank size-1 (z=local_nz is global nz-1)
    // Actually, based on original code logic:
    // Original loop: z from 0 to nz-1. Interior 1 to nz-2.
    // Boundary copy: z=0, z=nz-1, y=0, y=ny-1, x=0, x=nx-1.
    
    // For MPI:
    // If rank == 0, local z=1 corresponds to global z=0. This slice should be copied from input to output.
    // If rank == size-1, local z=local_nz corresponds to global z=nz-1. This slice should be copied.
    
    // Also X and Y boundaries for all local Z layers.
    
    for (size_t z = 1; z <= local_nz; ++z) {
        bool is_global_z_boundary = false;
        if (rank == 0 && z == 1) is_global_z_boundary = true;
        if (rank == size - 1 && z == local_nz) is_global_z_boundary = true;
        
        if (is_global_z_boundary) {
             for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
             }
             continue; // Entire slice is boundary
        }

        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    // Check local part [1, local_nz]
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
             for (size_t x = 0; x < nx; ++x) {
                 const auto& val = grid[idx3(x, y, z, nx, ny)];
                 if (std::isnan(val) || std::isinf(val)) {
                    printf("Validation failed: found NaN or Inf value\n");
                    return false;
                }
             }
        }
    }
    
    // 2. Values should be reasonable (bounded)
    // Need global min/max
    Real localMin = 1e30; // Large value
    Real localMax = -1e30; // Small value
    
    if (local_nz > 0) {
        localMin = grid[idx3(0,0,1,nx,ny)];
        localMax = grid[idx3(0,0,1,nx,ny)];
        
        for (size_t z = 1; z <= local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const auto& val = grid[idx3(x, y, z, nx, ny)];
                    localMin = std::min(localMin, val);
                    localMax = std::max(localMax, val);
                }
            }
        }
    }
    
    Real globalMin, globalMax;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        
        // After averaging, values should be somewhat bounded
        if (globalMax > 1e6 || globalMin < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
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
    printf("  -r           Print results for external validation (only rank 0)\n");
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
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain decomposition
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t start_z = 0;
    
    if (rank < (int)remainder) {
        local_nz++;
        start_z = rank * local_nz;
    } else {
        start_z = remainder * (local_nz + 1) + (rank - remainder) * local_nz;
    }
    
    // Allocate grids (double buffering)
    // Add 2 ghost layers (top and bottom)
    size_t local_grid_size = nx * ny * (local_nz + 2);
    
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_nz, start_z);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, local_nz, rank, size);
        } else {
            stencilIteration(grid2, grid1, nx, ny, local_nz, rank, size);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double localDuration = std::chrono::duration<double, std::milli>(end - start).count();
    double duration;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration);
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    const std::vector<Real>& finalLocalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    // Print results for external validation
    if (printResults) {
        // Gather to rank 0 for printing
        // This is complex because local sizes might differ
        // For simplicity, handle only if size=1 or just print local part
        // Or implement gather logic
        
        // Let's gather all data to rank 0
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = local_nz * nx * ny;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        std::vector<Real> globalGrid;
        if (rank == 0) globalGrid.resize(nx * ny * nz);
        
        // Extract inner part of local grid to send buffer
        std::vector<Real> sendBuf(my_count);
        for(size_t z=1; z<=local_nz; ++z) {
             std::copy(finalLocalGrid.begin() + idx3(0,0,z,nx,ny),
                       finalLocalGrid.begin() + idx3(0,0,z+1,nx,ny),
                       sendBuf.begin() + (z-1)*nx*ny);
        }
        
        MPI_Gatherv(sendBuf.data(), my_count, MPI_DOUBLE, 
                    globalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            print_results(globalGrid, "Grid");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(finalLocalGrid, nx, ny, local_nz);
        
        // Need to combine validity
        int localValid = valid ? 1 : 0;
        int globalValid;
        MPI_Reduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (globalValid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                // Don't return 1 here to avoid messing up MPI_Finalize
            }
        }
        // Wait for rank 0 to print
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0 && !globalValid) {
             MPI_Finalize();
             return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
