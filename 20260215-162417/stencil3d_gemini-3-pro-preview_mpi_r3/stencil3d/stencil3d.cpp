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
    // Fill the local grid, including ghost layers.
    // The grid includes ghost layers at z=0 and z=local_nz-1.
    // z index 1 corresponds to global start_z.
    
    for (size_t z = 0; z < local_nz; ++z) {
        // Global Z coordinate
        long global_z = (long)start_z + (long)z - 1;
        
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                if (global_z >= 0) { 
                   long global_idx = global_z * (nx * ny) + y * nx + x;
                   grid[idx] = (global_idx % 19) * 1.0;
                } else {
                   grid[idx] = 0.0; 
                }
            }
        }
    }
}

// Exchange ghost layers
void exchangeGhosts(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t my_nz, 
                   int rank, int num_procs) {
    // grid size is nx * ny * (my_nz + 2)
    // z=1..my_nz are owned.
    // z=0 is ghost from below.
    // z=my_nz+1 is ghost from above.
    
    size_t plane_size = nx * ny;
    MPI_Status status;
    
    // Send to down (rank-1), Receive from up (rank+1)
    // We send our bottom owned layer (z=1) to fill rank-1's top ghost.
    // We receive from rank+1's bottom owned layer to fill our top ghost (z=my_nz+1).
    
    // Send to up (rank+1), Receive from down (rank-1)
    // We send our top owned layer (z=my_nz) to fill rank+1's bottom ghost.
    // We receive from rank-1's top owned layer to fill our bottom ghost (z=0).
    
    // Let's use Sendrecv
    
    // 1. Exchange with Up (Rank + 1)
    // Send my top (z=my_nz) to Up, Recv from Up into my top ghost (z=my_nz+1)
    if (num_procs > 1) {
        int up_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;
        int down_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        
        // Send z=my_nz (offset my_nz * plane_size), Recv into z=0 (offset 0) from down?
        // Let's do even/odd or just chained Sendrecv.
        
        // Send DOWN (my z=1), Recv from UP (my z=my_nz+1) ?? 
        // No.
        
        // Communication 1: Send local z=1 to Rank-1 (who puts it in my_nz+1).
        //                  Recv from Rank+1 (their z=1) into local z=my_nz+1.
        
        // Let's stick to standard Halo exchange directions.
        
        // Step 1: Send to Rank-1 (Down), Recv from Rank+1 (Up)
        // Rank 0 has no one below. Rank N-1 has no one above.
        
        // We need:
        // - Ghost at z=0 (from Rank-1).
        // - Ghost at z=my_nz+1 (from Rank+1).
        
        // Send buffer: z=1 (start of owned data).
        // Recv buffer: z=my_nz+1 (top ghost).
        MPI_Sendrecv(
            &grid[1 * plane_size], plane_size, MPI_DOUBLE, down_rank, 0,
            &grid[(my_nz + 1) * plane_size], plane_size, MPI_DOUBLE, up_rank, 0,
            MPI_COMM_WORLD, &status
        );
        
        // Step 2: Send to Rank+1 (Up), Recv from Rank-1 (Down)
        // Send buffer: z=my_nz (end of owned data).
        // Recv buffer: z=0 (bottom ghost).
        MPI_Sendrecv(
            &grid[my_nz * plane_size], plane_size, MPI_DOUBLE, up_rank, 1,
            &grid[0 * plane_size], plane_size, MPI_DOUBLE, down_rank, 1,
            MPI_COMM_WORLD, &status
        );
    }
}

// 7-point stencil computation
void stencilIteration(std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, 
                      const size_t my_nz, const size_t total_nz,
                      const size_t start_z) {
    // input/output size: nx * ny * (my_nz + 2)
    // owned range: z=1 to z=my_nz
    
    // Determine the range of z to compute
    // We compute for global_z in [1, total_nz-2]
    
    size_t z_start_compute = 1;
    size_t z_end_compute = my_nz;
    
    // If we are at the global bottom boundary (start_z == 0), 
    // z=1 corresponds to global z=0. The computation starts at global z=1.
    // So if start_z=0, global z=1 is at local z=2.
    // Wait, let's recheck the indexing.
    // rank 0: starts at global z=0.
    // local z=1 -> global z=0.
    // local z=2 -> global z=1.
    // Original loop: z from 1 to nz-2 (inclusive).
    // So global z=1 is the first computed layer.
    // So yes, if start_z=0, we start computing at local z=2.
    if (start_z == 0) {
        z_start_compute = 2;
    }
    
    // If we are at the global top boundary (start_z + my_nz == total_nz), 
    // local z=my_nz -> global z=total_nz-1.
    // Original loop stops at global z=total_nz-2.
    // So global z=total_nz-1 is NOT computed.
    // So we stop computing at local z=my_nz-1.
    // Yes, this is correct.
    if (start_z + my_nz == total_nz) {
        z_end_compute = my_nz - 1;
    }
    
    // Loop over the compute range
    if (z_start_compute <= z_end_compute) {
        for (size_t z = z_start_compute; z <= z_end_compute; ++z) {
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
    }
    
    // Copy boundary values (local and global)
    
    // 1. If we skipped z=1 (global bottom at start_z=0), copy it.
    if (start_z == 0) {
         size_t z = 1;
         for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
    
    // 2. If we skipped z=my_nz (global top at end), copy it.
    if (start_z + my_nz == total_nz) {
         size_t z = my_nz;
         for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                output[idx] = input[idx];
            }
        }
    }

    // 3. For all other layers in [z_start_compute, z_end_compute], we computed the interior (x=1..nx-2, y=1..ny-2).
    // We need to copy the X and Y boundaries for these layers.
    if (z_start_compute <= z_end_compute) {
        for (size_t z = z_start_compute; z <= z_end_compute; ++z) {
            // Copy X boundaries (x=0, x=nx-1) for all y
            for (size_t y = 0; y < ny; ++y) {
                // x=0
                size_t idx = idx3(0, y, z, nx, ny);
                output[idx] = input[idx];
                // x=nx-1
                idx = idx3(nx-1, y, z, nx, ny);
                output[idx] = input[idx];
            }
            // Copy Y boundaries (y=0, y=ny-1) for interior x (already did corners above)
            // To be safe and simple, just iterate x=1..nx-2
             for (size_t x = 1; x < nx - 1; ++x) {
                // y=0
                size_t idx = idx3(x, 0, z, nx, ny);
                output[idx] = input[idx];
                // y=ny-1
                idx = idx3(x, ny-1, z, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
}

bool validateResult(const std::vector<Real>& local_grid, const size_t nx, const size_t ny, const size_t my_nz) {
    // Validating locally first
    // Note: We are checking owned parts only
    
    // 1. No NaN or Inf values
    for (size_t z = 1; z <= my_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const auto& val = local_grid[idx];
                if (std::isnan(val) || std::isinf(val)) {
                    printf("Validation failed: found NaN or Inf value\n");
                    return false;
                }
            }
        }
    }
    
    // Gather min/max across all procs
    Real localMin = local_grid[idx3(0,0,1,nx,ny)];
    Real localMax = local_grid[idx3(0,0,1,nx,ny)];
    
    for (size_t z = 1; z <= my_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const auto& val = local_grid[idx];
                localMin = std::min(localMin, val);
                localMax = std::max(localMax, val);
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
        
        if (globalMax > 1e6 || globalMin < -1e6) {
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
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI Parallel)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", num_procs);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Domain Decomposition
    size_t base_nz = nz / num_procs;
    size_t remainder = nz % num_procs;
    size_t my_nz = base_nz + (rank < (int)remainder ? 1 : 0);
    size_t start_z = rank * base_nz + (rank < (int)remainder ? rank : remainder);
    
    // Allocate grids (double buffering)
    // Add 2 for ghost layers (top and bottom)
    size_t local_allocated_nz = my_nz + 2;
    size_t gridSize = nx * ny * local_allocated_nz;
    
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_allocated_nz, start_z);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange ghosts on the current input grid
        if (iter % 2 == 0) {
            exchangeGhosts(grid1, nx, ny, my_nz, rank, num_procs);
            stencilIteration(grid1, grid2, nx, ny, my_nz, nz, start_z);
        } else {
            exchangeGhosts(grid2, nx, ny, my_nz, rank, num_procs);
            stencilIteration(grid2, grid1, nx, ny, my_nz, nz, start_z);
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
    
    // Print results for external validation
    const std::vector<Real>& finalLocalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    
    if (printResults) {
        // Gather to rank 0
        // We only gather the OWNED parts (skip ghosts)
        std::vector<Real> globalGrid;
        if (rank == 0) globalGrid.resize(nx * ny * nz);
        
        std::vector<int> recvcounts(num_procs);
        std::vector<int> displs(num_procs);
        
        // We need to gather element counts
        int my_count = nx * ny * my_nz;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < num_procs; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        // MPI_Gatherv needs contiguous memory. Our owned data starts at index nx*ny (skip bottom ghost)
        // and has size nx*ny*my_nz.
        MPI_Gatherv(&finalLocalGrid[nx * ny], my_count, MPI_DOUBLE, 
                    globalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
             print_results(globalGrid, "Grid");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(finalLocalGrid, nx, ny, my_nz);
        
        if (rank == 0) {
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
