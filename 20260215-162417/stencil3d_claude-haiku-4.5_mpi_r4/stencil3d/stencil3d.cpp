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

// MPI-related global variables
int mpi_rank = 0;
int mpi_size = 1;
int mpi_neighbors[2] = {MPI_PROC_NULL, MPI_PROC_NULL}; // prev and next in z-direction

// 3D index calculation (local coordinates)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Halo exchange for distributed stencil computation
void haloExchange(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz_local) {
    // grid has nz_local+2 layers (halo at z=0 and z=nz_local+1)
    // nz_local does NOT include halo layers
    // Send layers at z=1 (bottom of compute region) and z=nz_local (top of compute region)
    const size_t layer_size = nx * ny;
    
    MPI_Request send_requests[2], recv_requests[2];
    int send_count = 0, recv_count = 0;
    
    // Exchange with previous neighbor (lower z)
    if (mpi_neighbors[0] != MPI_PROC_NULL) {
        // Send z=1 (bottom interior) to previous
        MPI_Isend(&grid[idx3(0, 0, 1, nx, ny)], layer_size, MPI_DOUBLE, 
                  mpi_neighbors[0], 0, MPI_COMM_WORLD, &send_requests[send_count++]);
        // Receive from previous into z=0 (bottom halo)
        MPI_Irecv(&grid[idx3(0, 0, 0, nx, ny)], layer_size, MPI_DOUBLE, 
                  mpi_neighbors[0], 1, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    
    // Exchange with next neighbor (higher z)
    if (mpi_neighbors[1] != MPI_PROC_NULL) {
        // Send z=nz_local (top interior) to next
        MPI_Isend(&grid[idx3(0, 0, nz_local, nx, ny)], layer_size, MPI_DOUBLE, 
                  mpi_neighbors[1], 1, MPI_COMM_WORLD, &send_requests[send_count++]);
        // Receive from next into z=nz_local+1 (top halo)
        MPI_Irecv(&grid[idx3(0, 0, nz_local + 1, nx, ny)], layer_size, MPI_DOUBLE, 
                  mpi_neighbors[1], 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    
    // Wait for all communications to complete
    for (int i = 0; i < send_count; ++i) {
        MPI_Wait(&send_requests[i], MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < recv_count; ++i) {
        MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz_local, const size_t z_offset) {
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t global_z = z_offset + z;
                const size_t global_idx = global_z * (nx * ny) + y * nx + x;
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation (distributed version)
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz_local,
                      const bool is_first_process, const bool is_last_process) {
    // local grid has indices 0 to nz_local+1 (with halos)
    // compute region: z in [1, nz_local], excluding boundary at z=0 (halo) and z=nz_local+1 (halo)
    
    // Process interior points (not on boundaries)
    for (size_t z = 1; z <= nz_local; ++z) {
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
    
    // Copy boundary values in x-y plane (for x and y boundaries)
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            // x boundaries
            {
                const size_t idx = idx3(0, y, z, nx, ny);
                output[idx] = input[idx];
            }
            {
                const size_t idx = idx3(nx-1, y, z, nx, ny);
                output[idx] = input[idx];
            }
        }
        for (size_t x = 0; x < nx; ++x) {
            // y boundaries
            {
                const size_t idx = idx3(x, 0, z, nx, ny);
                output[idx] = input[idx];
            }
            {
                const size_t idx = idx3(x, ny-1, z, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
    
    // Handle z-direction boundaries
    if (is_first_process) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, 0, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
    
    if (is_last_process) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, nz_local + 1, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz_local) {
    // Simple sanity checks on local data (excludes halos)
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("[Rank %d] Validation failed: found NaN or Inf value\n", mpi_rank);
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
    
    printf("[Rank %d] Value range: [%.6f, %.6f]\n", mpi_rank, minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("[Rank %d] Validation failed: values out of expected range\n", mpi_rank);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Compute domain decomposition along z-axis
    size_t nz_base = nz / mpi_size;
    size_t nz_remainder = nz % mpi_size;
    
    // Each process gets a different chunk
    size_t z_offset = 0;
    size_t nz_local = nz_base;
    
    for (int i = 0; i < mpi_rank; ++i) {
        z_offset += nz_base + (i < (int)nz_remainder ? 1 : 0);
    }
    
    if (mpi_rank < (int)nz_remainder) {
        nz_local = nz_base + 1;
    }
    
    // Set up neighbors for halo exchange
    mpi_neighbors[0] = (mpi_rank > 0) ? mpi_rank - 1 : MPI_PROC_NULL;
    mpi_neighbors[1] = (mpi_rank < mpi_size - 1) ? mpi_rank + 1 : MPI_PROC_NULL;
    
    // Allocate local grid with halo regions (extra layer at top and bottom for z)
    size_t nz_with_halo = nz_local + 2;
    size_t gridSize = nx * ny * nz_with_halo;
    
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize local grid (offset indices to account for global positions)
    if (mpi_rank == 0) {
        printf("3D Stencil Benchmark (MPI Distributed Memory)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Processes: %d\n", mpi_size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    if (mpi_rank == 0) {
        printf("Initializing grid...\n");
    }
    
    initializeGrid(grid1, nx, ny, nz_local, z_offset);
    
    // Initialize halo regions (boundaries from neighboring processes)
    // Top halo from process itself's top boundary
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            grid1[idx3(x, y, nz_local + 1, nx, ny)] = 0.0;
            grid1[idx3(x, y, 0, nx, ny)] = 0.0;
        }
    }
    
    if (mpi_rank == 0) {
        printf("Running stencil computation...\n");
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run stencil iterations
    bool is_first = (mpi_rank == 0);
    bool is_last = (mpi_rank == mpi_size - 1);
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            haloExchange(grid1, nx, ny, nz_local);
            stencilIteration(grid1, grid2, nx, ny, nz_local, is_first, is_last);
        } else {
            haloExchange(grid2, nx, ny, nz_local);
            stencilIteration(grid2, grid1, nx, ny, nz_local, is_first, is_last);
        }
    }
    
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation (from rank 0 only)
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    
    if (printResults && mpi_rank == 0) {
        // Only print rank 0's contribution for now
        std::vector<Real> local_data(finalGrid.begin() + nx * ny, 
                                     finalGrid.begin() + nx * ny * (nz_local + 1));
        print_results(local_data, "Grid");
    }
    
    // Validation on each rank
    if (validate) {
        if (mpi_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(finalGrid, nx, ny, nz_with_halo);
        
        // Reduce validation result across all ranks
        int local_valid = valid ? 1 : 0;
        int global_valid = 0;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            if (global_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return global_valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
