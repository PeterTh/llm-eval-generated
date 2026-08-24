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

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_offset, const size_t global_nz) {
    for (size_t z = 1; z <= local_nz; ++z) { // z=1 to local_nz are the real data
        size_t global_z = z_offset + (z - 1);
        if (global_z >= global_nz) continue; 
        
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Original initialization used global index: z * (nx * ny) + y * nx + x
                const size_t global_idx = global_z * (nx * ny) + y * nx + x;
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Exchange ghost layers between MPI ranks
void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, int rank, int size) {
    size_t plane_size = nx * ny;
    MPI_Status status;

    // Send down (rank to rank-1), receive from up (rank+1 to rank)
    // We send our bottom real layer (z=1) to rank-1's top halo (z=local_nz+1)
    // We receive into our top halo (z=local_nz+1) from rank+1's bottom real layer (z=1)
    
    // Send up (rank to rank+1), receive from down (rank-1 to rank)
    // We send our top real layer (z=local_nz) to rank+1's bottom halo (z=0)
    // We receive into our bottom halo (z=0) from rank-1's top real layer (z=local_nz)

    // Tag 0: sending down / receiving from up
    // Tag 1: sending up / receiving from down
    
    int up_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int down_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;

    // Send to up, receive from down
    MPI_Sendrecv(
        &grid[idx3(0, 0, local_nz, nx, ny)], plane_size, MPI_DOUBLE, up_neighbor, 0,
        &grid[idx3(0, 0, 0, nx, ny)], plane_size, MPI_DOUBLE, down_neighbor, 0,
        MPI_COMM_WORLD, &status
    );

    // Send to down, receive from up
    MPI_Sendrecv(
        &grid[idx3(0, 0, 1, nx, ny)], plane_size, MPI_DOUBLE, down_neighbor, 1,
        &grid[idx3(0, 0, local_nz + 1, nx, ny)], plane_size, MPI_DOUBLE, up_neighbor, 1,
        MPI_COMM_WORLD, &status
    );
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      int rank, int size) {
    
    // Process local interior points
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Global boundary conditions:
                // If on global boundary, copy value instead of computing stencil
                
                // Determine if this point is on a global boundary
                bool on_boundary = false;
                
                // X boundaries
                if (x == 0 || x == nx-1) on_boundary = true;
                
                // Y boundaries
                if (y == 0 || y == ny-1) on_boundary = true;
                
                // Z boundaries
                // Bottom global boundary is at rank 0, local z=1
                if (rank == 0 && z == 1) on_boundary = true;
                // Top global boundary is at rank size-1, local z=local_nz
                if (rank == size-1 && z == local_nz) on_boundary = true;
                
                const size_t idx = idx3(x, y, z, nx, ny);
                
                if (on_boundary) {
                    output[idx] = input[idx];
                } else {
                    const Real center = input[idx];
                    const Real left = input[idx3(x-1, y, z, nx, ny)];
                    const Real right = input[idx3(x+1, y, z, nx, ny)];
                    const Real front = input[idx3(x, y-1, z, nx, ny)];
                    const Real back = input[idx3(x, y+1, z, nx, ny)];
                    const Real bottom = input[idx3(x, y, z-1, nx, ny)]; // Can access z=0 (ghost)
                    const Real top = input[idx3(x, y, z+1, nx, ny)];    // Can access z=local_nz+1 (ghost)
                    
                    // Simple averaging stencil
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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

std::vector<Real> gather_grid(const std::vector<Real>& local_grid, int rank, int size, size_t nx, size_t ny, size_t local_nz, size_t global_nz) {
    std::vector<Real> global_grid;
    if (rank == 0) {
        global_grid.resize(global_nz * nx * ny);
    }
    
    // Calculate counts and displacements for Gatherv
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    
    // We need to re-calculate local_nz for each rank to know exactly how much data to expect
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        size_t r_local_nz = global_nz / size;
        if (r < (int)(global_nz % size)) {
            r_local_nz++;
        }
        counts[r] = r_local_nz * nx * ny;
        displs[r] = offset * nx * ny;
        offset += r_local_nz;
    }

    // Prepare send buffer (skip bottom ghost layer)
    // The local grid has size (local_nz + 2) * nx * ny
    // The data starts at index nx * ny (skipping z=0 ghost layer)
    const Real* sendbuf = &local_grid[nx * ny];
    int sendcount = local_nz * nx * ny;
    
    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE,
                global_grid.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
                
    return global_grid;
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
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition
    size_t base_nz = nz / size;
    size_t remainder = nz % size;
    size_t local_nz = base_nz;
    size_t z_offset = 0;
    
    if (rank < (int)remainder) {
        local_nz++;
        z_offset = rank * local_nz;
    } else {
        z_offset = remainder * (base_nz + 1) + (rank - remainder) * base_nz;
    }
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI Parallel)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate grids (double buffering) with ghost layers
    // Size: (local_nz + 2) * nx * ny
    size_t local_grid_size = (local_nz + 2) * nx * ny;
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_nz, z_offset, nz);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchangeHalos(grid1, nx, ny, local_nz, rank, size);
            stencilIteration(grid1, grid2, nx, ny, local_nz, rank, size);
        } else {
            exchangeHalos(grid2, nx, ny, local_nz, rank, size);
            stencilIteration(grid2, grid1, nx, ny, local_nz, rank, size);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);
        
        // Calculate performance metrics
        // Note: Global boundaries are not updated, so subtract 2 from each dim
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Prepare for validation/printing
    std::vector<Real>& finalLocalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    
    if (printResults || validate) {
        std::vector<Real> globalGrid = gather_grid(finalLocalGrid, rank, size, nx, ny, local_nz, nz);
        
        if (rank == 0) {
            if (printResults) {
                print_results(globalGrid, "Grid");
            }
            
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(globalGrid, nx, ny, nz);
                
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
