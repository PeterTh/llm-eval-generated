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

// 3D index calculation for local grid (includes ghost layers)
// z is local index, 0 to nz_local+1
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t z_start, const size_t nz_local) {
    // Fill local part (indices 1 to nz_local)
    for (size_t z = 0; z < nz_local; ++z) {
        size_t global_z = z_start + z;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Map to local buffer index (z+1 because of ghost layer)
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Initialization based on global coordinates to match serial version
                // Original: grid[idx] = (idx % 19) * 1.0; where idx is global index
                // We need to reconstruct global index or just use the same formula logic?
                // Original used flattened index: z * (nx * ny) + y * nx + x
                size_t global_idx = global_z * (nx * ny) + y * nx + x;
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz_local,
                      const size_t z_start, const size_t nz_global,
                      int rank, int size) {

    // 1. Ghost Exchange
    size_t plane_size = nx * ny;
    int up_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int down_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;

    // Use non-blocking communication to overlap (potential future optimization)
    // For now, use Sendrecv for simplicity and correctness as first step.
    // Actually, overlapping is better.
    // Let's implement standard Isend/Irecv with wait.
    
    MPI_Request requests[4];
    int num_requests = 0;
    
    // Exchange with UP neighbor (rank + 1)
    if (up_neighbor != MPI_PROC_NULL) {
        // Recv from UP into my top ghost (nz_local+1)
        MPI_Irecv(&input[idx3(0, 0, nz_local + 1, nx, ny)], plane_size, MPI_DOUBLE, up_neighbor, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        // Send my top valid (nz_local) to UP
        MPI_Isend(&input[idx3(0, 0, nz_local, nx, ny)], plane_size, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    
    // Exchange with DOWN neighbor (rank - 1)
    if (down_neighbor != MPI_PROC_NULL) {
        // Recv from DOWN into my bottom ghost (0)
        MPI_Irecv(&input[idx3(0, 0, 0, nx, ny)], plane_size, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        // Send my bottom valid (1) to DOWN
        MPI_Isend(&input[idx3(0, 0, 1, nx, ny)], plane_size, MPI_DOUBLE, down_neighbor, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

    // 2. Computation
    // Process local interior points
    // Loop over valid local z: 1 to nz_local
    for (size_t z = 1; z <= nz_local; ++z) {
        size_t global_z = z_start + (z - 1);
        
        // Determine if we are at global boundary
        bool is_global_z_min = (global_z == 0);
        bool is_global_z_max = (global_z == nz_global - 1);
        
        // If we are at global boundary z=0 or z=nz-1, we just copy (according to original logic)
        // Original: if (z == 0 || z == nz-1) copy
        // Also if x==0, x==nx-1, y==0, y==ny-1 copy
        
        // Iterate over y and x
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || is_global_z_min || is_global_z_max) {
                    // Boundary condition: copy
                    output[idx] = input[idx];
                } else {
                    // Interior: stencil
                    const Real center = input[idx];
                    const Real left = input[idx3(x-1, y, z, nx, ny)];
                    const Real right = input[idx3(x+1, y, z, nx, ny)];
                    const Real front = input[idx3(x, y-1, z, nx, ny)];
                    const Real back = input[idx3(x, y+1, z, nx, ny)];
                    const Real bottom = input[idx3(x, y, z-1, nx, ny)]; // Accesses ghost if z=1
                    const Real top = input[idx3(x, y, z+1, nx, ny)];    // Accesses ghost if z=nz_local
                    
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
        // MPI might add its own arguments, so we should ignore unknown ones or check if they are standard?
        // Usually mpirun handles arguments before passing to app, but let's be safe.
        // Actually, we should probably ignore unknown arguments in MPI programs if we can't be sure.
        // But for this benchmark, let's assume standard behavior.
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI Parallel)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain Decomposition
    size_t nz_local = nz / size;
    size_t remainder = nz % size;
    size_t z_start = 0;
    
    // Distribute remainder
    if (static_cast<size_t>(rank) < remainder) {
        nz_local++; // Rank gets one extra plane
        z_start = rank * nz_local;
    } else {
        // Ranks < remainder each got (base_nz + 1) planes.
        // There are 'remainder' such ranks.
        // Total covered by them: remainder * (base_nz + 1)
        // Current rank is (rank - remainder) away from the first non-remainder rank.
        // Each of those previous non-remainder ranks had base_nz planes.
        z_start = remainder * (nz_local + 1) + (rank - remainder) * nz_local;
    }

    // Allocate local grids (including 2 ghost layers)
    size_t local_grid_size = nx * ny * (nz_local + 2);
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, z_start, nz_local);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD); // Sync before timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz_local, z_start, nz, rank, size);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz_local, z_start, nz, rank, size);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);
        
        // Calculate performance metrics
        // Note: Global update count
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for validation/output if needed
    if (validate || printResults) {
        std::vector<Real> full_grid;
        if (rank == 0) {
            full_grid.resize(nx * ny * nz);
        }
        
        // Prepare for Gatherv
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_count = static_cast<int>(nz_local * nx * ny);
        
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        // Send buffer: skip the first ghost layer (offset nx*ny)
        const std::vector<Real>& finalLocalGrid = (iterations % 2 == 0) ? grid1 : grid2;
        const Real* sendbuf = &finalLocalGrid[nx * ny]; 
        
        MPI_Gatherv(sendbuf, local_count, MPI_DOUBLE,
                    full_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            if (printResults) {
                print_results(full_grid, "Grid");
            }
            
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full_grid, nx, ny, nz);
                
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
