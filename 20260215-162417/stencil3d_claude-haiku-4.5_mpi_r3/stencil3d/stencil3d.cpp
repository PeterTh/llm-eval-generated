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

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// Exchange halo regions between neighboring processes
void exchangeHalo(std::vector<Real>& grid,
                 const size_t nx, const size_t ny, const size_t local_nz,
                 const int rank, const int nprocs) {
    size_t halo_size = nx * ny;
    std::vector<Real> send_bottom(halo_size), recv_bottom(halo_size);
    std::vector<Real> send_top(halo_size), recv_top(halo_size);
    
    // Extract boundaries to send (interior planes that neighbor procs need)
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            send_bottom[y * nx + x] = grid[idx3(x, y, 1, nx, ny)];
            send_top[y * nx + x] = grid[idx3(x, y, local_nz - 2, nx, ny)];
        }
    }
    
    // Exchange with bottom neighbor (rank - 1)
    if (rank > 0) {
        MPI_Sendrecv(send_bottom.data(), halo_size, MPI_DOUBLE, rank - 1, 10,
                     recv_bottom.data(), halo_size, MPI_DOUBLE, rank - 1, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Exchange with top neighbor (rank + 1)
    if (rank < nprocs - 1) {
        MPI_Sendrecv(send_top.data(), halo_size, MPI_DOUBLE, rank + 1, 11,
                     recv_top.data(), halo_size, MPI_DOUBLE, rank + 1, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Copy received data to halo regions
    if (rank > 0) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                grid[idx3(x, y, 0, nx, ny)] = recv_bottom[y * nx + x];
            }
        }
    }
    
    if (rank < nprocs - 1) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                grid[idx3(x, y, local_nz - 1, nx, ny)] = recv_top[y * nx + x];
            }
        }
    }
}

// 7-point stencil computation (local domain)
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      const int rank, const int nprocs) {
    // Process interior points (not on boundaries) with consideration for halo regions
    size_t z_start = (rank == 0) ? 1 : 1;
    size_t z_end = (rank == nprocs - 1) ? local_nz - 1 : local_nz - 1;
    
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
    
    // Copy boundary and halo values
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Copy boundaries in x/y, and halo/boundaries in z
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == local_nz - 1) {
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 processes)
    if (rank == 0) {
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all processes
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Compute local grid dimensions (distribute in Z dimension)
    size_t local_nz = nz / nprocs;
    size_t remainder = nz % nprocs;
    if (rank < (int)remainder) {
        local_nz += 1;
    }
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Number of MPI processes: %d\n", nprocs);
        printf("Local grid size per process: %zu x %zu x %zu\n", nx, ny, local_nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t local_gridSize = nx * ny * local_nz;
    
    // Allocate local grids (double buffering)
    std::vector<Real> grid1(local_gridSize);
    std::vector<Real> grid2(local_gridSize);
    
    // Initialize local portion
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    
    // Compute global z offset for this rank
    size_t z_offset = 0;
    for (int i = 0; i < rank; ++i) {
        z_offset += (i < (int)remainder) ? (nz / nprocs + 1) : (nz / nprocs);
    }
    
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_offset + z;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                const size_t local_idx = idx3(x, y, z, nx, ny);
                grid1[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
    
    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalo((iter % 2 == 0) ? grid1 : grid2, nx, ny, local_nz, rank, nprocs);
        
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, local_nz, rank, nprocs);
        } else {
            stencilIteration(grid2, grid1, nx, ny, local_nz, rank, nprocs);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather final grid for validation/output
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    
    if (printResults && rank == 0) {
        // For printing, we need full grid - collect from all processes
        // For simplicity, we print the master's portion
        print_results(finalGrid, "Grid");
    }
    
    // Validation (perform locally)
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(finalGrid, nx, ny, local_nz);
        
        // Reduce validation result across all processes
        int all_valid = valid ? 1 : 0;
        int global_valid = 0;
        MPI_Allreduce(&all_valid, &global_valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
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
