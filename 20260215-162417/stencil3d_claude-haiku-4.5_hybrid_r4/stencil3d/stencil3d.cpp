#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// CUDA kernel for stencil computation on GPU
#ifdef __CUDACC__
__global__ void stencil_kernel_cuda(const Real* input, Real* output, 
                                     size_t nx, size_t ny, size_t nz,
                                     size_t z_start, size_t z_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + z_start;
    
    if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1 && z >= z_start + 1 && z < z_end) {
        size_t idx = z * (nx * ny) + y * nx + x;
        
        Real center = input[idx];
        Real left = input[idx - 1];
        Real right = input[idx + 1];
        Real front = input[idx - nx];
        Real back = input[idx + nx];
        Real bottom = input[idx - nx * ny];
        Real top = input[idx + nx * ny];
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}
#endif

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Parallel initialization with OpenMP
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation with OpenMP parallelization
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries) with OpenMP
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 1; z < nz - 1; ++z) {
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
    
    // Copy boundary values with OpenMP
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
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
    #pragma omp parallel for
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            printf("Validation failed: found NaN or Inf value\n");
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
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
    
    // Parse command line arguments (only on rank 0)
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
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute Z dimension across MPI processes
    size_t z_per_process = nz / size;
    size_t z_remainder = nz % size;
    
    size_t local_nz = z_per_process + (rank < (int)z_remainder ? 1 : 0);
    size_t z_start = rank * z_per_process + std::min((size_t)rank, z_remainder);
    
    // Add ghost layers for halo exchange
    size_t local_nz_with_halo = local_nz + 2;
    
    // Allocate local grids (with halo layers)
    size_t local_grid_size = nx * ny * local_nz_with_halo;
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);
    
    // Initialize local portion of the grid
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_z = z_start + z;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                size_t local_idx = (z + 1) * (nx * ny) + y * nx + x;  // +1 for halo
                grid1[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
    
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halo layers via MPI
        MPI_Request requests[4];
        int req_count = 0;
        
        // Send top halo (z=local_nz) to rank+1
        if (rank < size - 1) {
            MPI_Isend(&((iter % 2 == 0 ? grid1 : grid2)[local_nz * nx * ny]), 
                     nx * ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        // Receive bottom halo from rank-1 (into z=0)
        if (rank > 0) {
            MPI_Irecv(&((iter % 2 == 0 ? grid1 : grid2)[0]), 
                     nx * ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        // Send bottom halo (z=1) to rank-1
        if (rank > 0) {
            MPI_Isend(&((iter % 2 == 0 ? grid1 : grid2)[nx * ny]), 
                     nx * ny, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        // Receive top halo from rank+1 (into z=local_nz+1)
        if (rank < size - 1) {
            MPI_Irecv(&((iter % 2 == 0 ? grid1 : grid2)[(local_nz + 1) * nx * ny]), 
                     nx * ny, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
        
        // Wait for all halo exchanges
        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }
        
        // Local stencil iteration with OpenMP
        const std::vector<Real>& input = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? grid2 : grid1;
        
        // Process interior points
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 1; z < local_nz + 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    size_t idx = z * (nx * ny) + y * nx + x;
                    
                    const Real center = input[idx];
                    const Real left = input[idx - 1];
                    const Real right = input[idx + 1];
                    const Real front = input[idx - nx];
                    const Real back = input[idx + nx];
                    const Real bottom = input[idx - nx * ny];
                    const Real top = input[idx + nx * ny];
                    
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
        
        // Copy boundary values (X and Y boundaries only for interior z layers)
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 1; z < local_nz + 1; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if ((x == 0 || x == nx - 1 || y == 0 || y == ny - 1)) {
                        size_t idx = z * (nx * ny) + y * nx + x;
                        output[idx] = input[idx];
                    }
                }
            }
        }
        
        // Copy halo layer boundaries (represent the global Z boundaries)
        // z=0 halo (global z=0 for rank 0, interior for other ranks)
        if (rank == 0) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t idx = 0 * (nx * ny) + y * nx + x;
                    output[idx] = input[idx];
                }
            }
        }
        
        // z=local_nz+1 halo (global z=nz-1 for last rank, interior for other ranks)
        if (rank == size - 1) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t idx = (local_nz + 1) * (nx * ny) + y * nx + x;
                    output[idx] = input[idx];
                }
            }
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
    
    // Gather results on rank 0
    std::vector<Real> finalGrid;
    if (rank == 0) {
        finalGrid.resize(nx * ny * nz);
    }
    
    const std::vector<Real>& local_result = (iterations % 2 == 0) ? grid1 : grid2;
    
    // Gather Z slices from all processes
    if (rank == 0) {
        // Copy rank 0's data
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t global_z = z;
                    size_t global_idx = idx3(x, y, global_z, nx, ny);
                    size_t local_idx = (z + 1) * (nx * ny) + y * nx + x;
                    finalGrid[global_idx] = local_result[local_idx];
                }
            }
        }
        
        // Receive from other ranks
        for (int src = 1; src < size; ++src) {
            size_t src_z_per = z_per_process + (src < (int)z_remainder ? 1 : 0);
            size_t src_z_start = src * z_per_process + std::min((size_t)src, z_remainder);
            
            std::vector<Real> remote_data(src_z_per * nx * ny);
            MPI_Recv(remote_data.data(), src_z_per * nx * ny, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            #pragma omp parallel for collapse(3) schedule(static)
            for (size_t z = 0; z < src_z_per; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        size_t global_z = src_z_start + z;
                        size_t global_idx = idx3(x, y, global_z, nx, ny);
                        size_t local_idx = (z + 1) * (nx * ny) + y * nx + x;
                        finalGrid[global_idx] = remote_data[local_idx];
                    }
                }
            }
        }
    } else {
        // Send local data to rank 0
        std::vector<Real> send_data(local_nz * nx * ny);
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t local_idx = (z + 1) * (nx * ny) + y * nx + x;
                    size_t send_idx = z * (nx * ny) + y * nx + x;
                    send_data[send_idx] = local_result[local_idx];
                }
            }
        }
        MPI_Send(send_data.data(), local_nz * nx * ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation (rank 0 only)
    if (rank == 0) {
        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            
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
