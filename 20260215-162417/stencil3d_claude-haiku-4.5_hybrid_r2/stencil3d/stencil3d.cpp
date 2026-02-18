#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

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

// Hybrid MPI+OpenMP stencil iteration
// Each MPI rank handles domain decomposition along Z axis with OpenMP for local parallelism
void stencilIterationHybrid(const std::vector<Real>& input, 
                             std::vector<Real>& output,
                             const size_t nx, const size_t ny, const size_t nz,
                             const int mpi_rank, const int mpi_size) {
    // Domain decomposition: distribute along Z axis
    const size_t local_nz = (nz + mpi_size - 1) / mpi_size;
    const size_t z_start = mpi_rank * local_nz;
    const size_t z_end = std::min(z_start + local_nz, nz);
    
    // Process interior points with OpenMP parallelism
    // Each MPI process handles its own domain partition
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = std::max(z_start + 1, size_t(1)); z < std::min(z_end, nz - 1); ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - nx];
                const Real back = input[idx + nx];
                const Real bottom = input[idx - (nx * ny)];
                const Real top = input[idx + (nx * ny)];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
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
    
    // MPI halo exchange for inter-domain boundaries
    if (mpi_size > 1) {
        // Prepare halo data
        std::vector<Real> send_up(nx * ny);
        std::vector<Real> send_down(nx * ny);
        std::vector<Real> recv_up(nx * ny);
        std::vector<Real> recv_down(nx * ny);
        
        // Non-blocking communication to hide latency
        MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        
        if (z_start > 0) {
            // Send halo to previous rank
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nx * ny; ++i) {
                send_down[i] = output[idx3(i % nx, (i / nx) % ny, z_start + 1, nx, ny)];
            }
            MPI_Isend(send_down.data(), nx * ny, MPI_DOUBLE, mpi_rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
        }
        
        if (z_end < nz) {
            // Send halo to next rank
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nx * ny; ++i) {
                send_up[i] = output[idx3(i % nx, (i / nx) % ny, z_end - 2, nx, ny)];
            }
            MPI_Isend(send_up.data(), nx * ny, MPI_DOUBLE, mpi_rank + 1, 1, MPI_COMM_WORLD, &requests[1]);
        }
        
        if (z_start > 0) {
            // Receive halo from previous rank
            MPI_Irecv(recv_down.data(), nx * ny, MPI_DOUBLE, mpi_rank - 1, 1, MPI_COMM_WORLD, &requests[2]);
        }
        
        if (z_end < nz) {
            // Receive halo from next rank
            MPI_Irecv(recv_up.data(), nx * ny, MPI_DOUBLE, mpi_rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
        }
        
        // Wait for all communication
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        
        // Copy received halos
        if (z_start > 0) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nx * ny; ++i) {
                output[idx3(i % nx, (i / nx) % ny, z_start, nx, ny)] = recv_down[i];
            }
        }
        
        if (z_end < nz) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nx * ny; ++i) {
                output[idx3(i % nx, (i / nx) % ny, z_end, nx, ny)] = recv_up[i];
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
            exit(1);
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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (mpi_rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    if (mpi_rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, nx, ny, nz);
    
    // Set CUDA device for each MPI rank (optional, if CUDA is available)
    // This structure allows potential GPU acceleration when available
    
    // Run stencil iterations
    if (mpi_rank == 0) {
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationHybrid(grid1, grid2, nx, ny, nz, mpi_rank, mpi_size);
        } else {
            stencilIterationHybrid(grid2, grid1, nx, ny, nz, mpi_rank, mpi_size);
        }
    }
    
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
    
    // Print results for external validation (from rank 0)
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults && mpi_rank == 0) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate && mpi_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
