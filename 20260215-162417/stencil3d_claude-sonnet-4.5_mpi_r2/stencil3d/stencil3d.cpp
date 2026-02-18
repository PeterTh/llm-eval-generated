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

// 7-point stencil computation with MPI halo exchange
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const bool has_lower_neighbor, const bool has_upper_neighbor,
                      std::vector<Real>& halo_send_bottom, std::vector<Real>& halo_send_top,
                      std::vector<Real>& halo_recv_bottom, std::vector<Real>& halo_recv_top,
                      const int rank, const int size) {
    
    // Exchange halo data with neighbors
    MPI_Request requests[4];
    int num_requests = 0;
    const size_t halo_size = nx * ny;
    
    // Non-blocking sends and receives
    if (has_lower_neighbor) {
        // Send bottom plane to lower neighbor
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                halo_send_bottom[y * nx + x] = input[idx3(x, y, 1, nx, ny)];
            }
        }
        MPI_Isend(halo_send_bottom.data(), halo_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(halo_recv_bottom.data(), halo_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    
    if (has_upper_neighbor) {
        // Send top plane to upper neighbor
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                halo_send_top[y * nx + x] = input[idx3(x, y, nz - 2, nx, ny)];
            }
        }
        MPI_Isend(halo_send_top.data(), halo_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(halo_recv_top.data(), halo_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    
    // Compute interior points that don't need halo data (z = 2 to nz-3)
    const size_t z_start_interior = has_lower_neighbor ? 2 : 1;
    const size_t z_end_interior = has_upper_neighbor ? nz - 2 : nz - 1;
    
    for (size_t z = z_start_interior; z < z_end_interior; ++z) {
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
    
    // Wait for halo exchange to complete
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
    
    // Copy received halo data into grid boundaries
    if (has_lower_neighbor) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                output[idx3(x, y, 0, nx, ny)] = halo_recv_bottom[y * nx + x];
            }
        }
    }
    
    if (has_upper_neighbor) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                output[idx3(x, y, nz - 1, nx, ny)] = halo_recv_top[y * nx + x];
            }
        }
    }
    
    // Compute boundary layers that depend on halo data
    if (has_lower_neighbor) {
        size_t z = 1;
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = halo_recv_bottom[y * nx + x];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    if (has_upper_neighbor) {
        size_t z = nz - 2;
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = halo_recv_top[y * nx + x];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy X and Y boundary values (not computed)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
    
    // Copy global Z boundaries
    if (!has_lower_neighbor) {
        size_t z = 0;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
    
    if (!has_upper_neighbor) {
        size_t z = nz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                output[idx] = input[idx];
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
    size_t nz_global = 0;  // Will be set to nx if not specified
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
            nz_global = atoi(argv[++i]);
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz_global == 0) nz_global = nx;
    
    // Domain decomposition in Z direction
    size_t nz_base = nz_global / size;
    size_t nz_remainder = nz_global % size;
    
    // Distribute remainder across first few ranks
    size_t nz_local = nz_base + (rank < (int)nz_remainder ? 1 : 0);
    size_t z_offset = rank * nz_base + std::min((size_t)rank, nz_remainder);
    
    // Add halo layers (top and bottom) except at global boundaries
    bool has_lower_neighbor = (rank > 0);
    bool has_upper_neighbor = (rank < size - 1);
    
    size_t nz_with_halo = nz_local + (has_lower_neighbor ? 1 : 0) + (has_upper_neighbor ? 1 : 0);
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", size);
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, nz_global);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t gridSize = nx * ny * nz_with_halo;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Allocate halo buffers
    size_t halo_size = nx * ny;
    std::vector<Real> halo_send_bottom(halo_size);
    std::vector<Real> halo_send_top(halo_size);
    std::vector<Real> halo_recv_bottom(halo_size);
    std::vector<Real> halo_recv_top(halo_size);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    size_t init_z_offset = z_offset - (has_lower_neighbor ? 1 : 0);
    initializeGrid(grid1, nx, ny, nz_with_halo, init_z_offset);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz_with_halo, 
                           has_lower_neighbor, has_upper_neighbor,
                           halo_send_bottom, halo_send_top, 
                           halo_recv_bottom, halo_recv_top,
                           rank, size);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz_with_halo,
                           has_lower_neighbor, has_upper_neighbor,
                           halo_send_bottom, halo_send_top,
                           halo_recv_bottom, halo_recv_top,
                           rank, size);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics (global)
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz_global-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for validation/printing
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    
    std::vector<Real> globalGrid;
    if (printResults || validate) {
        if (rank == 0) {
            globalGrid.resize(nx * ny * nz_global);
        }
        
        // Each rank sends its local data (excluding halos)
        std::vector<Real> local_interior;
        size_t start_z = has_lower_neighbor ? 1 : 0;
        size_t end_z = has_upper_neighbor ? nz_with_halo - 1 : nz_with_halo;
        
        for (size_t z = start_z; z < end_z; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    local_interior.push_back(finalGrid[idx3(x, y, z, nx, ny)]);
                }
            }
        }
        
        // Gather sizes
        std::vector<int> recv_counts(size);
        std::vector<int> displs(size);
        int local_count = local_interior.size();
        
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }
        
        MPI_Gatherv(local_interior.data(), local_count, MPI_DOUBLE,
                   globalGrid.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(globalGrid, "Grid");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(globalGrid, nx, ny, nz_global);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
