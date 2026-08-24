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
                const size_t global_z = z + z_offset;
                const size_t global_idx = global_z * (nx * ny) + y * nx + x;
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation with MPI halo exchange
void stencilIteration(std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      const int rank, const int size,
                      const bool has_lower, const bool has_upper) {
    // Halo buffers for communication
    std::vector<Real> send_lower(nx * ny);
    std::vector<Real> recv_lower(nx * ny);
    std::vector<Real> send_upper(nx * ny);
    std::vector<Real> recv_upper(nx * ny);
    
    // Prepare data to send
    if (has_upper) {
        // Copy the top plane (local_nz-2) to send buffer
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, local_nz - 2, nx, ny);
                send_upper[y * nx + x] = input[idx];
            }
        }
    }
    
    if (has_lower) {
        // Copy the bottom plane (1) to send buffer
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, 1, nx, ny);
                send_lower[y * nx + x] = input[idx];
            }
        }
    }
    
    // Exchange halos
    MPI_Request requests[4];
    int req_count = 0;
    
    if (has_lower) {
        MPI_Irecv(recv_lower.data(), nx * ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(send_lower.data(), nx * ny, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    if (has_upper) {
        MPI_Irecv(recv_upper.data(), nx * ny, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(send_upper.data(), nx * ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Process interior points while communication happens
    const size_t z_start = has_lower ? 2 : 1;
    const size_t z_end = has_upper ? local_nz - 2 : local_nz - 1;
    
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
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Wait for communication to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }
    
    // Update input halo regions with received data
    if (has_lower) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, 0, nx, ny);
                input[idx] = recv_lower[y * nx + x];
            }
        }
    }
    
    if (has_upper) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, local_nz - 1, nx, ny);
                input[idx] = recv_upper[y * nx + x];
            }
        }
    }
    
    // Process boundary layer that depends on halo data
    if (has_lower) {
        const size_t z = 1;
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
    
    if (has_upper) {
        const size_t z = local_nz - 2;
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
    
    // Copy X and Y boundary values
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
    
    // Copy Z boundary values for first and last ranks
    if (!has_lower) {
        // First rank, copy bottom plane
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, 0, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
    
    if (!has_upper) {
        // Last rank, copy top plane
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, local_nz - 1, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz, const int rank) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            if (rank == 0) {
                printf("Validation failed: found NaN or Inf value\n");
            }
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
    
    // Global reduction
    Real global_min, global_max;
    MPI_Allreduce(&minVal, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        
        // After averaging, values should be somewhat bounded
        if (global_max > 1e6 || global_min < -1e6) {
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
    
    // Domain decomposition in Z dimension
    size_t base_local_nz = nz / size;
    size_t remainder = nz % size;
    
    // Calculate local dimensions with extra rows for halos
    size_t local_nz = base_local_nz;
    size_t z_offset = rank * base_local_nz;
    
    // Distribute remainder to first processes
    if ((size_t)rank < remainder) {
        local_nz++;
        z_offset += rank;
    } else {
        z_offset += remainder;
    }
    
    // Add halo layers
    const bool has_lower = (rank > 0);
    const bool has_upper = (rank < size - 1);
    size_t local_nz_with_halo = local_nz;
    if (has_lower) local_nz_with_halo++;
    if (has_upper) local_nz_with_halo++;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("MPI Processes: %d\n", size);
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t local_gridSize = nx * ny * local_nz_with_halo;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(local_gridSize);
    std::vector<Real> grid2(local_gridSize);
    
    // Initialize
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, nx, ny, local_nz_with_halo, z_offset - (has_lower ? 1 : 0));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, local_nz_with_halo, rank, size, has_lower, has_upper);
        } else {
            stencilIteration(grid2, grid1, nx, ny, local_nz_with_halo, rank, size, has_lower, has_upper);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for printing
    const std::vector<Real>& finalLocalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    
    if (printResults) {
        // Gather all data to rank 0
        std::vector<Real> finalGrid;
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        // Calculate receive counts and displacements (excluding halos)
        for (int r = 0; r < size; ++r) {
            size_t r_base_local_nz = nz / size;
            size_t r_local_nz = r_base_local_nz;
            if ((size_t)r < remainder) {
                r_local_nz++;
            }
            recvcounts[r] = nx * ny * r_local_nz;
        }
        
        displs[0] = 0;
        for (int r = 1; r < size; ++r) {
            displs[r] = displs[r-1] + recvcounts[r-1];
        }
        
        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
        }
        
        // Extract data without halos
        std::vector<Real> sendData(nx * ny * local_nz);
        size_t send_z_start = has_lower ? 1 : 0;
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t local_idx = idx3(x, y, z + send_z_start, nx, ny);
                    const size_t send_idx = idx3(x, y, z, nx, ny);
                    sendData[send_idx] = finalLocalGrid[local_idx];
                }
            }
        }
        
        MPI_Gatherv(sendData.data(), nx * ny * local_nz, MPI_DOUBLE,
                    finalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(finalGrid, "Grid");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(finalLocalGrid, nx, ny, local_nz_with_halo, rank);
        
        // All processes must agree
        int local_valid = valid ? 1 : 0;
        int global_valid = 0;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
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
