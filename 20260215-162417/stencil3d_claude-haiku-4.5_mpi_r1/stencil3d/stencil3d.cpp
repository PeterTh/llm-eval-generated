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

// MPI-related globals
int mpi_rank = 0;
int mpi_size = 1;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Distribute z dimension across MPI ranks
struct ZDecomposition {
    size_t z_start;
    size_t z_end;
    size_t z_local;
    int rank;
};

ZDecomposition computeZDecomposition(size_t nz, int rank, int size) {
    size_t base_z = nz / size;
    size_t remainder = nz % size;
    
    ZDecomposition decomp;
    decomp.rank = rank;
    
    if (rank < (int)remainder) {
        decomp.z_start = rank * (base_z + 1);
        decomp.z_end = decomp.z_start + base_z + 1;
        decomp.z_local = base_z + 1;
    } else {
        decomp.z_start = remainder * (base_z + 1) + (rank - remainder) * base_z;
        decomp.z_end = decomp.z_start + base_z;
        decomp.z_local = base_z;
    }
    
    return decomp;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, 
                    size_t z_start, size_t z_end) {
    for (size_t z = z_start; z < z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z - z_start, nx, ny);
                // Use global z index to match sequential initialization
                const size_t global_idx = idx3(x, y, z, nx, ny);
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation with MPI halo exchange
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t z_local,
                      bool is_first_layer, bool is_last_layer) {
    // Exchange halo regions with neighbors using non-blocking communication
    std::vector<Real> halo_recv_prev(nx * ny);
    std::vector<Real> halo_recv_next(nx * ny);
    MPI_Request recv_prev_req = MPI_REQUEST_NULL;
    MPI_Request recv_next_req = MPI_REQUEST_NULL;
    MPI_Request send_prev_req = MPI_REQUEST_NULL;
    MPI_Request send_next_req = MPI_REQUEST_NULL;
    
    // Initiate receives
    if (!is_first_layer) {
        MPI_Irecv(halo_recv_prev.data(), nx * ny, MPI_DOUBLE, mpi_rank - 1, 0, MPI_COMM_WORLD, &recv_prev_req);
    }
    if (!is_last_layer) {
        MPI_Irecv(halo_recv_next.data(), nx * ny, MPI_DOUBLE, mpi_rank + 1, 1, MPI_COMM_WORLD, &recv_next_req);
    }
    
    // Initiate sends
    if (!is_first_layer) {
        MPI_Isend(const_cast<Real*>(input.data()), nx * ny, MPI_DOUBLE, mpi_rank - 1, 1, MPI_COMM_WORLD, &send_prev_req);
    }
    if (!is_last_layer) {
        const size_t last_layer_idx = (z_local - 1) * (nx * ny);
        MPI_Isend(const_cast<Real*>(input.data()) + last_layer_idx, nx * ny, MPI_DOUBLE, mpi_rank + 1, 0, MPI_COMM_WORLD, &send_next_req);
    }
    
    // Compute interior points (not dependent on halo)
    for (size_t z = 1; z < z_local - 1; ++z) {
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
    
    // Wait for receives
    if (recv_prev_req != MPI_REQUEST_NULL) {
        MPI_Wait(&recv_prev_req, MPI_STATUS_IGNORE);
    }
    if (recv_next_req != MPI_REQUEST_NULL) {
        MPI_Wait(&recv_next_req, MPI_STATUS_IGNORE);
    }
    
    // Compute boundaries between local layers and halos
    if (!is_first_layer && z_local > 1) {
        size_t z = 0;
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = halo_recv_prev[y * nx + x];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    if (!is_last_layer && z_local > 1) {
        size_t z = z_local - 1;
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = halo_recv_next[y * nx + x];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Wait for sends to complete
    if (send_prev_req != MPI_REQUEST_NULL) {
        MPI_Wait(&send_prev_req, MPI_STATUS_IGNORE);
    }
    if (send_next_req != MPI_REQUEST_NULL) {
        MPI_Wait(&send_next_req, MPI_STATUS_IGNORE);
    }
    
    // Copy boundary values on domain edges
    for (size_t z = 0; z < z_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
    
    // Copy MPI boundary faces (top and bottom of local domain)
    if (is_first_layer) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, 0, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
    if (is_last_layer) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z_local - 1, nx, ny);
                output[idx] = input[idx];
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, 
                   [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t z_local) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            if (mpi_rank == 0) printf("Validation failed: found NaN or Inf value\n");
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
    
    // Reduce to find global min/max
    Real global_min, global_max;
    MPI_Reduce(&minVal, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxVal, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        
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
        } else if (mpi_rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (mpi_rank == 0) {
        printf("3D Stencil Benchmark (MPI Parallelized)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI processes: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Compute domain decomposition
    ZDecomposition decomp = computeZDecomposition(nz, mpi_rank, mpi_size);
    bool is_first_layer = (mpi_rank == 0);
    bool is_last_layer = (mpi_rank == mpi_size - 1);
    
    size_t z_local = decomp.z_local;
    size_t local_grid_size = nx * ny * z_local;
    
    // Allocate local grids
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);
    
    // Initialize
    if (mpi_rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, decomp.z_start, decomp.z_end);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (mpi_rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, z_local, is_first_layer, is_last_layer);
        } else {
            stencilIteration(grid2, grid1, nx, ny, z_local, is_first_layer, is_last_layer);
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
    
    // Gather results for validation/output (only rank 0 needs full grid)
    std::vector<Real> full_grid;
    if (printResults || validate) {
        full_grid.resize(nx * ny * nz);
        
        // Determine send counts and displacements
        std::vector<int> sendcounts(mpi_size);
        std::vector<int> displacements(mpi_size, 0);
        
        for (int r = 0; r < mpi_size; ++r) {
            ZDecomposition d = computeZDecomposition(nz, r, mpi_size);
            sendcounts[r] = nx * ny * d.z_local;
            if (r > 0) {
                displacements[r] = displacements[r-1] + sendcounts[r-1];
            }
        }
        
        const std::vector<Real>& final_grid = (iterations % 2 == 0) ? grid1 : grid2;
        MPI_Gatherv(const_cast<Real*>(final_grid.data()), sendcounts[mpi_rank], MPI_DOUBLE,
                   full_grid.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }
    
    // Print results
    if (mpi_rank == 0 && printResults) {
        print_results(full_grid, "Grid");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResult((iterations % 2 == 0) ? grid1 : grid2, nx, ny, z_local);
        
        if (mpi_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
