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

// Decomposition structure
struct Decomposition {
    int rank;
    int nprocs;
    size_t local_nz;
    size_t local_z_start;
    int bottom_neighbor;
    int top_neighbor;
};

// Initialize decomposition
Decomposition init_decomposition(size_t nz, MPI_Comm comm) {
    Decomposition decomp;
    MPI_Comm_rank(comm, &decomp.rank);
    MPI_Comm_size(comm, &decomp.nprocs);
    
    // Distribute nz layers among processes
    size_t base = nz / decomp.nprocs;
    size_t remainder = nz % decomp.nprocs;
    
    // Ranks 0..remainder-1 get base+1 layers
    // Ranks remainder..nprocs-1 get base layers
    if ((size_t)decomp.rank < remainder) {
        decomp.local_nz = base + 1;
        decomp.local_z_start = decomp.rank * (base + 1);
    } else {
        decomp.local_nz = base;
        decomp.local_z_start = remainder * (base + 1) + (decomp.rank - remainder) * base;
    }
    
    // Set neighbors
    decomp.bottom_neighbor = (decomp.rank > 0) ? decomp.rank - 1 : MPI_PROC_NULL;
    decomp.top_neighbor = (decomp.rank < decomp.nprocs - 1) ? decomp.rank + 1 : MPI_PROC_NULL;
    
    return decomp;
}

// Initialize local grid with global indexing
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const Decomposition& decomp) {
    for (size_t lz = 1; lz <= decomp.local_nz; ++lz) {
        size_t gz = decomp.local_z_start + (lz - 1); // global z
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = gz * (nx * ny) + y * nx + x;
                const size_t local_idx = idx3(x, y, lz, nx, ny);
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Halo exchange using non-blocking communication
void exchange_halos(std::vector<Real>& grid, const Decomposition& decomp, const size_t nx, const size_t ny) {
    const int plane_size = static_cast<int>(nx * ny);
    
    MPI_Request reqs[4];
    int nreqs = 0;
    
    // Tag convention:
    // - tag 0: messages going upward (from lower rank to higher rank)
    // - tag 1: messages going downward (from higher rank to lower rank)
    const int tag_up = 0;
    const int tag_down = 1;
    
    // Post receives first (to avoid deadlock)
    if (decomp.bottom_neighbor != MPI_PROC_NULL) {
        // Receive bottom halo from bottom neighbor (they sent their top layer up to us)
        MPI_Irecv(&grid[0], plane_size, MPI_DOUBLE, decomp.bottom_neighbor, tag_up, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (decomp.top_neighbor != MPI_PROC_NULL) {
        // Receive top halo from top neighbor (they sent their bottom layer down to us)
        MPI_Irecv(&grid[(decomp.local_nz + 1) * plane_size], plane_size, MPI_DOUBLE, decomp.top_neighbor, tag_down, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    
    // Post sends
    if (decomp.top_neighbor != MPI_PROC_NULL) {
        // Send top real layer to top neighbor (going up with tag_up)
        MPI_Isend(&grid[decomp.local_nz * plane_size], plane_size, MPI_DOUBLE, decomp.top_neighbor, tag_up, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (decomp.bottom_neighbor != MPI_PROC_NULL) {
        // Send bottom real layer to bottom neighbor (going down with tag_down)
        MPI_Isend(&grid[plane_size], plane_size, MPI_DOUBLE, decomp.bottom_neighbor, tag_down, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    
    if (nreqs > 0) {
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    }
}

// 7-point stencil computation on local grid
void stencilIteration(std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const Decomposition& decomp) {
    // Exchange halos before computation
    exchange_halos(input, decomp, nx, ny);
    
    // Determine z range for interior computation
    size_t z_start = 1;
    size_t z_end = decomp.local_nz;
    
    // Rank 0: skip local z=1 (global z=0, physical boundary)
    if (decomp.rank == 0) {
        z_start = 2;
    }
    // Last rank: skip local z=local_nz (global z=nz-1, physical boundary)
    if (decomp.rank == decomp.nprocs - 1) {
        z_end = decomp.local_nz - 1;
    }
    
    // Process interior points
    for (size_t z = z_start; z <= z_end; ++z) {
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
    
    // Copy boundary values
    // Lateral boundaries (x=0, x=nx-1, y=0, y=ny-1) for all local real z
    for (size_t z = 1; z <= decomp.local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            // x=0
            output[idx3(0, y, z, nx, ny)] = input[idx3(0, y, z, nx, ny)];
            // x=nx-1
            output[idx3(nx-1, y, z, nx, ny)] = input[idx3(nx-1, y, z, nx, ny)];
        }
        for (size_t x = 0; x < nx; ++x) {
            // y=0
            output[idx3(x, 0, z, nx, ny)] = input[idx3(x, 0, z, nx, ny)];
            // y=ny-1
            output[idx3(x, ny-1, z, nx, ny)] = input[idx3(x, ny-1, z, nx, ny)];
        }
    }
    
    // Physical Z boundaries
    if (decomp.rank == 0) {
        // Local z=1 is global z=0
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                output[idx3(x, y, 1, nx, ny)] = input[idx3(x, y, 1, nx, ny)];
            }
        }
    }
    if (decomp.rank == decomp.nprocs - 1) {
        // Local z=local_nz is global z=nz-1
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                output[idx3(x, y, decomp.local_nz, nx, ny)] = input[idx3(x, y, decomp.local_nz, nx, ny)];
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, const Decomposition& decomp, const size_t nx, const size_t ny) {
    // Local validation
    bool local_valid = true;
    Real local_min = 1e300;
    Real local_max = -1e300;
    
    for (size_t lz = 1; lz <= decomp.local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const Real val = grid[idx3(x, y, lz, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_valid = false;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }
    
    // Global reduction
    bool global_valid;
    Real global_min, global_max;
    MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (decomp.rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
    }
    
    if (!global_valid) {
        if (decomp.rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }
    
    if (global_max > 1e6 || global_min < -1e6) {
        if (decomp.rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
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
    
    // Initialize decomposition
    Decomposition decomp = init_decomposition(nz, MPI_COMM_WORLD);
    
    if (decomp.rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d\n", decomp.nprocs);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Check if decomposition is valid
    if (decomp.local_nz == 0) {
        if (decomp.rank == 0) {
            printf("Error: nz (%zu) is smaller than number of processes (%d)\n", nz, decomp.nprocs);
        }
        MPI_Finalize();
        return 1;
    }
    
    const size_t local_nz_total = decomp.local_nz + 2;
    const size_t local_grid_size = nx * ny * local_nz_total;
    
    // Allocate local grids (double buffering)
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);
    
    // Initialize
    if (decomp.rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, nx, ny, decomp);
    
    // Run stencil iterations
    if (decomp.rank == 0) {
        printf("Running stencil computation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, decomp);
        } else {
            stencilIteration(grid2, grid1, nx, ny, decomp);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    // Get maximum elapsed time across all processes
    double local_time = end_time - start_time;
    double max_time;
    MPI_Allreduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double duration_ms = max_time * 1000.0;
    
    if (decomp.rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for printing
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    
    if (printResults) {
        std::vector<Real> global_grid;
        if (decomp.rank == 0) {
            global_grid.resize(nx * ny * nz);
        }
        
        // Prepare recvcounts and displacements
        std::vector<int> recvcounts(decomp.nprocs);
        std::vector<int> displs(decomp.nprocs);
        
        std::vector<size_t> all_local_nz(decomp.nprocs);
        std::vector<size_t> all_local_z_start(decomp.nprocs);
        MPI_Allgather(&decomp.local_nz, 1, MPI_UNSIGNED_LONG, all_local_nz.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
        MPI_Allgather(&decomp.local_z_start, 1, MPI_UNSIGNED_LONG, all_local_z_start.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
        
        for (int r = 0; r < decomp.nprocs; ++r) {
            recvcounts[r] = all_local_nz[r] * nx * ny;
            displs[r] = all_local_z_start[r] * nx * ny;
        }
        
        // Gather real data (excluding halos)
        MPI_Gatherv(&finalGrid[nx * ny], decomp.local_nz * nx * ny, MPI_DOUBLE,
                    global_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (decomp.rank == 0) {
            print_results(global_grid, "Grid");
        }
    }
    
    // Validation
    if (validate) {
        if (decomp.rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(finalGrid, decomp, nx, ny);
        
        if (valid) {
            if (decomp.rank == 0) {
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (decomp.rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
