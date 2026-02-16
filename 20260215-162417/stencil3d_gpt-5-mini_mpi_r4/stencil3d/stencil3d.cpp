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

// Initialize local subgrid using global coordinates
void initializeLocalGrid(std::vector<Real>& localGrid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_start_global) {
    // localGrid has shape (local_nz + 2) x ny x nx with halo planes at z=0 and z=local_nz+1
    const size_t local_nx = nx;
    const size_t local_ny = ny;
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        size_t gz = z_start_global + (lz - 1);
        for (size_t y = 0; y < local_ny; ++y) {
            for (size_t x = 0; x < local_nx; ++x) {
                const size_t idx = idx3(x, y, lz, local_nx, local_ny);
                const size_t global_idx = idx3(x, y, gz, nx, ny);
                localGrid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
    // Initialize halo planes to copies of adjacent boundary (will be overwritten by MPI exchange when applicable)
    // Lower halo (lz=0) copy from lz=1
    std::memcpy(&localGrid[idx3(0,0,0,local_nx,local_ny)], &localGrid[idx3(0,0,1,local_nx,local_ny)], sizeof(Real)*local_nx*local_ny);
    // Upper halo (lz=local_nz+1) copy from lz=local_nz
    std::memcpy(&localGrid[idx3(0,0,local_nz+1,local_nx,local_ny)], &localGrid[idx3(0,0,local_nz,local_nx,local_ny)], sizeof(Real)*local_nx*local_ny);
}

// 7-point stencil computation on local grid (with halos)
void stencilIterationLocal(const std::vector<Real>& input, std::vector<Real>& output, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_start_global, const size_t global_nz) {
    const size_t local_nx = nx;
    const size_t local_ny = ny;
    const size_t nz_with_halo = local_nz + 2;

    // Process interior points (excluding halo planes)
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        size_t gz = z_start_global + (lz - 1);
        if (gz == 0 || gz == global_nz - 1) {
            // global boundary planes: copy values through (will be handled below)
            continue;
        }
        for (size_t y = 1; y < local_ny - 1; ++y) {
            for (size_t x = 1; x < local_nx - 1; ++x) {
                const size_t idx = idx3(x, y, lz, local_nx, local_ny);
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, lz, local_nx, local_ny)];
                const Real right = input[idx3(x+1, y, lz, local_nx, local_ny)];
                const Real front = input[idx3(x, y-1, lz, local_nx, local_ny)];
                const Real back = input[idx3(x, y+1, lz, local_nx, local_ny)];
                const Real bottom = input[idx3(x, y, lz-1, local_nx, local_ny)];
                const Real top = input[idx3(x, y, lz+1, local_nx, local_ny)];
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Copy boundary values for x,y and global z boundaries
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        size_t gz = z_start_global + (lz - 1);
        for (size_t y = 0; y < local_ny; ++y) {
            for (size_t x = 0; x < local_nx; ++x) {
                if (x == 0 || x == local_nx-1 || y == 0 || y == local_ny-1 || gz == 0 || gz == global_nz-1) {
                    const size_t idx = idx3(x, y, lz, local_nx, local_ny);
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size = 1;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition along Z dimension
    const size_t base = nz / static_cast<size_t>(world_size);
    const size_t rem = nz % static_cast<size_t>(world_size);
    size_t local_nz = base + (static_cast<size_t>(world_rank) < rem ? 1 : 0);
    size_t z_start = static_cast<size_t>(world_rank) * base + std::min(static_cast<size_t>(world_rank), rem);

    const size_t local_nx = nx;
    const size_t local_ny = ny;

    // Allocate local grids with halos
    std::vector<Real> local1((local_nz + 2) * local_ny * local_nx);
    std::vector<Real> local2((local_nz + 2) * local_ny * local_nx);

    // Initialize local grid
    if (world_rank == 0) printf("Initializing grid...\n");
    initializeLocalGrid(local1, local_nx, local_ny, local_nz, z_start);

    // Buffers for send/recv (planes)
    const int planeCount = static_cast<int>(local_nx * local_ny);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Run stencil iterations
    if (world_rank == 0) printf("Running stencil computation...\n");
    for (int iter = 0; iter < iterations; ++iter) {
        // Determine input/output buffers
        std::vector<Real>& in = (iter % 2 == 0) ? local1 : local2;
        std::vector<Real>& out = (iter % 2 == 0) ? local2 : local1;

        // Exchange halos with neighbors
        int prev = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
        int next = (world_rank == world_size - 1) ? MPI_PROC_NULL : world_rank + 1;

        // Send lower real plane (lz=1) to prev, receive into halo lz=0
        MPI_Sendrecv(&in[idx3(0,0,1,local_nx,local_ny)], planeCount, MPI_DOUBLE, prev, 0,
                     &in[idx3(0,0,0,local_nx,local_ny)], planeCount, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send upper real plane (lz=local_nz) to next, receive into halo lz=local_nz+1
        MPI_Sendrecv(&in[idx3(0,0,local_nz,local_nx,local_ny)], planeCount, MPI_DOUBLE, next, 1,
                     &in[idx3(0,0,local_nz+1,local_nx,local_ny)], planeCount, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute local stencil
        stencilIterationLocal(in, out, local_nx, local_ny, local_nz, z_start, nz);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;
    double elapsed_max = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed_max * 1000.0);
        double cellUpdates = static_cast<double>((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (elapsed_max) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid to rank 0 for printing/validation
    // Prepare sendcounts and displacements
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        size_t lnz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        sendcounts[r] = static_cast<int>(lnz * local_nx * local_ny);
    }
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) displs[r] = displs[r-1] + sendcounts[r-1];

    std::vector<Real> finalGrid;
    if (world_rank == 0) finalGrid.resize(static_cast<size_t>(nx) * ny * nz);

    // Choose pointer to final local data (exclude halos)
    std::vector<Real>& finalLocal = (iterations % 2 == 0) ? local1 : local2;
    MPI_Gatherv(&finalLocal[idx3(0,0,1,local_nx,local_ny)], static_cast<int>(local_nz*local_nx*local_ny), MPI_DOUBLE,
                finalGrid.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print results for external validation
    if (printResults && world_rank == 0) {
        print_results(finalGrid, "Grid");
    }

    // Validation on rank 0
    if (validate && world_rank == 0) {
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
