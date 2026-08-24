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

// Initialize local grid using global indices
// Local grid layout: ghost layer at z_l=0, real data at z_l=1..local_nz, ghost at z_l=local_nz+1
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t local_nz, const size_t z_start) {
    const size_t stride = nx * ny;
    for (size_t z_l = 1; z_l <= local_nz; ++z_l) {
        const size_t z_global = z_start + (z_l - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = idx3(x, y, z_global, nx, ny);
                grid[z_l * stride + y * nx + x] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Compute stencil for a single z-plane
inline void computeStencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                                const size_t z_l, const size_t nx, const size_t ny, const size_t stride) {
    for (size_t y = 1; y < ny - 1; ++y) {
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t idx = z_l * stride + y * nx + x;
            output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                           input[idx - nx] + input[idx + nx] +
                           input[idx - stride] + input[idx + stride]) / 7.0;
        }
    }
}

// Combined halo exchange + stencil with communication/computation overlap
void stencilIterationMPI(std::vector<Real>& input, std::vector<Real>& output,
                         const size_t nx, const size_t ny, const size_t local_nz,
                         const size_t z_start, const size_t nz,
                         const int rank_below, const int rank_above, MPI_Comm comm) {
    const size_t stride = nx * ny;

    if (local_nz == 0) return;

    // Determine which local z-planes need stencil computation (global z in [1, nz-2])
    // z_global = z_start + (z_l - 1), so z_l = z_global - z_start + 1
    const size_t z_l_first = (z_start == 0) ? 2 : 1;
    const size_t z_global_last = z_start + local_nz - 1;
    const size_t z_l_last = (z_global_last == nz - 1) ? local_nz - 1 : local_nz;

    // Post non-blocking halo exchange (receives first for best overlap)
    MPI_Request reqs[4];
    int nreqs = 0;

    if (rank_below != MPI_PROC_NULL) {
        MPI_Irecv(&input[0], stride, MPI_DOUBLE, rank_below, 1, comm, &reqs[nreqs++]);
        MPI_Isend(&input[stride], stride, MPI_DOUBLE, rank_below, 0, comm, &reqs[nreqs++]);
    }
    if (rank_above != MPI_PROC_NULL) {
        MPI_Irecv(&input[(local_nz + 1) * stride], stride, MPI_DOUBLE, rank_above, 0, comm, &reqs[nreqs++]);
        MPI_Isend(&input[local_nz * stride], stride, MPI_DOUBLE, rank_above, 1, comm, &reqs[nreqs++]);
    }

    // Compute stencil on interior z-planes (z_l=2..local_nz-1) that don't touch ghost layers
    if (z_l_first <= z_l_last) {
        const size_t interior_start = std::max(z_l_first, (size_t)2);
        const size_t interior_end = std::min(z_l_last, local_nz - 1);
        for (size_t z_l = interior_start; z_l <= interior_end; ++z_l) {
            computeStencilPlane(input, output, z_l, nx, ny, stride);
        }
    }

    // Wait for halo exchange to complete
    if (nreqs > 0) MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Compute stencil on halo-dependent z-planes
    if (z_l_first <= z_l_last) {
        // z_l=1 reads from ghost z_l=0
        if (z_l_first == 1) {
            computeStencilPlane(input, output, 1, nx, ny, stride);
        }
        // z_l=local_nz reads from ghost z_l=local_nz+1
        if (z_l_last == local_nz && local_nz >= 2) {
            computeStencilPlane(input, output, local_nz, nx, ny, stride);
        }
    }

    // Copy boundary values for this rank's portion
    for (size_t z_l = 1; z_l <= local_nz; ++z_l) {
        const size_t z_global = z_start + (z_l - 1);
        const bool z_boundary = (z_global == 0 || z_global == nz - 1);
        if (z_boundary) {
            // Entire plane is boundary
            for (size_t y = 0; y < ny; ++y)
                for (size_t x = 0; x < nx; ++x)
                    output[z_l * stride + y * nx + x] = input[z_l * stride + y * nx + x];
        } else {
            // Only x/y edges are boundary
            for (size_t y = 0; y < ny; ++y) {
                output[z_l * stride + y * nx + 0] = input[z_l * stride + y * nx + 0];
                output[z_l * stride + y * nx + (nx - 1)] = input[z_l * stride + y * nx + (nx - 1)];
            }
            for (size_t x = 0; x < nx; ++x) {
                output[z_l * stride + 0 * nx + x] = input[z_l * stride + 0 * nx + x];
                output[z_l * stride + (ny - 1) * nx + x] = input[z_l * stride + (ny - 1) * nx + x];
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
    
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D domain decomposition along Z axis
    const size_t base_nz = nz / static_cast<size_t>(nprocs);
    const size_t remainder = nz % static_cast<size_t>(nprocs);
    const size_t local_nz = base_nz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t z_start = base_nz * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    // Neighbor ranks (MPI_PROC_NULL for no-neighbor boundaries)
    const int rank_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rank_above = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Allocate local grids with ghost layers (local_nz + 2 z-planes)
    const size_t local_size = nx * ny * (local_nz + 2);
    std::vector<Real> grid1(local_size, 0.0);
    std::vector<Real> grid2(local_size, 0.0);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, local_nz, z_start);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in  = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2 : grid1;
        stencilIterationMPI(in, out, nx, ny, local_nz, z_start, nz,
                            rank_below, rank_above, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double max_duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_duration_ms);
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather full grid on rank 0 for results/validation
    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    if (printResults || validate) {
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            size_t r_local_nz = base_nz + (static_cast<size_t>(r) < remainder ? 1 : 0);
            size_t r_z_start = base_nz * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder);
            recvcounts[r] = static_cast<int>(r_local_nz * nx * ny);
            displs[r] = static_cast<int>(r_z_start * nx * ny);
        }

        std::vector<Real> fullGrid;
        if (rank == 0) fullGrid.resize(nx * ny * nz);

        // Send real data (skip bottom ghost at offset 0)
        MPI_Gatherv(&finalLocal[nx * ny], static_cast<int>(local_nz * nx * ny), MPI_DOUBLE,
                     fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
