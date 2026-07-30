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

void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t local_nz,
                    const size_t global_z_offset) {
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = global_z_offset + z;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid[idx3(x, y, z, nx, ny)] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Exchange ghost planes between neighboring ranks (1D slab decomposition along Z)
// Buffer layout: [ghost_bottom(1 plane) | owned planes(1..local_nz) | ghost_top(1 plane)]
// Ghost planes are at indices 0 and local_nz+1
static void exchangeGhostPlanes(Real* buf, size_t plane_size, size_t local_nz,
                                int rank_above, int rank_below, MPI_Comm comm) {
    // Send top owned plane (index local_nz) to rank_above, receive bottom ghost (index 0) from rank_below
    MPI_Sendrecv(buf + local_nz * plane_size, (int)plane_size, MPI_DOUBLE, rank_above, 0,
                 buf + 0,                     (int)plane_size, MPI_DOUBLE, rank_below, 0,
                 comm, MPI_STATUS_IGNORE);

    // Send bottom owned plane (index 1) to rank_below, receive top ghost (index local_nz+1) from rank_above
    MPI_Sendrecv(buf + 1 * plane_size,               (int)plane_size, MPI_DOUBLE, rank_below, 1,
                 buf + (local_nz + 1) * plane_size,  (int)plane_size, MPI_DOUBLE, rank_above, 1,
                 comm, MPI_STATUS_IGNORE);
}

// 7-point stencil computation on local slab with ghost cells
// Buffer layout: indices 0..local_nz+1, where 0 and local_nz+1 are ghost planes
// Owned planes are 1..local_nz
static void stencilIteration(const Real* __restrict__ input,
                             Real* __restrict__ output,
                             const size_t nx, const size_t ny, const size_t local_nz,
                             const size_t global_z_offset, const size_t global_nz) {
    const size_t plane_size = nx * ny;

    for (size_t k = 1; k <= local_nz; ++k) {
        const size_t global_z = global_z_offset + (k - 1);

        if (global_z == 0 || global_z == global_nz - 1) {
            // Global boundary plane: copy entire plane
            std::memcpy(output + k * plane_size, input + k * plane_size, plane_size * sizeof(Real));
        } else {
            // Interior plane: compute stencil for interior (x,y) points
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = k * plane_size + y * nx + x;
                    const Real center = input[idx];
                    const Real left   = input[idx - 1];
                    const Real right  = input[idx + 1];
                    const Real front  = input[idx - nx];
                    const Real back   = input[idx + nx];
                    const Real bottom = input[idx - plane_size];
                    const Real top    = input[idx + plane_size];
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
            // Copy x/y boundary values for this plane
            // y=0 row
            std::memcpy(output + k * plane_size, input + k * plane_size, nx * sizeof(Real));
            // y=ny-1 row
            std::memcpy(output + k * plane_size + (ny - 1) * nx,
                        input + k * plane_size + (ny - 1) * nx, nx * sizeof(Real));
            // x=0 and x=nx-1 columns for interior y
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t idx = k * plane_size + y * nx;
                output[idx] = input[idx];
                output[idx + nx - 1] = input[idx + nx - 1];
            }
        }
    }
}

bool validateResult(const Real* grid, const size_t nx, const size_t ny, const size_t local_nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    const size_t local_size = nx * ny * local_nz;
    for (size_t i = 0; i < local_size; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (size_t i = 0; i < local_size; ++i) {
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 needs to parse)
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

    // Broadcast parameters from rank 0 to all ranks
    MPI_Bcast(&nx, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D slab decomposition along Z axis
    // Distribute nz planes across 'size' ranks as evenly as possible
    const size_t base_planes = nz / size;
    const size_t remainder = nz % size;
    const size_t local_nz = base_planes + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t global_z_offset = base_planes * rank + std::min(static_cast<size_t>(rank), remainder);

    // Buffer layout with ghost planes: [ghost_bottom | owned planes | ghost_top]
    // Owned planes are at indices 1..local_nz
    // Ghost planes are at indices 0 and local_nz+1
    const size_t plane_size = nx * ny;
    const size_t buffer_size = (local_nz + 2) * plane_size;

    // Allocate grids (double buffering) with ghost planes
    std::vector<Real> grid1(buffer_size, 0.0);
    std::vector<Real> grid2(buffer_size, 0.0);

    // Initialize owned planes (indices 1..local_nz)
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1.data() + plane_size, nx, ny, local_nz, global_z_offset);

    // Determine neighbor ranks
    const int rank_above = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    const int rank_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0) ? grid1.data() : grid2.data();
        Real* output = (iter % 2 == 0) ? grid2.data() : grid1.data();

        // Exchange ghost planes
        exchangeGhostPlanes(input, plane_size, local_nz, rank_above, rank_below, MPI_COMM_WORLD);

        // Compute stencil
        stencilIteration(input, output, nx, ny, local_nz, global_z_offset, nz);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Get maximum time across all ranks
    double local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double max_duration;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_duration);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_duration / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for validation and printing
    const Real* finalGrid = (iterations % 2 == 0) ? grid1.data() + plane_size : grid2.data() + plane_size;

    if (printResults || validate) {
        // Gather all data to rank 0
        std::vector<Real> fullGrid;
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int r = 0; r < size; ++r) {
            const size_t r_local_nz = base_planes + (static_cast<size_t>(r) < remainder ? 1 : 0);
            recvcounts[r] = static_cast<int>(nx * ny * r_local_nz);
            displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1];
        }

        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
        }

        // Convert local size_t counts to int for MPI
        const int local_count = static_cast<int>(nx * ny * local_nz);

        MPI_Gatherv(finalGrid, local_count, MPI_DOUBLE,
                    fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullGrid.data(), nx, ny, nz);

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
