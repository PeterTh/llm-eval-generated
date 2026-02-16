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

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries)
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
    
    // Copy boundary values
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

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Partition the grid along the z-dimension
    size_t z_per_rank = nz / size;
    size_t z_rem = nz % size;
    size_t z_start = rank * z_per_rank + (rank < z_rem ? rank : z_rem);
    size_t local_nz = z_per_rank + (rank < z_rem ? 1 : 0);
    size_t z_end = z_start + local_nz;

    // Add halo layers
    size_t local_nz_with_halo = local_nz + 2;
    size_t local_grid_size = nx * ny * local_nz_with_halo;

    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);

    // Initialize local grid (excluding halos)
    std::vector<Real> tmp_init(nx * ny * local_nz);
    initializeGrid(tmp_init, nx, ny, local_nz);
    std::copy(tmp_init.begin(), tmp_init.end(), grid1.begin() + nx * ny);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Initializing grid...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // Halo exchange
        MPI_Request reqs[4];
        int req_count = 0;
        // Send up, receive from down
        if (rank > 0) {
            MPI_Isend(&grid1[nx * ny], nx * ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&grid1[0], nx * ny, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Send down, receive from up
        if (rank < size - 1) {
            MPI_Isend(&grid1[nx * ny * local_nz], nx * ny, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&grid1[nx * ny * (local_nz + 1)], nx * ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Compute interior (excluding halos)
        for (size_t z = 1; z < local_nz_with_halo - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    size_t idx = z * nx * ny + y * nx + x;
                    Real center = grid1[idx];
                    Real left = grid1[idx - 1];
                    Real right = grid1[idx + 1];
                    Real front = grid1[idx - nx];
                    Real back = grid1[idx + nx];
                    Real bottom = grid1[idx - nx * ny];
                    Real top = grid1[idx + nx * ny];
                    grid2[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
        // Wait for halo exchange to complete
        if (req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        // Copy boundary values (global boundaries only)
        for (size_t z = 0; z < local_nz_with_halo; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    bool is_global_z0 = (rank == 0 && z == 0);
                    bool is_global_zn = (rank == size - 1 && z == local_nz_with_halo - 1);
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || is_global_z0 || is_global_zn) {
                        size_t idx = z * nx * ny + y * nx + x;
                        grid2[idx] = grid1[idx];
                    }
                }
            }
        }
        grid1.swap(grid2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double local_time = duration.count();
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)max_time);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_time / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for validation/printing
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx * ny * nz);
    std::vector<int> recvcounts(size), displs(size);
    for (int r = 0, z = 0; r < size; ++r) {
        size_t lz = nz / size + (r < (int)(nz % size) ? 1 : 0);
        recvcounts[r] = nx * ny * lz;
        displs[r] = z * nx * ny;
        z += lz;
    }
    MPI_Gatherv(&grid1[nx * ny], nx * ny * local_nz, MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        print_results(finalGrid, "Grid");
    }
    if (rank == 0 && validate) {
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
