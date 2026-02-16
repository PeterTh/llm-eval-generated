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
    }

    // Partition the grid along the Z dimension
    size_t local_z = nz / size;
    size_t remainder = nz % size;
    size_t z_start = rank * local_z + (rank < remainder ? rank : remainder);
    size_t z_count = local_z + (rank < remainder ? 1 : 0);
    size_t z_end = z_start + z_count;

    size_t localGridSize = nx * ny * z_count;
    std::vector<Real> local_grid1(localGridSize);
    std::vector<Real> local_grid2(localGridSize);

    // Initialize local grid
    // Each rank initializes its own subdomain
    for (size_t lz = 0; lz < z_count; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_z = z_start + lz;
                size_t idx = lz * (nx * ny) + y * nx + x;
                local_grid1[idx] = ((global_z * nx * ny + y * nx + x) % 19) * 1.0;
            }
        }
    }

    // Helper lambda for local idx3
    auto lidx3 = [nx, ny](size_t x, size_t y, size_t lz) {
        return lz * (nx * ny) + y * nx + x;
    };

    // Buffers for halo exchange
    std::vector<Real> send_top(nx * ny), send_bottom(nx * ny);
    std::vector<Real> recv_top(nx * ny), recv_bottom(nx * ny);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real> *in = (iter % 2 == 0) ? &local_grid1 : &local_grid2;
        std::vector<Real> *out = (iter % 2 == 0) ? &local_grid2 : &local_grid1;

        // Halo exchange
        if (size > 1) {
            // Send bottom slice to rank-1, receive top halo from rank-1
            if (rank > 0) {
                std::copy_n(in->data(), nx*ny, send_bottom.data());
                MPI_Sendrecv(send_bottom.data(), nx*ny, MPI_DOUBLE, rank-1, 0,
                             recv_top.data(), nx*ny, MPI_DOUBLE, rank-1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Send top slice to rank+1, receive bottom halo from rank+1
            if (rank < size-1) {
                std::copy_n(in->data() + (z_count-1)*(nx*ny), nx*ny, send_top.data());
                MPI_Sendrecv(send_top.data(), nx*ny, MPI_DOUBLE, rank+1, 1,
                             recv_bottom.data(), nx*ny, MPI_DOUBLE, rank+1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Stencil computation (interior)
        for (size_t lz = 0; lz < z_count; ++lz) {
            size_t global_z = z_start + lz;
            for (size_t y = 1; y < ny-1; ++y) {
                for (size_t x = 1; x < nx-1; ++x) {
                    // Only process interior (not global boundaries)
                    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || global_z == 0 || global_z == nz-1)
                        continue;
                    Real center = (*in)[lidx3(x, y, lz)];
                    Real left = (*in)[lidx3(x-1, y, lz)];
                    Real right = (*in)[lidx3(x+1, y, lz)];
                    Real front = (*in)[lidx3(x, y-1, lz)];
                    Real back = (*in)[lidx3(x, y+1, lz)];
                    Real bottom, top;
                    if (lz == 0) {
                        if (global_z == 0) bottom = center;
                        else bottom = (rank == 0) ? center : recv_top[lidx3(x, y, 0)];
                    } else {
                        bottom = (*in)[lidx3(x, y, lz-1)];
                    }
                    if (lz == z_count-1) {
                        if (global_z == nz-1) top = center;
                        else top = (rank == size-1) ? center : recv_bottom[lidx3(x, y, 0)];
                    } else {
                        top = (*in)[lidx3(x, y, lz+1)];
                    }
                    (*out)[lidx3(x, y, lz)] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
        // Copy boundary values (global boundaries only)
        for (size_t lz = 0; lz < z_count; ++lz) {
            size_t global_z = z_start + lz;
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || global_z == 0 || global_z == nz-1) {
                        (*out)[lidx3(x, y, lz)] = (*in)[lidx3(x, y, lz)];
                    }
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics (global)
    double local_cellUpdates = (double)((nx-2) * (ny-2) * (z_count)) * iterations;
    double global_cellUpdates = 0.0;
    MPI_Reduce(&local_cellUpdates, &global_cellUpdates, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double mcups = global_cellUpdates / (duration.count() / 1000.0) / 1e6;
    if (rank == 0) printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Gather results to rank 0 for validation/printing
    std::vector<Real> finalGrid;
    std::vector<Real> *finalLocal = (iterations % 2 == 0) ? &local_grid1 : &local_grid2;
    std::vector<int> recvcounts(size), displs(size);
    if (rank == 0) {
        finalGrid.resize(nx * ny * nz);
        for (int r = 0; r < size; ++r) {
            size_t lz = nz / size + (r < (nz % size) ? 1 : 0);
            recvcounts[r] = lz * nx * ny;
            displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1];
        }
    }
    MPI_Gatherv(finalLocal->data(), z_count*nx*ny, MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }

    if (validate && rank == 0) {
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
