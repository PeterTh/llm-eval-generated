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
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }
    
    // Partition grid along Z dimension
    size_t z_per_rank = nz / size;
    size_t z_rem = nz % size;
    size_t z_start = rank * z_per_rank + std::min((size_t)rank, z_rem);
    size_t z_count = z_per_rank + (rank < z_rem ? 1 : 0);
    size_t z_end = z_start + z_count;
    size_t localGridSize = nx * ny * z_count;

    std::vector<Real> grid1(localGridSize);
    std::vector<Real> grid2(localGridSize);

    // Initialize local grid
    for (size_t z = z_start; z < z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t localIdx = (z - z_start) * (nx * ny) + y * nx + x;
                size_t globalIdx = idx3(x, y, z, nx, ny);
                grid1[localIdx] = (globalIdx % 19) * 1.0;
            }
        }
    }

    // Buffers for halo exchange
    std::vector<Real> topHalo(nx * ny), bottomHalo(nx * ny);
    std::vector<Real> recvTop(nx * ny), recvBottom(nx * ny);

    // Timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halos
        if (size > 1) {
            // Send/recv top
            if (rank < size - 1 && z_count > 0) {
                std::copy(grid1.end() - nx * ny, grid1.end(), topHalo.begin());
                MPI_Send(topHalo.data(), nx * ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            }
            if (rank > 0 && z_count > 0) {
                MPI_Recv(recvBottom.data(), nx * ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Send/recv bottom
            if (rank > 0 && z_count > 0) {
                std::copy(grid1.begin(), grid1.begin() + nx * ny, bottomHalo.begin());
                MPI_Send(bottomHalo.data(), nx * ny, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
            }
            if (rank < size - 1 && z_count > 0) {
                MPI_Recv(recvTop.data(), nx * ny, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Stencil iteration
        for (size_t z = 0; z < z_count; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    size_t idx = z * (nx * ny) + y * nx + x;
                    Real center = grid1[idx];
                    Real left = grid1[z * (nx * ny) + y * nx + (x - 1)];
                    Real right = grid1[z * (nx * ny) + y * nx + (x + 1)];
                    Real front = grid1[z * (nx * ny) + (y - 1) * nx + x];
                    Real back = grid1[z * (nx * ny) + (y + 1) * nx + x];
                    Real bottom, top;
                    // Handle halo for bottom/top
                    if (z == 0) {
                        if (rank == 0) bottom = center;
                        else bottom = recvBottom[y * nx + x];
                    } else {
                        bottom = grid1[(z - 1) * (nx * ny) + y * nx + x];
                    }
                    if (z == z_count - 1) {
                        if (rank == size - 1) top = center;
                        else top = recvTop[y * nx + x];
                    } else {
                        top = grid1[(z + 1) * (nx * ny) + y * nx + x];
                    }
                    grid2[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
        // Copy boundary values
        for (size_t z = 0; z < z_count; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                        (z == 0 && rank == 0) || (z == z_count - 1 && rank == size - 1)) {
                        size_t idx = z * (nx * ny) + y * nx + x;
                        grid2[idx] = grid1[idx];
                    }
                }
            }
        }
        std::swap(grid1, grid2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
    }

    // Calculate performance metrics
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    if (rank == 0) {
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results for validation/output
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx * ny * nz);
    MPI_Gather(grid1.data(), localGridSize, MPI_DOUBLE,
               rank == 0 ? finalGrid.data() : nullptr, localGridSize, MPI_DOUBLE,
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
