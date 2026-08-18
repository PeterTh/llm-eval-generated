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

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, size_t globalZ) {
    for (size_t z = 1; z <= localNz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIdx = (globalZ + z - 1) * (nx * ny) + y * nx + x;
                grid[idx3(x, y, z, nx, ny)] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      size_t nx, size_t ny, size_t localNz, size_t globalZ, size_t globalNz) {
    // The destination starts as a copy so X/Y boundary values are preserved.
    std::copy(input.begin(), input.end(), output.begin());
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t gz = globalZ + z - 1;
        if (gz == 0 || gz + 1 == globalNz) continue;
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    
    if (nz < static_cast<size_t>(ranks) || nx < 3 || ny < 3 || iterations < 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be >= 3, Z must be >= MPI ranks, and iterations nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI, %d ranks)\n", ranks);
        printf("Grid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t globalZ = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny;
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, localNz, globalZ);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? grid1 : grid2;
        MPI_Sendrecv(in.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
                     rank > 0 ? rank - 1 : MPI_PROC_NULL, 0,
                     in.data(), static_cast<int>(plane), MPI_DOUBLE,
                     rank > 0 ? rank - 1 : MPI_PROC_NULL, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(in.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE,
                     rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL, 1,
                     in.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
                     rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, localNz, globalZ, nz);
        } else {
            stencilIteration(grid2, grid1, nx, ny, localNz, globalZ, nz);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long long elapsed = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %lld ms\n", elapsed);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    if (rank == 0) printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(gridSize);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { size_t n = base + (static_cast<size_t>(r) < extra); counts[r] = static_cast<int>(n * plane); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * plane); }
    MPI_Gatherv(localFinal.data() + plane, counts[rank], MPI_DOUBLE, rank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) { MPI_Finalize(); return 0; }
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
