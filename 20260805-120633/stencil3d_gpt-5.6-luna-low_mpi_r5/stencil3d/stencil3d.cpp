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

// Decompose complete Z planes.  Local storage has one ghost plane at each end.
void stencilIterationMPI(std::vector<Real>& input, std::vector<Real>& output,
                         size_t nx, size_t ny, size_t localNz, size_t globalZ0,
                         size_t globalNz, int rank, int ranks) {
    const size_t plane = nx * ny;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Request requests[4];
    MPI_Irecv(input.data(), static_cast<int>(plane), MPI_DOUBLE,
              lower, 17, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(input.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
              upper, 18, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
              lower, 18, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(input.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE,
              upper, 17, MPI_COMM_WORLD, &requests[3]);

    auto updatePlane = [&](size_t lz) {
        const size_t gz = globalZ0 + lz - 1;
        const size_t base = lz * plane;
        if (gz == 0 || gz + 1 == globalNz) {
            std::copy_n(input.data() + base, plane, output.data() + base);
            return;
        }
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = base + y * nx + x;
                if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny)
                    output[p] = input[p];
                else
                    output[p] = (input[p] + input[p-1] + input[p+1] +
                                 input[p-nx] + input[p+nx] + input[p-plane] +
                                 input[p+plane]) / 7.0;
            }
        }
    };
    for (size_t lz = 2; lz + 1 < localNz; ++lz) updatePlane(lz);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    if (localNz == 1) updatePlane(1);
    else { updatePlane(1); updatePlane(localNz); }
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
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) { if (rank == 0) printf("Invalid grid or iteration count\n"); MPI_Finalize(); return 1; }
    if (static_cast<size_t>(ranks) > nz) { if (rank == 0) printf("MPI rank count cannot exceed Z dimension\n"); MPI_Finalize(); return 1; }
    const size_t base = nz / ranks, remainder = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZ0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    if (rank == 0) { printf("3D Stencil Benchmark (MPI)\n"); printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz); printf("Iterations: %d\n", iterations); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    for (size_t lz = 1; lz <= localNz; ++lz)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                grid1[idx3(x, y, lz, nx, ny)] = (idx3(x, y, globalZ0 + lz - 1, nx, ny) % 19) * 1.0;
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationMPI(grid1, grid2, nx, ny, localNz, globalZ0, nz, rank, ranks);
        } else {
            stencilIterationMPI(grid2, grid1, nx, ny, localNz, globalZ0, nz, rank, ranks);
        }
    }
    
    const double localSeconds = MPI_Wtime() - start;
    double maxSeconds = 0.0; MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %.3f ms\n", maxSeconds * 1000.0);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / maxSeconds / 1e6;  // Million cell updates per second
    if (rank == 0) printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> gathered;
        if (rank == 0) gathered.resize(nx * ny * nz);
        std::vector<int> counts(ranks), displs(ranks);
        for (int r = 0; r < ranks; ++r) { size_t n = base + (static_cast<size_t>(r) < remainder ? 1 : 0); counts[r] = static_cast<int>(n * plane); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * plane); }
        MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE, gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(gathered, "Grid");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        std::vector<Real> gathered;
        if (rank == 0) gathered.resize(nx * ny * nz);
        std::vector<int> counts(ranks), displs(ranks);
        for (int r = 0; r < ranks; ++r) { size_t n = base + (static_cast<size_t>(r) < remainder ? 1 : 0); counts[r] = static_cast<int>(n * plane); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * plane); }
        MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE, gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        bool valid = rank == 0 ? validateResult(gathered, nx, ny, nz) : true;
        
        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
