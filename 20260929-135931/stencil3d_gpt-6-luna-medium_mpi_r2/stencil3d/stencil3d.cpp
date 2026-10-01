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
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
    const int activeSize = static_cast<int>(std::min<size_t>(nz, worldSize));
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    const bool active = worldRank < activeSize;
    size_t zStart = 0, localNz = 0;
    int rank = 0;
    if (active) {
        MPI_Comm_rank(activeComm, &rank);
        const size_t base = nz / activeSize, rem = nz % activeSize;
        localNz = base + (static_cast<size_t>(rank) < rem);
        zStart = static_cast<size_t>(rank) * base + std::min<size_t>(rank, rem);
    }
    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
        printf("Initializing grid...\nRunning stencil computation...\n");
    }

    const size_t plane = nx * ny;
    const size_t localSize = (localNz + 2) * plane;
    std::vector<Real> grid1(active ? localSize : 0), grid2(active ? localSize : 0);
    if (active) {
        for (size_t lz = 0; lz < localNz; ++lz) {
            const size_t gz = zStart + lz;
            for (size_t p = 0; p < plane; ++p)
                grid1[(lz + 1) * plane + p] = ((gz * plane + p) % 19) * 1.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    if (active) {
        const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int next = rank + 1 == activeSize ? MPI_PROC_NULL : rank + 1;
        for (int iter = 0; iter < iterations; ++iter) {
            auto& in = (iter % 2 == 0) ? grid1 : grid2;
            auto& out = (iter % 2 == 0) ? grid2 : grid1;
            MPI_Sendrecv(in.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                         in.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0, activeComm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(in.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                         in.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1, activeComm, MPI_STATUS_IGNORE);
            for (size_t lz = 0; lz < localNz; ++lz) {
                const size_t gz = zStart + lz;
                const size_t base = (lz + 1) * plane;
                for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = base + y * nx + x;
                    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == nz) out[idx] = in[idx];
                    else out[idx] = (in[idx] + in[idx-1] + in[idx+1] + in[idx-nx] + in[idx+nx] + in[idx-plane] + in[idx+plane]) / 7.0;
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long localDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(maxDurationMs);
    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = static_cast<double>((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = duration.count() ? cellUpdates / (duration.count() / 1000.0) / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        std::vector<int> counts, displs;
        if (worldRank == 0) { counts.resize(activeSize); displs.resize(activeSize); }
        size_t offset = 0;
        for (int r = 0; r < activeSize; ++r) {
            const size_t countZ = nz / activeSize + (static_cast<size_t>(r) < nz % activeSize);
            if (worldRank == 0) { counts[r] = static_cast<int>(countZ * plane); displs[r] = static_cast<int>(offset); }
            offset += countZ * plane;
        }
        std::vector<Real> packed;
        if (active) packed.assign(localFinal.begin() + plane, localFinal.begin() + (localNz + 1) * plane);
        if (worldRank == 0) finalGrid.resize(nx * ny * nz);
        MPI_Gatherv(active ? packed.data() : nullptr, static_cast<int>(packed.size()), MPI_DOUBLE,
                    worldRank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (worldRank == 0 && printResults) print_results(finalGrid, "Grid");
    if (validate) {
        int valid = 1;
        if (worldRank == 0) {
            printf("Validating result...\n");
            valid = validateResult(finalGrid, nx, ny, nz) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (active) MPI_Comm_free(&activeComm);
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    if (active) MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return 0;
}
