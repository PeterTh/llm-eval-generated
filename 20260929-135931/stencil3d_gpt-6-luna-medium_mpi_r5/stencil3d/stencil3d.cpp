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

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (!worldRank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!worldRank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0) {
        if (!worldRank) fprintf(stderr, "Grid dimensions must be at least 2 and iterations nonnegative\n");
        MPI_Finalize(); return 1;
    }
    // Keep at least one z plane on each participating rank.
    const int activeSize = static_cast<int>(std::min<size_t>(nz, static_cast<size_t>(worldSize)));
    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    if (worldRank >= activeSize) { MPI_Finalize(); return 0; }
    int rank = 0, size = 1;
    MPI_Comm_rank(activeComm, &rank); MPI_Comm_size(activeComm, &size);

    const size_t base = nz / static_cast<size_t>(size), rem = nz % static_cast<size_t>(size);
    const size_t localNz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min<size_t>(rank, rem);
    const size_t plane = nx * ny;
    const size_t slab = (localNz + 2) * plane;
    std::vector<Real> a(slab), b(slab);
    // Index 1..localNz are owned planes; 0 and localNz+1 are halo planes.
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t gz = zStart + lz - 1;
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIdx = idx3(x, y, gz, nx, ny);
                a[lz * plane + y * nx + x] = static_cast<Real>(globalIdx % 19);
            }
    }
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(activeComm);
    const auto startTime = std::chrono::high_resolution_clock::now();
    const int prev = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? a : b;
        std::vector<Real>& out = (iter % 2 == 0) ? b : a;
        MPI_Sendrecv(in.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                     in.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0, activeComm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(in.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                     in.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1, activeComm, MPI_STATUS_IGNORE);
        for (size_t lz = 1; lz <= localNz; ++lz) {
            const size_t gz = zStart + lz - 1;
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
                const size_t q = lz * plane + y * nx + x;
                if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == nz) out[q] = in[q];
                else out[q] = (in[q] + in[q-1] + in[q+1] + in[q-nx] + in[q+nx] + in[q-plane] + in[q+plane]) / 7.0;
            }
        }
    }
    MPI_Barrier(activeComm);
    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    long long maxMs = 0; MPI_Reduce(&ms, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, activeComm);
    if (!rank) {
        printf("Computation time: %lld ms\n", maxMs);
        const double updates = static_cast<double>((nx-2) * (ny-2) * (nz-2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", maxMs ? updates / (maxMs / 1000.0) / 1e6 : 0.0);
    }

    const std::vector<Real>& finalSlab = (iterations % 2 == 0) ? a : b;
    std::vector<Real> global;
    std::vector<int> counts(size), displs(size);
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t n = (nz / size + (static_cast<size_t>(r) < nz % size ? 1 : 0)) * plane;
        counts[r] = static_cast<int>(n); displs[r] = static_cast<int>(offset); offset += n;
    }
    if (!rank && (validate || printResults)) global.resize(nx * ny * nz);
    MPI_Gatherv(finalSlab.data() + plane, counts[rank], MPI_DOUBLE, rank == 0 ? global.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, activeComm);
    int result = 0;
    if (!rank && (validate || printResults)) {
        if (printResults) print_results(global, "Grid");
        if (validate) {
            bool valid = validateResult(global, nx, ny, nz);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) result = 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, activeComm);
    MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return result;
}
