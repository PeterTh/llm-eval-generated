#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t sliceSize = nx * ny;
    Real* data = grid.data();
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * sliceSize + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                data[idx] = static_cast<Real>(idx % 19);
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const Real* __restrict__ input,
                      Real* __restrict__ output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const size_t sliceSize = nx * ny;
    const Real inv7 = 1.0 / 7.0;

    // Process interior points (not on boundaries)
    #pragma omp parallel for schedule(static)
    for (size_t z = 1; z < nz - 1; ++z) {
        const size_t zOffset = z * sliceSize;
        const size_t zOffsetPrev = (z - 1) * sliceSize;
        const size_t zOffsetNext = (z + 1) * sliceSize;
        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t base = zOffset + y * nx;
            const size_t baseYprev = zOffset + (y - 1) * nx;
            const size_t baseYnext = zOffset + (y + 1) * nx;
            const size_t baseZprev = zOffsetPrev + y * nx;
            const size_t baseZnext = zOffsetNext + y * nx;
            for (size_t x = 1; x < nx - 1; ++x) {
                const Real center = input[base + x];
                const Real left   = input[base + x - 1];
                const Real right  = input[base + x + 1];
                const Real front  = input[baseYprev + x];
                const Real back   = input[baseYnext + x];
                const Real bottom = input[baseZprev + x];
                const Real top    = input[baseZnext + x];

                output[base + x] = (center + left + right + front + back + bottom + top) * inv7;
            }
        }
    }

    // Copy boundary values - optimized to avoid branching
    #pragma omp parallel
    {
        // Top and bottom Z faces
        #pragma omp for schedule(static) collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = y * nx + x;
                output[idx] = input[idx];                          // z=0
                output[(nz - 1) * sliceSize + idx] = input[(nz - 1) * sliceSize + idx]; // z=nz-1
            }
        }

        // Front and back Y faces (interior z only)
        #pragma omp for schedule(static) collapse(2)
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = z * sliceSize + x;
                output[idx] = input[idx];                          // y=0
                output[idx + (ny - 1) * nx] = input[idx + (ny - 1) * nx]; // y=ny-1
            }
        }

        // Left and right X faces (interior z and y only)
        #pragma omp for schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            const size_t zBase = z * sliceSize;
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t idx = zBase + y * nx;
                output[idx] = input[idx];       // x=0
                output[idx + nx - 1] = input[idx + nx - 1]; // x=nx-1
            }
        }
    }
}

// Overload that accepts vectors (convenience wrapper)
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    stencilIteration(input.data(), output.data(), nx, ny, nz);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t N = grid.size();
    const Real* data = grid.data();

    // 1. No NaN or Inf values
    int badCount = 0;
    #pragma omp parallel for reduction(+:badCount) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            ++badCount;
        }
    }
    if (badCount > 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = data[0];
    Real maxVal = data[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 1; i < N; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
