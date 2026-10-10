#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t count = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < count; ++idx) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Run all 7-point stencil iterations with a single persistent thread team.
void stencilIterations(std::vector<Real>& grid1,
                       std::vector<Real>& grid2,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int iterations) {
    if (iterations <= 0 || nx == 0 || ny == 0 || nz == 0) return;

    const size_t plane = nx * ny;
    const size_t xEnd = nx > 1 ? nx - 1 : 1;
    const size_t yEnd = ny > 1 ? ny - 1 : 1;
    const size_t zEnd = nz > 1 ? nz - 1 : 1;
    Real* const first = grid1.data();
    Real* const second = grid2.data();

    #pragma omp parallel
    {
        // Boundaries never change. Seed the second buffer once, visiting only
        // boundary elements rather than scanning every cell on every iteration.
        // No barrier is needed here: the first iteration reads only grid1 and
        // writes disjoint interior elements of grid2.
        #pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                const size_t row = idx3(0, y, z, nx, ny);
                if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                    std::copy_n(first + row, nx, second + row);
                } else {
                    second[row] = first[row];
                    second[row + nx - 1] = first[row + nx - 1];
                }
            }
        }

        for (int iter = 0; iter < iterations; ++iter) {
            const Real* const input = (iter % 2 == 0) ? first : second;
            Real* const output = (iter % 2 == 0) ? second : first;

            // Collapse the outer dimensions so shallow grids also expose enough
            // work. Static scheduling preserves locality, with contiguous SIMD
            // updates along X and the original floating-point operation order.
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 1; z < zEnd; ++z) {
                for (size_t y = 1; y < yEnd; ++y) {
                    const size_t row = idx3(0, y, z, nx, ny);
                    #pragma omp simd
                    for (size_t x = 1; x < xEnd; ++x) {
                        const size_t idx = row + x;
                        const Real center = input[idx];
                        const Real left = input[idx - 1];
                        const Real right = input[idx + 1];
                        const Real front = input[idx - nx];
                        const Real back = input[idx + nx];
                        const Real bottom = input[idx - plane];
                        const Real top = input[idx + plane];
                        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                    }
                }
            }
            // The implicit worksharing barrier completes every output (and the
            // initial boundary copies) before any thread starts the next step.
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
    
    stencilIterations(grid1, grid2, nx, ny, nz, iterations);
    
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
