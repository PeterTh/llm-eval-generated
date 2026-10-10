#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// Leave storage untouched until parallel initialization, so each worker first
// touches its own pages on NUMA machines instead of placing them all on one CPU.
template <typename T>
struct UninitializedAllocator : std::allocator<T> {
    template <typename U>
    struct rebind { using other = UninitializedAllocator<U>; };

    template <typename U>
    void construct(U* ptr) {
        ::new (static_cast<void*>(ptr)) U;
    }
};

using Grid = std::vector<Real, UninitializedAllocator<Real>>;

int gridThreadCount(const size_t gridSize) {
    const int maximum = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS")) return maximum;

    // Small volumes cannot amortize large-team synchronization. Without
    // affinity, leave more work per worker to reduce scheduling overhead.
    const size_t cellsPerThread = omp_get_proc_bind() == omp_proc_bind_false
        ? 131072 : 32768;
    const size_t workers = std::max<size_t>(1, gridSize / cellsPerThread);
    return static_cast<int>(std::min<size_t>(maximum, workers));
}

void initializeGrid(Grid& grid, Grid& scratch, const int iterations, const int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        const Real value = (idx % 19) * 1.0;
        grid[idx] = value;
        // Boundaries never change. Seed both buffers once instead of scanning
        // the whole volume to copy boundaries after every iteration.
        scratch[idx] = iterations > 0 ? value : 0.0;
    }
}

inline void stencilRow(const Real* input, Real* output,
                       const size_t nx, const size_t plane, const size_t row,
                       const size_t xBegin, const size_t xEnd) {
    #pragma omp simd
    for (size_t x = xBegin; x < xEnd; ++x) {
        const size_t idx = row + x;
        const Real center = input[idx];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];
        const Real bottom = input[idx - plane];
        const Real top = input[idx + plane];

        // Preserve the original operation order and division.
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

inline void stencilRows(const Real* input, Real* output,
                        const size_t nx, const size_t plane,
                        const size_t zBegin, const size_t zEnd,
                        const size_t yBegin, const size_t yEnd) {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = yBegin; y < yEnd; ++y) {
            stencilRow(input, output, nx, plane, z * plane + y * nx, 1, nx - 1);
        }
    }
}

// Called collectively by the persistent OpenMP team. The worksharing barrier
// completes all output writes before the next iteration reads that buffer.
void stencilIteration(const Real* input, Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) return;

    const size_t plane = nx * ny;
    const size_t threads = static_cast<size_t>(omp_get_num_threads());
    const size_t tileZ = 16;
    // Keep the working set of three input strips and one output strip small
    // enough for private caches, while retaining contiguous SIMD rows.
    const size_t tileY = std::min(ny - 2, std::max<size_t>(1, 8192 / nx));
    const size_t tiles = ((nz - 3) / tileZ + 1) * ((ny - 3) / tileY + 1);
    if ((ny - 2) * (nz - 2) < threads && nx > 1024) {
        // Very wide, shallow grids need worksharing within rows as well.
        const size_t tileX = 512;
        #pragma omp for collapse(3) schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; x += tileX) {
                    stencilRow(input, output, nx, plane, z * plane + y * nx,
                               x, std::min(x + tileX, nx - 1));
                }
            }
        }
    } else if (plane >= 131072 && threads >= 16 && tiles >= threads) {
        #pragma omp for collapse(2) schedule(static)
        for (size_t z = 1; z < nz - 1; z += tileZ) {
            for (size_t y = 1; y < ny - 1; y += tileY) {
                stencilRows(input, output, nx, plane,
                            z, std::min(z + tileZ, nz - 1),
                            y, std::min(y + tileY, ny - 1));
            }
        }
    } else {
        #pragma omp for collapse(2) schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                stencilRows(input, output, nx, plane, z, z + 1, y, y + 1);
            }
        }
    }
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz, const int threads) {
    // Simple sanity checks
    
    // Check finiteness and the value range in a single parallel pass.
    int nonFinite = 0;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:nonFinite) reduction(min:minVal) reduction(max:maxVal)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        const Real val = grid[idx];
        nonFinite |= std::isnan(val) || std::isinf(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (nonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // Boundary values remain fixed in both buffers.
    
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
    const int threads = gridThreadCount(gridSize);
    
    // Allocate grids (double buffering)
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, grid2, iterations, threads);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(threads)
    {
        Real* input = grid1.data();
        Real* output = grid2.data();
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration(input, output, nx, ny, nz);
            // Each worker swaps its own pointers after the worksharing barrier.
            std::swap(input, output);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (nx > 2 && ny > 2 && nz > 2 && iterations > 0)
        ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations : 0.0;
    double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / seconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const Grid& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz, threads);
        
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
