#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <unistd.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Parallel initialization with the same layout as the compute loops
// so pages are first-touched by the threads that will use them.
void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t nz, const Real fill, const bool useIndexPattern) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = useIndexPattern ? (idx % 19) * 1.0 : fill;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const Real* __restrict in,
                      Real* __restrict out,
                      const size_t nx, const size_t ny, const size_t nz) {

    // Interior and boundary writes touch disjoint indices and only read from
    // the input grid, so all loops can run concurrently in one parallel region.
    #pragma omp parallel
    {
        // Process interior points (not on boundaries)
        #pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);

                    const Real center = in[idx];
                    const Real left = in[idx3(x-1, y, z, nx, ny)];
                    const Real right = in[idx3(x+1, y, z, nx, ny)];
                    const Real front = in[idx3(x, y-1, z, nx, ny)];
                    const Real back = in[idx3(x, y+1, z, nx, ny)];
                    const Real bottom = in[idx3(x, y, z-1, nx, ny)];
                    const Real top = in[idx3(x, y, z+1, nx, ny)];

                    // Simple averaging stencil
                    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }

        // Copy boundary values: z = 0 and z = nz-1 faces (full planes)
        #pragma omp for schedule(static) nowait
        for (size_t y = 0; y < ny; ++y) {
            std::memcpy(&out[idx3(0, y, 0, nx, ny)], &in[idx3(0, y, 0, nx, ny)], nx * sizeof(Real));
            std::memcpy(&out[idx3(0, y, nz-1, nx, ny)], &in[idx3(0, y, nz-1, nx, ny)], nx * sizeof(Real));
        }

        // Remaining boundary values on interior z planes:
        // y = 0 and y = ny-1 rows, plus x = 0 and x = nx-1 columns
        #pragma omp for schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            std::memcpy(&out[idx3(0, 0, z, nx, ny)], &in[idx3(0, 0, z, nx, ny)], nx * sizeof(Real));
            std::memcpy(&out[idx3(0, ny-1, z, nx, ny)], &in[idx3(0, ny-1, z, nx, ny)], nx * sizeof(Real));
            for (size_t y = 1; y < ny - 1; ++y) {
                out[idx3(0, y, z, nx, ny)] = in[idx3(0, y, z, nx, ny)];
                out[idx3(nx-1, y, z, nx, ny)] = in[idx3(nx-1, y, z, nx, ny)];
            }
        }
    }
}

bool validateResult(const Real* grid, const size_t gridSize, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool finite = true;
    #pragma omp parallel for schedule(static) reduction(&&:finite)
    for (size_t i = 0; i < gridSize; ++i) {
        finite = finite && !std::isnan(grid[i]) && !std::isinf(grid[i]);
    }
    if (!finite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < gridSize; ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
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
    // Default to one thread per physical core, bound spread across sockets.
    // The OpenMP runtime reads these variables before main() runs, so set
    // them and re-exec once; explicit user settings take precedence
    // (overwrite = 0).
    if (getenv("STENCIL3D_OMP_INIT") == nullptr) {
        setenv("STENCIL3D_OMP_INIT", "1", 1);
        setenv("OMP_PROC_BIND", "spread", 0);
        setenv("OMP_PLACES", "cores", 0);
        execv("/proc/self/exe", argv);
        // If re-exec fails, continue with whatever environment we have.
    }
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int places = omp_get_num_places();
        if (places > 0 && places < omp_get_max_threads()) {
            omp_set_num_threads(places);
        }
    }

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
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering). Raw uninitialized allocations so the
    // parallel initialization below performs the NUMA first touch of the pages.
    Real* grid1 = new Real[gridSize];
    Real* grid2 = new Real[gridSize];

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz, 0.0, true);
    initializeGrid(grid2, nx, ny, nz, 0.0, false);  // first-touch only; fully overwritten below

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
    const Real* finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        print_results(std::vector<Real>(finalGrid, finalGrid + gridSize), "Grid");
    }

    int ret = 0;

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, gridSize, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    delete[] grid1;
    delete[] grid2;
    return ret;
}
