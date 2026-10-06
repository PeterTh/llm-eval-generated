#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Pin each OpenMP thread to its own CPU (spread over the CPUs available to the
// process) unless the user controls affinity via OMP_PROC_BIND / OMP_PLACES.
// Stable placement keeps the first-touch NUMA locality valid for all sweeps.
void pinThreads() {
    if (std::getenv("OMP_PROC_BIND") || std::getenv("OMP_PLACES")) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
    }
    if (cpus.empty()) return;

    #pragma omp parallel
    {
        const size_t nthreads = omp_get_num_threads();
        const size_t tid = omp_get_thread_num();
        const size_t ncpus = cpus.size();
        const int cpu = nthreads <= ncpus ? cpus[tid * ncpus / nthreads] : cpus[tid % ncpus];
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

// All grid loops share the same static distribution of (z, y) rows across
// threads, so pages are first-touched (NUMA-placed) by the thread that later
// computes on them.
void initializeGrid(Real* __restrict grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

void zeroGrid(Real* __restrict grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            Real* __restrict row = grid + idx3(0, y, z, nx, ny);
            for (size_t x = 0; x < nx; ++x) {
                row[x] = 0.0;
            }
        }
    }
}

// 7-point stencil computation (called from inside a parallel region)
void stencilIteration(const Real* __restrict input,
                      Real* __restrict output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const size_t plane = nx * ny;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            const Real* __restrict in = input + base;
            Real* __restrict out = output + base;

            if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                // Boundary row: copy
                std::memcpy(out, in, nx * sizeof(Real));
                continue;
            }

            const Real* __restrict inF = in - nx;
            const Real* __restrict inB = in + nx;
            const Real* __restrict inD = in - plane;
            const Real* __restrict inU = in + plane;

            // Process interior points (not on boundaries)
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const Real center = in[x];
                const Real left = in[x - 1];
                const Real right = in[x + 1];
                const Real front = inF[x];
                const Real back = inB[x];
                const Real bottom = inD[x];
                const Real top = inU[x];

                // Simple averaging stencil
                out[x] = (center + left + right + front + back + bottom + top) / 7.0;
            }

            // Copy boundary values in x
            out[0] = in[0];
            out[nx - 1] = in[nx - 1];
        }
    }
}

bool validateResult(const Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    // Simple sanity checks
    const size_t n = nx * ny * nz;

    // 1. No NaN or Inf values
    bool bad = false;
    #pragma omp parallel for schedule(static) reduction(||:bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
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
    
    // Allocate grids (double buffering). Left uninitialized here so that the
    // parallel initialization performs the NUMA first touch.
    std::unique_ptr<Real[]> buf1(new Real[gridSize]);
    std::unique_ptr<Real[]> buf2(new Real[gridSize]);
    Real* const grid1 = buf1.get();
    Real* const grid2 = buf2.get();
    
    pinThreads();

    // Initialize
    printf("Initializing grid...\n");
    #pragma omp parallel
    {
        initializeGrid(grid1, nx, ny, nz);
        zeroGrid(grid2, nx, ny, nz);
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, nx, ny, nz);
            } else {
                stencilIteration(grid2, grid1, nx, ny, nz);
            }
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
        const std::vector<Real> finalVec(finalGrid, finalGrid + gridSize);
        print_results(finalVec, "Grid");
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
