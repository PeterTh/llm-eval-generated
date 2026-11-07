#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    // Parallel initialization: also performs NUMA first-touch so each thread's
    // pages are placed close to the core that will later process them.
    const long long nzl = (long long)nz;
    const long long nyl = (long long)ny;
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (long long z = 0; z < nzl; ++z) {
        for (long long y = 0; y < nyl; ++y) {
            Real* row = grid.data() + (size_t)z * plane + (size_t)y * nx;
            const size_t base = (size_t)z * plane + (size_t)y * nx;
            for (size_t x = 0; x < nx; ++x) {
                row[x] = ((base + x) % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation.
// Must be called from inside an OpenMP parallel region by all threads of the
// team (orphaned worksharing loops); the final loop carries the barrier that
// separates consecutive iterations. Called from serial code it simply runs
// on the calling thread.
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const Real* __restrict__ in = input.data();
    Real* __restrict__ out = output.data();
    const size_t plane = nx * ny;
    const long long nzl = (long long)nz;
    const long long nyl = (long long)ny;

    {
        // Process interior points (not on boundaries).
        // Work is distributed over (z, y) rows with a static schedule so each
        // thread touches the same rows every iteration (cache/NUMA locality);
        // the innermost x loop is contiguous in memory and vectorizes.
#pragma omp for collapse(2) schedule(static) nowait
        for (long long z = 1; z < nzl - 1; ++z) {
            for (long long y = 1; y < nyl - 1; ++y) {
                const size_t rowBase = (size_t)z * plane + (size_t)y * nx;
                const Real* __restrict__ c  = in + rowBase;
                const Real* __restrict__ ym = c - nx;
                const Real* __restrict__ yp = c + nx;
                const Real* __restrict__ zm = c - plane;
                const Real* __restrict__ zp = c + plane;
                Real* __restrict__ o = out + rowBase;
#pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const Real center = c[x];
                    const Real left = c[x-1];
                    const Real right = c[x+1];
                    const Real front = ym[x];
                    const Real back = yp[x];
                    const Real bottom = zm[x];
                    const Real top = zp[x];

                    // Simple averaging stencil
                    o[x] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }

        // Copy boundary values (disjoint from the interior region above).
        // z = 0 and z = nz-1 planes: full contiguous copies.
#pragma omp for schedule(static) nowait
        for (long long y = 0; y < nyl; ++y) {
            const size_t r0 = (size_t)y * nx;
            std::memcpy(out + r0, in + r0, nx * sizeof(Real));
            if (nz > 1) {
                const size_t r1 = (size_t)(nz - 1) * plane + (size_t)y * nx;
                std::memcpy(out + r1, in + r1, nx * sizeof(Real));
            }
        }
        // Remaining planes: y = 0 / y = ny-1 rows and x = 0 / x = nx-1 columns.
        // (No nowait: the implicit barrier here orders this iteration's writes
        // before the next iteration's reads.)
#pragma omp for schedule(static)
        for (long long z = 1; z < nzl - 1; ++z) {
            const size_t zb = (size_t)z * plane;
            std::memcpy(out + zb, in + zb, nx * sizeof(Real));
            if (ny > 1) {
                const size_t rl = zb + (ny - 1) * nx;
                std::memcpy(out + rl, in + rl, nx * sizeof(Real));
            }
            for (size_t y = 1; y + 1 < ny; ++y) {
                const size_t rb = zb + y * nx;
                out[rb] = in[rb];
                if (nx > 1) out[rb + nx - 1] = in[rb + nx - 1];
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    const long long n = (long long)grid.size();
    const Real* data = grid.data();
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(||:bad)
    for (long long i = 0; i < n; ++i) {
        const Real val = data[i];
        if (std::isnan(val) || std::isinf(val)) {
            bad = true;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (long long i = 0; i < n; ++i) {
        const Real val = data[i];
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

// Pin each OpenMP thread to one CPU of the process's allowed set unless the
// user already configured affinity through the runtime's environment
// variables. Unpinned threads migrate between cores/NUMA nodes, which hurts
// this memory-bound kernel badly.
void pinThreads() {
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") ||
        getenv("GOMP_CPU_AFFINITY") || getenv("KMP_AFFINITY")) {
        return;
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
    }
    if (cpus.size() < 2) return;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[(size_t)tid % cpus.size()], &one);
        sched_setaffinity(0, sizeof(one), &one);
    }
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
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    pinThreads();

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // One parallel region for all iterations: avoids per-iteration fork/join
    // and keeps the static row-to-thread mapping fixed across iterations.
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
