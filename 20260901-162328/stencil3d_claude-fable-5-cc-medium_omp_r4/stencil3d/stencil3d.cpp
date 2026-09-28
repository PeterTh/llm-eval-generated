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
#include <unistd.h>

// One CPU per physical core: cpus that are listed first among their SMT
// siblings in sysfs. Falls back to all online cpus if sysfs is unavailable.
static std::vector<int> physicalCoreCpus() {
    const long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    std::vector<int> cpus;
    for (long c = 0; c < ncpus; ++c) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings_list", c);
        FILE* f = fopen(path, "r");
        if (!f) {
            cpus.clear();
            for (long i = 0; i < ncpus; ++i) cpus.push_back((int)i);
            return cpus;
        }
        int first = -1;
        if (fscanf(f, "%d", &first) == 1 && first == c) cpus.push_back((int)c);
        fclose(f);
    }
    if (cpus.empty()) cpus.push_back(0);
    return cpus;
}

// Pin each OpenMP thread to a fixed CPU, spread evenly across the physical
// cores. The OpenMP runtime does not bind threads unless OMP_PROC_BIND is
// set in the environment (which is parsed before main), so without this
// threads migrate between NUMA nodes and destroy first-touch locality.
// Unless the user requested a thread count explicitly, use one thread per
// physical core: SMT siblings only add contention for this memory-bound
// kernel.
static void bindThreads() {
    const std::vector<int> cpus = physicalCoreCpus();
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads((int)cpus.size());
    }
    #pragma omp parallel
    {
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[(tid * cpus.size()) / nthreads], &set);
        sched_setaffinity(0, sizeof(set), &set);  // best effort
    }
}

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    // Parallel initialization also establishes NUMA-friendly first-touch page
    // placement matching the z-partitioning used in the stencil kernel.
    #pragma omp parallel for schedule(static)
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
void stencilIteration(const Real* const in,
                      Real* const out,
                      const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel
    {
        // Process interior points (not on boundaries)
        #pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t row = idx3(0, y, z, nx, ny);
                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = row + x;

                    const Real center = in[idx];
                    const Real left = in[idx - 1];
                    const Real right = in[idx + 1];
                    const Real front = in[idx - nx];
                    const Real back = in[idx + nx];
                    const Real bottom = in[idx - nx * ny];
                    const Real top = in[idx + nx * ny];

                    // Simple averaging stencil
                    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }

        // Copy boundary values: only the boundary faces are touched, which
        // writes exactly the same cells as the original full-grid scan.
        #pragma omp for collapse(2) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                const size_t row = idx3(0, y, z, nx, ny);
                if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                    // Entire row lies on the boundary
                    std::memcpy(out + row, in + row, nx * sizeof(Real));
                } else {
                    // Interior row: only the x-faces are boundary cells
                    out[row] = in[row];
                    out[row + nx - 1] = in[row + nx - 1];
                }
            }
        }
    }
}

bool validateResult(const Real* grid, const size_t gridSize) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (size_t i = 0; i < gridSize; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
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
    
    bindThreads();

    size_t gridSize = nx * ny * nz;

    // Allocate grids (double buffering). Raw arrays are used instead of
    // std::vector so the memory is not zero-filled serially: first touch
    // happens in the parallel init/stencil loops, distributing pages across
    // NUMA nodes to match the thread partitioning.
    std::unique_ptr<Real[]> grid1(new Real[gridSize]);
    std::unique_ptr<Real[]> grid2(new Real[gridSize]);

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1.get(), nx, ny, nz);
    initializeGrid(grid2.get(), nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1.get(), grid2.get(), nx, ny, nz);
        } else {
            stencilIteration(grid2.get(), grid1.get(), nx, ny, nz);
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
    const Real* finalGrid = (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        const std::vector<Real> finalCopy(finalGrid, finalGrid + gridSize);
        print_results(finalCopy, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, gridSize);
        
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
