#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <unistd.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    // Parallel initialization also first-touches pages across threads for NUMA locality
    #pragma omp parallel for schedule(static) proc_bind(spread)
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
    const Real* __restrict in = input.data();
    Real* __restrict out = output.data();

    #pragma omp parallel proc_bind(spread)
    {
        // Process interior points (not on boundaries)
        #pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t rowIdx = idx3(0, y, z, nx, ny);
                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = rowIdx + x;

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

        // Copy boundary values: z-faces (full planes)
        #pragma omp for schedule(static) nowait
        for (size_t y = 0; y < ny; ++y) {
            memcpy(out + idx3(0, y, 0, nx, ny), in + idx3(0, y, 0, nx, ny), nx * sizeof(Real));
            memcpy(out + idx3(0, y, nz - 1, nx, ny), in + idx3(0, y, nz - 1, nx, ny), nx * sizeof(Real));
        }

        // y-faces and x-faces of the remaining interior slabs
        #pragma omp for schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            memcpy(out + idx3(0, 0, z, nx, ny), in + idx3(0, 0, z, nx, ny), nx * sizeof(Real));
            memcpy(out + idx3(0, ny - 1, z, nx, ny), in + idx3(0, ny - 1, z, nx, ny), nx * sizeof(Real));
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t row = idx3(0, y, z, nx, ny);
                out[row] = in[row];
                out[row + nx - 1] = in[row + nx - 1];
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
    // Pin each OpenMP thread to a distinct CPU; unbound threads migrate and
    // performance collapses on multi-socket/SMT machines. Skipped if the user
    // already requested a binding policy via OMP_PROC_BIND/OMP_PLACES.
    if (getenv("OMP_PROC_BIND") == nullptr && getenv("OMP_PLACES") == nullptr) {
        const long ncpus = sysconf(_SC_NPROCESSORS_ONLN);

        // Find one CPU per physical core (the first entry of each core's
        // sibling list); SMT siblings add contention, not bandwidth, here.
        std::vector<int> cores;
        for (long cpu = 0; cpu < ncpus; ++cpu) {
            char path[128];
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings_list", cpu);
            FILE* f = fopen(path, "r");
            if (!f) { cores.clear(); break; }
            long first = -1;
            if (fscanf(f, "%ld", &first) == 1 && first == cpu) cores.push_back((int)cpu);
            fclose(f);
        }
        if (cores.empty())
            for (long cpu = 0; cpu < ncpus; ++cpu) cores.push_back((int)cpu);

        // Default to one thread per physical core unless the user chose a count
        if (getenv("OMP_NUM_THREADS") == nullptr)
            omp_set_num_threads((int)cores.size());

        #pragma omp parallel
        {
            const long nthreads = omp_get_num_threads();
            const long tid = omp_get_thread_num();
            const long ncores = (long)cores.size();
            cpu_set_t set;
            CPU_ZERO(&set);
            if (nthreads <= ncores) {
                // Spread evenly across physical cores (covers all sockets/NUMA
                // nodes even at partial thread counts)
                CPU_SET(cores[tid * ncores / std::max(nthreads, 1L)], &set);
            } else {
                CPU_SET((int)(tid % ncpus), &set);
            }
            sched_setaffinity(0, sizeof(set), &set);
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
