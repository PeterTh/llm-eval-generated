#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Pin each OpenMP worker thread to its own CPU unless the user has already
// configured affinity through the OpenMP runtime. Unpinned threads migrate
// between cores and NUMA nodes, which costs a large factor on a bandwidth-bound
// stencil. Thread t is bound to the t-th CPU of the process' affinity mask, so
// taskset/cgroup restrictions are respected and thread->memory placement stays
// stable across iterations (matching the parallel first touch below).
void setupThreadAffinity() {
    if (std::getenv("OMP_PROC_BIND") || std::getenv("OMP_PLACES") || std::getenv("GOMP_CPU_AFFINITY")) {
        return;  // user is in control of binding
    }
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
        const int t = omp_get_thread_num();
        const int cpu = cpus[static_cast<size_t>(t) % cpus.size()];
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    }
}

// Allocate a grid and perform the first touch of its pages in parallel so that,
// on NUMA systems, memory ends up on the node of the thread that later works on
// the same z-planes. std::vector<Real>(n) would zero-fill serially from the
// main thread and place every page on a single NUMA node.
std::vector<Real> allocateGrid(const size_t nx, const size_t ny, const size_t nz) {
    const size_t gridSize = nx * ny * nz;
    std::vector<Real> grid;
    grid.reserve(gridSize);
    Real* const data = grid.data();
    const size_t plane = nx * ny;
    // Touch the raw storage in the same static z/y distribution used by the kernels.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            std::memset(static_cast<void*>(data + z * plane + y * nx), 0, nx * sizeof(Real));
        }
    }
    // Now value-initialize (zero) the elements; pages are already mapped locally.
    grid.resize(gridSize);
    return grid;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* const __restrict g = grid.data();
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowBase = idx3(0, y, z, nx, ny);
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = rowBase + x;
                g[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation.
// Interior points receive the averaged value; every boundary point (any index on
// a face of the box) is copied from the input, exactly as in the original code.
// Interior update and boundary copy are fused into one parallel sweep over
// (z, y) rows so each output row is written exactly once, contiguously.
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const Real* const __restrict in = input.data();
    Real* const __restrict out = output.data();
    const size_t plane = nx * ny;
    const size_t rowBytes = nx * sizeof(Real);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowBase = z * plane + y * nx;
            const bool boundaryRow = (z == 0) || (z == nz - 1) || (y == 0) || (y == ny - 1);
            if (boundaryRow || nx <= 2) {
                // Whole row lies on the boundary: straight copy.
                std::memcpy(out + rowBase, in + rowBase, rowBytes);
                continue;
            }

            const Real* const __restrict c  = in + rowBase;          // center / left / right
            const Real* const __restrict yp = in + rowBase - nx;     // front  (y-1)
            const Real* const __restrict yn = in + rowBase + nx;     // back   (y+1)
            const Real* const __restrict zp = in + rowBase - plane;  // bottom (z-1)
            const Real* const __restrict zn = in + rowBase + plane;  // top    (z+1)
            Real* const __restrict o = out + rowBase;

            o[0] = c[0];
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const Real center = c[x];
                const Real left   = c[x - 1];
                const Real right  = c[x + 1];
                const Real front  = yp[x];
                const Real back   = yn[x];
                const Real bottom = zp[x];
                const Real top    = zn[x];
                // Simple averaging stencil (same operation order as the original)
                o[x] = (center + left + right + front + back + bottom + top) / 7.0;
            }
            o[nx - 1] = c[nx - 1];
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const Real* const data = grid.data();
    const size_t n = grid.size();
    
    // 1. No NaN or Inf values
    bool bad = false;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for (size_t i = 0; i < n; ++i) {
        bad |= (std::isnan(data[i]) || std::isinf(data[i]));
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
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
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
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    setupThreadAffinity();
    
    // Allocate grids (double buffering), first-touched in parallel
    std::vector<Real> grid1 = allocateGrid(nx, ny, nz);
    std::vector<Real> grid2 = allocateGrid(nx, ny, nz);
    
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
