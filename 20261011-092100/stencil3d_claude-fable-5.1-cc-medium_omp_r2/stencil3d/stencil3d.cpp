#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Pin each OpenMP worker to its own CPU (thread t -> t-th CPU of the process affinity
// mask). Unpinned threads migrate between cores and NUMA nodes, which destroys the
// first-touch locality set up below. Any affinity policy the user has set through the
// usual environment variables takes precedence and is left untouched.
void bindThreadsToCpus() {
#ifdef __linux__
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
    if (cpus.empty()) return;

#pragma omp parallel
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t mine;
        CPU_ZERO(&mine);
        CPU_SET(cpus[t % cpus.size()], &mine);
        sched_setaffinity(0, sizeof(mine), &mine);
    }
#endif
}

// std::vector value-initialises its storage from the constructing (main) thread, which
// on a NUMA system places every page on that thread's node. Discard those pages so the
// following parallel write re-faults them on the node of the thread that will use them.
// The buffer reads as all-zero both before and after this call, so semantics are kept.
void releasePagesForFirstTouch(std::vector<Real>& grid) {
#ifdef __linux__
    if (grid.empty()) return;
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) return;
    const uintptr_t pg = static_cast<uintptr_t>(pageSize);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(grid.data());
    const uintptr_t end = begin + grid.size() * sizeof(Real);
    const uintptr_t alignedBegin = (begin + pg - 1) & ~(pg - 1);
    const uintptr_t alignedEnd = end & ~(pg - 1);
    if (alignedEnd > alignedBegin) {
        madvise(reinterpret_cast<void*>(alignedBegin), alignedEnd - alignedBegin, MADV_DONTNEED);
    }
#else
    (void)grid;
#endif
}

// Parallel first-touch of a grid. The (z, y) row partition and static schedule are
// identical to the ones used by stencilIteration, so every thread initialises exactly
// the rows it will later compute on, keeping pages NUMA-local and warm in cache.
template <typename RowFn>
inline void forEachRowParallel(const size_t nx, const size_t ny, const size_t nz, RowFn&& rowFn) {
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            rowFn(z, y, z * plane + y * nx);
        }
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    releasePagesForFirstTouch(grid);
    Real* __restrict g = grid.data();
    forEachRowParallel(nx, ny, nz, [=](size_t, size_t, const size_t rowBase) {
        for (size_t x = 0; x < nx; ++x) {
            const size_t idx = rowBase + x;
            g[idx] = (idx % 19) * 1.0;
        }
    });
}

// First-touch the scratch grid with the same thread/row mapping (contents stay zero).
void touchGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    releasePagesForFirstTouch(grid);
    Real* __restrict g = grid.data();
    forEachRowParallel(nx, ny, nz, [=](size_t, size_t, const size_t rowBase) {
        std::memset(g + rowBase, 0, nx * sizeof(Real));
    });
}

// 7-point stencil computation
//
// Interior points receive the 7-point average; boundary points are copied from the
// input. Both are handled in a single fused pass over (z, y) rows so that each output
// element is written exactly once, there is no second sweep over the grid, and the
// work is distributed across threads in contiguous, cache-friendly chunks.
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) return;

    const Real* __restrict in = input.data();
    Real* __restrict out = output.data();
    const size_t plane = nx * ny;
    const bool hasInterior = (nx >= 3 && ny >= 3 && nz >= 3);

    forEachRowParallel(nx, ny, nz, [=](const size_t z, const size_t y, const size_t rowBase) {
        const Real* __restrict inRow = in + rowBase;
        Real* __restrict outRow = out + rowBase;

        const bool boundaryRow = !hasInterior ||
                                 z == 0 || z == nz - 1 ||
                                 y == 0 || y == ny - 1;
        if (boundaryRow) {
            // Entire row lies on the boundary: straight copy.
            std::memcpy(outRow, inRow, nx * sizeof(Real));
            return;
        }

        // x-boundary points of this row.
        outRow[0] = inRow[0];
        outRow[nx - 1] = inRow[nx - 1];

        const Real* __restrict front  = inRow - nx;      // y-1
        const Real* __restrict back   = inRow + nx;      // y+1
        const Real* __restrict bottom = inRow - plane;   // z-1
        const Real* __restrict top    = inRow + plane;   // z+1

        // Interior: contiguous in x, vectorizes cleanly.
#pragma omp simd
        for (size_t x = 1; x < nx - 1; ++x) {
            const Real center = inRow[x];
            const Real left   = inRow[x - 1];
            const Real right  = inRow[x + 1];
            outRow[x] = (center + left + right + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
        }
    });
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    const Real* __restrict g = grid.data();
    const size_t n = grid.size();

    // 1. No NaN or Inf values
    bool nonFinite = false;
#pragma omp parallel for schedule(static) reduction(||:nonFinite)
    for (size_t i = 0; i < n; ++i) {
        nonFinite = nonFinite || !std::isfinite(g[i]);
    }
    if (nonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, g[i]);
        maxVal = std::max(maxVal, g[i]);
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
    bindThreadsToCpus();
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    touchGrid(grid2, nx, ny, nz);
    
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
