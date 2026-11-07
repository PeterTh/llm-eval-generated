#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <omp.h>

#if defined(__linux__)
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

// Pin each OpenMP thread to its own CPU unless the user already configured
// affinity through the usual environment variables. Without binding, threads
// migrate between NUMA nodes and lose the locality that the first-touch
// allocation below establishes. Purely a performance measure.
static void pinThreadsIfUnbound() {
#if defined(__linux__)
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") ||
        getenv("GOMP_CPU_AFFINITY") || getenv("KMP_AFFINITY")) {
        return;
    }
    cpu_set_t avail;
    CPU_ZERO(&avail);
    if (sched_getaffinity(0, sizeof(avail), &avail) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &avail)) cpus.push_back(c);
    }
    if (cpus.empty()) return;
    const int ncpus = static_cast<int>(cpus.size());
#pragma omp parallel
    {
        cpu_set_t mine;
        CPU_ZERO(&mine);
        CPU_SET(cpus[omp_get_thread_num() % ncpus], &mine);
        sched_setaffinity(0, sizeof(mine), &mine);
    }
#endif
}

// Allocate a grid of gridSize elements and place its pages NUMA-aware.
//
// std::vector<Real>::resize() zero-fills the storage from the calling thread,
// which under Linux first-touch policy pins every page to that thread's NUMA
// node. To undo that we release the (zero) pages with MADV_DONTNEED; anonymous
// private memory re-faults as zero pages on the next access, so the vector's
// contents are unchanged. Then every page is re-touched in parallel using the
// same static (z,y)-row partitioning the stencil uses, so each thread's
// working set lands on its own memory node. Transparent huge pages are
// requested to cut TLB pressure on the large grids. All of this is purely a
// performance hint; on non-Linux systems it degrades to a plain resize.
static void allocateGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t gridSize = nx * ny * nz;
    grid.resize(gridSize);
    if (gridSize == 0) return;

#if defined(__linux__)
    {
        const long pageSizeL = sysconf(_SC_PAGESIZE);
        const size_t pageSize = pageSizeL > 0 ? static_cast<size_t>(pageSizeL) : 4096;
        const uintptr_t begin = reinterpret_cast<uintptr_t>(grid.data());
        const uintptr_t end = begin + gridSize * sizeof(Real);
        const uintptr_t abegin = (begin + pageSize - 1) & ~(pageSize - 1);
        const uintptr_t aend = end & ~(pageSize - 1);
        if (aend > abegin) {
            void* p = reinterpret_cast<void*>(abegin);
            const size_t len = aend - abegin;
#ifdef MADV_HUGEPAGE
            madvise(p, len, MADV_HUGEPAGE);
#endif
            // Return values deliberately ignored: failure just means pages keep
            // their current placement, which is still correct.
            madvise(p, len, MADV_DONTNEED);
        }
    }
#endif

    Real* __restrict__ g = grid.data();
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            Real* __restrict__ row = g + idx3(0, y, z, nx, ny);
            for (size_t x = 0; x < nx; ++x) {
                row[x] = 0.0;
            }
        }
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* __restrict__ g = grid.data();
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowBase = idx3(0, y, z, nx, ny);
            Real* __restrict__ row = g + rowBase;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = rowBase + x;
                row[x] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
//
// One parallel sweep over all (z,y) rows. Rows lying on the z or y boundary
// are copied verbatim; interior rows copy their two x-boundary cells and
// compute the 7-point average for the interior cells. This fuses the original
// interior loop and boundary-copy loop into a single pass with no data
// dependencies between rows, so every row can be processed independently.
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) return;

    const Real* __restrict__ in = input.data();
    Real* __restrict__ out = output.data();
    const size_t sxy = nx * ny;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowBase = z * sxy + y * nx;
            const Real* __restrict__ c = in + rowBase;
            Real* __restrict__ o = out + rowBase;

            if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                // Boundary row: copy all values
                for (size_t x = 0; x < nx; ++x) {
                    o[x] = c[x];
                }
                continue;
            }

            const Real* __restrict__ ym = c - nx;    // y-1
            const Real* __restrict__ yp = c + nx;    // y+1
            const Real* __restrict__ zm = c - sxy;   // z-1
            const Real* __restrict__ zp = c + sxy;   // z+1

            // x boundaries are copied
            o[0] = c[0];
            o[nx - 1] = c[nx - 1];

            // Interior: same expression and evaluation order as the original
#pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const Real center = c[x];
                const Real left   = c[x - 1];
                const Real right  = c[x + 1];
                const Real front  = ym[x];
                const Real back   = yp[x];
                const Real bottom = zm[x];
                const Real top    = zp[x];
                o[x] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();
    const Real* __restrict__ g = grid.data();

    // 1. No NaN or Inf values
    // 2. Values should be reasonable (bounded)
    bool bad = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(||:bad) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        const Real val = g[i];
        if (std::isnan(val) || std::isinf(val)) {
            bad = true;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
    pinThreadsIfUnbound();


    // Allocate grids (double buffering), NUMA-aware first touch
    std::vector<Real> grid1;
    std::vector<Real> grid2;
    allocateGrid(grid1, nx, ny, nz);
    allocateGrid(grid2, nx, ny, nz);

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
