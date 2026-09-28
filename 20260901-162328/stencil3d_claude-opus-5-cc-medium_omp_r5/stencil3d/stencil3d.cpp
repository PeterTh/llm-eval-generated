#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Drop the physical pages backing a freshly allocated (zero-initialized) buffer.
// The contents stay all-zero, but the pages are re-faulted by whichever thread
// touches them first, so the parallel initialization below determines the NUMA
// placement of the grids instead of the thread that ran the allocation.
void releasePages(std::vector<Real>& grid) {
    const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t begin = reinterpret_cast<uintptr_t>(grid.data());
    const uintptr_t end = begin + grid.size() * sizeof(Real);
    const uintptr_t alignedBegin = (begin + pageSize - 1) & ~static_cast<uintptr_t>(pageSize - 1);
    const uintptr_t alignedEnd = end & ~static_cast<uintptr_t>(pageSize - 1);
    if (alignedEnd > alignedBegin) {
        madvise(reinterpret_cast<void*>(alignedBegin), alignedEnd - alignedBegin, MADV_DONTNEED);
    }
}

// Number of SMT siblings sharing one physical core (1 if it cannot be determined).
int threadsPerCore() {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (!f) return 1;
    char buf[512] = {0};
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return 1;

    // The list looks like "0,128" or "0-1"; count the CPUs it describes.
    int siblings = 0;
    const char* p = buf;
    while (*p) {
        char* next = nullptr;
        const long first = strtol(p, &next, 10);
        if (next == p) break;
        p = next;
        long last = first;
        if (*p == '-') {
            last = strtol(p + 1, &next, 10);
            p = next;
        }
        siblings += static_cast<int>(last - first + 1);
        if (*p == ',') ++p; else break;
    }
    return siblings > 0 ? siblings : 1;
}

// This stencil is memory bandwidth bound, so running two threads on the SMT
// siblings of one core buys nothing and makes the per-iteration barriers much
// more expensive. Unless the user asked for a specific thread count, use one
// thread per physical core.
void selectThreadCount() {
    if (getenv("OMP_NUM_THREADS") != nullptr) return;
    const int available = omp_get_max_threads();
    const int cores = available / threadsPerCore();
    if (cores >= 1 && cores < available) {
        omp_set_num_threads(cores);
    }
}

// Decomposition of the (z, y) planes into rectangular tiles. Every parallel
// loop in this program iterates over the very same tile list with the same
// static schedule, so a thread always works on the memory it first touched
// (NUMA locality) and the tiles are sized so that the three y-rows a stencil
// point needs stay in cache while the tile is swept along z.
struct Tiling {
    size_t yBlock;
    size_t zBlock;
    size_t nyTiles;
    size_t nzTiles;
    size_t nTiles;
};

Tiling makeTiling(const size_t nx, const size_t ny, const size_t nz, const int threads) {
    Tiling t{};

    // Keep the ~3 active y-rows of a tile (plus halo) within L2: aim at 128 KiB.
    const size_t rowBytes = std::max<size_t>(nx, 1) * sizeof(Real);
    size_t yb = (128 * 1024) / (3 * rowBytes);
    yb = std::clamp<size_t>(yb, 8, std::max<size_t>(ny, 1));
    t.yBlock = yb;
    t.nyTiles = (ny + yb - 1) / yb;
    if (t.nyTiles == 0) t.nyTiles = 1;

    // Pick the z block so that there is enough work to balance across threads,
    // while staying long enough that the z halo of a tile stays amortized.
    const size_t wanted = static_cast<size_t>(std::max(threads, 1)) * 3;
    size_t nzTiles = (wanted + t.nyTiles - 1) / t.nyTiles;
    nzTiles = std::clamp<size_t>(nzTiles, 1, std::max<size_t>(nz / 8, 1));
    size_t zb = (std::max<size_t>(nz, 1) + nzTiles - 1) / nzTiles;
    zb = std::max<size_t>(zb, 1);
    t.zBlock = zb;
    t.nzTiles = (nz + zb - 1) / zb;
    if (t.nzTiles == 0) t.nzTiles = 1;

    t.nTiles = t.nyTiles * t.nzTiles;
    return t;
}

// Parallel first touch + initialization. Must be called from inside a parallel
// region.
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz, const Tiling& t) {
    Real* __restrict const g = grid.data();
    const size_t slice = nx * ny;

#pragma omp for schedule(static)
    for (size_t tile = 0; tile < t.nTiles; ++tile) {
        const size_t z0 = (tile / t.nyTiles) * t.zBlock;
        const size_t z1 = std::min(z0 + t.zBlock, nz);
        const size_t y0 = (tile % t.nyTiles) * t.yBlock;
        const size_t y1 = std::min(y0 + t.yBlock, ny);

        for (size_t z = z0; z < z1; ++z) {
            for (size_t y = y0; y < y1; ++y) {
                const size_t base = z * slice + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = base + x;
                    g[idx] = (idx % 19) * 1.0;
                }
            }
        }
    }
}

// Parallel first touch of the second buffer, using the same distribution.
void touchGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz, const Tiling& t) {
    Real* __restrict const g = grid.data();
    const size_t slice = nx * ny;

#pragma omp for schedule(static)
    for (size_t tile = 0; tile < t.nTiles; ++tile) {
        const size_t z0 = (tile / t.nyTiles) * t.zBlock;
        const size_t z1 = std::min(z0 + t.zBlock, nz);
        const size_t y0 = (tile % t.nyTiles) * t.yBlock;
        const size_t y1 = std::min(y0 + t.yBlock, ny);

        for (size_t z = z0; z < z1; ++z) {
            std::fill_n(g + z * slice + y0 * nx, (y1 - y0) * nx, Real(0));
        }
    }
}

// 7-point stencil computation. Called from inside a parallel region; the
// `omp for` provides the barrier that separates consecutive iterations.
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz, const Tiling& t) {
    const Real* __restrict const in = input.data();
    Real* __restrict const out = output.data();
    const size_t slice = nx * ny;
    if (nx == 0 || ny == 0 || nz == 0) return;  // nothing to do (all threads agree)
    const size_t xEnd = nx - 1;

    // A single pass over the grid: interior points get the stencil, boundary
    // points are copied (same result as the original two-pass formulation).
#pragma omp for schedule(static)
    for (size_t tile = 0; tile < t.nTiles; ++tile) {
        const size_t z0 = (tile / t.nyTiles) * t.zBlock;
        const size_t z1 = std::min(z0 + t.zBlock, nz);
        const size_t y0 = (tile % t.nyTiles) * t.yBlock;
        const size_t y1 = std::min(y0 + t.yBlock, ny);

        for (size_t z = z0; z < z1; ++z) {
            if (z == 0 || z == nz - 1) {
                // Entire slice is boundary
                std::memcpy(out + z * slice + y0 * nx, in + z * slice + y0 * nx, (y1 - y0) * nx * sizeof(Real));
                continue;
            }

            for (size_t y = y0; y < y1; ++y) {
                const size_t base = z * slice + y * nx;
                const Real* __restrict const center = in + base;
                Real* __restrict const dst = out + base;

                if (y == 0 || y == ny - 1) {
                    // Entire row is boundary
                    std::memcpy(dst, center, nx * sizeof(Real));
                    continue;
                }

                const Real* __restrict const front = center - nx;
                const Real* __restrict const back = center + nx;
                const Real* __restrict const bottom = center - slice;
                const Real* __restrict const top = center + slice;

                dst[0] = center[0];  // x = 0 boundary

#pragma omp simd
                for (size_t x = 1; x < xEnd; ++x) {
                    dst[x] = (center[x] + center[x - 1] + center[x + 1] + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                }

                dst[nx - 1] = center[nx - 1];  // x = nx-1 boundary
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    const Real* __restrict const g = grid.data();
    const size_t n = grid.size();

    // 1. No NaN or Inf values
    bool badValue = false;
#pragma omp parallel for schedule(static) reduction(|| : badValue)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(g[i]) || std::isinf(g[i])) {
            badValue = true;
        }
    }
    if (badValue) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
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
    selectThreadCount();
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    size_t gridSize = nx * ny * nz;

    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    // Undo the serial first touch performed by the allocation, so the parallel
    // initialization below distributes the pages across the NUMA domains.
    releasePages(grid1);
    releasePages(grid2);

    // Initialize
    printf("Initializing grid...\n");
    printf("Running stencil computation...\n");

    std::chrono::high_resolution_clock::time_point start, end;

    const Tiling tiling = makeTiling(nx, ny, nz, omp_get_max_threads());

    // One parallel region covers initialization and all iterations: threads keep
    // working on the data they first touched, and there is no repeated fork/join.
#pragma omp parallel proc_bind(spread)
    {
        initializeGrid(grid1, nx, ny, nz, tiling);
        touchGrid(grid2, nx, ny, nz, tiling);

#pragma omp master
        start = std::chrono::high_resolution_clock::now();
#pragma omp barrier

        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, nx, ny, nz, tiling);
            } else {
                stencilIteration(grid2, grid1, nx, ny, nz, tiling);
            }
        }

#pragma omp master
        end = std::chrono::high_resolution_clock::now();
    }

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
