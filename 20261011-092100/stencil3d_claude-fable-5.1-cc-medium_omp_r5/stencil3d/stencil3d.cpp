#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Read a small integer-list sysfs file such as "0-3,128-131"; returns false on failure.
static bool readCpuList(const char* path, std::vector<int>& out) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    char buf[4096];
    size_t len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[len] = '\0';
    char* p = buf;
    while (*p) {
        char* end;
        long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        p = end;
        if (*p == '-') {
            b = strtol(p + 1, &end, 10);
            p = end;
        }
        for (long c = a; c <= b; ++c) out.push_back((int)c);
        while (*p == ',' || *p == ' ' || *p == '\n') ++p;
    }
    return true;
}

// Choose the CPUs for T threads and the thread -> CPU mapping.
// Selection: physical cores first (one CPU per core), spread evenly over NUMA
// nodes, then SMT siblings, so any thread count uses all memory controllers.
// Mapping: the selected CPUs are then grouped by NUMA node so that threads
// with adjacent IDs (which own adjacent z-planes under the static schedule and
// exchange halo planes) sit on the same node and the first-touch placement
// keeps nearly all stencil traffic node-local.
static std::vector<int> buildCpuOrder(const std::vector<int>& allowed, const size_t nthreads) {
    std::vector<int> nodeOf(CPU_SETSIZE, 0);
    std::vector<char> primary(CPU_SETSIZE, 1);
    int maxNode = 0;
    char path[256];
    for (int node = 0; node < 1024; ++node) {
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
        std::vector<int> cpus;
        if (!readCpuList(path, cpus)) {
            if (node > 0) break;
            continue;
        }
        for (int c : cpus) if (c >= 0 && c < CPU_SETSIZE) nodeOf[c] = node;
        maxNode = node;
    }
    for (int c : allowed) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        std::vector<int> sib;
        if (readCpuList(path, sib) && !sib.empty()) {
            // The lowest-numbered *allowed* sibling is the primary for its core
            int lowest = -1;
            for (int sc : sib) {
                if (std::binary_search(allowed.begin(), allowed.end(), sc) && (lowest < 0 || sc < lowest)) lowest = sc;
            }
            primary[c] = (lowest == c) ? 1 : 0;
        }
    }
    // Spread order: primaries interleaved across nodes, then siblings
    std::vector<int> spread;
    spread.reserve(allowed.size());
    for (int pass = 1; pass >= 0; --pass) {
        std::vector<std::vector<int>> perNode(maxNode + 1);
        for (int c : allowed) {
            if (primary[c] == pass) perNode[nodeOf[c]].push_back(c);
        }
        bool more = true;
        for (size_t i = 0; more; ++i) {
            more = false;
            for (auto& v : perNode) {
                if (i < v.size()) { spread.push_back(v[i]); more = true; }
            }
        }
    }
    // Select the CPUs actually used and group them by node (stable, so the
    // within-node order stays primaries-then-siblings, ascending).
    const size_t use = std::min(nthreads, spread.size());
    std::vector<int> order(spread.begin(), spread.begin() + (ptrdiff_t)use);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return nodeOf[a] < nodeOf[b]; });
    return order;
}

// Pin each OpenMP thread to a distinct CPU from the process's allowed CPU set.
// Unpinned threads migrate between cores/NUMA nodes, which destroys cache
// locality and first-touch NUMA placement for a bandwidth-bound stencil. If
// the user already controls binding via OMP_PROC_BIND / OMP_PLACES /
// GOMP_CPU_AFFINITY, leave it to the runtime.
void pinThreads() {
    omp_set_dynamic(0);  // keep team size fixed so the mapping below stays valid
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY")) return;
    cpu_set_t allowedSet;
    CPU_ZERO(&allowedSet);
    if (sched_getaffinity(0, sizeof(allowedSet), &allowedSet) != 0) return;
    std::vector<int> allowed;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowedSet)) allowed.push_back(c);
    }
    if (allowed.empty()) return;
    int nthreads = 1;
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
    const std::vector<int> order = buildCpuOrder(allowed, (size_t)nthreads);
    if (order.empty()) return;
#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(order[(size_t)t % order.size()], &one);
        sched_setaffinity(0, sizeof(one), &one);
    }
}

// Allocate a grid without serial zero-fill: reserve, request transparent huge
// pages, and let the parallel first-touch (in the same z/y distribution the
// stencil uses) place pages on the NUMA node of the thread that will use them.
void allocateGrid(std::vector<Real>& grid, const size_t n) {
    grid.reserve(n);
    const size_t bytes = n * sizeof(Real);
    const uintptr_t beg = reinterpret_cast<uintptr_t>(grid.data());
    const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    const uintptr_t abeg = (beg + pg - 1) & ~(pg - 1);
    const uintptr_t aend = (beg + bytes) & ~(pg - 1);
    if (aend > abeg) {
        madvise(reinterpret_cast<void*>(abeg), aend - abeg, MADV_HUGEPAGE);  // best-effort hint
    }
}

// Resize after the storage has been first-touched; for a trivial element type
// this is a (serial, untimed) zero-fill that does not change page placement.
void commitGrid(std::vector<Real>& grid, const size_t n) {
    grid.resize(n);
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* __restrict g = grid.data();
    const ptrdiff_t NX = (ptrdiff_t)nx, NY = (ptrdiff_t)ny, NZ = (ptrdiff_t)nz;
    // Parallel initialization: also performs NUMA first-touch placement of pages
    // in the same z/y distribution used by the stencil loop.
#pragma omp parallel for collapse(2) schedule(static)
    for (ptrdiff_t z = 0; z < NZ; ++z) {
        for (ptrdiff_t y = 0; y < NY; ++y) {
            Real* __restrict row = g + idx3(0, y, z, nx, ny);
            const size_t base = idx3(0, y, z, nx, ny);
            for (ptrdiff_t x = 0; x < NX; ++x) {
                row[x] = ((base + (size_t)x) % 19) * 1.0;
            }
        }
    }
}

// Touch (zero) a buffer in parallel with the same distribution as the stencil
// so that both grids have matching NUMA page placement. Operates on reserved
// storage, so `g` is passed explicitly.
void firstTouchGrid(Real* __restrict g, const size_t nx, const size_t ny, const size_t nz) {
    const ptrdiff_t NX = (ptrdiff_t)nx, NY = (ptrdiff_t)ny, NZ = (ptrdiff_t)nz;
#pragma omp parallel for collapse(2) schedule(static)
    for (ptrdiff_t z = 0; z < NZ; ++z) {
        for (ptrdiff_t y = 0; y < NY; ++y) {
            Real* __restrict row = g + idx3(0, y, z, nx, ny);
            for (ptrdiff_t x = 0; x < NX; ++x) {
                row[x] = 0.0;
            }
        }
    }
}

// 7-point stencil computation.
// Must be called from inside an active OpenMP parallel region (orphaned
// worksharing); the implicit barrier at the end of the `omp for` separates
// successive iterations.
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const Real* __restrict in = input.data();
    Real* __restrict out = output.data();
    const ptrdiff_t NX = (ptrdiff_t)nx, NY = (ptrdiff_t)ny, NZ = (ptrdiff_t)nz;
    const ptrdiff_t sy = NX;        // stride in y
    const ptrdiff_t sz = NX * NY;   // stride in z

    // Each (z, y) row is handled by exactly one thread. Boundary rows
    // (z or y on the boundary) are copied wholesale; interior rows compute
    // the stencil for 1..nx-2 and copy the two x-boundary cells. This fuses
    // the original interior loop and boundary-copy loop into one pass.
#pragma omp for collapse(2) schedule(static) nowait
    for (ptrdiff_t z = 0; z < NZ; ++z) {
        for (ptrdiff_t y = 0; y < NY; ++y) {
            const ptrdiff_t rowBase = z * sz + y * sy;
            const Real* __restrict ir = in + rowBase;
            Real* __restrict orow = out + rowBase;
            if (z == 0 || z == NZ - 1 || y == 0 || y == NY - 1) {
                // Copy boundary row
                for (ptrdiff_t x = 0; x < NX; ++x) {
                    orow[x] = ir[x];
                }
            } else {
                // Copy x boundaries
                orow[0] = ir[0];
                orow[NX - 1] = ir[NX - 1];
                const Real* __restrict front  = ir - sy;
                const Real* __restrict back   = ir + sy;
                const Real* __restrict bottom = ir - sz;
                const Real* __restrict top    = ir + sz;
#pragma omp simd
                for (ptrdiff_t x = 1; x < NX - 1; ++x) {
                    const Real center = ir[x];
                    const Real left   = ir[x - 1];
                    const Real right  = ir[x + 1];
                    // Simple averaging stencil (same evaluation order as the original)
                    orow[x] = (center + left + right + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                }
            }
        }
    }
#pragma omp barrier
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const Real* __restrict g = grid.data();
    const ptrdiff_t N = (ptrdiff_t)grid.size();

    // 1. No NaN or Inf values
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(|:bad)
    for (ptrdiff_t i = 0; i < N; ++i) {
        bad |= (std::isnan(g[i]) || std::isinf(g[i])) ? 1 : 0;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (ptrdiff_t i = 0; i < N; ++i) {
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
    
    size_t gridSize = nx * ny * nz;
    
    pinThreads();

    // Allocate grids (double buffering)
    std::vector<Real> grid1;
    std::vector<Real> grid2;
    allocateGrid(grid1, gridSize);
    allocateGrid(grid2, gridSize);
    
    // Initialize (parallel first touch), then commit the vector sizes
    printf("Initializing grid...\n");
    firstTouchGrid(grid1.data(), nx, ny, nz);
    firstTouchGrid(grid2.data(), nx, ny, nz);
    commitGrid(grid1, gridSize);
    commitGrid(grid2, gridSize);
    initializeGrid(grid1, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // One parallel region for all iterations; each stencilIteration contains an
    // orphaned `omp for` followed by a barrier, so there is no per-iteration
    // fork/join overhead.
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
