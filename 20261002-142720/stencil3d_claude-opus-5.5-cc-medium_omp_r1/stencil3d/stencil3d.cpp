#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <new>
#include <utility>
#include <vector>

#include <omp.h>
#include <sched.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Cache-line aligned allocator that default-initializes (does not zero) elements,
// so that the first touch of every page happens inside the parallel initialization
// (NUMA first-touch placement matching the compute decomposition).
template <typename T>
struct NumaAllocator {
    using value_type = T;
    NumaAllocator() noexcept = default;
    template <typename U>
    NumaAllocator(const NumaAllocator<U>&) noexcept {}
    T* allocate(size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64}));
    }
    void deallocate(T* p, size_t) noexcept { ::operator delete(p, std::align_val_t{64}); }
    template <typename U>
    void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
    template <typename U>
    bool operator==(const NumaAllocator<U>&) const noexcept { return true; }
};

using Grid = std::vector<Real, NumaAllocator<Real>>;

// Thread-to-CPU pinning. OpenMP threads are unbound by default, which lets the OS
// migrate them away from the NUMA node holding their first-touched data. Unless the
// user controls binding via OMP_PLACES / OMP_PROC_BIND, pin thread t to a fixed CPU of
// the process affinity mask, preferring one hardware thread per physical core.
struct ThreadPinning {
    std::vector<int> cpuOfThread;  // empty: no pinning

    explicit ThreadPinning(const size_t nthreads) {
        if (getenv("OMP_PLACES") || getenv("OMP_PROC_BIND")) return;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
        std::vector<int> primary, secondary;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &mask)) continue;
            // First CPU in the sibling list is the core's primary hardware thread
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            int first = c;
            if (f) f >> first;
            (first == c ? primary : secondary).push_back(c);
        }
        if (primary.empty()) return;
        std::vector<int> all(primary);
        all.insert(all.end(), secondary.begin(), secondary.end());
        cpuOfThread.resize(nthreads);
        for (size_t t = 0; t < nthreads; ++t) {
            if (nthreads <= primary.size()) {
                cpuOfThread[t] = primary[t * primary.size() / nthreads];
            } else {
                cpuOfThread[t] = all[t % all.size()];
            }
        }
    }

    void pin(const size_t tid) const {
        if (tid >= cpuOfThread.size()) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpuOfThread[tid], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
};

// Static 2D (z, y) domain decomposition shared by initialization and computation,
// so that each thread always works on the memory it first-touched.
struct Decomposition {
    size_t pz = 1, py = 1;  // thread grid
    size_t tileY = 1;       // y sub-tile size (keeps 3 planes of a tile in L2)
    bool streaming = false; // use non-temporal stores (grids too large for caches)

    Decomposition(const size_t nthreads, const size_t nx, const size_t ny, const size_t nz) {
        // Choose pz * py == nthreads minimizing the largest block incl. halo
        size_t bestCost = SIZE_MAX;
        for (size_t a = 1; a <= nthreads; ++a) {
            if (nthreads % a != 0) continue;
            const size_t b = nthreads / a;
            const size_t bz = (nz + a - 1) / a;
            const size_t by = (ny + b - 1) / b;
            const size_t cost = (bz + 2) * (by + 2);
            if (cost < bestCost) {
                bestCost = cost;
                pz = a;
                py = b;
            }
        }
        const size_t rowBytes = std::max<size_t>(nx, 1) * sizeof(Real);
        tileY = std::max<size_t>(1, (256 * 1024) / (3 * rowBytes));
        streaming = 2.0 * double(nx) * double(ny) * double(nz) * sizeof(Real) > 1024.0 * 1024 * 1024;
    }

    // Range [begin, end) of thread-coordinate i out of p over n elements
    static void range(const size_t n, const size_t p, const size_t i, size_t& b, size_t& e) {
        const size_t q = n / p, r = n % p;
        b = i * q + std::min(i, r);
        e = b + q + (i < r ? 1 : 0);
    }

    void block(const size_t tid, const size_t ny, const size_t nz,
               size_t& z0, size_t& z1, size_t& y0, size_t& y1) const {
        range(nz, pz, tid / py, z0, z1);
        range(ny, py, tid % py, y0, y1);
    }
};

void initializeGrid(Grid& grid, Grid& other, const Decomposition& dec, const ThreadPinning& pinning,
                    const size_t nx, const size_t ny, const size_t nz) {
    Real* __restrict g = grid.data();
    Real* __restrict o = other.data();
    #pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        pinning.pin(tid);
        size_t z0, z1, y0, y1;
        dec.block(tid, ny, nz, z0, z1, y0, y1);
        for (size_t z = z0; z < z1; ++z) {
            for (size_t y = y0; y < y1; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    g[idx] = (idx % 19) * 1.0;
                    o[idx] = 0.0;
                }
            }
        }
    }
}

// 7-point stencil computation on the rows [y0, y1) x [z0, z1)
static inline void stencilBlock(const Real* __restrict input,
                                Real* __restrict output,
                                const size_t nx, const size_t ny, const size_t nz,
                                const size_t z0, const size_t z1,
                                const size_t y0, const size_t y1,
                                [[maybe_unused]] const bool streaming) {
    const size_t plane = nx * ny;
    for (size_t z = z0; z < z1; ++z) {
        const bool zBoundary = (z == 0 || z == nz - 1);
        for (size_t y = y0; y < y1; ++y) {
            const size_t row = idx3(0, y, z, nx, ny);
            const Real* __restrict in = input + row;
            Real* __restrict out = output + row;
            if (zBoundary || y == 0 || y == ny - 1) {
                // Copy boundary row
                for (size_t x = 0; x < nx; ++x) out[x] = in[x];
                continue;
            }
            const Real* __restrict inF = in - nx;
            const Real* __restrict inB = in + nx;
            const Real* __restrict inD = in - plane;
            const Real* __restrict inU = in + plane;
            size_t x = 1;
            const size_t xEnd = nx - 1;
#if defined(__AVX__)
            // Streaming (non-temporal) stores avoid read-for-ownership traffic on the
            // output grid; the arithmetic per element is identical to the scalar path.
            const __m256d seventh = _mm256_set1_pd(7.0);
            for (; streaming && x < xEnd && (reinterpret_cast<uintptr_t>(out + x) & 31) != 0; ++x) {
                out[x] = (in[x] + in[x - 1] + in[x + 1] + inF[x] + inB[x] + inD[x] + inU[x]) / 7.0;
            }
            for (; streaming && x + 4 <= xEnd; x += 4) {
                __m256d s = _mm256_loadu_pd(in + x);
                s = _mm256_add_pd(s, _mm256_loadu_pd(in + x - 1));
                s = _mm256_add_pd(s, _mm256_loadu_pd(in + x + 1));
                s = _mm256_add_pd(s, _mm256_loadu_pd(inF + x));
                s = _mm256_add_pd(s, _mm256_loadu_pd(inB + x));
                s = _mm256_add_pd(s, _mm256_loadu_pd(inD + x));
                s = _mm256_add_pd(s, _mm256_loadu_pd(inU + x));
                _mm256_stream_pd(out + x, _mm256_div_pd(s, seventh));
            }
#endif
            for (; x < xEnd; ++x) {
                // Simple averaging stencil (same operation order as the reference)
                out[x] = (in[x] + in[x - 1] + in[x + 1] + inF[x] + inB[x] + inD[x] + inU[x]) / 7.0;
            }
            // Copy boundary values in x
            out[0] = in[0];
            out[nx - 1] = in[nx - 1];
        }
    }
}

// Runs all iterations inside one persistent parallel region; returns the final grid
const Grid& runStencil(Grid& grid1, Grid& grid2, const Decomposition& dec, const ThreadPinning& pinning,
                       const size_t nx, const size_t ny, const size_t nz, const int iterations) {
    Real* const g1 = grid1.data();
    Real* const g2 = grid2.data();
    #pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        pinning.pin(tid);
        size_t z0, z1, y0, y1;
        dec.block(tid, ny, nz, z0, z1, y0, y1);
        const size_t tileY = dec.tileY;
        const bool streaming = dec.streaming;
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* in = (iter % 2 == 0) ? g1 : g2;
            Real* out = (iter % 2 == 0) ? g2 : g1;
            for (size_t ty = y0; ty < y1; ty += tileY) {
                stencilBlock(in, out, nx, ny, nz, z0, z1, ty, std::min(y1, ty + tileY), streaming);
            }
#if defined(__AVX__)
            _mm_sfence();  // make streaming stores visible before the barrier
#endif
            #pragma omp barrier
        }
    }
    return (iterations % 2 == 0) ? grid1 : grid2;
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const Real* g = grid.data();
    const size_t n = grid.size();

    // 1. No NaN or Inf values
    bool bad = false;
    #pragma omp parallel for reduction(||: bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(g[i]) || std::isinf(g[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = g[0];
    Real maxVal = g[0];
    #pragma omp parallel for reduction(min: minVal) reduction(max: maxVal) schedule(static)
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
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    const size_t nthreads = static_cast<size_t>(omp_get_max_threads());
    const Decomposition dec(nthreads, nx, ny, nz);
    const ThreadPinning pinning(nthreads);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, grid2, dec, pinning, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    const Grid& finalGrid = runStencil(grid1, grid2, dec, pinning, nx, ny, nz, iterations);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(std::vector<Real>(finalGrid.begin(), finalGrid.end()), "Grid");
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
