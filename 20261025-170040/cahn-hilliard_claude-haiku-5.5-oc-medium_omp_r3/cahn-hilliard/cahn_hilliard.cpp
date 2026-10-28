#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Allocator whose value-construction is a no-op for trivial types. Grid buffers are
// therefore not zero-filled by the master thread; their pages are first written by the
// parallel loops, which places each page on the NUMA node of the thread that uses it.
template <typename T>
struct default_init_allocator : std::allocator<T> {
    using std::allocator<T>::allocator;
    template <typename U>
    struct rebind { using other = default_init_allocator<U>; };

    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

using Grid = std::vector<double, default_init_allocator<double>>;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Laplacian of one cell with clamped boundary conditions. cr is the (y,z) row of the field,
// cyp/cym the rows at y+1/y-1 and czp/czm the rows at z+1/z-1 (all clamped). xn/xp are the
// clamped x-neighbour indices. Operation order matches the reference formulation exactly.
inline double laplacianCell(const double* cr, const double* cyp, const double* cym,
                            const double* czp, const double* czm,
                            const size_t x, const size_t xn, const size_t xp,
                            const double dx, const double dy, const double dz) {
    const double cc = cr[x];
    const double cxx = (cr[xp] + cr[xn] - 2.0 * cc) / (dx * dx);
    const double cyy = (cyp[x] + cym[x] - 2.0 * cc) / (dy * dy);
    const double czz = (czp[x] + czm[x] - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

// Invokes fn(x, xn, xp) for every x in a row. Interior cells use direct neighbours so the
// loop vectorizes; the two x-boundary cells use clamped neighbours.
template <typename Fn>
inline void forEachRowCell(const size_t nx, Fn&& fn) {
    if (nx == 1) {
        fn(size_t{0}, size_t{0}, size_t{0});
        return;
    }
    fn(size_t{0}, size_t{0}, size_t{1});
    #pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        fn(x, x - 1, x + 1);
    }
    fn(nx - 1, nx - 2, nx - 1);
}

// Compute chemical potential
void computeChemicalPotential(const Grid& c, Grid& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t zn = (z > 0) ? z - 1 : 0;

            const double* cr = c.data() + idx3(0, y, z, nx, ny);
            const double* cyp = c.data() + idx3(0, yp, z, nx, ny);
            const double* cym = c.data() + idx3(0, yn, z, nx, ny);
            const double* czp = c.data() + idx3(0, y, zp, nx, ny);
            const double* czm = c.data() + idx3(0, y, zn, nx, ny);
            double* mr = mu.data() + idx3(0, y, z, nx, ny);

            forEachRowCell(nx, [&](const size_t x, const size_t xn, const size_t xp) {
                const double cv = cr[x];
                mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * laplacianCell(cr, cyp, cym, czp, czm, x, xn, xp, dx, dy, dz);
            });
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(Grid& cnew, const Grid& cold,
                        const Grid& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t zn = (z > 0) ? z - 1 : 0;

            const double* mr = mu.data() + idx3(0, y, z, nx, ny);
            const double* myp = mu.data() + idx3(0, yp, z, nx, ny);
            const double* mym = mu.data() + idx3(0, yn, z, nx, ny);
            const double* mzp = mu.data() + idx3(0, y, zp, nx, ny);
            const double* mzm = mu.data() + idx3(0, y, zn, nx, ny);
            const double* co = cold.data() + idx3(0, y, z, nx, ny);
            double* nr = cnew.data() + idx3(0, y, z, nx, ny);

            forEachRowCell(nx, [&](const size_t x, const size_t xn, const size_t xp) {
                nr[x] = co[x] + dt * D * laplacianCell(mr, myp, mym, mzp, mzm, x, xn, xp, dx, dy, dz);
            });
        }
    }
}

// Initialize concentration field
void initializeConcentration(Grid& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const Grid& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    long long badCount = 0;
    #pragma omp parallel for schedule(static) proc_bind(close) reduction(+:badCount)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            ++badCount;
        }
    }
    if (badCount > 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for schedule(static) proc_bind(close) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    Grid cold(gridSize);
    Grid cnew(gridSize);
    Grid mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> out(cold.begin(), cold.end());
        print_results(out, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
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
