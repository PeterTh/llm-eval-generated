#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Allocator that default-initializes (no serial zero-fill), so the first touch
// of each page happens in the parallel initialization loops (NUMA locality).
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template <typename U> struct rebind { using other = DefaultInitAllocator<U>; };
    using std::allocator<T>::allocator;
    template <typename U>
    void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
};
using Field = std::vector<double, DefaultInitAllocator<double>>;

// Laplacian of one x-row (fixed y,z) with clamped boundary conditions.
// out[x] = base(x) + scale * lap(x), where base is either a polynomial of c (mode 0) or cold (mode 1).
template <int MODE>
static inline void rowKernel(double* __restrict out, const double* __restrict f,
                             const double* __restrict ypr, const double* __restrict ynr,
                             const double* __restrict zpr, const double* __restrict znr,
                             const double* __restrict cold, const size_t nx,
                             const double dx2, const double dy2, const double dz2,
                             const double gamma, const double e_AA, const double e_BB, const double e_AB,
                             const double dtD) {
    auto point = [&](const size_t x, const size_t xp, const size_t xn) {
        const double cv = f[x];
        const double cxx = (f[xp] + f[xn] - 2.0 * cv) / dx2;
        const double cyy = (ypr[x] + ynr[x] - 2.0 * cv) / dy2;
        const double czz = (zpr[x] + znr[x] - 2.0 * cv) / dz2;
        const double lap = cxx + cyy + czz;
        if (MODE == 0) {
            out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                     + 3.0 * cv + cv * cv * cv - gamma * lap;
        } else {
            out[x] = cold[x] + dtD * lap;
        }
    };
    point(0, nx > 1 ? 1 : 0, 0);
    if (nx > 1) {
#pragma omp simd
        for (size_t x = 1; x < nx - 1; ++x) point(x, x + 1, x - 1);
        point(nx - 1, nx - 1, nx - 2);
    }
}

// Compute chemical potential
void computeChemicalPotential(const Field& c, Field& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double* cp = c.data();
    double* mp = mu.data();
    const double sx = dx * dx, sy = dy * dy, sz = dz * dz;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t zp = (z < nz - 1) ? z + 1 : z, zn = (z > 0) ? z - 1 : 0;
            const size_t yp = (y < ny - 1) ? y + 1 : y, yn = (y > 0) ? y - 1 : 0;
            rowKernel<0>(mp + idx3(0, y, z, nx, ny), cp + idx3(0, y, z, nx, ny),
                         cp + idx3(0, yp, z, nx, ny), cp + idx3(0, yn, z, nx, ny),
                         cp + idx3(0, y, zp, nx, ny), cp + idx3(0, y, zn, nx, ny),
                         nullptr, nx, sx, sy, sz, gamma, e_AA, e_BB, e_AB, 0.0);
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(Field& cnew, const Field& cold, const Field& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double* mp = mu.data();
    const double* op = cold.data();
    double* np = cnew.data();
    const double sx = dx * dx, sy = dy * dy, sz = dz * dz;
    const double dtD = dt * D;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t zp = (z < nz - 1) ? z + 1 : z, zn = (z > 0) ? z - 1 : 0;
            const size_t yp = (y < ny - 1) ? y + 1 : y, yn = (y > 0) ? y - 1 : 0;
            rowKernel<1>(np + idx3(0, y, z, nx, ny), mp + idx3(0, y, z, nx, ny),
                         mp + idx3(0, yp, z, nx, ny), mp + idx3(0, yn, z, nx, ny),
                         mp + idx3(0, y, zp, nx, ny), mp + idx3(0, y, zn, nx, ny),
                         op + idx3(0, y, z, nx, ny), nx, sx, sy, sz, 0.0, 0.0, 0.0, 0.0, dtD);
        }
    }
}

// Initialize concentration field
void initializeConcentration(Field& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double* cp = c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vol; ++i) {
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((i + 1) * 1299709) % vol) / static_cast<double>(vol));
        cp[i] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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
    Field cold(gridSize);
    Field cnew(gridSize);
    Field mu(gridSize);
    // First-touch the remaining buffers in parallel
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < gridSize; ++i) {
        cnew[i] = 0.0;
        mu[i] = 0.0;
    }
    
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
    
    std::vector<double> result;
    if (printResults || validate) result.assign(cold.begin(), cold.end());

    // Print results for external validation
    if (printResults) {
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, nx, ny, nz);
        
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
