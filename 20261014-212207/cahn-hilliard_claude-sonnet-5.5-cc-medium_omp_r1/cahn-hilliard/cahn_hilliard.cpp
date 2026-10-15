#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Laplacian of one x-row with clamped boundary conditions.
// Rows are given as pointers to the center row and its y/z neighbours (already clamped).
static inline double lapAt(const double* __restrict cc, const double* __restrict cyp, const double* __restrict cyn,
                           const double* __restrict czp, const double* __restrict czn,
                           const size_t x, const size_t xp, const size_t xn,
                           const double dx2, const double dy2, const double dz2) noexcept {
    const double v = cc[x];
    const double cxx = (cc[xp] + cc[xn] - 2.0 * v) / dx2;
    const double cyy = (cyp[x] + cyn[x] - 2.0 * v) / dy2;
    const double czz = (czp[x] + czn[x] - 2.0 * v) / dz2;
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& cvec, std::vector<double>& muvec,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double* __restrict c = cvec.data();
    double* __restrict mu = muvec.data();
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t zn = (z > 0) ? z - 1 : 0;
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const double* cc = c + z * plane + y * nx;
            const double* cyp = c + z * plane + yp * nx;
            const double* cyn = c + z * plane + yn * nx;
            const double* czp = c + zp * plane + y * nx;
            const double* czn = c + zn * plane + y * nx;
            double* mr = mu + z * plane + y * nx;
            auto pt = [&](size_t x, size_t xp, size_t xn) {
                const double cv = cc[x];
                mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * lapAt(cc, cyp, cyn, czp, czn, x, xp, xn, dx2, dy2, dz2);
            };
            pt(0, nx > 1 ? 1 : 0, 0);
            if (nx > 1) {
#pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) pt(x, x + 1, x - 1);
                pt(nx - 1, nx - 1, nx - 2);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnewv, const std::vector<double>& coldv,
                        const std::vector<double>& muvec,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double* __restrict cold = coldv.data();
    const double* __restrict mu = muvec.data();
    double* __restrict cnew = cnewv.data();
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t zn = (z > 0) ? z - 1 : 0;
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const double* mc = mu + z * plane + y * nx;
            const double* myp = mu + z * plane + yp * nx;
            const double* myn = mu + z * plane + yn * nx;
            const double* mzp = mu + zp * plane + y * nx;
            const double* mzn = mu + zn * plane + y * nx;
            const double* co = cold + z * plane + y * nx;
            double* cn = cnew + z * plane + y * nx;
            auto pt = [&](size_t x, size_t xp, size_t xn) {
                cn[x] = co[x] + dt * D * lapAt(mc, myp, myn, mzp, mzn, x, xp, xn, dx2, dy2, dz2);
            };
            pt(0, nx > 1 ? 1 : 0, 0);
            if (nx > 1) {
#pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) pt(x, x + 1, x - 1);
                pt(nx - 1, nx - 1, nx - 2);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
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
    const long long n = static_cast<long long>(c.size());
    const double* cp = c.data();
    int bad = 0;
#pragma omp parallel for reduction(|:bad) schedule(static)
    for (long long i = 0; i < n; ++i) {
        if (std::isnan(cp[i]) || std::isinf(cp[i])) bad = 1;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (long long i = 0; i < n; ++i) {
        minVal = std::min(minVal, cp[i]);
        maxVal = std::max(maxVal, cp[i]);
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
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
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
        print_results(cold, "Concentration");
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
