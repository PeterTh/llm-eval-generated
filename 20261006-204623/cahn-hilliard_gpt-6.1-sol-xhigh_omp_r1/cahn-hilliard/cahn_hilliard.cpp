#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Row pointers keep boundary/index calculations out of the SIMD loop.
inline double rowLaplacian(const double* row, const double* yn, const double* yp,
                          const double* zn, const double* zp,
                          const size_t x, const size_t xn, const size_t xp,
                          const double dx2, const double dy2, const double dz2) {
    const double cxx = (row[xp] + row[xn] - 2.0 * row[x]) / dx2;
    const double cyy = (yp[x] + yn[x] - 2.0 * row[x]) / dy2;
    const double czz = (zp[x] + zn[x] - 2.0 * row[x]) / dz2;
    return cxx + cyy + czz;
}

// All threads in the simulation team call this worksharing kernel. The implicit
// barrier completes the entire stencil pass before its output is read.
// Inline the kernels so the fixed physical parameters (including unit grid
// spacing) are folded into the vectorized loops rather than divided at runtime.
template <typename Store>
[[gnu::always_inline]] inline void applyLaplacian(
    const double* field, const size_t nx, const size_t ny, const size_t nz,
    const double dx, const double dy, const double dz, Store store) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            if (nx == 0) continue;
            const size_t base = z * plane + y * nx;
            const double* row = field + base;
            const double* yn = (y > 0) ? row - nx : row;
            const double* yp = (y + 1 < ny) ? row + nx : row;
            const double* zn = (z > 0) ? row - plane : row;
            const double* zp = (z + 1 < nz) ? row + plane : row;

            store(base, rowLaplacian(row, yn, yp, zn, zp, 0, 0, nx > 1 ? 1 : 0, dx2, dy2, dz2));

            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                store(base + x, rowLaplacian(row, yn, yp, zn, zp, x, x - 1, x + 1, dx2, dy2, dz2));
            }

            if (nx > 1) {
                const size_t x = nx - 1;
                store(base + x, rowLaplacian(row, yn, yp, zn, zp, x, x - 1, x, dx2, dy2, dz2));
            }
        }
    }
}

// Compute chemical potential within the persistent OpenMP team.
[[gnu::always_inline]] inline void computeChemicalPotential(
    const double* c, double* mu, const size_t nx, const size_t ny, const size_t nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    applyLaplacian(c, nx, ny, nz, dx, dy, dz, [=](size_t idx, double laplacian) {
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                  + 3.0 * cv + cv * cv * cv - gamma * laplacian;
    });
}

// Cahn-Hilliard update step
[[gnu::always_inline]] inline void cahnHilliardUpdate(
    double* cnew, const double* cold, const double* mu,
    const size_t nx, const size_t ny, const size_t nz,
    const double D, const double dt, const double dx, const double dy, const double dz) {
    applyLaplacian(mu, nx, ny, nz, dx, dy, dz, [=](size_t idx, double laplacian) {
        cnew[idx] = cold[idx] + dt * D * laplacian;
    });
}

// Initialize concentration and first-touch scratch buffers before timing.
void initializeConcentration(double* c, double* cnew, double* mu,
                             const size_t nx, const size_t ny, const size_t nz, const int threadCount) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for schedule(static) num_threads(threadCount)
    for (size_t idx = 0; idx < vol; ++idx) {
        // Preserve the original deterministic pseudo-random sequence.
        const double pseudo = ((((idx + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
        cnew[idx] = 0.0;
        mu[idx] = 0.0;
    }
}

bool validateResult(const std::span<const double> c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

    // Keep small default grids from spending more time synchronizing an idle
    // team than computing. Explicit OpenMP thread counts remain authoritative.
    int threadCount = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        const size_t workThreads = std::max(size_t{1}, gridSize / 16384);
        threadCount = static_cast<int>(std::min(static_cast<size_t>(threadCount), workThreads));
    }
    
    // Allocate without serial zero-fills. Parallel initialization places pages
    // on the workers' NUMA nodes and keeps page faults outside the timed loop.
    auto cold = std::make_unique_for_overwrite<double[]>(gridSize);
    auto cnew = std::make_unique_for_overwrite<double[]>(gridSize);
    auto mu = std::make_unique_for_overwrite<double[]>(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold.get(), cnew.get(), mu.get(), nx, ny, nz, threadCount);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(threadCount)
    {
        // Private buffer pointers avoid a shared swap and an extra barrier.
        double* current = cold.get();
        double* next = cnew.get();
        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(current, mu.get(), nx, ny, nz, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdate(next, current, mu.get(), nx, ny, nz, D, dt, dx, dy, dz);
            std::swap(current, next);
        }
    }
    if (iterations > 0 && iterations % 2 != 0) std::swap(cold, cnew);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / elapsedSeconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        // Reuse the existing reporting interface outside the timed simulation.
        // Release scratch buffers first to keep peak memory within three grids.
        cnew.reset();
        mu.reset();
        const std::vector<double> result(cold.get(), cold.get() + gridSize);
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(std::span<const double>(cold.get(), gridSize), nx, ny, nz);
        
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
