#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

#ifndef _OPENMP
#error "This benchmark requires OpenMP support"
#endif

// Neighboring rows are selected once per row, including clamped Y/Z boundaries.
// Keep the original arithmetic order and handle X boundaries outside SIMD loops.
inline double computeLaplacian(const double* row, const double* yp, const double* yn,
                               const double* zp, const double* zn,
                               const size_t x, const size_t xp, const size_t xn,
                               const double dx2, const double dy2, const double dz2) {
    const double center = row[x];
    const double cxx = (row[xp] + row[xn] - 2.0 * center) / dx2;
    const double cyy = (yp[x] + yn[x] - 2.0 * center) / dy2;
    const double czz = (zp[x] + zn[x] - 2.0 * center) / dz2;
    return cxx + cyy + czz;
}

// These worksharing routines are called by every thread in the persistent team.
inline void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            if (nx == 0) continue;
            const size_t base = z * plane + y * nx;
            const double* row = c + base;
            const double* yp = row + (y + 1 < ny ? nx : 0);
            const double* yn = row - (y > 0 ? nx : 0);
            const double* zp = row + (z + 1 < nz ? plane : 0);
            const double* zn = row - (z > 0 ? plane : 0);
            double* out = mu + base;
            const auto cell = [=](size_t x, size_t xp, size_t xn) {
                const double cv = row[x];
                out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(row, yp, yn, zp, zn, x, xp, xn, dx2, dy2, dz2);
            };
            cell(0, nx > 1 ? 1 : 0, 0);
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                cell(x, x + 1, x - 1);
            }
            if (nx > 1) cell(nx - 1, nx - 1, nx - 2);
        }
    }
    // The implicit barrier makes all of mu available to the update stencil.
}

// Cahn-Hilliard update step
inline void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                        const double* __restrict mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            if (nx == 0) continue;
            const size_t base = z * plane + y * nx;
            const double* row = mu + base;
            const double* yp = row + (y + 1 < ny ? nx : 0);
            const double* yn = row - (y > 0 ? nx : 0);
            const double* zp = row + (z + 1 < nz ? plane : 0);
            const double* zn = row - (z > 0 ? plane : 0);
            const double* old = cold + base;
            double* out = cnew + base;
            const auto cell = [=](size_t x, size_t xp, size_t xn) {
                out[x] = old[x] + dt * D *
                         computeLaplacian(row, yp, yn, zp, zn, x, xp, xn, dx2, dy2, dz2);
            };
            cell(0, nx > 1 ? 1 : 0, 0);
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                cell(x, x + 1, x - 1);
            }
            if (nx > 1) cell(nx - 1, nx - 1, nx - 2);
        }
    }
    // The implicit barrier completes cnew before it becomes the next input.
}

// Initialize all buffers with the same row ownership as the simulation.
void initializeConcentration(double* __restrict c, double* __restrict cnew, double* __restrict mu,
                             const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    #pragma omp parallel for collapse(2) schedule(static) default(none) shared(c, cnew, mu) firstprivate(nx, ny, nz, vol)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * (nx * ny) + y * nx;
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                // Generate pseudo-random value in [-1, 1]
                const double pseudo = ((((idx + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
                cnew[idx] = 0.0;
                mu[idx] = 0.0;
            }
        }
    }
}

bool validateResult(const double* c, const size_t size) {
    // Check for NaN or Inf
    for (size_t i = 0; i < size; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (size_t i = 0; i < size; ++i) {
        const double val = c[i];
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
    
    // Leave storage uninitialized so the owning threads first-touch its pages.
    auto cold = std::make_unique_for_overwrite<double[]>(gridSize);
    auto cnew = std::make_unique_for_overwrite<double[]>(gridSize);
    auto mu = std::make_unique_for_overwrite<double[]>(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold.get(), cnew.get(), mu.get(), nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    double* current = cold.get();
    double* next = cnew.get();
    double* potential = mu.get();
    #pragma omp parallel default(none) firstprivate(current, next, potential, nx, ny, nz, iterations, dx, dy, dz, gamma, e_AA, e_BB, e_AB, D, dt)
    {
        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(current, potential, nx, ny, nz, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdate(next, current, potential, nx, ny, nz, D, dt, dx, dy, dz);
            // Private pointers avoid a shared buffer swap and an extra barrier.
            std::swap(current, next);
        }
    }
    if (iterations > 0 && iterations % 2 != 0) std::swap(cold, cnew);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        // Release scratch storage before constructing the output vector.
        cnew.reset();
        mu.reset();
        const std::vector<double> result(cold.get(), cold.get() + gridSize);
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold.get(), gridSize);
        
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
