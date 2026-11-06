#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute a Laplacian using preselected rows for the clamped boundaries.
inline double rowLaplacian(const double* center, const double* ym, const double* yp,
                           const double* zm, const double* zp, const size_t x,
                           const size_t xm, const size_t xp,
                           const double dx2, const double dy2, const double dz2) noexcept {
    const double value = center[x];
    const double cxx = (center[xp] + center[xm] - 2.0 * value) / dx2;
    const double cyy = (yp[x] + ym[x] - 2.0 * value) / dy2;
    const double czz = (zp[x] + zm[x] - 2.0 * value) / dz2;
    return cxx + cyy + czz;
}

// These worksharing loops are called by every thread in the simulation team.
// Each implicit barrier makes the complete field available to the next stage.
void computeChemicalPotential(const double* c, double* mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    const double* const data = c;
    double* const output = mu;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            if (nx == 0) continue;
            const size_t row = z * plane + y * nx;
            const double* const center = data + row;
            const double* const ym = data + z * plane + (y > 0 ? y - 1 : y) * nx;
            const double* const yp = data + z * plane + (y + 1 < ny ? y + 1 : y) * nx;
            const double* const zm = data + (z > 0 ? z - 1 : z) * plane + y * nx;
            const double* const zp = data + (z + 1 < nz ? z + 1 : z) * plane + y * nx;
            const auto update = [&](size_t x, size_t xm, size_t xp) {
                const double cv = center[x];
                output[row + x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                  + 3.0 * cv + cv * cv * cv
                                  - gamma * rowLaplacian(center, ym, yp, zm, zp, x, xm, xp, dx2, dy2, dz2);
            };
            update(0, 0, nx > 1 ? 1 : 0);
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                update(x, x - 1, x + 1);
            }
            if (nx > 1) update(nx - 1, nx - 2, nx - 1);
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(double* cnew, const double* cold,
                        const double* mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    const double* const data = mu;
    const double* const old = cold;
    double* const output = cnew;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            if (nx == 0) continue;
            const size_t row = z * plane + y * nx;
            const double* const center = data + row;
            const double* const ym = data + z * plane + (y > 0 ? y - 1 : y) * nx;
            const double* const yp = data + z * plane + (y + 1 < ny ? y + 1 : y) * nx;
            const double* const zm = data + (z > 0 ? z - 1 : z) * plane + y * nx;
            const double* const zp = data + (z + 1 < nz ? z + 1 : z) * plane + y * nx;
            const auto update = [&](size_t x, size_t xm, size_t xp) {
                output[row + x] = old[row + x] + dt * D *
                    rowLaplacian(center, ym, yp, zm, zp, x, xm, xp, dx2, dy2, dz2);
            };
            update(0, 0, nx > 1 ? 1 : 0);
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                update(x, x - 1, x + 1);
            }
            if (nx > 1) update(nx - 1, nx - 2, nx - 1);
        }
    }
}

// Initialize concentration field
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < vol; ++idx) {
        // Generate the same deterministic pseudo-random value in [-1, 1].
        const double pseudo = (((idx + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t gridSize = nx * ny * nz;
    // Check for NaN or Inf
    for (size_t i = 0; i < gridSize; ++i) {
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
    for (size_t i = 0; i < gridSize; ++i) {
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
    
    // Uninitialized storage is first touched by the threads that fill each field.
    std::unique_ptr<double[]> cold(new double[gridSize]);
    std::unique_ptr<double[]> cnew(new double[gridSize]);
    std::unique_ptr<double[]> mu(new double[gridSize]);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold.get(), nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        // Private pointer swaps avoid an additional team-wide barrier per step.
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
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> results(cold.get(), cold.get() + gridSize);
        print_results(results, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold.get(), nx, ny, nz);
        
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
