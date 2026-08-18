#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t z = 0; z < static_cast<ptrdiff_t>(nz); ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, static_cast<size_t>(z), nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = static_cast<size_t>(z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
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
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    const size_t planeSize = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    std::vector<double>* current = &cold;
    std::vector<double>* next = &cnew;

    // Keep the worker team alive across all time steps.  Each sweep writes a
    // distinct row, so the only synchronization required is between sweeps
    // and after exchanging the two concentration buffers.
    #pragma omp parallel default(none) shared(current, next, mu, nx, ny, nz, planeSize, dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB, D, dt, iterations)
    {
        for (int t = 0; t < iterations; ++t) {
            const double* const c = current->data();
            double* const chemicalPotential = mu.data();

            #pragma omp for collapse(2) schedule(static)
            for (ptrdiff_t z = 0; z < static_cast<ptrdiff_t>(nz); ++z) {
                for (ptrdiff_t y = 0; y < static_cast<ptrdiff_t>(ny); ++y) {
                    const size_t zs = static_cast<size_t>(z);
                    const size_t ys = static_cast<size_t>(y);
                    const size_t zBase = zs * planeSize;
                    const size_t zmBase = (zs == 0 ? 0 : zs - 1) * planeSize;
                    const size_t zpBase = (zs + 1 == nz ? zs : zs + 1) * planeSize;
                    const size_t row = zBase + ys * nx;
                    const size_t ymRow = zBase + (ys == 0 ? 0 : ys - 1) * nx;
                    const size_t ypRow = zBase + (ys + 1 == ny ? ys : ys + 1) * nx;

                    for (size_t x = 0; x < nx; ++x) {
                        const size_t idx = row + x;
                        const double cv = c[idx];
                        const size_t xm = x == 0 ? 0 : x - 1;
                        const size_t xp = x + 1 == nx ? x : x + 1;
                        const double cxx = (c[row + xp] + c[row + xm] - 2.0 * cv) / dx2;
                        const double cyy = (c[ypRow + x] + c[ymRow + x] - 2.0 * cv) / dy2;
                        const double czz = (c[zpBase + ys * nx + x] + c[zmBase + ys * nx + x] - 2.0 * cv) / dz2;
                        chemicalPotential[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                               + 3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
                    }
                }
            }

            const double* const potential = mu.data();
            double* const updated = next->data();

            #pragma omp for collapse(2) schedule(static)
            for (ptrdiff_t z = 0; z < static_cast<ptrdiff_t>(nz); ++z) {
                for (ptrdiff_t y = 0; y < static_cast<ptrdiff_t>(ny); ++y) {
                    const size_t zs = static_cast<size_t>(z);
                    const size_t ys = static_cast<size_t>(y);
                    const size_t zBase = zs * planeSize;
                    const size_t zmBase = (zs == 0 ? 0 : zs - 1) * planeSize;
                    const size_t zpBase = (zs + 1 == nz ? zs : zs + 1) * planeSize;
                    const size_t row = zBase + ys * nx;
                    const size_t ymRow = zBase + (ys == 0 ? 0 : ys - 1) * nx;
                    const size_t ypRow = zBase + (ys + 1 == ny ? ys : ys + 1) * nx;

                    for (size_t x = 0; x < nx; ++x) {
                        const size_t idx = row + x;
                        const double mv = potential[idx];
                        const size_t xm = x == 0 ? 0 : x - 1;
                        const size_t xp = x + 1 == nx ? x : x + 1;
                        const double mxx = (potential[row + xp] + potential[row + xm] - 2.0 * mv) / dx2;
                        const double myy = (potential[ypRow + x] + potential[ymRow + x] - 2.0 * mv) / dy2;
                        const double mzz = (potential[zpBase + ys * nx + x] + potential[zmBase + ys * nx + x] - 2.0 * mv) / dz2;
                        updated[idx] = c[idx] + dt * D * (mxx + myy + mzz);
                    }
                }
            }

            #pragma omp single
            {
                std::swap(current, next);
            }
        }
    }

    // `current` holds the newest field after the final buffer exchange.
    if (current != &cold) {
        cold.swap(cnew);
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
