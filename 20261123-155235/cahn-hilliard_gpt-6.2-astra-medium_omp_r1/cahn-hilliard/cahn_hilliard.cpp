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

// Apply a seven-point stencil a row at a time on the unit-spaced grid.
template<class Update>
static inline void stencil(const double* field, double* result, size_t nx, size_t ny,
                           size_t nz, Update update) {
    const size_t plane = nx * ny;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * plane + y * nx;
            const double* row = field + base;
            const double* yp = y + 1 < ny ? row + nx : row;
            const double* yn = y > 0 ? row - nx : row;
            const double* zp = z + 1 < nz ? row + plane : row;
            const double* zn = z > 0 ? row - plane : row;
            auto cell = [&](size_t x, size_t xp, size_t xn) {
                const double twice = 2.0 * row[x];
                const double xx = row[xp] + row[xn] - twice;
                const double yy = yp[x] + yn[x] - twice;
                const double zz = zp[x] + zn[x] - twice;
                result[base + x] = update(base + x, xx + yy + zz);
            };

            cell(0, nx > 1 ? 1 : 0, 0);
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                cell(x, x + 1, x - 1);
            }
            if (nx > 1) cell(nx - 1, nx - 1, nx - 2);
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
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Select the initial buffer so the final field always lands in concentration.
    std::vector<double> concentration(gridSize);
    auto scratch = std::make_unique_for_overwrite<double[]>(gridSize);
    auto potential = std::make_unique_for_overwrite<double[]>(gridSize);
    double* initial = iterations > 0 && iterations % 2 ? scratch.get() : concentration.data();
    double* other = initial == concentration.data() ? scratch.get() : concentration.data();
    double* mu = potential.get();

    // Keep enough work per thread to amortize the two barriers per time step.
    const size_t usefulThreads = 1 + gridSize / 16384;
    const int threads = std::min<size_t>(omp_get_max_threads(), usefulThreads);

    printf("Initializing concentration field...\n");
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < gridSize; ++i) {
        const double pseudo = (((i + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        initial[i] = -1.0 + 2.0 * pseudo;
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(threads)
    {
        double* current = initial;
        double* next = other;
        for (int t = 0; t < iterations && gridSize > 0; ++t) {
            stencil(current, mu, nx, ny, nz, [&](size_t i, double laplacian) {
                const double cv = current[i];
                return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            });
            stencil(mu, next, nx, ny, nz, [&](size_t i, double laplacian) {
                return current[i] + dt * D * laplacian;
            });
            // Each thread swaps its own pointers after the update barrier.
            std::swap(current, next);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(concentration, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(concentration, nx, ny, nz);
        
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
