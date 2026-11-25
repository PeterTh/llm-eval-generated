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

// Apply the unit-spacing Laplacian with clamped boundaries, then transform
// the result. Every thread in the team participates in each stencil pass.
template <typename Transform>
void applyStencil(double* output, const double* input,
                  size_t nx, size_t ny, size_t nz,
                  size_t tiles, size_t tileWidth, Transform transform) {
    const size_t plane = nx * ny;
    #pragma omp for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t tile = 0; tile < tiles; ++tile) {
                const size_t first = tile * tileWidth;
                const size_t last = std::min(nx, first + tileWidth);
                if (first >= last) continue;

                const size_t base = z * plane + y * nx;
                const double* center = input + base;
                const double* north = y > 0 ? center - nx : center;
                const double* south = y + 1 < ny ? center + nx : center;
                const double* below = z > 0 ? center - plane : center;
                const double* above = z + 1 < nz ? center + plane : center;

                const auto updateCell = [&](size_t x, size_t xp, size_t xn) {
                    const double twiceCenter = 2.0 * center[x];
                    const double cxx = center[xp] + center[xn] - twiceCenter;
                    const double cyy = south[x] + north[x] - twiceCenter;
                    const double czz = above[x] + below[x] - twiceCenter;
                    output[base + x] = transform(base + x, cxx + cyy + czz);
                };

                if (first == 0) updateCell(0, nx > 1 ? 1 : 0, 0);
                const size_t interiorBegin = std::max(first, size_t{1});
                const size_t interiorEnd = std::min(last, nx - 1);
                #pragma omp simd
                for (size_t x = interiorBegin; x < interiorEnd; ++x) {
                    updateCell(x, x + 1, x - 1);
                }
                if (last == nx && nx > 1) updateCell(nx - 1, nx - 1, nx - 2);
            }
        }
    }
}

bool validateResult(std::span<const double> c) {
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

    const size_t gridSize = nx * ny * nz;
    auto initial = std::make_unique_for_overwrite<double[]>(gridSize);
    auto next = std::make_unique_for_overwrite<double[]>(gridSize);
    auto potential = std::make_unique_for_overwrite<double[]>(gridSize);

    // Keep enough cells per worker to amortize the stencil barriers on small grids.
    constexpr size_t cellsPerThread = 8192;
    const int threads = static_cast<int>(std::min(
        static_cast<size_t>(omp_get_max_threads()),
        std::max(size_t{1}, gridSize / cellsPerThread)
    ));
    // Split long rows only when there are too few rows to occupy the team.
    const size_t rows = std::max(size_t{1}, ny * nz);
    const size_t tiles = 1 + (static_cast<size_t>(threads) - 1) / rows;
    const size_t tileWidth = nx / tiles + (nx % tiles != 0);

    printf("Initializing concentration field...\n");
    // First-touch all three fields in the same row order used by the solver.
    #pragma omp parallel for collapse(3) schedule(static) num_threads(threads)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t tile = 0; tile < tiles; ++tile) {
                const size_t base = (z * ny + y) * nx;
                const size_t first = tile * tileWidth;
                const size_t last = std::min(nx, first + tileWidth);
                for (size_t x = first; x < last; ++x) {
                    const size_t i = base + x;
                    const double pseudo = ((i + 1) * 1299709 % gridSize) / static_cast<double>(gridSize);
                    initial[i] = -1.0 + 2.0 * pseudo;
                    next[i] = 0.0;
                    potential[i] = 0.0;
                }
            }
        }
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(threads)
    {
        double* cold = initial.get();
        double* cnew = next.get();
        double* mu = potential.get();

        for (int t = 0; t < iterations; ++t) {
            applyStencil(mu, cold, nx, ny, nz, tiles, tileWidth, [&](size_t i, double laplacian) {
                const double cv = cold[i];
                return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            });

            applyStencil(cnew, mu, nx, ny, nz, tiles, tileWidth, [&](size_t i, double laplacian) {
                return cold[i] + dt * D * laplacian;
            });

            // The stencil barrier completes all writes before each worker swaps
            // its private pointers for the next time step.
            std::swap(cold, cnew);
        }
    }

    const std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / elapsed.count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    if (iterations > 0 && iterations % 2 != 0) std::swap(initial, next);
    next.reset();
    potential.reset();
    const std::span<const double> concentration(initial.get(), gridSize);

    // Print results for external validation
    if (printResults) {
        print_results(std::vector<double>(concentration.begin(), concentration.end()), "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(concentration);
        
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
