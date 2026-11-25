#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// All workers traverse the same stencil, applying the phase-specific update
// to disjoint output cells. The worksharing barrier completes each phase.
template<class Update>
void applyStencil(const double* field, double* output, size_t nx, size_t ny, size_t nz,
                  const Update& update) {
    constexpr size_t tileWidth = 256;
    const size_t tiles = nx / tileWidth + (nx % tileWidth != 0);
    const size_t plane = nx * ny;
    #pragma omp for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t tile = 0; tile < tiles; ++tile) {
                const size_t base = z * plane + y * nx;
                const double* row = field + base;
                const double* north = y ? row - nx : row;
                const double* south = y + 1 < ny ? row + nx : row;
                const double* below = z ? row - plane : row;
                const double* above = z + 1 < nz ? row + plane : row;
                auto cell = [&](size_t x, size_t left, size_t right) {
                    const double twice = 2.0 * row[x];
                    // Grid spacing is one in all three directions.
                    const double xx = row[right] + row[left] - twice;
                    const double yy = south[x] + north[x] - twice;
                    const double zz = above[x] + below[x] - twice;
                    output[base + x] = update(base + x, xx + yy + zz);
                };
                const size_t begin = tile * tileWidth;
                const size_t end = std::min(begin + tileWidth, nx);
                if (begin == 0) cell(0, 0, nx > 1 ? 1 : 0);
                const size_t interiorEnd = std::min(end, nx - 1);
                #pragma omp simd
                for (size_t x = std::max(begin, size_t{1}); x < interiorEnd; ++x) {
                    cell(x, x - 1, x + 1);
                }
                if (end == nx && nx > 1) cell(nx - 1, nx - 2, nx - 1);
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
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    const size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    // Bound team overhead on small grids while respecting the runtime limit.
    constexpr size_t cellsPerThread = 16384;
    const size_t usefulThreads = gridSize ? 1 + (gridSize - 1) / cellsPerThread : 1;
    const int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(), usefulThreads));
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < gridSize; ++i) {
        const double pseudo = (((i + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[i] = -1.0 + 2.0 * pseudo;
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(threads)
    {
        double* potential = mu.data();
        double* current = cold.data();
        double* next = cnew.data();
        for (int t = 0; t < iterations; ++t) {
            applyStencil(current, potential, nx, ny, nz, [&](size_t i, double laplacian) {
                const double cv = current[i];
                return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            });
            applyStencil(potential, next, nx, ny, nz, [&](size_t i, double laplacian) {
                return current[i] + dt * D * laplacian;
            });
            std::swap(current, next);
        }
    }
    if (iterations > 0 && iterations % 2) cold.swap(cnew);

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
