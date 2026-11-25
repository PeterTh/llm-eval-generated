#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Apply the unit-spacing Laplacian along a row segment. Handle the two clamped
// endpoints separately so every interior SIMD lane uses consecutive loads.
template <class Update>
inline void stencilRow(const double* field, double* output, size_t nx, size_t ny,
                       size_t nz, size_t y, size_t z, size_t begin, size_t end, Update update) {
    const size_t plane = nx * ny;
    const size_t offset = z * plane + y * nx;
    const double* row = field + offset;
    const double* yp = y + 1 < ny ? row + nx : row;
    const double* yn = y > 0 ? row - nx : row;
    const double* zp = z + 1 < nz ? row + plane : row;
    const double* zn = z > 0 ? row - plane : row;

    auto cell = [&](size_t x, size_t right, size_t left) {
        const double twice = 2.0 * row[x];
        const double cxx = (row[right] + row[left]) - twice;
        const double cyy = (yp[x] + yn[x]) - twice;
        const double czz = (zp[x] + zn[x]) - twice;
        output[offset + x] = update(offset + x, row[x], cxx + cyy + czz);
    };

    if (begin == 0) cell(0, nx > 1 ? 1 : 0, 0);
    const size_t firstInterior = std::max<size_t>(1, begin);
    const size_t endInterior = std::min(nx - 1, end);
    #pragma omp simd
    for (size_t x = firstInterior; x < endInterior; ++x) {
        cell(x, x + 1, x - 1);
    }
    if (end == nx && nx > 1) cell(nx - 1, nx - 1, nx - 2);
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
    
    constexpr size_t rowBlock = 1024;
    const size_t blocksPerRow = nx / rowBlock + (nx % rowBlock != 0);
    const size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    // Limit barrier overhead on small grids while respecting the runtime limit.
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), 1 + gridSize / 32768));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < gridSize; ++i) {
        const double pseudo = (((i + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[i] = -1.0 + 2.0 * pseudo;
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(threads)
    {
        double* current = cold.data();
        double* next = cnew.data();
        for (int t = 0; t < iterations; ++t) {
            #pragma omp for collapse(3) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t block = 0; block < blocksPerRow; ++block) {
                        const size_t begin = block * rowBlock;
                        const size_t end = std::min(begin + rowBlock, nx);
                        stencilRow(current, mu.data(), nx, ny, nz, y, z, begin, end,
                            [&](size_t, double cv, double laplacian) {
                                return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                     + 3.0 * cv + cv * cv * cv - gamma * laplacian;
                            });
                    }
                }
            }
            #pragma omp for collapse(3) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t block = 0; block < blocksPerRow; ++block) {
                        const size_t begin = block * rowBlock;
                        const size_t end = std::min(begin + rowBlock, nx);
                        stencilRow(mu.data(), next, nx, ny, nz, y, z, begin, end,
                            [&](size_t i, double, double laplacian) {
                                return current[i] + dt * D * laplacian;
                            });
                    }
                }
            }
            std::swap(current, next);
        }
    }
    if (iterations > 0 && iterations % 2 != 0) cold.swap(cnew);

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
