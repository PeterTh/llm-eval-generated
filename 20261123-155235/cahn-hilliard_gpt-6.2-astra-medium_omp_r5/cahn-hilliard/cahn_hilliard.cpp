#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Apply the unit-spacing Laplacian along a row. Handle its endpoints
// separately so all six neighbor streams in the interior have unit stride.
template <class Update>
inline void stencilRow(double* dest, const double* center, const double* north,
                       const double* south, const double* below, const double* above,
                       size_t nx, Update update) {
    auto cell = [&](size_t x, size_t left, size_t right) {
        const double twice = 2.0 * center[x];
        const double xx = center[right] + center[left] - twice;
        const double yy = south[x] + north[x] - twice;
        const double zz = above[x] + below[x] - twice;
        dest[x] = update(x, xx + yy + zz);
    };
    cell(0, 0, nx > 1 ? 1 : 0);
    const size_t last = nx - 1;
    #pragma omp simd
    for (size_t x = 1; x < last; ++x) {
        cell(x, x - 1, x + 1);
    }
    if (last > 0) cell(last, last - 1, last);
}

constexpr size_t tileRows = 12;
constexpr size_t tilePlanes = 12;

// Keep three planes of chemical potential for this tile in thread-local
// scratch. Neighboring tiles recompute their one-cell halos independently.
void advanceTile(const double* cold, double* cnew, double* scratch,
                 size_t nx, size_t ny, size_t nz, size_t y0, size_t z0) {
    constexpr double dt = 0.01, D = 1.0;
    const size_t yEnd = std::min(y0 + tileRows, ny);
    const size_t zEnd = std::min(z0 + tilePlanes, nz);
    const size_t haloStart = y0 > 0 ? y0 - 1 : 0;
    const size_t haloEnd = std::min(yEnd + 1, ny);
    const size_t stride = (haloEnd - haloStart) * nx;

    auto potentialPlane = [&](size_t z, double* mu) {
        constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
        constexpr double gamma = 0.5;
        const size_t zn = z > 0 ? z - 1 : 0;
        const size_t zp = std::min(z + 1, nz - 1);
        for (size_t y = haloStart; y < haloEnd; ++y) {
            const double* row = cold + (z * ny + y) * nx;
            stencilRow(mu + (y - haloStart) * nx, row,
                       y > 0 ? row - nx : row, y + 1 < ny ? row + nx : row,
                       cold + (zn * ny + y) * nx, cold + (zp * ny + y) * nx, nx,
                       [=](size_t x, double lap) {
                           const double cv = row[x];
                           return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                  + 3.0 * cv + cv * cv * cv - gamma * lap;
                       });
        }
    };

    double* prev = scratch;
    double* curr = prev + stride;
    double* next = curr + stride;
    if (z0 > 0) potentialPlane(z0 - 1, prev);
    potentialPlane(z0, curr);
    for (size_t z = z0; z < zEnd; ++z) {
        if (z + 1 < nz) potentialPlane(z + 1, next);
        for (size_t y = y0; y < yEnd; ++y) {
            const size_t i = (z * ny + y) * nx;
            const size_t offset = (y - haloStart) * nx;
            const double* row = curr + offset;
            stencilRow(cnew + i, row, y > 0 ? row - nx : row,
                       y + 1 < ny ? row + nx : row,
                       (z > 0 ? prev : curr) + offset, (z + 1 < nz ? next : curr) + offset,
                       nx, [=](size_t x, double lap) { return cold[i + x] + dt * D * lap; });
        }
        std::swap(prev, curr);
        std::swap(curr, next);
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
    
    const size_t gridSize = nx * ny * nz;
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    const size_t yTiles = (ny + tileRows - 1) / tileRows;
    const size_t zTiles = (nz + tilePlanes - 1) / tilePlanes;
    const size_t tiles = nx == 0 ? 0 : yTiles * zTiles;
    const int threads = std::max<size_t>(1, std::min<size_t>(
        omp_get_max_threads(), std::min(tiles, 1 + gridSize / 8192)));

    printf("Initializing concentration field...\n");
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < gridSize; ++i) {
        const double pseudo = (((i + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[i] = -1.0 + 2.0 * pseudo;
    }

    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> scratch(3 * std::min(ny, tileRows + 2) * nx);
        double* source = cold.data();
        double* dest = cnew.data();
        for (int t = 0; t < iterations; ++t) {
            #pragma omp for schedule(static)
            for (size_t tile = 0; tile < tiles; ++tile) {
                advanceTile(source, dest, scratch.data(), nx, ny, nz,
                            (tile % yTiles) * tileRows, (tile / yTiles) * tilePlanes);
            }
            std::swap(source, dest);
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
