#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

using Clock = std::chrono::steady_clock;

Clock::duration runStencil(Real* grid1, Real* grid2, const size_t nx, const size_t ny,
                           const size_t nz, const int iterations, const int threads) {
    const size_t plane = nx * ny;
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t first = hasInterior ? 1 : 0;
    const size_t yEnd = ny - first;
    const size_t zEnd = nz - first;
    const size_t rows = (yEnd - first) * (zEnd - first);
    // Split wide rows only when there are too few rows to keep the team busy.
    const size_t blocksPerRow = rows == 0 ? 1 : 1 + (threads - 1) / rows;
    constexpr size_t cellsPerCacheLine = 8;
    const size_t blockWidth = blocksPerRow == 1 ? std::max<size_t>(nx, 1)
        : std::max<size_t>(256, ((nx - 1) / (blocksPerRow * cellsPerCacheLine) + 1) *
                                    cellsPerCacheLine);
    const size_t blocks = nx == 0 ? 0 : 1 + (nx - 1) / blockWidth;
    Clock::time_point start;
    Clock::duration elapsed;

    #pragma omp parallel num_threads(threads)
    {
        // First-touch both buffers using the same row distribution as the stencil.
        // The boundary values are fixed and already present in both buffers.
        #pragma omp for collapse(3) schedule(static)
        for (size_t z = first; z < zEnd; ++z) {
            for (size_t y = first; y < yEnd; ++y) {
                for (size_t block = 0; block < blocks; ++block) {
                    const size_t xBegin = block * blockWidth;
                    const size_t xEnd = std::min(xBegin + blockWidth, nx);
                    // Each edge row also initializes its adjacent boundary rows.
                    const size_t initZBegin = (z == first) ? 0 : z;
                    const size_t initZEnd = (z + 1 == zEnd) ? nz : z + 1;
                    const size_t initYBegin = (y == first) ? 0 : y;
                    const size_t initYEnd = (y + 1 == yEnd) ? ny : y + 1;
                    for (size_t initZ = initZBegin; initZ < initZEnd; ++initZ) {
                        for (size_t initY = initYBegin; initY < initYEnd; ++initY) {
                            const size_t row = initZ * plane + initY * nx;
                            #pragma omp simd
                            for (size_t x = xBegin; x < xEnd; ++x) {
                                const Real value = static_cast<Real>((row + x) % 19);
                                grid1[row + x] = value;
                                grid2[row + x] = iterations < 0 ? 0.0 : value;
                            }
                        }
                    }
                }
            }
        }

        #pragma omp master
        {
            printf("Running stencil computation...\n");
            start = Clock::now();
        }
        #pragma omp barrier

        Real* input = grid1;
        Real* output = grid2;
        for (int iter = 0; hasInterior && iter < iterations; ++iter) {
            #pragma omp for collapse(3) schedule(static)
            for (size_t z = first; z < zEnd; ++z) {
                for (size_t y = first; y < yEnd; ++y) {
                    for (size_t block = 0; block < blocks; ++block) {
                        const size_t xBegin = block * blockWidth;
                        const size_t xEnd = std::min(xBegin + blockWidth, nx - 1);
                        const size_t row = z * plane + y * nx;
                        const Real* center = input + row;
                        const Real* front = center - nx;
                        const Real* back = center + nx;
                        const Real* bottom = center - plane;
                        const Real* top = center + plane;
                        Real* result = output + row;

                        #pragma omp simd
                        for (size_t x = std::max<size_t>(1, xBegin); x < xEnd; ++x) {
                            result[x] = (center[x] + center[x - 1] + center[x + 1] +
                                         front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                        }
                    }
                }
            }
            // The worksharing barrier completes all writes before the buffers swap roles.
            std::swap(input, output);
        }

        #pragma omp master
        elapsed = Clock::now() - start;
    }
    return elapsed;
}

bool validateResult(const Real* grid, const size_t gridSize, const int threads) {
    if (gridSize == 0) {
        printf("Validation failed: grid is empty\n");
        return false;
    }

    bool finite = true;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for simd num_threads(threads) schedule(static) \
        reduction(&&:finite) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < gridSize; ++i) {
        const Real value = grid[i];
        finite = finite && std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }

    if (!finite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    // Leave small grids enough work per thread to amortize the iteration barriers.
    constexpr size_t cellsPerThread = 16384;
    const int threads = static_cast<int>(std::min(
        static_cast<size_t>(omp_get_max_threads()),
        std::max<size_t>(1, gridSize / cellsPerThread)
    ));

    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);

    printf("Initializing grid...\n");
    const auto elapsed = runStencil(grid1.get(), grid2.get(), nx, ny, nz, iterations, threads);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %ld ms\n", milliseconds.count());

    const double cellUpdates = nx > 2 && ny > 2 && nz > 2
        ? static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * std::max(0, iterations)
        : 0.0;
    const double seconds = std::chrono::duration<double>(elapsed).count();
    const double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* finalGrid = (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        print_results(finalGrid, gridSize, "Grid");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, gridSize, threads);

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
