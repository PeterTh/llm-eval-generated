#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

using Clock = std::chrono::high_resolution_clock;

// Keep contiguous rows together, but split very wide rows to expose parallel work.
constexpr size_t xBlockSize = 4096;

Clock::duration runStencil(Real* grid1, Real* grid2, const size_t nx,
                           const size_t ny, const size_t nz, const int iterations,
                           const int threads) {
    const size_t planeSize = nx * ny;
    const size_t xBlocks = nx / xBlockSize + (nx % xBlockSize != 0);
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    Clock::time_point start;

    #pragma omp parallel num_threads(threads)
    {
        // First-touch both grids using the same partition as the stencil sweeps.
        // Negative iteration counts can select the original zero-filled second buffer.
        #pragma omp for collapse(3) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t block = 0; block < xBlocks; ++block) {
                    const size_t row = z * planeSize + y * nx;
                    const size_t begin = block * xBlockSize;
                    const size_t end = std::min(begin + xBlockSize, nx);

                    #pragma omp simd
                    for (size_t x = begin; x < end; ++x) {
                        const size_t index = row + x;
                        const Real value = static_cast<Real>(index % 19);
                        grid1[index] = value;
                        grid2[index] = iterations < 0 ? 0.0 : value;
                    }
                }
            }
        }

        #pragma omp single
        {
            printf("Running stencil computation...\n");
            start = Clock::now();
        }

        Real* input = grid1;
        Real* output = grid2;
        for (int iter = 0; hasInterior && iter < iterations; ++iter) {
            #pragma omp for collapse(3) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t block = 0; block < xBlocks; ++block) {
                        if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                            continue;
                        }

                        const size_t row = z * planeSize + y * nx;
                        const Real* center = input + row;
                        const Real* front = center - nx;
                        const Real* back = center + nx;
                        const Real* bottom = center - planeSize;
                        const Real* top = center + planeSize;
                        Real* destination = output + row;
                        const size_t begin = std::max(block * xBlockSize, size_t{1});
                        const size_t end = std::min((block + 1) * xBlockSize, nx - 1);

                        // Both buffers already hold the fixed boundary values.
                        #pragma omp simd
                        for (size_t x = begin; x < end; ++x) {
                            destination[x] = (center[x] + center[x - 1] + center[x + 1] +
                                              front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                        }
                    }
                }
            }

            // The worksharing barrier completes the sweep before the buffers switch roles.
            std::swap(input, output);
        }
    }

    return Clock::now() - start;
}

bool validateResult(const Real* grid, const size_t count, const int threads) {
    if (count == 0) {
        printf("Validation failed: empty grid\n");
        return false;
    }

    bool finite = true;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(&&: finite) reduction(min: minVal) reduction(max: maxVal)
    for (size_t i = 0; i < count; ++i) {
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
    // Amortize worksharing barriers on small grids.
    constexpr size_t cellsPerThread = 65536;
    const size_t usefulThreads = std::max(size_t{1}, gridSize / cellsPerThread);
    const int threads = static_cast<int>(std::min(usefulThreads, static_cast<size_t>(omp_get_max_threads())));

    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);

    printf("Initializing grid...\n");
    const auto elapsed = runStencil(grid1.get(), grid2.get(), nx, ny, nz, iterations, threads);
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %ld ms\n", duration.count());

    const double cellUpdates = nx > 2 && ny > 2 && nz > 2
        ? static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations
        : 0.0;
    const double seconds = std::chrono::duration<double>(elapsed).count();
    const double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* finalGrid = iterations % 2 == 0 ? grid1.get() : grid2.get();
    if (printResults) {
        print_results(finalGrid, gridSize, "Grid");
    }

    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, gridSize, threads);

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
