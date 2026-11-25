#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Each worker handles a contiguous range, including partial rows on thin grids.
template <bool ComputePotential>
void stencilSweep(const double* field, const double* concentration, double* output,
                  size_t nx, size_t ny, size_t nz, size_t begin, size_t end) {
    if (begin == end) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t firstRow = begin / nx;
    size_t y = firstRow % ny;
    size_t z = firstRow / ny;
    size_t x = begin % nx;
    for (size_t row = firstRow * nx; row < end; row += nx) {
        const size_t stop = std::min(nx, end - row);

        const double* center = field + row;
        const double* south = y == 0 ? center : center - nx;
        const double* north = y + 1 == ny ? center : center + nx;
        const double* below = z == 0 ? center : center - plane;
        const double* above = z + 1 == nz ? center : center + plane;

        const auto updateCell = [&](size_t j, size_t left, size_t right) {
            // The grid spacing is 1.0 in each direction.
            const double cv = center[j];
            const double cxx = center[right] + center[left] - 2.0 * cv;
            const double cyy = north[j] + south[j] - 2.0 * cv;
            const double czz = above[j] + below[j] - 2.0 * cv;
            const double laplacian = cxx + cyy + czz;
            if constexpr (ComputePotential) {
                constexpr double e_AA = -(2.0 / 9.0);
                constexpr double e_BB = -(2.0 / 9.0);
                constexpr double e_AB = 2.0 / 9.0;
                constexpr double gamma = 0.5;
                output[row + j] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            } else {
                constexpr double dt = 0.01;
                constexpr double D = 1.0;
                output[row + j] = concentration[row + j] + dt * D * laplacian;
            }
        };

        // Peel the clamped endpoints so the contiguous interior can be vectorized.
        if (x == 0) {
            updateCell(0, 0, nx == 1 ? 0 : 1);
        }
        const size_t interiorEnd = std::min(stop, nx - 1);
        #pragma omp simd
        for (size_t j = std::max(x, size_t{1}); j < interiorEnd; ++j) {
            updateCell(j, j - 1, j + 1);
        }
        if (nx > 1 && stop == nx) {
            updateCell(nx - 1, nx - 2, nx - 1);
        }
        x = 0;
        if (++y == ny) {
            y = 0;
            ++z;
        }
    }
}

bool validateResult(std::span<const double> c, int threads) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -minVal;
    bool allFinite = true;
    #pragma omp parallel for simd num_threads(threads) schedule(static) reduction(&&:allFinite) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        allFinite &= std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (!allFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
    auto first = std::make_unique_for_overwrite<double[]>(gridSize);
    auto second = std::make_unique_for_overwrite<double[]>(gridSize);
    auto potential = std::make_unique_for_overwrite<double[]>(gridSize);

    // Avoid dispatching more workers than this grid can keep busy.
    constexpr size_t cellsPerThread = 16384;
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max(size_t{1}, gridSize / cellsPerThread)));

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
    printf("Initializing concentration field...\n");
    #pragma omp parallel num_threads(threads)
    {
        const size_t worker = omp_get_thread_num();
        const size_t teamSize = omp_get_num_threads();
        // Split on cache-line-sized blocks to limit false sharing at chunk boundaries.
        constexpr size_t blockSize = 8;
        const size_t blocks = gridSize / blockSize + (gridSize % blockSize != 0);
        const size_t blocksPerWorker = blocks / teamSize;
        const size_t extra = blocks % teamSize;
        const size_t firstBlock = worker * blocksPerWorker + std::min(worker, extra);
        const size_t lastBlock = firstBlock + blocksPerWorker + (worker < extra);
        const size_t begin = std::min(gridSize, firstBlock * blockSize);
        const size_t end = std::min(gridSize, lastBlock * blockSize);

        double* current = first.get();
        double* next = second.get();
        double* mu = potential.get();

        // First-touch all fields on the workers that will write them.
        for (size_t i = begin; i < end; ++i) {
            const double pseudo = ((i + 1) * 1299709 % gridSize) / static_cast<double>(gridSize);
            current[i] = -1.0 + 2.0 * pseudo;
            next[i] = 0.0;
            mu[i] = 0.0;
        }

        #pragma omp barrier
        #pragma omp single
        {
            printf("Running Cahn-Hilliard simulation...\n");
            start = Clock::now();
        }

        for (int t = 0; t < iterations; ++t) {
            stencilSweep<true>(current, current, mu, nx, ny, nz, begin, end);
            #pragma omp barrier
            stencilSweep<false>(mu, current, next, nx, ny, nz, begin, end);
            #pragma omp barrier
            // Every worker swaps its own pointers after the update is complete.
            std::swap(current, next);
        }
    }

    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    if (iterations > 0 && iterations % 2 != 0) {
        std::swap(first, second);
    }
    second.reset();
    potential.reset();
    const std::span<const double> concentration(first.get(), gridSize);

    // Keep the original order for external validation statistics and hashing.
    if (printResults) {
        print_results(std::vector<double>(concentration.begin(), concentration.end()), "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(concentration, threads);
        
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
