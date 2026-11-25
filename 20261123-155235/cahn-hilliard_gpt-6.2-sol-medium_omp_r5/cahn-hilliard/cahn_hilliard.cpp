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

// The grid spacing is one in all three directions.
inline double laplacian(const double* row, const double* south, const double* north,
                        const double* below, const double* above, size_t x,
                        size_t left, size_t right) {
    const double twiceCenter = 2.0 * row[x];
    const double cxx = row[right] + row[left] - twiceCenter;
    const double cyy = north[x] + south[x] - twiceCenter;
    const double czz = above[x] + below[x] - twiceCenter;
    return cxx + cyy + czz;
}

// Called by every thread in the simulation team. Each sweep ends with a barrier.
template <typename Update>
void sweepStencil(const double* field, size_t nx, size_t ny, size_t nz,
                  size_t blockWidth, Update update) {
    const size_t plane = nx * ny;
    const size_t blocksPerRow = nx == 0 ? 0 : (nx - 1) / blockWidth + 1;

    #pragma omp for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t block = 0; block < blocksPerRow; ++block) {
                const size_t offset = z * plane + y * nx;
                const size_t begin = block * blockWidth;
                const size_t end = std::min(begin + blockWidth, nx);
                const double* row = field + offset;
                const double* south = y > 0 ? row - nx : row;
                const double* north = y + 1 < ny ? row + nx : row;
                const double* below = z > 0 ? row - plane : row;
                const double* above = z + 1 < nz ? row + plane : row;

                // Clamp the two X boundaries outside the contiguous SIMD loop.
                if (begin == 0) {
                    update(offset, row[0], laplacian(row, south, north, below, above, 0, 0, nx > 1));
                }

                const size_t interiorEnd = std::min(end, nx - 1);
                #pragma omp simd
                for (size_t x = std::max(begin, size_t{1}); x < interiorEnd; ++x) {
                    update(offset + x, row[x], laplacian(row, south, north, below, above, x, x - 1, x + 1));
                }

                if (end == nx && nx > 1) {
                    const size_t x = nx - 1;
                    update(offset + x, row[x], laplacian(row, south, north, below, above, x, x - 1, x));
                }
            }
        }
    }
}

bool validateResult(const double* c, size_t size, int threads) {
    if (size == 0) {
        printf("Validation failed: empty concentration field\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    int nonfinite = 0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:minVal) reduction(max:maxVal) reduction(|:nonfinite)
    for (size_t i = 0; i < size; ++i) {
        const double value = c[i];
        if (std::isfinite(value)) {
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
        } else {
            nonfinite = 1;
        }
    }

    if (nonfinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
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
    constexpr double dt = 0.01;
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double D = 1.0;

    const size_t gridSize = nx * ny * nz;
    // Limit barrier overhead on small grids and honor the OpenMP thread limit.
    constexpr size_t cellsPerThread = 16384;
    const size_t usefulThreads = gridSize / cellsPerThread + (gridSize % cellsPerThread != 0);
    const int threads = static_cast<int>(std::min(
        static_cast<size_t>(omp_get_max_threads()), std::max(size_t{1}, usefulThreads)));
    // Split long rows when there are too few rows to occupy the team.
    const size_t blockWidth = nx != 0 && ny * nz < static_cast<size_t>(threads)
        ? std::min(nx, size_t{256}) : nx;

    // Let the worker threads first-touch the arrays they will use.
    auto cold = std::make_unique_for_overwrite<double[]>(gridSize);
    auto cnew = std::make_unique_for_overwrite<double[]>(gridSize);
    auto mu = std::make_unique_for_overwrite<double[]>(gridSize);

    printf("Initializing concentration field...\n");
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;

    #pragma omp parallel num_threads(threads)
    {
        double* current = cold.get();
        double* next = cnew.get();
        double* potential = mu.get();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < gridSize; ++i) {
            const double pseudo = ((i + 1) * 1299709) % gridSize / static_cast<double>(gridSize);
            current[i] = -1.0 + 2.0 * pseudo;
        }

        #pragma omp master
        {
            printf("Running Cahn-Hilliard simulation...\n");
            start = Clock::now();
        }
        #pragma omp barrier

        for (int t = 0; t < iterations; ++t) {
            sweepStencil(current, nx, ny, nz, blockWidth, [=](size_t i, double cv, double lap) {
                potential[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                             + 3.0 * cv + cv * cv * cv - gamma * lap;
            });

            sweepStencil(potential, nx, ny, nz, blockWidth, [=](size_t i, double, double lap) {
                next[i] = current[i] + dt * D * lap;
            });

            // Each thread swaps its own pointers after the update barrier.
            std::swap(current, next);
        }
    }

    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    if (iterations > 0 && iterations % 2 != 0) {
        cold.swap(cnew);
    }
    cnew.reset();
    mu.reset();

    // Print results for external validation
    if (printResults) {
        print_results(std::vector<double>(cold.get(), cold.get() + gridSize), "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold.get(), gridSize, threads);
        
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
