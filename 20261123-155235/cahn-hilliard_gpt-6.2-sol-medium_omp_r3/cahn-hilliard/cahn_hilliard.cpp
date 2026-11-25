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

// Compute the Laplacian for the benchmark's unit grid spacing.
inline double computeLaplacian(const double* center, const double* north, const double* south,
                               const double* above, const double* below,
                               size_t x, size_t left, size_t right) {
    const double twiceCenter = 2.0 * center[x];
    const double cxx = center[right] + center[left] - twiceCenter;
    const double cyy = south[x] + north[x] - twiceCenter;
    const double czz = above[x] + below[x] - twiceCenter;
    return cxx + cyy + czz;
}

template <typename Update>
void sweepStencil(const double* field, size_t begin, size_t end,
                  size_t nx, size_t ny, size_t nz, const Update& update) {
    if (begin == end) return;

    const size_t planeSize = nx * ny;
    const size_t firstRow = begin / nx;
    size_t xBegin = begin % nx;
    size_t y = firstRow % ny;
    size_t z = firstRow / ny;

    for (size_t rowStart = begin - xBegin; rowStart < end; rowStart += nx) {
        const double* center = field + rowStart;
        const double* north = y > 0 ? center - nx : center;
        const double* south = y + 1 < ny ? center + nx : center;
        const double* below = z > 0 ? center - planeSize : center;
        const double* above = z + 1 < nz ? center + planeSize : center;
        const size_t xEnd = std::min(nx, end - rowStart);

        const auto updateCell = [&](size_t x, size_t left, size_t right) {
            update(rowStart + x, center[x],
                   computeLaplacian(center, north, south, above, below, x, left, right));
        };

        // Handle clamped X neighbors at the row edges.
        if (xBegin == 0) updateCell(0, 0, nx > 1 ? 1 : 0);

        const size_t interiorBegin = std::max(xBegin, size_t{1});
        const size_t interiorEnd = std::min(xEnd, nx - 1);
        #pragma omp simd
        for (size_t x = interiorBegin; x < interiorEnd; ++x) {
            updateCell(x, x - 1, x + 1);
        }

        if (xEnd == nx && nx > 1) updateCell(nx - 1, nx - 2, nx - 1);

        xBegin = 0;
        if (++y == ny) {
            y = 0;
            ++z;
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
    auto concentration = std::make_unique_for_overwrite<double[]>(gridSize);
    auto next = std::make_unique_for_overwrite<double[]>(gridSize);
    auto potential = std::make_unique_for_overwrite<double[]>(gridSize);

    // Small grids need enough work per thread to amortize the stencil barriers.
    constexpr size_t cellsPerThread = 8192;
    size_t threadLimit = omp_get_max_threads();
    // Large unbound teams can spend more time synchronizing than computing.
    if (!std::getenv("OMP_NUM_THREADS") && omp_get_proc_bind() == omp_proc_bind_false) {
        threadLimit = std::min(threadLimit, size_t{32});
    }
    const int threads = static_cast<int>(std::clamp(gridSize / cellsPerThread, size_t{1}, threadLimit));

    printf("Initializing concentration field...\n");
    using Clock = std::chrono::steady_clock;
    Clock::time_point start;

    #pragma omp parallel num_threads(threads)
    {
        const size_t thread = omp_get_thread_num();
        const size_t teamSize = omp_get_num_threads();
        const size_t cells = gridSize / teamSize;
        const size_t remainder = gridSize % teamSize;
        const size_t begin = thread * cells + std::min(thread, remainder);
        const size_t end = begin + cells + (thread < remainder);

        double* cold = concentration.get();
        double* cnew = next.get();
        double* mu = potential.get();

        // Initialize the same contiguous range that this thread will update.
        for (size_t i = begin; i < end; ++i) {
            const double pseudo = ((i + 1) * 1299709 % gridSize) / static_cast<double>(gridSize);
            cold[i] = -1.0 + 2.0 * pseudo;
        }
        std::fill(cnew + begin, cnew + end, 0.0);
        std::fill(mu + begin, mu + end, 0.0);

        #pragma omp barrier
        #pragma omp master
        {
            printf("Running Cahn-Hilliard simulation...\n");
            start = Clock::now();
        }
        #pragma omp barrier

        for (int t = 0; t < iterations; ++t) {
            sweepStencil(cold, begin, end, nx, ny, nz, [&](size_t i, double cv, double laplacian) {
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            });

            // The update reads potential values from neighboring threads' ranges.
            #pragma omp barrier
            sweepStencil(mu, begin, end, nx, ny, nz, [&](size_t i, double, double laplacian) {
                cnew[i] = cold[i] + dt * D * laplacian;
            });

            // Finish writing the new concentration before the next stencil reads it.
            #pragma omp barrier
            std::swap(cold, cnew);
        }
    }

    const auto elapsed = Clock::now() - start;
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %ld ms\n", duration.count());

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / std::chrono::duration<double>(elapsed).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (iterations > 0 && iterations % 2 != 0) std::swap(concentration, next);

    // The reporting routine uses a vector and preserves the original hash order.
    if (printResults) {
        next.reset();
        potential.reset();
        print_results(std::vector<double>(concentration.get(), concentration.get() + gridSize), "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult({concentration.get(), gridSize});
        
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
