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

// Apply the clamped stencil on a contiguous block of cells. The interior of
// each row has unit-stride neighbors and can be vectorized independently.
template <typename Store>
void applyLaplacian(const double* field, size_t begin, size_t end,
                    size_t nx, size_t ny, size_t nz, Store store) {
    const size_t plane = nx * ny;
    size_t z = begin / plane;
    size_t y = (begin % plane) / nx;
    size_t xBegin = begin % nx;
    for (size_t row = begin - xBegin; row < end; row += nx) {
        const double* center = field + row;
        const double* yn = y > 0 ? center - nx : center;
        const double* yp = y + 1 < ny ? center + nx : center;
        const double* zn = z > 0 ? center - plane : center;
        const double* zp = z + 1 < nz ? center + plane : center;
        const size_t xEnd = std::min(nx, end - row);

        // The grid spacing is one in all three directions.
        const auto update = [&](size_t x, size_t xn, size_t xp) {
            const double cv = center[x];
            const double cxx = center[xp] + center[xn] - 2.0 * cv;
            const double cyy = yp[x] + yn[x] - 2.0 * cv;
            const double czz = zp[x] + zn[x] - 2.0 * cv;
            store(row + x, cv, cxx + cyy + czz);
        };

        if (xBegin == 0) {
            update(0, 0, nx > 1 ? 1 : 0);
            xBegin = 1;
        }

        const size_t interiorEnd = std::min(xEnd, nx - 1);
        #pragma omp simd
        for (size_t x = xBegin; x < interiorEnd; ++x) {
            update(x, x - 1, x + 1);
        }

        if (xEnd == nx && nx > 1) {
            update(nx - 1, nx - 2, nx - 1);
        }

        xBegin = 0;
        if (++y == ny) {
            y = 0;
            ++z;
        }
    }
}

bool validateResult(const double* c, size_t size, int threads) {
    if (size == 0) {
        printf("Validation failed: concentration field is empty\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    bool nonfinite = false;
    #pragma omp parallel for simd schedule(static) num_threads(threads) reduction(min:minVal) reduction(max:maxVal) reduction(|:nonfinite)
    for (size_t i = 0; i < size; ++i) {
        const double val = c[i];
        nonfinite |= !std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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
    // Large unbound teams can spend more time waiting at barriers. Honor
    // explicit runtime settings and use a smaller team by default in that case.
    int maxThreads = omp_get_max_threads();
    if (omp_get_proc_bind() == omp_proc_bind_false && std::getenv("OMP_NUM_THREADS") == nullptr) {
        maxThreads = std::min(maxThreads, 32);
    }
    // Keep enough work per thread to amortize the two barriers per time step.
    constexpr size_t cellsPerThread = 16384;
    const int threads = static_cast<int>(std::min<size_t>(
        maxThreads, std::max<size_t>(1, gridSize / cellsPerThread)));

    // First touch each allocation in the thread that will update those cells.
    const auto first = std::make_unique_for_overwrite<double[]>(gridSize);
    const auto second = std::make_unique_for_overwrite<double[]>(gridSize);
    const auto potential = std::make_unique_for_overwrite<double[]>(gridSize);

    printf("Initializing concentration field...\n");
    std::chrono::high_resolution_clock::time_point start;

    #pragma omp parallel num_threads(threads)
    {
        const size_t teamSize = omp_get_num_threads();
        const size_t thread = omp_get_thread_num();
        const size_t blockSize = gridSize / teamSize;
        const size_t extra = gridSize % teamSize;
        const size_t begin = thread * blockSize + std::min(thread, extra);
        const size_t end = begin + blockSize + (thread < extra);

        double* cold = first.get();
        double* cnew = second.get();
        double* mu = potential.get();
        for (size_t i = begin; i < end; ++i) {
            const double pseudo = ((i + 1) * 1299709 % gridSize) / static_cast<double>(gridSize);
            cold[i] = -1.0 + 2.0 * pseudo;
        }
        std::fill(cnew + begin, cnew + end, 0.0);
        std::fill(mu + begin, mu + end, 0.0);

        #pragma omp barrier
        #pragma omp single
        {
            printf("Running Cahn-Hilliard simulation...\n");
            start = std::chrono::high_resolution_clock::now();
        }

        for (int t = 0; t < iterations && gridSize != 0; ++t) {
            applyLaplacian(cold, begin, end, nx, ny, nz,
                           [&](size_t i, double cv, double laplacian) {
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            });

            // Neighboring blocks must finish writing the potential before it is read.
            #pragma omp barrier
            applyLaplacian(mu, begin, end, nx, ny, nz,
                           [&](size_t i, double, double laplacian) {
                cnew[i] = cold[i] + dt * D * laplacian;
            });

            // Every thread swaps its own pointers after the new field is complete.
            #pragma omp barrier
            std::swap(cold, cnew);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    const double* result = iterations > 0 && iterations % 2 != 0 ? second.get() : first.get();

    // Print results for external validation
    if (printResults) {
        print_results(result, gridSize, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, gridSize, threads);
        
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
