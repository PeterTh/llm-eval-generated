#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t linear_id = 0; linear_id < vol; ++linear_id) {
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[linear_id] = -1.0 + 2.0 * pseudo;
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
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    const size_t planeSize = nx * ny;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double updateScale = dt * D;

    // Keep one OpenMP team alive for the whole solve.  Each sweep writes a
    // distinct cell, and the implicit workshare barriers enforce the stencil
    // dependency between the chemical-potential and update sweeps.
    #pragma omp parallel default(none) shared(cold, cnew, mu, nx, ny, nz, planeSize, invDx2, invDy2, invDz2, updateScale, gamma, e_AA, e_BB, e_AB, iterations)
    {
        for (int t = 0; t < iterations; ++t) {
            const double* const c = cold.data();
            double* const m = mu.data();

            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t zBase = z * planeSize;
                    const size_t zmBase = (z == 0 ? z : z - 1) * planeSize;
                    const size_t zpBase = (z + 1 < nz ? z + 1 : z) * planeSize;
                    const size_t rowBase = zBase + y * nx;
                    const size_t ymBase = zBase + (y == 0 ? y : y - 1) * nx;
                    const size_t ypBase = zBase + (y + 1 < ny ? y + 1 : y) * nx;

                    #pragma omp simd
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t i = rowBase + x;
                        const size_t xm = x == 0 ? x : x - 1;
                        const size_t xp = x + 1 < nx ? x + 1 : x;
                        const double cv = c[i];
                        const double laplacian =
                            (c[rowBase + xp] + c[rowBase + xm] - 2.0 * cv) * invDx2 +
                            (c[ypBase + x] + c[ymBase + x] - 2.0 * cv) * invDy2 +
                            (c[zpBase + y * nx + x] + c[zmBase + y * nx + x] - 2.0 * cv) * invDz2;
                        m[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                             + 3.0 * cv + cv * cv * cv - gamma * laplacian;
                    }
                }
            }

            const double* const mRead = mu.data();
            const double* const cRead = cold.data();
            double* const cWrite = cnew.data();

            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t zBase = z * planeSize;
                    const size_t zmBase = (z == 0 ? z : z - 1) * planeSize;
                    const size_t zpBase = (z + 1 < nz ? z + 1 : z) * planeSize;
                    const size_t rowBase = zBase + y * nx;
                    const size_t ymBase = zBase + (y == 0 ? y : y - 1) * nx;
                    const size_t ypBase = zBase + (y + 1 < ny ? y + 1 : y) * nx;

                    #pragma omp simd
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t i = rowBase + x;
                        const size_t xm = x == 0 ? x : x - 1;
                        const size_t xp = x + 1 < nx ? x + 1 : x;
                        const double mv = mRead[i];
                        const double laplacian =
                            (mRead[rowBase + xp] + mRead[rowBase + xm] - 2.0 * mv) * invDx2 +
                            (mRead[ypBase + x] + mRead[ymBase + x] - 2.0 * mv) * invDy2 +
                            (mRead[zpBase + y * nx + x] + mRead[zmBase + y * nx + x] - 2.0 * mv) * invDz2;
                        cWrite[i] = cRead[i] + updateScale * laplacian;
                    }
                }
            }

            #pragma omp single
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
