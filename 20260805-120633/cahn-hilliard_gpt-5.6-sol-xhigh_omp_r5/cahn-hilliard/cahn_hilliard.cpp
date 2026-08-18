#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Compute chemical potential. This is an orphaned worksharing loop: every thread
// in the persistent parallel region calls it with the same arguments.
void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double invDx2, const double invDy2, const double invDz2,
                              const double gamma, const double e_AA, const double e_BB,
                              const double e_AB) {
    const size_t plane = nx * ny;

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t yMinusRow = z * plane + (y > 0 ? y - 1 : y) * nx;
            const size_t yPlusRow = z * plane + (y + 1 < ny ? y + 1 : y) * nx;
            const size_t zMinusRow = (z > 0 ? z - 1 : z) * plane + y * nx;
            const size_t zPlusRow = (z + 1 < nz ? z + 1 : z) * plane + y * nx;

            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = row + x;
                const size_t xMinus = x > 0 ? index - 1 : index;
                const size_t xPlus = x + 1 < nx ? index + 1 : index;
                const double cv = c[index];
                const double cxx = (c[xPlus] + c[xMinus] - 2.0 * cv) * invDx2;
                const double cyy = (c[yPlusRow + x] + c[yMinusRow + x] - 2.0 * cv) * invDy2;
                const double czz = (c[zPlusRow + x] + c[zMinusRow + x] - 2.0 * cv) * invDz2;

                mu[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                          + 3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update step. The implicit barrier at the end prevents the next
// timestep from reading a partially updated concentration field.
void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double dtD, const double invDx2, const double invDy2,
                        const double invDz2) {
    const size_t plane = nx * ny;

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t yMinusRow = z * plane + (y > 0 ? y - 1 : y) * nx;
            const size_t yPlusRow = z * plane + (y + 1 < ny ? y + 1 : y) * nx;
            const size_t zMinusRow = (z > 0 ? z - 1 : z) * plane + y * nx;
            const size_t zPlusRow = (z + 1 < nz ? z + 1 : z) * plane + y * nx;

            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = row + x;
                const size_t xMinus = x > 0 ? index - 1 : index;
                const size_t xPlus = x + 1 < nx ? index + 1 : index;
                const double muv = mu[index];
                const double muxx = (mu[xPlus] + mu[xMinus] - 2.0 * muv) * invDx2;
                const double muyy = (mu[yPlusRow + x] + mu[yMinusRow + x] - 2.0 * muv) * invDy2;
                const double muzz = (mu[zPlusRow + x] + mu[zMinusRow + x] - 2.0 * muv) * invDz2;

                cnew[index] = cold[index] + dtD * (muxx + muyy + muzz);
            }
        }
    }
}

void runSimulation(double*& cold, double*& cnew, double* const mu,
                   const size_t gridSize, const size_t nx, const size_t ny,
                   const size_t nz, const int iterations, const double D,
                   const double dt, const double dx, const double dy, const double dz,
                   const double gamma, const double e_AA, const double e_BB,
                   const double e_AB) {
    if (iterations <= 0 || gridSize == 0) {
        return;
    }

    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;
    double* const coldData = cold;
    double* const cnewData = cnew;

    // Keep one team alive for the full simulation. Each thread owns its pointer
    // pair, so changing buffers needs no extra single-thread region or barrier.
    #pragma omp parallel default(none) shared(coldData, cnewData, mu, nx, ny, nz, iterations, invDx2, invDy2, invDz2, gamma, e_AA, e_BB, e_AB, dtD)
    {
        double* current = coldData;
        double* next = cnewData;

        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(current, mu, nx, ny, nz, invDx2, invDy2,
                                     invDz2, gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdate(next, current, mu, nx, ny, nz, dtD, invDx2,
                               invDy2, invDz2);
            std::swap(current, next);
        }
    }

    // Expose the final buffer through cold, as in the original per-step swap.
    if ((iterations % 2) != 0) {
        std::swap(cold, cnew);
    }
}

// Initialize all three fields in parallel so their pages are placed close to
// the threads that process them on NUMA systems.
void initializeFields(double* const c, double* const cnew, double* const mu,
                      const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for simd default(none) shared(c, cnew, mu, vol) schedule(static)
    for (size_t linearId = 0; linearId < vol; ++linearId) {
        // Generate a deterministic pseudo-random value in [-1, 1].
        const double pseudo = (((linearId + 1) * 1299709) % vol) /
                              static_cast<double>(vol);
        c[linearId] = -1.0 + 2.0 * pseudo;
        cnew[linearId] = 0.0;
        mu[linearId] = 0.0;
    }
}

bool validateResult(const double* const c, const size_t gridSize,
                    [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    int nonFinite = 0;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();

    #pragma omp parallel for default(none) shared(c, gridSize) schedule(static) reduction(|:nonFinite) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < gridSize; ++i) {
        const double value = c[i];
        if (!std::isfinite(value)) {
            nonFinite = 1;
        } else {
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
        }
    }

    if (nonFinite != 0) {
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
    
    // Avoid serially touching the arrays so parallel initialization can provide
    // first-touch NUMA placement.
    auto coldStorage = std::make_unique_for_overwrite<double[]>(gridSize);
    auto cnewStorage = std::make_unique_for_overwrite<double[]>(gridSize);
    auto mu = std::make_unique_for_overwrite<double[]>(gridSize);
    double* cold = coldStorage.get();
    double* cnew = cnewStorage.get();
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeFields(cold, cnew, mu.get(), nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(cold, cnew, mu.get(), gridSize, nx, ny, nz, iterations, D, dt,
                  dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> output(gridSize);
        std::copy_n(cold, gridSize, output.data());
        print_results(output, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, gridSize, nx, ny, nz);
        
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
