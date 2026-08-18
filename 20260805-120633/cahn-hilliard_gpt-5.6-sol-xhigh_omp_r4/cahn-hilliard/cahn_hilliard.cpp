#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t cacheLineSize = 64;

using AlignedArray = std::unique_ptr<double, decltype(&std::free)>;

AlignedArray allocateAligned(const size_t count) {
    const size_t bytes = std::max(cacheLineSize, count * sizeof(double));
    const size_t paddedBytes = (bytes + cacheLineSize - 1) & ~(cacheLineSize - 1);
    void* const allocation = std::aligned_alloc(cacheLineSize, paddedBytes);
    if (allocation == nullptr) {
        throw std::bad_alloc();
    }
    return AlignedArray(static_cast<double*>(allocation), &std::free);
}

// The helper is inlined into the vectorized x loop. Neighbor indices implement
// the same clamped (zero normal derivative) boundaries as the original code.
__attribute__((always_inline)) inline double laplacianAt(
    const double* __restrict__ field, const size_t center,
    const size_t xm, const size_t xp, const size_t ym, const size_t yp,
    const size_t zm, const size_t zp, const double invDx2,
    const double invDy2, const double invDz2) noexcept {
    const double value = field[center];
    const double cxx = (field[xp] + field[xm] - 2.0 * value) * invDx2;
    const double cyy = (field[yp] + field[ym] - 2.0 * value) * invDy2;
    const double czz = (field[zp] + field[zm] - 2.0 * value) * invDz2;
    return cxx + cyy + czz;
}

__attribute__((always_inline)) inline double chemicalPotentialAt(
    const double cv, const double laplacian, const double gamma,
    const double e_AA, const double e_BB, const double e_AB) noexcept {
    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
         + 3.0 * cv + cv * cv * cv - gamma * laplacian;
}

// Orphaned worksharing kernels are called by every thread in the persistent
// parallel region in main. Their implicit end barriers enforce the two stencil
// dependencies in each time step without repeatedly creating OpenMP teams.
void computeChemicalPotential(const double* __restrict__ c,
                              double* __restrict__ mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double invDx2, const double invDy2,
                              const double invDz2, const double gamma,
                              const double e_AA, const double e_BB,
                              const double e_AB) {
    const size_t plane = nx * ny;

#pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t ymRow = (y == 0) ? row : row - nx;
            const size_t ypRow = (y + 1 == ny) ? row : row + nx;
            const size_t zmRow = (z == 0) ? row : row - plane;
            const size_t zpRow = (z + 1 == nz) ? row : row + plane;

            if (nx == 1) {
                const double cv = c[row];
                const double laplacian = laplacianAt(
                    c, row, row, row, ymRow, ypRow, zmRow, zpRow,
                    invDx2, invDy2, invDz2);
                mu[row] = chemicalPotentialAt(cv, laplacian, gamma, e_AA, e_BB, e_AB);
                continue;
            }

            double cv = c[row];
            double laplacian = laplacianAt(
                c, row, row, row + 1, ymRow, ypRow, zmRow, zpRow,
                invDx2, invDy2, invDz2);
            mu[row] = chemicalPotentialAt(cv, laplacian, gamma, e_AA, e_BB, e_AB);

#pragma omp simd aligned(c, mu : cacheLineSize)
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t i = row + x;
                cv = c[i];
                laplacian = laplacianAt(
                    c, i, i - 1, i + 1, ymRow + x, ypRow + x,
                    zmRow + x, zpRow + x, invDx2, invDy2, invDz2);
                mu[i] = chemicalPotentialAt(cv, laplacian, gamma, e_AA, e_BB, e_AB);
            }

            const size_t x = nx - 1;
            const size_t i = row + x;
            cv = c[i];
            laplacian = laplacianAt(
                c, i, i - 1, i, ymRow + x, ypRow + x,
                zmRow + x, zpRow + x, invDx2, invDy2, invDz2);
            mu[i] = chemicalPotentialAt(cv, laplacian, gamma, e_AA, e_BB, e_AB);
        }
    }
}

void cahnHilliardUpdate(double* __restrict__ cnew,
                        const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double dtD, const double invDx2,
                        const double invDy2, const double invDz2) {
    const size_t plane = nx * ny;

#pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t ymRow = (y == 0) ? row : row - nx;
            const size_t ypRow = (y + 1 == ny) ? row : row + nx;
            const size_t zmRow = (z == 0) ? row : row - plane;
            const size_t zpRow = (z + 1 == nz) ? row : row + plane;

            if (nx == 1) {
                cnew[row] = cold[row] + dtD * laplacianAt(
                    mu, row, row, row, ymRow, ypRow, zmRow, zpRow,
                    invDx2, invDy2, invDz2);
                continue;
            }

            cnew[row] = cold[row] + dtD * laplacianAt(
                mu, row, row, row + 1, ymRow, ypRow, zmRow, zpRow,
                invDx2, invDy2, invDz2);

#pragma omp simd aligned(cnew, cold, mu : cacheLineSize)
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t i = row + x;
                cnew[i] = cold[i] + dtD * laplacianAt(
                    mu, i, i - 1, i + 1, ymRow + x, ypRow + x,
                    zmRow + x, zpRow + x, invDx2, invDy2, invDz2);
            }

            const size_t x = nx - 1;
            const size_t i = row + x;
            cnew[i] = cold[i] + dtD * laplacianAt(
                mu, i, i - 1, i, ymRow + x, ypRow + x,
                zmRow + x, zpRow + x, invDx2, invDy2, invDz2);
        }
    }
}

void initializeFields(double* __restrict__ c, double* __restrict__ cnew,
                      double* __restrict__ mu, const size_t volume) {
    // Besides initialization, this parallel first-touch places all three fields
    // on the NUMA nodes of the threads that will operate on them.
#pragma omp parallel for schedule(static)
    for (size_t linearId = 0; linearId < volume; ++linearId) {
        const double pseudo = ((((linearId + 1) * 1299709) % volume)
                               / static_cast<double>(volume));
        c[linearId] = -1.0 + 2.0 * pseudo;
        cnew[linearId] = 0.0;
        mu[linearId] = 0.0;
    }
}

bool validateResult(const double* c, const size_t volume) {
    // Check for NaN or Inf
    for (size_t i = 0; i < volume; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (size_t i = 0; i < volume; ++i) {
        const double val = c[i];
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

} // namespace

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

    // Honor OMP_NUM_THREADS while preventing the runtime from silently
    // shrinking the team between the benchmark's worksharing regions.
    omp_set_dynamic(0);
    
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
    
    const size_t gridSize = nx * ny * nz;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;
    
    // Allocate cache-line-aligned, uninitialized arrays. Parallel first-touch
    // below gives the fields useful NUMA placement on multisocket machines.
    AlignedArray cold = allocateAligned(gridSize);
    AlignedArray cnew = allocateAligned(gridSize);
    AlignedArray mu = allocateAligned(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeFields(cold.get(), cnew.get(), mu.get(), gridSize);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    double* const coldData = cold.get();
    double* const cnewData = cnew.get();
    double* const muData = mu.get();

    // Each thread keeps the same private pair of buffer pointers and swaps them
    // after the update barrier. This removes an otherwise serial pointer swap
    // and its extra synchronization from every time step.
#pragma omp parallel default(none) \
    firstprivate(coldData, cnewData, muData, iterations, nx, ny, nz, \
                 invDx2, invDy2, invDz2, gamma, e_AA, e_BB, e_AB, dtD)
    {
        double* current = coldData;
        double* next = cnewData;

        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(current, muData, nx, ny, nz,
                                     invDx2, invDy2, invDz2,
                                     gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdate(next, current, muData, nx, ny, nz,
                               dtD, invDx2, invDy2, invDz2);
            std::swap(current, next);
        }
    }

    const double* const result = (iterations > 0 && iterations % 2 != 0)
                               ? cnewData : coldData;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> resultVector(result, result + gridSize);
        print_results(resultVector, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, gridSize);
        
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
