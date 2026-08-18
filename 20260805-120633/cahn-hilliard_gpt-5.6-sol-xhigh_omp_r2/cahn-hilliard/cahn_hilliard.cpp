#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using AlignedArray = std::unique_ptr<double, decltype(&std::free)>;

AlignedArray allocateArray(const size_t count) {
    if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(double)) {
        throw std::bad_alloc();
    }

    void* allocation = nullptr;
    constexpr size_t alignment = 64;
    if (posix_memalign(&allocation, alignment, count * sizeof(double)) != 0) {
        throw std::bad_alloc();
    }
    return AlignedArray(static_cast<double*>(allocation), &std::free);
}

// Initialize all three fields in parallel.  Besides generating the initial
// concentration, this gives each array a NUMA-friendly parallel first touch.
void initializeFields(double* const concentration, double* const next,
                      double* const chemicalPotential, const size_t volume) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < volume; ++i) {
        const double pseudo = (((i + 1) * 1299709) % volume) /
                              static_cast<double>(volume);
        concentration[i] = -1.0 + 2.0 * pseudo;
        next[i] = 0.0;
        chemicalPotential[i] = 0.0;
    }
}

inline double laplacianPoint(const double* const row,
                             const double* const rowMinusY,
                             const double* const rowPlusY,
                             const double* const rowMinusZ,
                             const double* const rowPlusZ,
                             const size_t x, const size_t xMinus,
                             const size_t xPlus, const double invDx2,
                             const double invDy2, const double invDz2) noexcept {
    const double center = row[x];
    const double dxx = (row[xPlus] + row[xMinus] - 2.0 * center) * invDx2;
    const double dyy = (rowPlusY[x] + rowMinusY[x] - 2.0 * center) * invDy2;
    const double dzz = (rowPlusZ[x] + rowMinusZ[x] - 2.0 * center) * invDz2;
    return dxx + dyy + dzz;
}

// Run both stencils in one persistent OpenMP region.  The implicit barrier at
// the end of each worksharing loop is required: the update needs every value
// of mu, and the following time step needs every value of the updated field.
double* runSimulation(double* const cold, double* const cnew, double* const mu,
                      const size_t nx, const size_t ny, const size_t nz,
                      const int iterations, const double D, const double dt,
                      const double dx, const double dy, const double dz,
                      const double gamma, const double eAA,
                      const double eBB, const double eAB) {
    if (iterations <= 0) {
        return cold;
    }

    const size_t plane = nx * ny;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double updateScale = dt * D;

#pragma omp parallel
    {
        for (int t = 0; t < iterations; ++t) {
            const double* const current = (t & 1) == 0 ? cold : cnew;
            double* const next = (t & 1) == 0 ? cnew : cold;

#pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t base = z * plane + y * nx;
                    const size_t minusYBase = y == 0 ? base : base - nx;
                    const size_t plusYBase = y + 1 == ny ? base : base + nx;
                    const size_t minusZBase = z == 0 ? base : base - plane;
                    const size_t plusZBase = z + 1 == nz ? base : base + plane;

                    const double* const row = current + base;
                    const double* const rowMinusY = current + minusYBase;
                    const double* const rowPlusY = current + plusYBase;
                    const double* const rowMinusZ = current + minusZBase;
                    const double* const rowPlusZ = current + plusZBase;
                    double* const output = mu + base;

                    const auto chemicalPotentialAt = [&](const size_t x,
                                                         const size_t xMinus,
                                                         const size_t xPlus) {
                        const double value = row[x];
                        const double laplacian = laplacianPoint(
                            row, rowMinusY, rowPlusY, rowMinusZ, rowPlusZ,
                            x, xMinus, xPlus, invDx2, invDy2, invDz2);
                        output[x] =
                            4.5 * ((value + 1.0) * eAA +
                                   (value - 1.0) * eBB -
                                   2.0 * value * eAB) +
                            3.0 * value + value * value * value -
                            gamma * laplacian;
                    };

                    chemicalPotentialAt(0, 0, nx == 1 ? 0 : 1);

                    if (nx > 2) {
#pragma omp simd
                        for (size_t x = 1; x < nx - 1; ++x) {
                            const double value = row[x];
                            const double dxx =
                                (row[x + 1] + row[x - 1] - 2.0 * value) * invDx2;
                            const double dyy =
                                (rowPlusY[x] + rowMinusY[x] - 2.0 * value) * invDy2;
                            const double dzz =
                                (rowPlusZ[x] + rowMinusZ[x] - 2.0 * value) * invDz2;
                            const double laplacian = dxx + dyy + dzz;
                            output[x] =
                                4.5 * ((value + 1.0) * eAA +
                                       (value - 1.0) * eBB -
                                       2.0 * value * eAB) +
                                3.0 * value + value * value * value -
                                gamma * laplacian;
                        }
                    }

                    if (nx > 1) {
                        chemicalPotentialAt(nx - 1, nx - 2, nx - 1);
                    }
                }
            }

#pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t base = z * plane + y * nx;
                    const size_t minusYBase = y == 0 ? base : base - nx;
                    const size_t plusYBase = y + 1 == ny ? base : base + nx;
                    const size_t minusZBase = z == 0 ? base : base - plane;
                    const size_t plusZBase = z + 1 == nz ? base : base + plane;

                    const double* const row = mu + base;
                    const double* const rowMinusY = mu + minusYBase;
                    const double* const rowPlusY = mu + plusYBase;
                    const double* const rowMinusZ = mu + minusZBase;
                    const double* const rowPlusZ = mu + plusZBase;
                    const double* const old = current + base;
                    double* const output = next + base;

                    const auto updateAt = [&](const size_t x,
                                              const size_t xMinus,
                                              const size_t xPlus) {
                        output[x] = old[x] + updateScale * laplacianPoint(
                            row, rowMinusY, rowPlusY, rowMinusZ, rowPlusZ,
                            x, xMinus, xPlus, invDx2, invDy2, invDz2);
                    };

                    updateAt(0, 0, nx == 1 ? 0 : 1);

                    if (nx > 2) {
#pragma omp simd
                        for (size_t x = 1; x < nx - 1; ++x) {
                            const double value = row[x];
                            const double dxx =
                                (row[x + 1] + row[x - 1] - 2.0 * value) * invDx2;
                            const double dyy =
                                (rowPlusY[x] + rowMinusY[x] - 2.0 * value) * invDy2;
                            const double dzz =
                                (rowPlusZ[x] + rowMinusZ[x] - 2.0 * value) * invDz2;
                            output[x] = old[x] + updateScale * (dxx + dyy + dzz);
                        }
                    }

                    if (nx > 1) {
                        updateAt(nx - 1, nx - 2, nx - 1);
                    }
                }
            }

            // On the last step the parallel-region barrier provides the
            // synchronization, so avoid paying for two consecutive barriers.
            if (t + 1 < iterations) {
#pragma omp barrier
            }
        }
    }

    return (iterations & 1) == 0 ? cold : cnew;
}

bool validateResult(const double* const concentration, const size_t volume) {
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    int invalid = 0;

#pragma omp parallel for schedule(static) reduction(min : minValue) reduction(max : maxValue) reduction(| : invalid)
    for (size_t i = 0; i < volume; ++i) {
        const double value = concentration[i];
        if (!std::isfinite(value)) {
            invalid = 1;
        } else {
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
    }

    if (invalid != 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* const programName) {
    printf("Usage: %s [options]\n", programName);
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

    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        printf("Invalid grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    AlignedArray cold(nullptr, &std::free);
    AlignedArray cnew(nullptr, &std::free);
    AlignedArray mu(nullptr, &std::free);
    try {
        cold = allocateArray(gridSize);
        cnew = allocateArray(gridSize);
        mu = allocateArray(gridSize);
    } catch (const std::bad_alloc&) {
        printf("Failed to allocate concentration fields\n");
        return 1;
    }

    printf("Initializing concentration field...\n");
    initializeFields(cold.get(), cnew.get(), mu.get(), gridSize);

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();

    double* const result = runSimulation(
        cold.get(), cnew.get(), mu.get(), nx, ny, nz, iterations, D, dt,
        dx, dy, dz, gamma, eAA, eBB, eAB);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();

    printf("Computation time: %ld ms\n", milliseconds.count());
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults) {
        // The shared result helper consumes a std::vector.  Keep the timed
        // stencil fields uninitialized/aligned for parallel first touch, and
        // materialize this compatibility copy only when output is requested.
        const std::vector<double> resultVector(result, result + gridSize);
        print_results(resultVector, "Concentration");
    }

    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(result, gridSize);
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        }

        printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
