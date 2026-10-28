#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Allocator whose element construction is default-initialization (a no-op for
// double). Grid buffers are therefore not zero-filled serially by the main
// thread; their pages are first touched by the parallel loops, which places
// each page on the NUMA node of the thread that uses it.
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template <typename U>
    struct rebind { using other = DefaultInitAllocator<U>; };

    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

// Grid field storage. Every field is fully written by the parallel passes
// before it is read, so skipping zero-initialization does not change results.
using Field = std::vector<double, DefaultInitAllocator<double>>;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Divide by a squared grid spacing. With UnitSpacing (spacing == 1.0, where
// division is an exact identity in IEEE arithmetic) the division is skipped.
// This keeps the results bit-identical while avoiding costly runtime divides
// inside the OpenMP-outlined loops.
template <bool UnitSpacing>
inline double scaleByInvSpacing(const double v, const double dd) noexcept {
    if constexpr (UnitSpacing) {
        return v;
    } else {
        return v / dd;
    }
}

// Visit every cell of the row (y, z) and pass its Laplacian to op(x, lap).
// The Laplacian uses clamped boundary conditions. The expression order matches
// the original per-cell formulation so results are bit-identical.
// Rows are addressed through raw pointers so the inner loop optimizes well
// inside OpenMP-outlined regions.
template <bool UnitSpacing, typename Op>
inline void forEachCellLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                 const double ddx, const double ddy, const double ddz,
                                 const size_t y, const size_t z, Op&& op) {
    const size_t nxy = nx * ny;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double* rowC  = c + z * nxy + y * nx;
    const double* rowYp = c + z * nxy + yp * nx;
    const double* rowYn = c + z * nxy + yn * nx;
    const double* rowZp = c + zp * nxy + y * nx;
    const double* rowZn = c + zn * nxy + y * nx;

    for (size_t x = 0; x < nx; ++x) {
        const size_t xp = (x < nx - 1) ? x + 1 : x;
        const size_t xn = (x > 0) ? x - 1 : 0;
        const double center = rowC[x];

        const double cxx = scaleByInvSpacing<UnitSpacing>(rowC[xp] + rowC[xn] - 2.0 * center, ddx);
        const double cyy = scaleByInvSpacing<UnitSpacing>(rowYp[x] + rowYn[x] - 2.0 * center, ddy);
        const double czz = scaleByInvSpacing<UnitSpacing>(rowZp[x] + rowZn[x] - 2.0 * center, ddz);

        op(x, cxx + cyy + czz);
    }
}

// Chemical potential kernel, specialized on unit grid spacing
template <bool UnitSpacing>
void chemicalPotentialKernel(const double* __restrict cp, double* __restrict mup,
                             const size_t nx, const size_t ny, const size_t nz,
                             const double ddx, const double ddy, const double ddz,
                             const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    // Each (z, y) row writes only its own cells of mu; c is read-only.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * (nx * ny) + y * nx;
            const double* cRow = cp + base;
            double* muRow = mup + base;

            forEachCellLaplacian<UnitSpacing>(cp, nx, ny, nz, ddx, ddy, ddz, y, z,
                [&](const size_t x, const double lap) {
                    const double cv = cRow[x];
                    muRow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                             + 3.0 * cv + cv * cv * cv
                             - gamma * lap;
                });
        }
    }
}

// Compute chemical potential
void computeChemicalPotential(const Field& c, Field& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double ddx = dx * dx;
    const double ddy = dy * dy;
    const double ddz = dz * dz;
    if (ddx == 1.0 && ddy == 1.0 && ddz == 1.0) {
        chemicalPotentialKernel<true>(c.data(), mu.data(), nx, ny, nz, ddx, ddy, ddz, gamma, e_AA, e_BB, e_AB);
    } else {
        chemicalPotentialKernel<false>(c.data(), mu.data(), nx, ny, nz, ddx, ddy, ddz, gamma, e_AA, e_BB, e_AB);
    }
}

// Cahn-Hilliard update kernel, specialized on unit grid spacing
template <bool UnitSpacing>
void cahnHilliardKernel(double* __restrict cnewP, const double* __restrict coldP, const double* __restrict muP,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double ddx, const double ddy, const double ddz) {
    // Each (z, y) row writes only its own cells of cnew; cold and mu are read-only.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * (nx * ny) + y * nx;
            const double* coldRow = coldP + base;
            double* cnewRow = cnewP + base;

            forEachCellLaplacian<UnitSpacing>(muP, nx, ny, nz, ddx, ddy, ddz, y, z,
                [&](const size_t x, const double lap) {
                    cnewRow[x] = coldRow[x] + dt * D * lap;
                });
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(Field& cnew, const Field& cold,
                        const Field& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double ddx = dx * dx;
    const double ddy = dy * dy;
    const double ddz = dz * dz;
    if (ddx == 1.0 && ddy == 1.0 && ddz == 1.0) {
        cahnHilliardKernel<true>(cnew.data(), cold.data(), mu.data(), nx, ny, nz, D, dt, ddx, ddy, ddz);
    } else {
        cahnHilliardKernel<false>(cnew.data(), cold.data(), mu.data(), nx, ny, nz, D, dt, ddx, ddy, ddz);
    }
}

// Initialize concentration field
void initializeConcentration(Field& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    // Same (z, y) row partitioning as the kernels so each thread first-touches its own rows
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const Field& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
    printf("OpenMP threads: %d\n", omp_get_max_threads());

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
    // Storage is left uninitialized here; initializeConcentration and the
    // first parallel passes write every cell (see DefaultInitAllocator).
    Field cold(gridSize);
    Field cnew(gridSize);
    Field mu(gridSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
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
        // Copy into a standard vector for the shared reporting helper (outside the timed region)
        const std::vector<double> result(cold.begin(), cold.end());
        print_results(result, "Concentration");
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
