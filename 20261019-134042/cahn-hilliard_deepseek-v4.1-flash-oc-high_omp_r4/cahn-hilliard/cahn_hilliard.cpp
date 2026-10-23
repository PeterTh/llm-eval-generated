#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Chemical potential from concentration and its Laplacian
inline double chemicalPotentialValue(const double cv, const double lap,
                                     const double gamma,
                                     const double e_AA, const double e_BB, const double e_AB) noexcept {
    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
         + 3.0 * cv + cv * cv * cv
         - gamma * lap;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nxy = nx * ny;
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double* __restrict cdat = c.data();
    double* __restrict mudat = mu.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * nxy + y * nx;
            const size_t zmbase = ((z > 0) ? z - 1 : z) * nxy + y * nx;
            const size_t zpbase = ((z + 1 < nz) ? z + 1 : z) * nxy + y * nx;
            const size_t ymbase = z * nxy + ((y > 0) ? y - 1 : y) * nx;
            const size_t ypbase = z * nxy + ((y + 1 < ny) ? y + 1 : y) * nx;

            // Interior cells: no boundary branches, vectorizable
            const size_t xlast = (nx > 0) ? nx - 1 : 0;
            #pragma omp simd
            for (size_t x = 1; x < xlast; ++x) {
                const double cv = cdat[base + x];
                const double lap =
                    (cdat[base + x + 1] + cdat[base + x - 1] - 2.0 * cv) * invdx2 +
                    (cdat[ypbase + x] + cdat[ymbase + x] - 2.0 * cv) * invdy2 +
                    (cdat[zpbase + x] + cdat[zmbase + x] - 2.0 * cv) * invdz2;
                mudat[base + x] = chemicalPotentialValue(cv, lap, gamma, e_AA, e_BB, e_AB);
            }

            // Boundary cells in x (clamped)
            for (size_t xi = 0; xi < 2; ++xi) {
                if (xi == 1 && nx == 1) break;
                const size_t x = (xi == 0) ? 0 : nx - 1;
                const size_t xm = (x > 0) ? x - 1 : 0;
                const size_t xp = (x + 1 < nx) ? x + 1 : x;
                const double cv = cdat[base + x];
                const double lap =
                    (cdat[base + xp] + cdat[base + xm] - 2.0 * cv) * invdx2 +
                    (cdat[ypbase + x] + cdat[ymbase + x] - 2.0 * cv) * invdy2 +
                    (cdat[zpbase + x] + cdat[zmbase + x] - 2.0 * cv) * invdz2;
                mudat[base + x] = chemicalPotentialValue(cv, lap, gamma, e_AA, e_BB, e_AB);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nxy = nx * ny;
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double factor = dt * D;
    const double* __restrict coldat = cold.data();
    const double* __restrict mudat = mu.data();
    double* __restrict cnewdat = cnew.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * nxy + y * nx;
            const size_t zmbase = ((z > 0) ? z - 1 : z) * nxy + y * nx;
            const size_t zpbase = ((z + 1 < nz) ? z + 1 : z) * nxy + y * nx;
            const size_t ymbase = z * nxy + ((y > 0) ? y - 1 : y) * nx;
            const size_t ypbase = z * nxy + ((y + 1 < ny) ? y + 1 : y) * nx;

            // Interior cells: no boundary branches, vectorizable
            const size_t xlast = (nx > 0) ? nx - 1 : 0;
            #pragma omp simd
            for (size_t x = 1; x < xlast; ++x) {
                const double lap =
                    (mudat[base + x + 1] + mudat[base + x - 1] - 2.0 * mudat[base + x]) * invdx2 +
                    (mudat[ypbase + x] + mudat[ymbase + x] - 2.0 * mudat[base + x]) * invdy2 +
                    (mudat[zpbase + x] + mudat[zmbase + x] - 2.0 * mudat[base + x]) * invdz2;
                cnewdat[base + x] = coldat[base + x] + factor * lap;
            }

            // Boundary cells in x (clamped)
            for (size_t xi = 0; xi < 2; ++xi) {
                if (xi == 1 && nx == 1) break;
                const size_t x = (xi == 0) ? 0 : nx - 1;
                const size_t xm = (x > 0) ? x - 1 : 0;
                const size_t xp = (x + 1 < nx) ? x + 1 : x;
                const double lap =
                    (mudat[base + xp] + mudat[base + xm] - 2.0 * mudat[base + x]) * invdx2 +
                    (mudat[ypbase + x] + mudat[ymbase + x] - 2.0 * mudat[base + x]) * invdy2 +
                    (mudat[zpbase + x] + mudat[zmbase + x] - 2.0 * mudat[base + x]) * invdz2;
                cnewdat[base + x] = coldat[base + x] + factor * lap;
            }
        }
    }
}

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
    if (c.empty()) {
        printf("Validation failed: empty concentration field\n");
        return false;
    }

    // Check for NaN or Inf and compute range in a single parallel pass
    int hasNonFinite = 0;
    double minVal = c[0];
    double maxVal = c[0];

    #pragma omp parallel for reduction(| : hasNonFinite) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) {
            hasNonFinite = 1;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (hasNonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
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

    // Ensure every run uses OpenMP with a stable thread count
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
