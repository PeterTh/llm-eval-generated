#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute a row's Laplacian without recomputing 3D indices per cell.
// Neighbor rows already account for clamped Y and Z boundaries.
inline double computeLaplacian(const double* row, const double* yp, const double* yn,
                              const double* zp, const double* zn,
                              const double dx2, const double dy2, const double dz2,
                              const size_t x, const size_t xp, const size_t xn) {
    const double cxx = (row[xp] + row[xn] - 2.0 * row[x]) / dx2;
    const double cyy = (yp[x] + yn[x] - 2.0 * row[x]) / dy2;
    const double czz = (zp[x] + zn[x] - 2.0 * row[x]) / dz2;
    return cxx + cyy + czz;
}

// Worksharing kernels are called by every thread in the persistent team.
// The implicit barrier at the end of each pass makes all stencil reads safe.
// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            const double* row = c.data() + base;
            const double* yp = row + (y + 1 < ny ? nx : 0);
            const double* yn = row - (y > 0 ? nx : 0);
            const double* zp = row + (z + 1 < nz ? plane : 0);
            const double* zn = row - (z > 0 ? plane : 0);
            double* out = mu.data() + base;
            const auto potential = [&](size_t x, size_t xp, size_t xn) {
                const double cv = row[x];
                return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * computeLaplacian(row, yp, yn, zp, zn, dx2, dy2, dz2, x, xp, xn);
            };
            if (nx > 0) out[0] = potential(0, nx > 1 ? 1 : 0, 0);
            // Handle the two X boundaries separately so the hot SIMD loop has
            // only contiguous loads, with no per-cell boundary tests/gathers.
            #pragma omp simd
            for (size_t x = 1; x < (nx > 0 ? nx - 1 : 0); ++x) {
                out[x] = potential(x, x + 1, x - 1);
            }
            if (nx > 1) out[nx - 1] = potential(nx - 1, nx - 1, nx - 2);
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            const double* row = mu.data() + base;
            const double* yp = row + (y + 1 < ny ? nx : 0);
            const double* yn = row - (y > 0 ? nx : 0);
            const double* zp = row + (z + 1 < nz ? plane : 0);
            const double* zn = row - (z > 0 ? plane : 0);
            const double* old = cold.data() + base;
            double* out = cnew.data() + base;
            const auto update = [&](size_t x, size_t xp, size_t xn) {
                return old[x] + dt * D *
                       computeLaplacian(row, yp, yn, zp, zn, dx2, dy2, dz2, x, xp, xn);
            };
            if (nx > 0) out[0] = update(0, nx > 1 ? 1 : 0, 0);
            #pragma omp simd
            for (size_t x = 1; x < (nx > 0 ? nx - 1 : 0); ++x) {
                out[x] = update(x, x + 1, x - 1);
            }
            if (nx > 1) out[nx - 1] = update(nx - 1, nx - 1, nx - 2);
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for simd schedule(static)
    for (size_t idx = 0; idx < vol; ++idx) {
        // Generate the same deterministic pseudo-random value in [-1, 1].
        const double pseudo = ((((idx + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
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
    
    #pragma omp parallel
    {
        // Each thread swaps its own pointers after the update barrier, avoiding
        // a shared vector swap and an additional synchronization per time step.
        std::vector<double>* current = &cold;
        std::vector<double>* next = &cnew;
        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(*current, mu, nx, ny, nz, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdate(*next, *current, mu, nx, ny, nz, D, dt, dx, dy, dz);
            std::swap(current, next);
        }
    }
    if (iterations > 0 && iterations % 2 != 0) {
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
