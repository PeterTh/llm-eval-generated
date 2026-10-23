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

// Compute chemical potential
void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nxy = nx * ny;
    const size_t nxm1 = nx - 1;
    const size_t nym1 = ny - 1;
    const size_t nzm1 = nz - 1;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    // Parallelize over (z, y) rows: ny*nz independent units of work give
    // good load balance while keeping the inner x loop contiguous.
    const size_t nyz = ny * nz;
    #pragma omp for schedule(static)
    for (size_t zy = 0; zy < nyz; ++zy) {
        const size_t z = zy / ny;
        const size_t y = zy - z * ny;
        const size_t zbase = z * nxy;
        const size_t zpbase = (z < nzm1 ? z + 1 : z) * nxy;
        const size_t znbase = (z > 0 ? z - 1 : 0) * nxy;
        const size_t ybase = zbase + y * nx;
        const size_t ypbase = zbase + (y < nym1 ? y + 1 : y) * nx;
        const size_t ynbase = zbase + (y > 0 ? y - 1 : y) * nx;
        const size_t zypbase = zpbase + y * nx;
        const size_t zynbase = znbase + y * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t idx = ybase + x;
            const size_t xp = (x < nxm1) ? x + 1 : x;
            const size_t xn = (x > 0) ? x - 1 : 0;
            const double cv = c[idx];

            const double cxx = (c[ybase + xp] + c[ybase + xn] - 2.0 * cv) / dx2;
            const double cyy = (c[ypbase + x] + c[ynbase + x] - 2.0 * cv) / dy2;
            const double czz = (c[zypbase + x] + c[zynbase + x] - 2.0 * cv) / dz2;

            mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                     + 3.0 * cv + cv * cv * cv
                     - gamma * (cxx + cyy + czz);
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nxy = nx * ny;
    const size_t nxm1 = nx - 1;
    const size_t nym1 = ny - 1;
    const size_t nzm1 = nz - 1;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double factor = dt * D;

    const size_t nyz = ny * nz;
    #pragma omp for schedule(static)
    for (size_t zy = 0; zy < nyz; ++zy) {
        const size_t z = zy / ny;
        const size_t y = zy - z * ny;
        const size_t zbase = z * nxy;
        const size_t zpbase = (z < nzm1 ? z + 1 : z) * nxy;
        const size_t znbase = (z > 0 ? z - 1 : 0) * nxy;
        const size_t ybase = zbase + y * nx;
        const size_t ypbase = zbase + (y < nym1 ? y + 1 : y) * nx;
        const size_t ynbase = zbase + (y > 0 ? y - 1 : y) * nx;
        const size_t zypbase = zpbase + y * nx;
        const size_t zynbase = znbase + y * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t idx = ybase + x;
            const size_t xp = (x < nxm1) ? x + 1 : x;
            const size_t xn = (x > 0) ? x - 1 : 0;
            const double c0 = mu[idx];

            const double cxx = (mu[ybase + xp] + mu[ybase + xn] - 2.0 * c0) / dx2;
            const double cyy = (mu[ypbase + x] + mu[ynbase + x] - 2.0 * c0) / dy2;
            const double czz = (mu[zypbase + x] + mu[zynbase + x] - 2.0 * c0) / dz2;

            cnew[idx] = cold[idx] + factor * (cxx + cyy + czz);
        }
    }
}

// Initialize concentration field
void initializeConcentration(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    const size_t nyz = ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t zy = 0; zy < nyz; ++zy) {
        const size_t row = zy * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t linear_id = row + x;
            // Generate pseudo-random value in [-1, 1]
            const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
            c[linear_id] = -1.0 + 2.0 * pseudo;
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool invalid = false;
    #pragma omp parallel for reduction(||:invalid) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        invalid = invalid || std::isnan(val) || std::isinf(val);
    }
    if (invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
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
    // Keep a fixed thread team for predictable parallel scaling
    omp_set_dynamic(0);

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
    initializeConcentration(cold.data(), nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run all time steps inside a single persistent parallel region to avoid
    // repeated thread-team startup/teardown costs between iterations.
    #pragma omp parallel
    {
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, nz, dx, dy, dz, 
                                    gamma, e_AA, e_BB, e_AB);
            
            // Update concentration
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, nz, D, dt, dx, dy, dz);
            
            // Swap buffers (implicit barriers around the single region ensure
            // all threads observe the swap before the next time step).
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
