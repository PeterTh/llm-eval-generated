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

// Compute Laplacian with clamped boundary conditions using raw pointers
inline double computeLaplacianPtr(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz,
                                  const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                  const size_t x, const size_t y, const size_t z) noexcept {
    const size_t nxy = nx * ny;
    const size_t idx_c = z * nxy + y * nx + x;

    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double c_center = c[idx_c];

    const double cxx = (c[z * nxy + y * nx + xp] + c[z * nxy + y * nx + xn] -
                        2.0 * c_center) * inv_dx2;
    const double cyy = (c[z * nxy + yp * nx + x] + c[z * nxy + yn * nx + x] -
                        2.0 * c_center) * inv_dy2;
    const double czz = (c[zp * nxy + y * nx + x] + c[zn * nxy + y * nx + x] -
                        2.0 * c_center) * inv_dz2;

    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double inv_dx2, const double inv_dy2, const double inv_dz2,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nxy = nx * ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = z * nxy + y * nx + x;
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacianPtr(c, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double DtD, const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const size_t nxy = nx * ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = z * nxy + y * nx + x;
                cnew[idx] = cold[idx] + DtD *
                           computeLaplacianPtr(mu, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const double inv_vol = 1.0 / static_cast<double>(vol);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((idx + 1) * 1299709) % vol) * inv_vol);
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* data = c.data();

    // Check for NaN or Inf and compute min/max in one parallel pass
    int found_invalid = 0;
    double minVal = c[0];
    double maxVal = c[0];

    #pragma omp parallel
    {
        int local_invalid = 0;
        double local_min = data[0];
        double local_max = data[0];

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            const double val = data[i];
            if (std::isnan(val) || std::isinf(val)) {
                local_invalid = 1;
            }
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }

        #pragma omp critical
        {
            if (local_invalid) found_invalid = 1;
            if (local_min < minVal) minVal = local_min;
            if (local_max > maxVal) maxVal = local_max;
        }
    }

    if (found_invalid) {
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
    
    // Precompute constants for performance
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double DtD = dt * D;
    
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
    
    double* cold_ptr = cold.data();
    double* cnew_ptr = cnew.data();
    double* mu_ptr = mu.data();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold_ptr, mu_ptr, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdate(cnew_ptr, cold_ptr, mu_ptr, nx, ny, nz, DtD, inv_dx2, inv_dy2, inv_dz2);
        
        // Swap buffers
        std::swap(cold_ptr, cnew_ptr);
    }
    
    // Update vectors to point to final data
    if (cold_ptr != cold.data()) {
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
